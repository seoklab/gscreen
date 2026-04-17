from pathlib import Path

import pandas as pd
from sklearn import metrics as skmetrics
from tqdm import tqdm
from typer import Typer

from shared_metrics import METHOD_SLUG_MAP, enrichment_factor

app = Typer(pretty_exceptions_enable=False)


def _load_gscreen_values(bench_home: Path):
    scores = pd.read_csv(bench_home / "gscreen.csv")
    scores = scores.rename(columns={"score": "GS-SP"})
    scores = scores[["target", "id", "is_active", "GS-SP"]]
    scores["dataset"] = bench_home.name
    scores["id"] = scores["id"].astype(str).str.strip()
    return scores


def _load_method_values(db_home: Path, method: str):
    method_name = METHOD_SLUG_MAP[method]
    scores: list[pd.DataFrame] = []
    for f in tqdm(list(db_home.glob(f"{method}/outputs/*/scores.csv"))):
        df = pd.read_csv(f)
        df["target"] = f.parent.name
        df = df.rename(columns={"score": method_name})
        scores.append(df[["target", "id", "is_active", method_name]])
    df = pd.concat(scores, ignore_index=True)

    df["dataset"] = db_home.name
    df["id"] = df["id"].astype(str).str.strip()
    return df


def _summarize_both(
    group: pd.DataFrame,
    score_col: str,
    active_col: str,
    ratios: list[float],
    metric_names: list[str],
):
    return {
        metric_names[0]: skmetrics.roc_auc_score(
            group[active_col], group[score_col]
        ),
        **{
            name: enrichment_factor(
                group[active_col],
                group[score_col],
                ratio=r,
                strict_mode=False,
            )
            for name, r in zip(metric_names[1:], ratios)
        },
        **{
            name: enrichment_factor(
                group[active_col],
                group[score_col],
                ratio=r,
                strict_mode=True,
            )
            for name, r in zip(metric_names[1 + len(ratios) :], ratios)
        },
    }


def _compute_metrics(
    df: pd.DataFrame,
    methods: list[str],
    ratios: list[float],
    metric_cols: list[str],
):
    all_metrics = []
    for (db, target), group in df.groupby(["dataset", "target"]):
        metrics = _summarize_both(
            group,
            score_col="GS-SP",
            active_col="is_active",
            ratios=ratios,
            metric_names=metric_cols,
        )
        metrics["dataset"] = db
        metrics["target"] = target
        metrics["method"] = "GS-SP"
        all_metrics.append(metrics)

        for method in methods:
            method = METHOD_SLUG_MAP[method]
            valid = group[group[method].notna()].copy()
            if valid.empty:
                continue

            mult = valid[method].max() * 1000
            valid["combined"] = valid["GS-SP"] * mult + valid[method]
            metrics = _summarize_both(
                valid,
                score_col="combined",
                active_col="is_active",
                ratios=ratios,
                metric_names=metric_cols,
            )
            metrics["dataset"] = db
            metrics["target"] = target
            metrics["method"] = method
            all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


@app.command()
def main(
    gscreen_home: Path,
    output: Path = Path("tiebreaking"),
    bench_home: Path = Path.home() / "benchmark",
    methods: str = "autodock-vina,pharmagist",
    db_names: str = "dud-e,lit-pcba,muv",
    ef_levels: str = "0.001,0.01,0.05",
):
    methods = methods.split(",")

    all_scores: list[pd.DataFrame] = []
    for db in db_names.split(","):
        gscreen_root = gscreen_home / db
        bench_root = bench_home / db

        scores = _load_gscreen_values(gscreen_root)
        for method in methods:
            scores = pd.merge(
                scores,
                _load_method_values(bench_root, method),
                on=["dataset", "target", "id", "is_active"],
                how="left",
            )
        scores["dataset"] = db.upper()
        all_scores.append(scores)

    df = pd.concat(all_scores, ignore_index=True)

    ratios = list(map(float, ef_levels.split(",")))
    metric_cols = [
        "AUROC",
        *[f"EF{ratio * 100:1g}%" for ratio in ratios],
        *[f"SEF{ratio * 100:1g}%" for ratio in ratios],
    ]
    metrics = _compute_metrics(df, methods, ratios, metric_cols)

    output.mkdir(exist_ok=True)
    metrics.to_csv(output / "tiebreaking_metrics.csv", index=False)


if __name__ == "__main__":
    app()
