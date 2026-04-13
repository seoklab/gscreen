import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn import metrics
from typer import Typer

app = Typer(pretty_exceptions_enable=False)

all_methods = [
    "GS-S",
    "GS-P",
    "GS-SP",
    "Flexi-LS-align",
    "PharmaGist",
    "Autodock Vina",
    "ECFP4",
]
all_methods_short = [
    "GS-S",
    "GS-P",
    "GS-SP",
    "LA",
    "PG",
    "Vina",
    "ECFP4",
]
_method_name_map = {
    "ls-align": "Flexi-LS-align",
    "pharmagist": "PharmaGist",
    "autodock-vina": "Autodock Vina",
}


def ecfp4_weight(df):
    ecfp4 = df["ecfp4"].to_numpy()
    return np.clip(6 * (ecfp4 - 0.1), 0, 0.9)


def enrichment_factor(labels, scores, ratio: float = 0.01):
    labels = np.array(labels)
    scores = np.array(scores)

    total_len = len(scores)

    kth = math.floor((1 - ratio) * total_len)
    idxs = np.argpartition(scores, kth)

    total_actives = sum(labels)
    selected_actives = sum(labels[idx] for idx in idxs[kth:])
    return (selected_actives / (total_len - kth)) / (total_actives / total_len)


def summarize_scores(
    dfs: dict[str, pd.DataFrame],
    method: str,
    ratios: list[float],
    metric_cols: list[str],
    score_col: str = "score",
    active_col: str = "is_active",
):
    summary = {}
    for target, df in sorted(dfs.items(), key=lambda x: x[0]):
        summary[target] = [
            metrics.roc_auc_score(df[active_col], df[score_col]),
            *(
                enrichment_factor(df[active_col], df[score_col], ratio=ratio)
                for ratio in ratios
            ),
        ]

    summary = pd.DataFrame.from_dict(
        summary,
        orient="index",
        columns=metric_cols,
    )
    summary = summary.rename_axis("target").reset_index(drop=False)
    summary = summary.melt(
        id_vars=["target"], var_name="metric", value_name="score"
    )
    summary.insert(0, "method", method)
    return summary


def target_average_tani_ratio(
    data: pd.DataFrame,
    target: str,
    cutoff: int,
    sort_by: str = "score",
):
    tgt = data[data["target"] == target].sort_values(sort_by, ascending=False)
    selected = tgt.iloc[:cutoff]
    return selected["ecfp4"].mean() / tgt["ecfp4"].mean()


def _load_gscreen_scores(
    results: Path,
    db_home: Path,
    fallback: Optional[Path] = None,
):
    gscreen_scores: dict[str, pd.DataFrame] = {}

    for db_target in db_home.iterdir():
        if not db_target.is_dir():
            continue

        target = results / db_target.name
        score_csv = target / "scores.csv"
        if not score_csv.is_file() and fallback is not None:
            typer.echo(
                f"WARNING: Missing scores for {db_target.name}, trying fallback",
                err=True,
            )
            score_csv = fallback / db_target.name / "scores.csv"

        df = pd.read_csv(score_csv)
        if "is_active" not in df.columns:
            df["is_active"] = df["type"] == "active"
            df = df.drop(columns=["type"])
            df = df.rename(
                columns={
                    "pharma_score": "pharma",
                    "shape_score": "shape",
                    "tani_sim": "ecfp4",
                }
            )

        weight = ecfp4_weight(df)
        df["score"] = df["shape"] * weight + df["pharma"] * (1 - weight)
        df["target"] = db_target.name

        gscreen_scores[db_target.name] = df

    return gscreen_scores


def _load_method_scores(
    bench_home: Path,
    method: str,
    ratios: list[float],
    metric_cols: list[str],
    similarities: pd.DataFrame,
    skip_missing: bool = False,
):
    method_scores = {}
    for target in bench_home.joinpath(method, "outputs").iterdir():
        if not target.is_dir():
            continue

        try:
            df = pd.read_csv(target / "scores.csv", index_col=0)
        except FileNotFoundError:
            if skip_missing:
                typer.echo(
                    f"WARNING: Missing scores for {target.name} in {method}, skipping",
                    err=True,
                )
                continue
            raise

        df["id"] = df["id"].astype(str).str.strip()
        df["target"] = target.name
        method_scores[target.name] = df

    method_metrics = summarize_scores(
        method_scores,
        method=_method_name_map.get(method, method),
        ratios=ratios,
        metric_cols=metric_cols,
    )

    method_scores: pd.DataFrame = pd.concat(
        method_scores.values(),
        ignore_index=True,
    )
    method_scores = method_scores.join(
        similarities, on=["target", "id"], how="inner"
    )

    return method_scores, method_metrics


@app.command()
def main(
    results: Path,
    db_home: Path = Path.home() / "db",
    bench_home: Path = Path.home() / "benchmark",
    fallback_home: Optional[Path] = None,
    ef_ratios: str = "0.001,0.01,0.05,0.1",
    similarity_enrichment_cutoff: float = 0.01,
    skip_missing: bool = False,
):
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Helvetica Neue",
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )

    db = results.name
    db_home = db_home / db
    bench_home = bench_home / db
    if fallback_home is not None:
        fallback_home = fallback_home / db

    ratios = list(map(float, ef_ratios.split(",")))
    metric_cols = ["aucroc", *[f"ef {ratio * 100:1g}%" for ratio in ratios]]

    typer.echo(f"Loading gscreen scores for {db}...")
    gscreen_scores = _load_gscreen_scores(
        results,
        db_home,
        fallback=fallback_home,
    )

    gss_metrics = summarize_scores(
        gscreen_scores,
        method="GS-S",
        ratios=ratios,
        metric_cols=metric_cols,
        score_col="shape",
    )
    gsp_metrics = summarize_scores(
        gscreen_scores,
        method="GS-P",
        ratios=ratios,
        metric_cols=metric_cols,
        score_col="pharma",
    )
    gssp_metrics = summarize_scores(
        gscreen_scores,
        method="GS-SP",
        ratios=ratios,
        metric_cols=metric_cols,
    )
    ecfp4_metrics = summarize_scores(
        gscreen_scores,
        method="ECFP4",
        ratios=ratios,
        metric_cols=metric_cols,
        score_col="ecfp4",
    )

    gscreen_scores = pd.concat(gscreen_scores.values(), ignore_index=True)
    gscreen_scores.to_csv(results / "gscreen.csv", index=False)

    gscreen_scores["id"] = gscreen_scores["id"].astype(str).str.strip()
    target_sims = gscreen_scores.set_index(["target", "id"])[["ecfp4"]]

    typer.echo(f"Loading other method scores for {db}...")
    ls_scores, ls_metrics = _load_method_scores(
        bench_home,
        "ls-align",
        ratios,
        metric_cols,
        target_sims,
        skip_missing=skip_missing,
    )
    pg_scores, pg_metrics = _load_method_scores(
        bench_home,
        "pharmagist",
        ratios,
        metric_cols,
        target_sims,
        skip_missing=skip_missing,
    )
    vina_scores, vina_metrics = _load_method_scores(
        bench_home,
        "autodock-vina",
        ratios,
        metric_cols,
        target_sims,
        skip_missing=skip_missing,
    )

    all_metrics = pd.concat(
        [
            gss_metrics,
            gsp_metrics,
            gssp_metrics,
            ls_metrics,
            pg_metrics,
            vina_metrics,
            ecfp4_metrics,
        ],
        ignore_index=True,
    )
    all_metrics.to_csv(results / "scores.csv", index=False)

    typer.echo("Summary:")
    typer.echo(all_metrics.groupby("method")["target"].nunique())

    pd.options.display.float_format = "{:.2f}".format
    typer.echo(
        all_metrics.groupby(["method", "metric"])
        .mean(numeric_only=True)
        .reorder_levels([1, 0])
        .loc[
            ["aucroc", *[f"ef {ratio * 100:1g}%" for ratio in ratios]],
            all_methods,
            :,
        ]
    )

    target_cutoff = {
        t: round(len(df) * similarity_enrichment_cutoff)
        for t, df in gscreen_scores.groupby("target")
    }

    averages = []
    for target in gscreen_scores["target"].unique():
        averages.append(
            (
                "GS-S",
                target,
                target_average_tani_ratio(
                    gscreen_scores,
                    target,
                    target_cutoff[target],
                    "shape",
                ),
            )
        )
        averages.append(
            (
                "GS-P",
                target,
                target_average_tani_ratio(
                    gscreen_scores,
                    target,
                    target_cutoff[target],
                    "pharma",
                ),
            )
        )
        averages.append(
            (
                "GS-SP",
                target,
                target_average_tani_ratio(
                    gscreen_scores,
                    target,
                    target_cutoff[target],
                ),
            )
        )

    for target in ls_scores["target"].unique():
        averages.append(
            (
                "LA",
                target,
                target_average_tani_ratio(
                    ls_scores,
                    target,
                    target_cutoff[target],
                ),
            )
        )

    for target in pg_scores["target"].unique():
        averages.append(
            (
                "PG",
                target,
                target_average_tani_ratio(
                    pg_scores,
                    target,
                    target_cutoff[target],
                ),
            )
        )

    for target in vina_scores["target"].unique():
        averages.append(
            (
                "Vina",
                target,
                target_average_tani_ratio(
                    vina_scores,
                    target,
                    target_cutoff[target],
                ),
            )
        )

    averages = pd.DataFrame(averages, columns=["method", "target", "ratio"])
    averages.to_csv(results / "enrichment.csv", index=False)

    pd.options.display.float_format = None
    typer.echo(averages.groupby("method", sort=False).count())

    pd.options.display.float_format = "{:+.1%}".format
    typer.echo(
        averages.groupby("method", sort=False).mean(numeric_only=True) - 1
    )


if __name__ == "__main__":
    app()
