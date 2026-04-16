from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from shared_metrics import (
    ALL_METHODS,
    GSCREEN_METHODS,
    METHOD_SLUG_MAP,
    compute_metrics,
    load_gscreen_scores,
    load_method_scores,
)
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


def _summarize_scores(
    dfs: dict[str, pd.DataFrame],
    method: str,
    ratios: list[float],
    metric_cols: list[str],
    score_col: str = "score",
    active_col: str = "is_active",
) -> pd.DataFrame:
    rows = []
    for target, df in sorted(dfs.items(), key=lambda x: x[0]):
        vals = compute_metrics(df, score_col, active_col, ratios, metric_cols)
        rows.extend(
            (method, target, metric, score) for metric, score in vals.items()
        )
    return pd.DataFrame(rows, columns=["method", "target", "metric", "score"])


def target_average_tani_ratio(
    data: pd.DataFrame,
    target: str,
    cutoff: int,
    sort_by: str = "score",
):
    tgt = data[data["target"] == target]
    scores = tgt[sort_by].values
    ecfp4 = tgt["ecfp4"].values

    n_select = max(1, cutoff)
    kth = len(scores) - n_select
    threshold = np.partition(scores, kth)[kth]

    above = scores > threshold
    tied = scores == threshold

    n_above = np.sum(above)
    n_tied = np.sum(tied)
    n_from_tied = n_select - n_above

    expected_ecfp4_sum = ecfp4[above].sum() + ecfp4[tied].sum() * (
        n_from_tied / n_tied
    )
    return (expected_ecfp4_sum / n_select) / ecfp4.mean()


def _load_method_scores(
    bench_home: Path,
    method: str,
    ratios: list[float],
    metric_cols: list[str],
    similarities: pd.DataFrame,
    skip_missing: bool = False,
):
    raw = load_method_scores(bench_home, method, skip_missing=skip_missing)

    method_metrics = _summarize_scores(
        raw,
        method=METHOD_SLUG_MAP.get(method, method),
        ratios=ratios,
        metric_cols=metric_cols,
    )

    method_scores: pd.DataFrame = pd.concat(
        raw.values(),
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
    ef_ratios: str = "0.001,0.01,0.05",
    similarity_enrichment_cutoff: float = 0.01,
    skip_missing: bool = False,
    only_gscreen: bool = False,
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
    gscreen_scores = load_gscreen_scores(
        results,
        db_home,
        fallback=fallback_home,
    )

    gss_metrics = _summarize_scores(
        gscreen_scores,
        method="GS-S",
        ratios=ratios,
        metric_cols=metric_cols,
        score_col="shape",
    )
    gsp_metrics = _summarize_scores(
        gscreen_scores,
        method="GS-P",
        ratios=ratios,
        metric_cols=metric_cols,
        score_col="pharma",
    )
    gssp_metrics = _summarize_scores(
        gscreen_scores,
        method="GS-SP",
        ratios=ratios,
        metric_cols=metric_cols,
    )

    gscreen_scores = pd.concat(gscreen_scores.values(), ignore_index=True)
    gscreen_scores.to_csv(results / "gscreen.csv", index=False)

    target_sims = gscreen_scores.set_index(["target", "id"])[["ecfp4"]]

    if only_gscreen:
        all_metrics = pd.concat(
            [gss_metrics, gsp_metrics, gssp_metrics],
            ignore_index=True,
        )
    else:
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
            (GSCREEN_METHODS if only_gscreen else ALL_METHODS),
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

    if not only_gscreen:
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
