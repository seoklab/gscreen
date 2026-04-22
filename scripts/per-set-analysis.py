import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from joblib import Parallel, delayed
from typer import Typer

from shared_metrics import (
    BASELINE_METHODS,
    GSCREEN_METHODS,
    METHOD_SLUG_MAP,
    TIEBREAK_METHODS,
    compute_metrics,
    load_gscreen_scores,
    load_method_scores,
)

app = Typer(pretty_exceptions_enable=False)


def _score_one_target(
    df: pd.DataFrame,
    method: str,
    target: str,
    score_col: str,
    active_col: str,
    ratios: list[float],
    metric_cols: list[str],
    strict_ef: bool,
):
    invalid = df[score_col].isna()
    df = df[~invalid]
    ndrop = invalid.sum()
    if ndrop > 0:
        typer.echo(
            f"Warning: Dropping {ndrop} rows with NaN scores for {method} on {target}",
            err=True,
        )
    if df.empty:
        typer.echo(
            f"Warning: No valid scores for {method} on {target} after dropping NaNs",
            err=True,
        )
        return []

    vals = compute_metrics(
        df,
        score_col,
        active_col,
        ratios,
        metric_cols,
        strict_mode=strict_ef,
    )
    return [(method, target, metric, score) for metric, score in vals.items()]


def _summarize_scores(
    dfs: pd.DataFrame | dict[str, pd.DataFrame],
    method: str,
    ratios: list[float],
    metric_cols: list[str],
    score_col: str = "score",
    active_col: str = "is_active",
    strict_ef: bool = True,
    nproc: int = 8,
) -> pd.DataFrame:
    if not isinstance(dfs, dict):
        dfs = dict(list(dfs.groupby("target")))

    rows = Parallel(n_jobs=nproc)(
        delayed(_score_one_target)(
            df,
            method,
            target,
            score_col,
            active_col,
            ratios,
            metric_cols,
            strict_ef,
        )
        for target, df in sorted(dfs.items(), key=lambda x: x[0])
    )

    rows = list(itertools.chain.from_iterable(rows))
    return pd.DataFrame(rows, columns=["method", "target", "metric", "score"])


def target_average_tani_ratio(
    data: pd.DataFrame,
    target: str,
    cutoff: int,
    sort_by: str = "score",
    strict_mode: bool = True,
):
    tgt = data[data["target"] == target]
    if tgt.empty:
        return np.nan

    scores = tgt[sort_by].values
    ecfp4 = tgt["ecfp4"].values

    n_select = max(1, cutoff)
    kth = len(scores) - n_select
    threshold = np.partition(scores, kth)[kth]

    if strict_mode:
        above = scores >= threshold
        n_above = above.sum()
        if n_above == 0:
            return np.nan
        return (ecfp4[above].sum() / n_above) / ecfp4.mean()

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
    strict_ef: bool = True,
    nproc: int = 8,
):
    raw = load_method_scores(bench_home, method, skip_missing=skip_missing)

    method_metrics = _summarize_scores(
        raw,
        method=METHOD_SLUG_MAP.get(method, method),
        ratios=ratios,
        metric_cols=metric_cols,
        strict_ef=strict_ef,
        nproc=nproc,
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
    strict_ef: bool = True,
    include_tiebreak: bool = False,
    nproc: int = 8,
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

    typer.echo(f"EF Strict mode: {'ON' if strict_ef else 'OFF'}")

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
        strict_ef=strict_ef,
        nproc=nproc,
    )
    gsp_metrics = _summarize_scores(
        gscreen_scores,
        method="GS-P",
        ratios=ratios,
        metric_cols=metric_cols,
        score_col="pharma",
        strict_ef=strict_ef,
        nproc=nproc,
    )
    gssp_metrics = _summarize_scores(
        gscreen_scores,
        method="GS-SP",
        ratios=ratios,
        metric_cols=metric_cols,
        strict_ef=strict_ef,
        nproc=nproc,
    )

    gscreen_scores = pd.concat(gscreen_scores.values(), ignore_index=True)
    gscreen_scores.to_csv(results / "gscreen.csv", index=False)

    target_sims = gscreen_scores.set_index(["target", "id"])[["ecfp4"]]

    methods = GSCREEN_METHODS
    all_metrics = [gss_metrics, gsp_metrics, gssp_metrics]
    if not only_gscreen:
        typer.echo(f"Loading other method scores for {db}...")

        methods = [*GSCREEN_METHODS, *BASELINE_METHODS]
        if include_tiebreak:
            methods += TIEBREAK_METHODS

        ls_scores, ls_metrics = _load_method_scores(
            bench_home,
            "ls-align",
            ratios,
            metric_cols,
            target_sims,
            skip_missing=skip_missing,
            strict_ef=strict_ef,
            nproc=nproc,
        )
        pg_scores, pg_metrics = _load_method_scores(
            bench_home,
            "pharmagist",
            ratios,
            metric_cols,
            target_sims,
            skip_missing=skip_missing,
            strict_ef=strict_ef,
            nproc=nproc,
        )
        vina_scores, vina_metrics = _load_method_scores(
            bench_home,
            "autodock-vina",
            ratios,
            metric_cols,
            target_sims,
            skip_missing=skip_missing,
            strict_ef=strict_ef,
            nproc=nproc,
        )
        all_metrics += [ls_metrics, pg_metrics, vina_metrics]

        if include_tiebreak:
            typer.echo("Computing tiebreaking scores...")

            tiebreak_scores = pd.merge(
                gscreen_scores,
                pg_scores[["target", "id", "score"]].rename(
                    columns={"score": "pharmagist"}
                ),
                on=["target", "id"],
                how="left",
            )
            tiebreak_scores = pd.merge(
                tiebreak_scores,
                vina_scores[["target", "id", "score"]].rename(
                    columns={"score": "autodock-vina"}
                ),
                on=["target", "id"],
                how="left",
            )

            div = tiebreak_scores.groupby("target")[
                ["pharmagist", "autodock-vina"]
            ].transform(lambda x: 1 / max(np.nanmax(x, initial=0) * 1000, 1))
            for method in ["pharmagist", "autodock-vina"]:
                tiebreak_scores[f"pharma + {method}"] = (
                    tiebreak_scores["pharma"]
                    + tiebreak_scores[method] * div[method]
                )
                tiebreak_scores[f"score + {method}"] = (
                    tiebreak_scores["score"]
                    + tiebreak_scores[method] * div[method]
                )
            tiebreak_scores.to_csv(
                results / "gscreen-tiebreak.csv", index=False
            )

            gsp_pg_tiebreak = _summarize_scores(
                tiebreak_scores,
                "GS-P + PG",
                ratios,
                metric_cols,
                score_col="pharma + pharmagist",
                strict_ef=strict_ef,
                nproc=nproc,
            )
            gsp_vina_tiebreak = _summarize_scores(
                tiebreak_scores,
                "GS-P + Vina",
                ratios,
                metric_cols,
                score_col="pharma + autodock-vina",
                strict_ef=strict_ef,
                nproc=nproc,
            )
            gssp_pg_tiebreak = _summarize_scores(
                tiebreak_scores,
                "GS-SP + PG",
                ratios,
                metric_cols,
                score_col="score + pharmagist",
                strict_ef=strict_ef,
                nproc=nproc,
            )
            gssp_vina_tiebreak = _summarize_scores(
                tiebreak_scores,
                "GS-SP + Vina",
                ratios,
                metric_cols,
                score_col="score + autodock-vina",
                strict_ef=strict_ef,
                nproc=nproc,
            )
            all_metrics += [
                gsp_pg_tiebreak,
                gsp_vina_tiebreak,
                gssp_pg_tiebreak,
                gssp_vina_tiebreak,
            ]

    all_metrics = pd.concat(all_metrics, ignore_index=True)
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
            methods,
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
                    strict_mode=strict_ef,
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
                    strict_mode=strict_ef,
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
                    strict_mode=strict_ef,
                ),
            )
        )

        if not only_gscreen:
            averages.append(
                (
                    "LA",
                    target,
                    target_average_tani_ratio(
                        ls_scores,
                        target,
                        target_cutoff[target],
                        strict_mode=strict_ef,
                    ),
                )
            )

            averages.append(
                (
                    "PG",
                    target,
                    target_average_tani_ratio(
                        pg_scores,
                        target,
                        target_cutoff[target],
                        strict_mode=strict_ef,
                    ),
                )
            )

            averages.append(
                (
                    "Vina",
                    target,
                    target_average_tani_ratio(
                        vina_scores,
                        target,
                        target_cutoff[target],
                        strict_mode=strict_ef,
                    ),
                )
            )

            if include_tiebreak:
                averages.append(
                    (
                        "GS-P + PG",
                        target,
                        target_average_tani_ratio(
                            tiebreak_scores,
                            target,
                            target_cutoff[target],
                            "pharma + pharmagist",
                            strict_mode=strict_ef,
                        ),
                    )
                )
                averages.append(
                    (
                        "GS-P + Vina",
                        target,
                        target_average_tani_ratio(
                            tiebreak_scores,
                            target,
                            target_cutoff[target],
                            "pharma + autodock-vina",
                            strict_mode=strict_ef,
                        ),
                    )
                )
                averages.append(
                    (
                        "GS-SP + PG",
                        target,
                        target_average_tani_ratio(
                            tiebreak_scores,
                            target,
                            target_cutoff[target],
                            "score + pharmagist",
                            strict_mode=strict_ef,
                        ),
                    )
                )
                averages.append(
                    (
                        "GS-SP + Vina",
                        target,
                        target_average_tani_ratio(
                            tiebreak_scores,
                            target,
                            target_cutoff[target],
                            "score + autodock-vina",
                            strict_mode=strict_ef,
                        ),
                    )
                )

    averages = pd.DataFrame(averages, columns=["method", "target", "ratio"])
    averages = averages[averages["ratio"].notna()]
    averages.to_csv(results / "enrichment.csv", index=False)

    pd.options.display.float_format = None
    typer.echo(averages.groupby("method", sort=False).count())

    pd.options.display.float_format = "{:+.1%}".format
    typer.echo(
        averages.groupby("method", sort=False).mean(numeric_only=True) - 1
    )


if __name__ == "__main__":
    app()
