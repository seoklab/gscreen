"""Cross-validation analysis of virtual screening performance.

Loads screening results produced with multiple reference structures per target,
computes per-reference metrics (ROC-AUC, EF), assesses within-target stability,
and performs paired statistical tests (Friedman, post-hoc Wilcoxon) between
methods, plus a CV-based stability comparison.  Generates publication-ready
violin plots with significance brackets.
"""

from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy import stats
from statsmodels.stats.multitest import multipletests

from shared_metrics import (
    ALL_METHODS,
    BASELINE_METHODS,
    GSCREEN_METHODS,
    METHOD_SLUG_MAP,
    TICK_LABELS,
    compute_metrics,
    load_gscreen_scores,
    load_method_scores,
)

app = typer.Typer(pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------
# Directory name parsing
# ---------------------------------------------------------------------------


def _parse_target_pdbid(dirname: str) -> tuple[str, str]:
    """Split 'TARGET-PDBID' on the last hyphen (handles targets like ESR1_ago)."""
    idx = dirname.rfind("-")
    if idx < 0:
        raise ValueError(
            f"Cannot parse target-pdbid from directory name: {dirname}"
        )
    return dirname[:idx], dirname[idx + 1 :]


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------


def _load_gscreen_scores(
    results: Path, db_home: Path
) -> dict[str, pd.DataFrame]:
    scores = load_gscreen_scores(results, db_home)
    for key, df in scores.items():
        target, pdbid = _parse_target_pdbid(key)
        df["target"] = target
        df["pdbid"] = pdbid
    return scores


def _load_method_scores(
    bench_home: Path,
    method: str,
    skip_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    scores = load_method_scores(bench_home, method, skip_missing=skip_missing)
    for key, df in scores.items():
        target, pdbid = _parse_target_pdbid(key)
        df["target"] = target
        df["pdbid"] = pdbid
    return scores


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


def _collect_all_metrics(
    gscreen_scores: dict[str, pd.DataFrame],
    external_methods: dict[str, dict[str, pd.DataFrame]],
    ratios: list[float],
    metric_names: list[str],
) -> pd.DataFrame:
    """Build a long-form DataFrame: target, pdbid, method, metric, score."""
    rows: list[dict] = []

    gscreen_submethods = {
        "GS-S": "shape",
        "GS-P": "pharma",
        "GS-SP": "score",
    }

    for key, df in gscreen_scores.items():
        target, pdbid = _parse_target_pdbid(key)
        for method_name, score_col in gscreen_submethods.items():
            vals = compute_metrics(
                df, score_col, "is_active", ratios, metric_names
            )
            for metric, score in vals.items():
                rows.append(
                    {
                        "target": target,
                        "pdbid": pdbid,
                        "method": method_name,
                        "metric": metric,
                        "score": score,
                    }
                )

    for method_slug, method_scores in external_methods.items():
        method_name = METHOD_SLUG_MAP.get(method_slug, method_slug)
        for key, df in method_scores.items():
            target, pdbid = _parse_target_pdbid(key)
            vals = compute_metrics(
                df, "score", "is_active", ratios, metric_names
            )
            for metric, score in vals.items():
                rows.append(
                    {
                        "target": target,
                        "pdbid": pdbid,
                        "method": method_name,
                        "metric": metric,
                        "score": score,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def _stability_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per (target, method, metric): mean, std, count across pdbids."""
    return (
        all_metrics.groupby(["target", "method", "metric"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _aggregate_per_target(all_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (target, method, metric), averaging over pdbids."""
    return (
        all_metrics.groupby(["target", "method", "metric"])["score"]
        .mean()
        .reset_index()
    )


def _friedman_test(
    agg: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    """Friedman test per metric across methods.

    Only uses targets where all requested methods are present.
    """
    rows: list[dict] = []
    for metric, mdf in agg.groupby("metric"):
        pivot = mdf.pivot_table(
            index="target",
            columns="method",
            values="score",
        )
        present = [m for m in methods if m in pivot.columns]
        if len(present) < 3:
            continue

        pivot = pivot[present].dropna()
        if len(pivot) < 3:
            continue

        stat, pval = stats.friedmanchisquare(
            *(pivot[m].values for m in present),
        )
        rows.append(
            {
                "metric": metric,
                "methods": ", ".join(present),
                "n_targets": len(pivot),
                "statistic": stat,
                "p_value": pval,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 1:
        _, corrected, _, _ = multipletests(df["p_value"], method="holm")
        df["p_corrected"] = corrected
    elif len(df) == 1:
        df["p_corrected"] = df["p_value"]
    return df


def _wilcoxon_exact(x, y, alternative="two-sided"):
    """Wilcoxon signed-rank test, dropping zeros before using exact p-values."""
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan, 1.0
    return stats.wilcoxon(diff, alternative=alternative, method="exact")


def _posthoc_wilcoxon(
    agg: pd.DataFrame,
    gscreen: list[str],
    baselines: list[str],
) -> pd.DataFrame:
    """Post-hoc Wilcoxon signed-rank tests: each G-screen vs each baseline.

    Returns one row per (metric, gscreen_method, baseline_method).
    p-values are BH FDR-corrected across all comparisons within each metric.
    """
    rows: list[dict] = []
    for metric, mdf in agg.groupby("metric"):
        pivot = mdf.pivot_table(
            index="target",
            columns="method",
            values="score",
        )

        metric_rows: list[dict] = []
        for gs, bl in product(gscreen, baselines):
            if gs not in pivot.columns or bl not in pivot.columns:
                continue

            paired = pivot[[gs, bl]].dropna()
            if len(paired) < 6:
                continue

            diff = paired[gs] - paired[bl]
            if (diff == 0).all():
                metric_rows.append(
                    {
                        "metric": metric,
                        "gscreen": gs,
                        "baseline": bl,
                        "n_targets": len(paired),
                        "statistic": np.nan,
                        "p_value": 1.0,
                        "mean_diff": 0.0,
                    }
                )
                continue

            stat, pval = _wilcoxon_exact(paired[gs], paired[bl])
            metric_rows.append(
                {
                    "metric": metric,
                    "gscreen": gs,
                    "baseline": bl,
                    "n_targets": len(paired),
                    "statistic": stat,
                    "p_value": pval,
                    "mean_diff": diff.mean(),
                }
            )

        if len(metric_rows) > 1:
            pvals = [r["p_value"] for r in metric_rows]
            _, corrected, _, _ = multipletests(pvals, method="fdr_bh")
            for r, pc in zip(metric_rows, corrected):
                r["p_corrected"] = pc
        elif len(metric_rows) == 1:
            metric_rows[0]["p_corrected"] = metric_rows[0]["p_value"]

        rows.extend(metric_rows)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stability test (dispersion comparison)
# ---------------------------------------------------------------------------


def _dispersion_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-target dispersion: sd for AUCROC, sd of log1p(score) for EF metrics."""
    parts = []
    for metric, mdf in all_metrics.groupby("metric"):
        if metric == "aucroc":
            disp = (
                mdf.groupby(["target", "method"])["score"]
                .std()
                .rename("dispersion")
                .reset_index()
            )
        else:
            tmp = mdf.copy()
            tmp["log_score"] = np.log1p(tmp["score"])
            disp = (
                tmp.groupby(["target", "method"])["log_score"]
                .std()
                .rename("dispersion")
                .reset_index()
            )
        disp["metric"] = metric
        parts.append(disp)

    return pd.concat(parts, ignore_index=True).dropna(subset=["dispersion"])


def _stability_test(
    all_metrics: pd.DataFrame,
    gscreen: list[str],
    baselines: list[str],
    epsilon: float = 0.0,
) -> pd.DataFrame:
    """Paired Wilcoxon on per-target dispersion: G-screen vs baselines.

    Tests whether dispersion differs between G-screen and baseline methods
    (two-sided), with an optional non-inferiority margin epsilon applied
    before testing.  Reports mean difference, 95% CI (bootstrap), two-sided
    p-value, and BH-FDR correction across all planned comparisons.
    """
    disp = _dispersion_table(all_metrics)

    rows: list[dict] = []
    for metric, mdf in disp.groupby("metric"):
        pivot = mdf.pivot_table(
            index="target",
            columns="method",
            values="dispersion",
        )

        for gs, bl in product(gscreen, baselines):
            if gs not in pivot.columns or bl not in pivot.columns:
                continue

            paired = pivot[[gs, bl]].dropna()
            if len(paired) < 6:
                continue

            diff = np.asarray(paired[gs]) - np.asarray(paired[bl])
            shifted = diff - epsilon

            mean_diff = diff.mean()
            ci_lo, ci_hi = _bootstrap_ci(diff)

            if (shifted == 0).all():
                rows.append(
                    {
                        "metric": metric,
                        "gscreen": gs,
                        "baseline": bl,
                        "n_targets": len(paired),
                        "mean_disp_gs": paired[gs].mean(),
                        "mean_disp_bl": paired[bl].mean(),
                        "mean_diff": mean_diff,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "statistic": np.nan,
                        "p_value": 1.0,
                    }
                )
                continue

            nonzero = shifted[shifted != 0]
            if len(nonzero) == 0:
                stat, pval = np.nan, 1.0
            else:
                stat, pval = stats.wilcoxon(
                    nonzero,
                    alternative="two-sided",
                    method="exact",
                )

            rows.append(
                {
                    "metric": metric,
                    "gscreen": gs,
                    "baseline": bl,
                    "n_targets": len(paired),
                    "mean_disp_gs": paired[gs].mean(),
                    "mean_disp_bl": paired[bl].mean(),
                    "mean_diff": mean_diff,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "statistic": stat,
                    "p_value": pval,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) > 1:
        _, corrected, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["p_corrected"] = corrected
    elif len(df) == 1:
        df["p_corrected"] = df["p_value"]
    return df


def _bootstrap_ci(
    diff: np.ndarray,
    n_boot: int = 9999,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = np.array(
        [
            rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(n_boot)
        ]
    )
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _pval_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


_GROUP_GAP = 0.5


def _grouped_x() -> dict[str, float]:
    """X positions: tight within groups, wider gap between groups."""
    pos = {}
    for i, m in enumerate(GSCREEN_METHODS):
        pos[m] = float(i)
    offset = len(GSCREEN_METHODS) + _GROUP_GAP
    for i, m in enumerate(BASELINE_METHODS):
        pos[m] = offset + i
    return pos


def _shift_boxes(ax: plt.Axes, old_x: list[float], new_x: list[float]):
    """Shift seaborn boxplot artists from default integer positions to new_x."""
    shift = {round(o): n - o for o, n in zip(old_x, new_x)}

    for patch in ax.patches:
        verts = patch.get_path().vertices
        cx = round(np.mean([verts[:, 0].min(), verts[:, 0].max()]))
        dx = shift.get(cx, 0)
        if dx:
            verts[:, 0] += dx
            patch.get_path().vertices = verts

    for line in ax.lines:
        xd = line.get_xdata()
        if len(xd) == 0:
            continue
        cx = round(np.mean(xd))
        dx = shift.get(cx, 0)
        if dx:
            line.set_xdata(np.asarray(xd, dtype=float) + dx)

    for coll in ax.collections:
        offs = coll.get_offsets()
        if len(offs) == 0:
            continue
        cx = round(offs[:, 0].mean())
        dx = shift.get(cx, 0)
        if dx:
            offs[:, 0] += dx
            coll.set_offsets(offs)


def _draw_boxplot(
    ax: plt.Axes,
    agg: pd.DataFrame,
    metric: str,
    wilcoxon_df: pd.DataFrame,
    lims: tuple[Optional[float], Optional[float]],
):
    """Draw a single box plot panel on the given axes."""
    data = agg[agg["metric"] == metric]
    colors = (
        list(sns.color_palette("magma_r"))[:3]
        + list(sns.color_palette("viridis_r"))[1:4]
    )
    method_to_x = _grouped_x()

    sns.boxplot(
        data=data,
        x="method",
        y="score",
        hue="method",
        order=ALL_METHODS,
        hue_order=ALL_METHODS,
        palette=colors,
        legend=False,
        ax=ax,
    )

    old_x = list(range(len(ALL_METHODS)))
    new_x = [method_to_x[m] for m in ALL_METHODS]
    _shift_boxes(ax, old_x, new_x)

    ax.set_xticks(new_x)
    ax.set_xticklabels([TICK_LABELS.get(m, m) for m in ALL_METHODS])
    ax.set_xlim(new_x[0] - 0.5, new_x[-1] + 0.5)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC" if metric == "aucroc" else metric.upper())
    ax.set_ylim(lims)

    if wilcoxon_df.empty or "metric" not in wilcoxon_df.columns:
        return
    wdf = wilcoxon_df[wilcoxon_df["metric"] == metric]
    if wdf.empty:
        return

    means = data.groupby("method")["score"].mean()
    gs_present = [m for m in GSCREEN_METHODS if m in means.index]
    if not gs_present:
        return

    best_gs = max(gs_present, key=lambda m: means[m])

    y_max = data["score"].max()
    y_step = (data["score"].max() - data["score"].min()) * 0.08
    y_cur = y_max + y_step

    bl_present = [m for m in BASELINE_METHODS if m in means.index]
    for bl in bl_present:
        row = wdf[(wdf["gscreen"] == best_gs) & (wdf["baseline"] == bl)]
        if row.empty:
            continue

        p_adj = row["p_corrected"].iloc[0]
        stars = _pval_stars(p_adj)

        x1 = method_to_x[best_gs]
        x2 = method_to_x[bl]
        ax.plot(
            [x1, x1, x2, x2],
            [y_cur, y_cur + y_step * 0.3, y_cur + y_step * 0.3, y_cur],
            lw=1,
            color="black",
        )
        ax.text(
            (x1 + x2) / 2,
            y_cur + y_step * 0.35,
            stars,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        y_cur += y_step

    ax.set_ylim(top=max(ax.get_ylim()[1], y_cur + y_step * 0.5))


def _plot_combined(
    agg: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    output: Path,
):
    """Two-column figure: AUROC (left) and EF 1% (right)."""
    fig, (ax_auc, ax_ef) = plt.subplots(1, 2, figsize=(12, 5))

    _draw_boxplot(ax_auc, agg, "aucroc", wilcoxon_df, lims=(0.0, 1.0))
    _draw_boxplot(ax_ef, agg, "ef 1%", wilcoxon_df, lims=(0.0, None))

    ax_auc.annotate(
        "a",
        xy=(-0.11, 1),
        xycoords="axes fraction",
        fontsize=18,
        fontweight=700,
    )
    ax_auc.axhline(y=0.5, color="grey", linestyle="--", zorder=-10)
    ax_ef.annotate(
        "b",
        xy=(-0.11, 1),
        xycoords="axes fraction",
        fontsize=18,
        fontweight=700,
    )
    ax_ef.axhline(y=1.0, color="grey", linestyle="--", zorder=-10)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=500)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    results: Path,
    db_home: Path = Path.home() / "db",
    bench_home: Path = Path.home() / "benchmark",
    skip_missing: bool = False,
    stability_epsilon: float = 0.0,
):
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Helvetica Neue",
            "ytick.left": True,
        },
    )

    db = results.name
    db_home = db_home / db
    bench_home = bench_home / db

    ratios = [0.01]
    metric_names = ["aucroc", "ef 1%"]

    typer.echo(f"Loading gscreen scores for {db} ...")
    gscreen_scores = _load_gscreen_scores(results, db_home)
    typer.echo(f"  {len(gscreen_scores)} entries loaded")

    typer.echo(f"Loading external method scores for {db} ...")
    ext_methods: dict[str, dict[str, pd.DataFrame]] = {}
    for slug in ("ls-align", "pharmagist", "autodock-vina"):
        ext_methods[slug] = _load_method_scores(bench_home, slug, skip_missing)
        name = METHOD_SLUG_MAP.get(slug, slug)
        typer.echo(f"  {name}: {len(ext_methods[slug])} entries")

    # ------------------------------------------------------------------
    typer.echo("Computing metrics ...")
    all_metrics = _collect_all_metrics(
        gscreen_scores,
        ext_methods,
        ratios,
        metric_names,
    )
    all_metrics.to_csv(results / "crossval_metrics.csv", index=False)

    methods_present = [
        m for m in ALL_METHODS if m in all_metrics["method"].unique()
    ]
    gs_present = [m for m in GSCREEN_METHODS if m in methods_present]
    bl_present = [m for m in BASELINE_METHODS if m in methods_present]

    # ------------------------------------------------------------------
    typer.echo("Stability analysis (std across references per target) ...")
    stability = _stability_table(all_metrics)
    stability.to_csv(results / "crossval_stability.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    typer.echo(
        stability.groupby(["method", "metric"])[["mean", "std"]]
        .mean()
        .reorder_levels([1, 0])
        .loc[metric_names, methods_present, :]
    )

    # ------------------------------------------------------------------
    typer.echo("\nAggregating per target (mean over references) ...")
    agg = _aggregate_per_target(all_metrics)

    pd.options.display.float_format = "{:.2f}".format
    pivot_summary = agg.pivot_table(
        index=["metric", "target"],
        columns="method",
        values="score",
    )
    typer.echo(
        pivot_summary[methods_present]
        .groupby("metric")
        .mean()
        .loc[metric_names]
    )

    # ------------------------------------------------------------------
    typer.echo("\nFriedman tests (omnibus) ...")
    friedman_df = _friedman_test(agg, methods_present)
    friedman_df.to_csv(results / "crossval_friedman.csv", index=False)
    if friedman_df.empty:
        typer.echo("  Not enough data for Friedman test")
    else:
        typer.echo(friedman_df.to_string(index=False))

    # ------------------------------------------------------------------
    significant_metrics = set()
    if not friedman_df.empty:
        significant_metrics = set(
            friedman_df.loc[friedman_df["p_corrected"] < 0.05, "metric"]
        )

    typer.echo("\nPost-hoc Wilcoxon tests (G-screen vs baselines, BH-FDR) ...")
    if not significant_metrics:
        typer.echo("  Skipped (no significant Friedman tests)")
        wilcoxon_df = pd.DataFrame()
    else:
        agg_sig = agg[agg["metric"].isin(significant_metrics)]
        wilcoxon_df = _posthoc_wilcoxon(agg_sig, gs_present, bl_present)

        if wilcoxon_df.empty:
            typer.echo("  Not enough paired data")
        else:
            pd.options.display.float_format = "{:.4f}".format
            typer.echo(wilcoxon_df.to_string(index=False))

    wilcoxon_df.to_csv(results / "crossval_wilcoxon.csv", index=False)

    # ------------------------------------------------------------------
    gs_stability = [m for m in ("GS-S", "GS-P", "GS-SP") if m in gs_present]
    eps_label = f", epsilon={stability_epsilon}" if stability_epsilon else ""
    typer.echo(f"\nStability test (dispersion, two-sided{eps_label}) ...")
    stab_df = _stability_test(
        all_metrics,
        gs_stability,
        bl_present,
        epsilon=stability_epsilon,
    )
    stab_df.to_csv(results / "crossval_stability_test.csv", index=False)
    if stab_df.empty:
        typer.echo("  Not enough data for stability test")
    else:
        pd.options.display.float_format = "{:.4f}".format
        typer.echo(stab_df.to_string(index=False))

    # ------------------------------------------------------------------
    typer.echo("\nGenerating box plots ...")
    out_path = results / "crossval_boxplots.svg"
    _plot_combined(agg, wilcoxon_df, out_path)
    typer.echo(f"  {out_path}")

    pd.options.display.float_format = None
    typer.echo(f"\nAll outputs written to {results}/")


if __name__ == "__main__":
    app()
