"""Correlate ligand flexibility with G-screen virtual screening performance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from shared_metrics import enrichment_factor

app = typer.Typer(pretty_exceptions_enable=False)

_METHODS = {
    "GS-S": "shape",
    "GS-P": "pharma",
    "GS-SP": "score",
}
_FLEX_COLS = {
    "nrotors": "Rotatable bonds",
    "rot_per_heavyatom": "Rotors / heavy atom",
    "rot_per_bond": "Rotors / bond",
}
_PERF_METRICS = ["aucroc", "ef 1%"]
_DATASET_STYLE: dict[str, dict] = {
    "dud-e": {"marker": "o", "color": "#4c72b0"},
    "lit-pcba": {"marker": "s", "color": "#dd8452"},
    "muv": {"marker": "D", "color": "#55a868"},
}


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------


def _load_data(
    data_dir: Path,
    db_home: Path,
    datasets: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for ds in datasets:
        gs = pd.read_csv(data_dir / ds / "gscreen.csv")
        gs["dataset"] = ds
        flex = pd.read_csv(db_home / ds / "ligand-flexibility-metrics.csv")

        merged = gs.merge(
            flex[["id", "target", "nheavy", "nbonds", "nrotors"]],
            on=["id", "target"],
            how="inner",
        )
        parts.append(merged)

    df = pd.concat(parts, ignore_index=True)
    df["rot_per_heavyatom"] = df["nrotors"] / df["nheavy"].clip(lower=1)
    df["rot_per_bond"] = df["nrotors"] / df["nbonds"].clip(lower=1)
    return df


# -------------------------------------------------------------------
# Analysis 1: per-target correlation
# -------------------------------------------------------------------


def _per_target_correlation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-target median flexibility + AUROC/EF1% per method.

    Returns (per_target_df, spearman_summary_df).
    """
    rows = []
    for (ds, target), tdf in df.groupby(["dataset", "target"]):
        med_flex = {fc: tdf[fc].median() for fc in _FLEX_COLS}
        for method, score_col in _METHODS.items():
            try:
                auc = roc_auc_score(tdf["is_active"], tdf[score_col])
            except ValueError:
                continue
            ef = enrichment_factor(
                tdf["is_active"].values,
                tdf[score_col].values,
                ratio=0.01,
            )
            rows.append(
                {
                    "dataset": ds,
                    "target": target,
                    "method": method,
                    **med_flex,
                    "aucroc": auc,
                    "ef 1%": ef,
                }
            )

    per_target = pd.DataFrame(rows)

    corr_rows = []
    for method in _METHODS:
        mdf = per_target[per_target["method"] == method]
        if len(mdf) < 5:
            continue
        for fc in _FLEX_COLS:
            for pm in _PERF_METRICS:
                rho, pval = spearmanr(mdf[fc], mdf[pm])
                corr_rows.append(
                    {
                        "method": method,
                        "flex_metric": fc,
                        "perf_metric": pm,
                        "spearman_rho": rho,
                        "p_value": pval,
                        "n_targets": len(mdf),
                    }
                )

    corr_summary = pd.DataFrame(corr_rows)
    return per_target, corr_summary


# -------------------------------------------------------------------
# Analysis 2: within-target stratified
# -------------------------------------------------------------------


def _stratified_analysis(
    df: pd.DataFrame,
    flex_col: str = "nrotors",
    n_bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin molecules by flexibility, compute AUROC/EF1% per bin.

    Returns (per_bin_df, stats_df).
    """
    bin_labels = ["low", "medium", "high"][:n_bins]
    rows = []

    for (ds, target), tdf in df.groupby(["dataset", "target"]):
        try:
            tdf = tdf.copy()
            tdf["flex_bin"] = pd.qcut(
                tdf[flex_col],
                n_bins,
                labels=bin_labels,
                duplicates="drop",
            )
        except ValueError:
            continue

        for method, score_col in _METHODS.items():
            for bname, bdf in tdf.groupby("flex_bin", observed=True):
                n_actives = bdf["is_active"].sum()
                if n_actives < 3 or n_actives == len(bdf):
                    continue
                try:
                    auc = roc_auc_score(bdf["is_active"], bdf[score_col])
                except ValueError:
                    continue
                ef = enrichment_factor(
                    bdf["is_active"].values,
                    bdf[score_col].values,
                    ratio=0.01,
                )
                rows.append(
                    {
                        "dataset": ds,
                        "target": target,
                        "method": method,
                        "flex_bin": bname,
                        "n_total": len(bdf),
                        "n_actives": int(n_actives),
                        "aucroc": auc,
                        "ef 1%": ef,
                    }
                )

    strat = pd.DataFrame(rows)

    stat_rows = []
    for method in _METHODS:
        mdf = strat[strat["method"] == method]
        targets_with_all = (
            mdf.groupby("target")["flex_bin"]
            .nunique()
            .loc[lambda s: s == n_bins]
            .index
        )
        mdf_full = mdf[mdf["target"].isin(targets_with_all)]

        for pm in _PERF_METRICS:
            groups = [
                mdf_full.loc[mdf_full["flex_bin"] == bl, pm].values
                for bl in bin_labels
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue

            row = {
                "method": method,
                "perf_metric": pm,
                "n_targets": len(targets_with_all),
            }

            if len(groups) >= 3 and all(len(g) > 1 for g in groups):
                min_len = min(len(g) for g in groups)
                try:
                    _, fp = friedmanchisquare(*(g[:min_len] for g in groups))
                    row["friedman_p"] = fp
                except Exception:
                    row["friedman_p"] = np.nan
            else:
                row["friedman_p"] = np.nan

            low = mdf_full.loc[
                mdf_full["flex_bin"] == bin_labels[0], pm
            ].values
            high = mdf_full.loc[
                mdf_full["flex_bin"] == bin_labels[-1], pm
            ].values
            n_paired = min(len(low), len(high))
            if n_paired >= 5:
                try:
                    _, wp = wilcoxon(low[:n_paired], high[:n_paired])
                    row["wilcoxon_low_vs_high_p"] = wp
                except Exception:
                    row["wilcoxon_low_vs_high_p"] = np.nan
            else:
                row["wilcoxon_low_vs_high_p"] = np.nan

            row["mean_low"] = np.mean(low) if len(low) else np.nan
            row["mean_high"] = np.mean(high) if len(high) else np.nan
            stat_rows.append(row)

    stats = pd.DataFrame(stat_rows)
    return strat, stats


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------


def _plot_correlation(
    per_target: pd.DataFrame,
    flex_col: str,
    out: Path,
):
    """Scatter: per-target flexibility vs performance, one col per method."""
    methods = [m for m in _METHODS if m in per_target["method"].unique()]
    fig, axes = plt.subplots(
        len(_PERF_METRICS),
        len(methods),
        figsize=(4 * len(methods), 3.8 * len(_PERF_METRICS)),
        squeeze=False,
    )

    for ci, method in enumerate(methods):
        mdf = per_target[per_target["method"] == method]
        for ri, pm in enumerate(_PERF_METRICS):
            ax = axes[ri, ci]
            for ds, sty in _DATASET_STYLE.items():
                mask = mdf["dataset"] == ds
                if not mask.any():
                    continue
                ax.scatter(
                    mdf.loc[mask, flex_col],
                    mdf.loc[mask, pm],
                    s=30,
                    marker=sty["marker"],
                    color=sty["color"],
                    edgecolors="white",
                    linewidths=0.3,
                    label=ds if ri == 0 and ci == 0 else None,
                    zorder=3,
                )

            rho, pval = spearmanr(mdf[flex_col], mdf[pm])
            ax.annotate(
                f"$\\rho$ = {rho:.2f}\np = {pval:.3f}",
                xy=(0.97, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

            if ri == len(_PERF_METRICS) - 1:
                ax.set_xlabel(_FLEX_COLS[flex_col])
            if ci == 0:
                label = "AUROC" if pm == "aucroc" else pm.upper()
                ax.set_ylabel(label)
            if ri == 0:
                ax.set_title(method)
            ax.grid(True, ls=":", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            fontsize=8,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("svg", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def _plot_stratified(strat: pd.DataFrame, out: Path):
    """Box plots: performance by flexibility bin, grouped by method."""
    methods = [m for m in _METHODS if m in strat["method"].unique()]
    fig, axes = plt.subplots(
        1,
        len(_PERF_METRICS),
        figsize=(5 * len(_PERF_METRICS), 4),
        squeeze=False,
    )

    bin_order = ["low", "medium", "high"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    for pi, pm in enumerate(_PERF_METRICS):
        ax = axes[0, pi]
        tick_positions = []
        tick_labels = []
        n_methods = len(methods)
        width = 0.7 / n_methods

        for bi, bname in enumerate(bin_order):
            center = bi
            tick_positions.append(center)
            tick_labels.append(bname)
            for mi, method in enumerate(methods):
                pos = center + (mi - (n_methods - 1) / 2) * width
                sub = strat[
                    (strat["method"] == method) & (strat["flex_bin"] == bname)
                ]
                if sub.empty:
                    continue
                bp = ax.boxplot(
                    sub[pm].dropna().values,
                    positions=[pos],
                    widths=width * 0.85,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(colors[mi % len(colors)])
                    patch.set_alpha(0.7)
                for element in ("whiskers", "caps", "medians"):
                    for line in bp[element]:
                        line.set_color("black")
                        line.set_linewidth(0.8)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Flexibility bin (nrotors)")
        label = "AUROC" if pm == "aucroc" else pm.upper()
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", ls=":", alpha=0.4)

    legend_patches = [
        Patch(facecolor=colors[i], alpha=0.7, label=m)
        for i, m in enumerate(methods)
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=len(methods),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("svg", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


@app.command()
def main(
    data_dir: Path = Path.home() / "repo/seoklab/gscreen-data/benchmark",
    db_home: Path = Path.home() / "db",
    output_dir: Path = Path("flexibility_results"),
    datasets: str = "dud-e,lit-pcba,muv",
):
    ds_list = [s.strip() for s in datasets.split(",")]

    typer.echo("Loading data...")
    df = _load_data(data_dir, db_home, ds_list)
    typer.echo(
        f"  {len(df)} molecules, "
        f"{df['target'].nunique()} targets, "
        f"{df['dataset'].nunique()} datasets"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Analysis 1: per-target correlation ---
    typer.echo("\n=== Per-target correlation ===")
    per_target, corr_summary = _per_target_correlation(df)

    per_target.to_csv(output_dir / "flexibility_correlation.csv", index=False)
    corr_summary.to_csv(output_dir / "flexibility_summary.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    typer.echo(corr_summary.to_string(index=False))

    for fc in _FLEX_COLS:
        _plot_correlation(
            per_target,
            fc,
            output_dir / f"flexibility_correlation_{fc}",
        )
    typer.echo(f"  Correlation plots saved to {output_dir}/")

    # --- Analysis 2: within-target stratified ---
    typer.echo("\n=== Stratified analysis (nrotors terciles) ===")
    strat, stats = _stratified_analysis(df, flex_col="nrotors", n_bins=3)

    strat.to_csv(output_dir / "flexibility_stratified.csv", index=False)
    stats.to_csv(output_dir / "flexibility_stratified_stats.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    typer.echo(stats.to_string(index=False))

    _plot_stratified(strat, output_dir / "flexibility_stratified")
    typer.echo(f"  Stratified plots saved to {output_dir}/")

    pd.options.display.float_format = None


if __name__ == "__main__":
    app()
