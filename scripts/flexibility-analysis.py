"""Correlate ligand flexibility with G-screen virtual screening performance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from shared_metrics import enrichment_factor

app = typer.Typer(pretty_exceptions_enable=False)

_METHODS = {
    "GS-S": "shape",
    "GS-P": "pharma",
    "GS-SP": "score",
}
_FLEX_COLS = {
    "nrotors": "# Rotatable bonds",
    "rot_per_heavyatom": "# Rotatable bonds / # Heavy atom",
}
_PERF_METRICS = {"aucroc": "AUROC", "ef 1%": "EF1%"}
_DATASET_STYLE: dict[str, dict] = {
    "DUD-E": {"marker": "o", "color": "#4c72b0"},
    "LIT-PCBA": {"marker": "s", "color": "#dd8452"},
    "MUV": {"marker": "D", "color": "#55a868"},
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
    df["id"] = df["id"].astype(str)
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
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Bin molecules by flexibility, compute AUROC/EF1% per bin.

    Returns (per_bin_df, stats_df, bin_edges).
    """
    bin_labels = ["low", "medium", "high"]
    flex_col = "nrotors"

    # Global tercile boundaries applied uniformly to all targets
    bins, bin_edges = pd.qcut(
        df[flex_col],
        3,
        bin_labels,
        retbins=True,
        duplicates="drop",
    )
    bin_edges = bin_edges.astype(int)
    bin_edges[0] = -1
    bin_edges[-1] = 10000
    typer.echo(
        f"  Global {flex_col} bin edges: "
        + ", ".join(f"{e}" for e in bin_edges[1:-1])
    )

    df = df.copy()
    df["flex_bin"] = bins

    rows = []
    for (ds, target), tdf in df.groupby(["dataset", "target"]):
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
        for pm in _PERF_METRICS:
            row = {"method": method, "perf_metric": pm}
            for bl in bin_labels:
                vals = mdf.loc[mdf["flex_bin"] == bl, pm].values
                row[f"n_{bl}"] = len(vals)
                row[f"mean_{bl}"] = np.mean(vals) if len(vals) else np.nan
                row[f"median_{bl}"] = np.median(vals) if len(vals) else np.nan
            stat_rows.append(row)

    stats = pd.DataFrame(stat_rows)
    return strat, stats, bin_edges


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------


_CORR_FLEX_COLS = [("nrotors", True), ("rot_per_heavyatom", False)]
_CORR_ROWS = [
    (pm, fc, jt) for pm in _PERF_METRICS for fc, jt in _CORR_FLEX_COLS
]


def _plot_correlation(per_target: pd.DataFrame, out: Path):
    """4-row x 3-col scatter with trendlines.

    Rows: (AUROC x nrotors), (AUROC x rot/heavy), (EF1% x nrotors),
          (EF1% x rot/heavy).
    Cols: GS-S, GS-P, GS-SP.
    """
    methods = [m for m in _METHODS if m in per_target["method"].unique()]
    nrows = len(_CORR_ROWS)
    ncols = len(methods)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.0 * nrows),
        squeeze=False,
    )
    rng = np.random.default_rng(42)

    for ci, method in enumerate(methods):
        mdf = per_target[per_target["method"] == method]
        for ri, (pm, fc, jt) in enumerate(_CORR_ROWS):
            ax = axes[ri, ci]

            x_range = mdf[fc].max() - mdf[fc].min()
            jitter_scale = x_range * 0.015

            for ds, sty in _DATASET_STYLE.items():
                mask = mdf["dataset"] == ds
                if not mask.any():
                    continue

                x_scatter = mdf.loc[mask, fc]
                if jt:
                    x_scatter = x_scatter + rng.uniform(
                        -jitter_scale,
                        jitter_scale,
                        mask.sum(),
                    )
                ax.scatter(
                    x_scatter,
                    mdf.loc[mask, pm],
                    s=24,
                    marker=sty["marker"],
                    color=sty["color"],
                    edgecolors="white",
                    linewidths=0.3,
                    label=ds if ri == 0 and ci == 0 else None,
                    zorder=3,
                    alpha=0.7,
                )

            sns.regplot(
                x=mdf[fc],
                y=mdf[pm],
                ax=ax,
                scatter=False,
                ci=None,
                color="grey",
            )

            rho, pval = spearmanr(mdf[fc], mdf[pm])
            stars = (
                "***"
                if pval < 0.001
                else "**"
                if pval < 0.01
                else "*"
                if pval < 0.05
                else ""
            )
            ax.annotate(
                f"$\\rho$={rho:+.2f}{stars}",
                xy=(0.97, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

            ax.set_xlabel(_FLEX_COLS[fc], fontsize=8)
            if ci == 0:
                label = "AUROC" if pm == "aucroc" else pm.upper()
                ax.set_ylabel(label, fontsize=8)
            else:
                ax.set_ylabel("")
            if ri == 0:
                ax.set_title(method, fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(True, ls=":", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("svg", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


_BIN_COLORS = {"low": "#7fbf7f", "medium": "#f0c05a", "high": "#e07070"}


def _plot_stratified(strat: pd.DataFrame, bin_edges: np.ndarray, out: Path):
    """Box plots grouped by method, with jittered overlay by dataset."""
    methods = [m for m in _METHODS if m in strat["method"].unique()]
    bin_order = ["low", "medium", "high"]

    range_labels = {}
    for i, b in enumerate(bin_order):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        lo_s = str(lo + 1)
        hi_s = str(hi) if hi < 10000 else "+"
        joiner = "\u2013" if hi < 10000 else ""
        range_labels[b] = f"{b.title()} ({lo_s}{joiner}{hi_s})"

    n_bins = len(bin_order)
    fig, axes = plt.subplots(
        len(_PERF_METRICS),
        len(methods),
        figsize=(3.5 * len(methods), 3.5 * len(_PERF_METRICS)),
        squeeze=False,
    )

    width = 0.7 / n_bins
    rng = np.random.default_rng(42)

    for pi, (pm, pm_label) in enumerate(_PERF_METRICS.items()):
        for ci, method in enumerate(methods):
            ax = axes[pi, ci]
            mdf = strat[strat["method"] == method]

            tick_positions = []

            for bi, bname in enumerate(bin_order):
                pos = bi
                tick_positions.append(pos)
                sub = mdf[mdf["flex_bin"] == bname]
                vals = sub[pm].dropna().values
                if len(vals) == 0:
                    continue

                bp = ax.boxplot(
                    vals,
                    positions=[pos],
                    widths=width * 2.5,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(_BIN_COLORS[bname])
                    patch.set_alpha(0.5)
                for el in ("whiskers", "caps", "medians"):
                    for line in bp[el]:
                        line.set_color("black")
                        line.set_linewidth(0.8)

                # Jittered overlay by dataset
                for ds, sty in _DATASET_STYLE.items():
                    ds_mask = sub["dataset"] == ds
                    if not ds_mask.any():
                        continue
                    jitter = rng.uniform(
                        -width * 0.8, width * 0.8, ds_mask.sum()
                    )
                    ax.scatter(
                        pos + jitter,
                        sub.loc[ds_mask, pm],
                        s=12,
                        marker=sty["marker"],
                        color=sty["color"],
                        edgecolors="white",
                        linewidths=0.2,
                        alpha=0.6,
                        zorder=4,
                        label=ds if pi == 0 and ci == 0 and bi == 0 else None,
                    )

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [range_labels[b] for b in bin_order],
                fontsize=7,
            )
            if ci == 0:
                ax.set_ylabel(pm_label, fontsize=8)
            if pi == 0:
                ax.set_title(method, fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", ls=":", alpha=0.4)

    ds_handles, ds_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles=ds_handles,
        labels=ds_labels,
        loc="lower center",
        ncol=len(ds_labels),
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "merged_cache.parquet"
    if cache.exists():
        typer.echo(f"Loading cached data from {cache} ...")
        df = pd.read_parquet(cache)
    else:
        typer.echo("Loading data (first run, will cache)...")
        ds_list = [s.strip().lower() for s in datasets.split(",")]
        df = _load_data(data_dir, db_home, ds_list)
        df.to_parquet(cache, index=False)
        typer.echo(f"  Cached to {cache}")

    df["dataset"] = df["dataset"].str.upper()

    typer.echo(
        f"  {len(df)} molecules, "
        f"{df['target'].nunique()} targets, "
        f"{df['dataset'].nunique()} datasets"
    )

    pd.options.display.float_format = "{:.3f}".format

    # --- Analysis 1: per-target correlation ---
    typer.echo("\n=== Per-target correlation ===")
    per_target, corr_summary = _per_target_correlation(df)
    per_target.to_csv(output_dir / "flexibility_correlation.csv", index=False)
    corr_summary.to_csv(output_dir / "flexibility_summary.csv", index=False)
    typer.echo(corr_summary.to_string(index=False))

    _plot_correlation(per_target, output_dir / "flexibility_correlation")
    typer.echo(f"  Correlation plot saved to {output_dir}/")

    # --- Analysis 2: within-target stratified ---
    typer.echo("\n=== Stratified analysis (nrotors terciles) ===")
    strat, stats, bin_edges = _stratified_analysis(df)
    strat.to_csv(output_dir / "flexibility_stratified.csv", index=False)
    stats.to_csv(output_dir / "flexibility_stratified_stats.csv", index=False)
    typer.echo(stats.to_string(index=False))

    _plot_stratified(strat, bin_edges, output_dir / "flexibility_stratified")
    typer.echo(f"  Stratified plots saved to {output_dir}/")


if __name__ == "__main__":
    app()
