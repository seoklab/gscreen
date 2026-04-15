"""Memory scaling analysis.

Reads per-target CSV outputs from the memory_scaling driver, produces
publication-ready figures and summary/feasibility tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_enable=False)

_LABEL_MAP = {
    "galign": "GS-S",
    "gscreen": "GS-P/SP",
    "lsalign": "Flexi-LS-align",
    "pharmagist": "PharmaGist",
}
_METHOD_ORDER = ["GS-S", "GS-P/SP", "Flexi-LS-align", "PharmaGist"]
_COLORS = {
    "GS-S": "#1f77b4",
    "GS-P/SP": "#ff7f0e",
    "Flexi-LS-align": "#2ca02c",
    "PharmaGist": "#d62728",
}


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------


def _load_data(data_dir: Path) -> pd.DataFrame:
    """Glob per-target CSVs and combine, adding target column."""
    frames = []
    for csv in sorted(data_dir.glob("*/*/memory_scaling.csv")):
        rel = csv.relative_to(data_dir)
        if len(rel.parts) < 3:
            continue
        db, target = rel.parts[0], rel.parts[1]
        df = pd.read_csv(csv)
        df["target"] = f"{db}/{target}"
        frames.append(df)

    if not frames:
        typer.echo(f"No memory_scaling.csv found under {data_dir}")
        raise typer.Exit(1)

    raw = pd.concat(frames, ignore_index=True)
    raw["peak_rss_mb"] = raw["memkb"] / 1024
    raw["method"] = raw["method"].map(_LABEL_MAP)
    raw = raw.dropna(subset=["method"])
    return raw


def _fill_missing_as_inf(
    df: pd.DataFrame, methods: list, sizes: list
) -> pd.DataFrame:
    """Ensure every (target, method, n_mols) exists; fill gaps with inf.

    Prevents survivor bias when baselines OOM at higher counts.
    """
    targets = df["target"].unique()
    idx = pd.MultiIndex.from_product(
        [targets, methods, sizes],
        names=["target", "method", "n_mols"],
    )
    full = idx.to_frame(index=False)
    merged = full.merge(df, on=["target", "method", "n_mols"], how="left")
    merged["peak_rss_mb"] = merged["peak_rss_mb"].fillna(np.inf)
    return merged


# -------------------------------------------------------------------
# Quantile aggregation helpers
# -------------------------------------------------------------------


def _quantile_table(df: pd.DataFrame, q: float) -> pd.DataFrame:
    return (
        df.groupby(["method", "n_mols"])["peak_rss_mb"]
        .quantile(q)
        .unstack("n_mols")
    )


def _agg_quantiles(df: pd.DataFrame):
    """Return median, q25, q75, q10, q90 pivot tables."""
    med = _quantile_table(df, 0.5)
    q25 = _quantile_table(df, 0.25)
    q75 = _quantile_table(df, 0.75)
    q10 = _quantile_table(df, 0.10)
    q90 = _quantile_table(df, 0.90)
    return med, q25, q75, q10, q90


# -------------------------------------------------------------------
# Panel helpers
# -------------------------------------------------------------------


def _plot_ribbon(ax, sizes, med, q25, q75, q10, q90, method, color):
    finite = np.isfinite(med)
    xs = np.array(sizes)[finite]
    if len(xs) == 0:
        return

    ym = np.array(med)[finite]
    y25 = np.array(q25)[finite]
    y75 = np.array(q75)[finite]
    y10 = np.array(q10)[finite]
    y90 = np.array(q90)[finite]

    ax.fill_between(xs, y10, y90, alpha=0.10, color=color)
    ax.fill_between(xs, y25, y75, alpha=0.25, color=color)
    ax.plot(xs, ym, "o-", color=color, label=method, markersize=4)


def _panel_absolute(ax, df, methods, sizes):
    """Panel A: absolute RSS vs molecule count."""
    med, q25, q75, q10, q90 = _agg_quantiles(df)
    for m in methods:
        if m not in med.index:
            continue
        cols = [s for s in sizes if s in med.columns]
        _plot_ribbon(
            ax,
            cols,
            med.loc[m, cols].values,
            q25.loc[m, cols].values,
            q75.loc[m, cols].values,
            q10.loc[m, cols].values,
            q90.loc[m, cols].values,
            m,
            _COLORS[m],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of molecules")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("A. Absolute RSS vs. Molecule Count")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def _panel_normalised(ax, df, methods, sizes):
    """Panel B: RSS normalised by smallest-n value per target."""
    records = []
    for (tgt, m), grp in df.groupby(["target", "method"]):
        grp = grp.sort_values("n_mols")
        base = grp["peak_rss_mb"].iloc[0]
        if not np.isfinite(base) or base <= 0:
            continue
        for _, row in grp.iterrows():
            records.append(
                {
                    "target": tgt,
                    "method": m,
                    "n_mols": row["n_mols"],
                    "norm_rss": row["peak_rss_mb"] / base,
                }
            )

    ndf = pd.DataFrame(records)
    if ndf.empty:
        return

    med, q25, q75, q10, q90 = _agg_quantiles(
        ndf.rename(columns={"norm_rss": "peak_rss_mb"})
    )
    for m in methods:
        if m not in med.index:
            continue
        cols = [s for s in sizes if s in med.columns]
        _plot_ribbon(
            ax,
            cols,
            med.loc[m, cols].values,
            q25.loc[m, cols].values,
            q75.loc[m, cols].values,
            q10.loc[m, cols].values,
            q90.loc[m, cols].values,
            m,
            _COLORS[m],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of molecules")
    ax.set_ylabel("Normalised RSS (fold change)")
    ax.set_title("B. Normalised Memory Growth")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def _panel_p90(ax, df, methods, sizes):
    """Panel C: 90th-pctile RSS with RAM-limit reference lines."""
    q90 = _quantile_table(df, 0.90)
    for m in methods:
        if m not in q90.index:
            continue
        cols = [s for s in sizes if s in q90.columns]
        vals = q90.loc[m, cols].values
        finite = np.isfinite(vals)
        xs = np.array(cols)[finite]
        ys = vals[finite]
        if len(xs):
            ax.plot(
                xs,
                ys,
                "s-",
                color=_COLORS[m],
                label=m,
                markersize=4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    for ram_gb, ls in [(256, "--"), (512, ":")]:
        ram_mb = ram_gb * 1024
        ax.axhline(
            ram_mb,
            color="grey",
            ls=ls,
            lw=1,
            alpha=0.7,
        )
        ax.text(
            ax.get_xlim()[0],
            ram_mb * 1.1,
            f"{ram_gb} GB",
            fontsize=7,
            color="grey",
        )
    ax.set_xlabel("Number of molecules")
    ax.set_ylabel("90th-pctile Peak RSS (MB)")
    ax.set_title("C. High-End Memory Burden (p90)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)


# -------------------------------------------------------------------
# Summary table
# -------------------------------------------------------------------


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, mdf in df.groupby("method"):
        all_sizes = sorted(mdf["n_mols"].unique())
        n_min, n_max = all_sizes[0], all_sizes[-1]
        n_targets = mdf["target"].nunique()

        med_min = mdf.loc[mdf["n_mols"] == n_min, "peak_rss_mb"].median()
        med_max = mdf.loc[mdf["n_mols"] == n_max, "peak_rss_mb"].median()

        # Per-target fold change
        fold_changes = []
        loglog_slopes = []
        for _, tgrp in mdf.groupby("target"):
            tgrp = tgrp.sort_values("n_mols")
            rss_vals = tgrp["peak_rss_mb"].values
            n_vals = tgrp["n_mols"].values
            base = rss_vals[0]
            top = rss_vals[-1]
            if np.isfinite(base) and base > 0:
                fold_changes.append(top / base)
            # log-log slope
            fin = np.isfinite(rss_vals) & (rss_vals > 0)
            if fin.sum() >= 2:
                slope = np.polyfit(
                    np.log10(n_vals[fin].astype(float)),
                    np.log10(rss_vals[fin]),
                    1,
                )[0]
                loglog_slopes.append(slope)

        fc = np.array(fold_changes)
        sl = np.array(loglog_slopes)
        n_success_max = int(
            np.isfinite(mdf.loc[mdf["n_mols"] == n_max, "peak_rss_mb"]).sum()
        )
        p90_max = mdf.loc[mdf["n_mols"] == n_max, "peak_rss_mb"].quantile(0.9)

        rows.append(
            {
                "method": m,
                "tested_n_range": f"{n_min}-{n_max}",
                "median_rss_min_n": round(med_min, 1),
                "median_rss_max_n": round(med_max, 1),
                "median_fold_change": (
                    round(float(np.median(fc)), 2) if len(fc) else np.nan
                ),
                "p90_fold_change": (
                    round(float(np.quantile(fc, 0.9)), 2)
                    if len(fc)
                    else np.nan
                ),
                "median_loglog_slope": (
                    round(float(np.median(sl)), 3) if len(sl) else np.nan
                ),
                "p90_rss_max_n": (
                    round(p90_max, 1) if np.isfinite(p90_max) else "inf"
                ),
                "n_targets_total": n_targets,
                "n_success_max_n": n_success_max,
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Feasibility table
# -------------------------------------------------------------------

_RAM_THRESHOLDS_GB = [32, 64, 128, 256, 512, 1024]


def _compute_feasibility(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, n), grp in df.groupby(["method", "n_mols"]):
        n_total = len(grp)
        rss = grp["peak_rss_mb"].values
        n_success = int(np.isfinite(rss).sum())
        n_missing = n_total - n_success

        row = {
            "method": m,
            "n_mols": n,
            "n_targets": n_total,
            "n_success": n_success,
            "n_missing_or_failed": n_missing,
        }

        # Single-thread thresholds
        for gb in _RAM_THRESHOLDS_GB:
            mb = gb * 1024
            row[f"n_exceed_{gb}gb_1t"] = int(np.sum(rss > mb))

        # 128-thread projection
        proj = rss * 128
        for gb in [256, 512, 1024]:
            mb = gb * 1024
            row[f"n_exceed_{gb}gb_128t"] = int(np.sum(proj > mb))

        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Supplementary: parallel nproc figure
# -------------------------------------------------------------------


def _plot_parallel(df_all: pd.DataFrame, out: Path):
    """Show flat memory for GS variants across nproc values."""
    gs = df_all[df_all["method"].isin(["GS-S", "GS-P/SP"])].copy()
    nprocs = sorted(gs["nproc"].unique())
    if len(nprocs) < 2:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharey=True,
    )
    for ax, m in zip(axes, ["GS-S", "GS-P/SP"]):
        mdf = gs[gs["method"] == m]
        for np_val in nprocs:
            sub = mdf[mdf["nproc"] == np_val]
            med = sub.groupby("n_mols")["peak_rss_mb"].median().sort_index()
            ax.plot(
                med.index,
                med.values,
                "o-",
                label=f"nproc={np_val}",
                markersize=4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of molecules")
        ax.set_ylabel("Median Peak RSS (MB)")
        ax.set_title(m)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


@app.command()
def main(
    data_dir: Path,
    output_dir: Path = Path("memory_scaling_results"),
):
    """Analyse memory scaling CSVs produced by the benchmark driver.

    DATA_DIR is the root output directory from memory_scaling.py,
    containing db/target/memory_scaling.csv files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_data(data_dir)

    typer.echo(f"Loaded {len(raw)} rows, {raw['target'].nunique()} targets")

    # Primary analysis on nproc=1
    df1 = raw[raw["nproc"] == 1].copy()
    methods = [m for m in _METHOD_ORDER if m in df1["method"].unique()]
    sizes = sorted(df1["n_mols"].unique())

    df1 = _fill_missing_as_inf(df1, methods, sizes)

    # --- Main figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _panel_absolute(axes[0], df1, methods, sizes)
    _panel_normalised(axes[1], df1, methods, sizes)
    _panel_p90(axes[2], df1, methods, sizes)
    fig.tight_layout()
    main_fig = output_dir / "memory_scaling_main"
    for ext in ("png", "pdf"):
        fig.savefig(main_fig.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)
    typer.echo(f"Saved {main_fig}.png/pdf")

    # --- Summary table ---
    summary = _compute_summary(df1)
    summary_path = output_dir / "memory_scaling_summary.csv"
    summary.to_csv(summary_path, index=False)
    typer.echo(f"Saved {summary_path}")
    typer.echo(summary.to_string(index=False))

    # --- Feasibility table ---
    feas = _compute_feasibility(df1)
    feas_path = output_dir / "memory_scaling_feasibility.csv"
    feas.to_csv(feas_path, index=False)
    typer.echo(f"\nSaved {feas_path}")
    typer.echo(feas.to_string(index=False))

    # --- Supplementary parallel figure ---
    _plot_parallel(raw, output_dir / "memory_scaling_parallel")


if __name__ == "__main__":
    app()
