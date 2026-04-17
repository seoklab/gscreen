from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    SymmetricalLogLocator,
)
from typer import Typer

from shared_metrics import DATASET_STYLES

app = Typer(pretty_exceptions_enable=False)

_BASELINE = "GS-SP"
_METHODS = ["PharmaGist", "AutoDock Vina"]
_ALL_METHODS = [_BASELINE, *_METHODS]

_METRICS = ["AUROC", "EF0.1%", "EF1%", "EF5%", "SEF0.1%", "SEF1%", "SEF5%"]
_DELTA_COLS = [f"d{m}" for m in _METRICS]
_REQUIRED_COLS = {*_METRICS, "dataset", "target", "method"}

_DATASET_ORDER = ["DUD-E", "LIT-PCBA", "MUV", "All"]
_METHOD_PALETTE = {
    "GS-SP": "#aaaaaa",
    "PharmaGist": "#c44e52",
    "AutoDock Vina": "#8172b2",
}


def _load_and_validate(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keys = df.groupby(["dataset", "target"])["method"].apply(set)
    no_baseline = keys[~keys.apply(lambda s: _BASELINE in s)].index.tolist()
    if no_baseline:
        typer.echo(
            f"WARNING: {len(no_baseline)} dataset-target pairs missing "
            f"{_BASELINE}, excluding them."
        )
        for ds, tgt in no_baseline:
            typer.echo(f"  - {ds}/{tgt}")
        mask = df.set_index(["dataset", "target"]).index.isin(no_baseline)
        df = df[~mask.values]

    return df


def _compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["method"] == _BASELINE].set_index(["dataset", "target"])[
        _METRICS
    ]
    rows = []
    for method in _METHODS:
        mdf = df[df["method"] == method].set_index(["dataset", "target"])[
            _METRICS
        ]
        shared = baseline.index.intersection(mdf.index)
        if shared.empty:
            continue
        delta = mdf.loc[shared] - baseline.loc[shared]
        delta.columns = _DELTA_COLS
        delta = delta.reset_index()
        delta["method"] = method
        rows.append(delta)

    deltas = pd.concat(rows, ignore_index=True)
    deltas["_src_dataset"] = deltas["dataset"]

    pooled = deltas.copy()
    pooled["dataset"] = "All"
    deltas = pd.concat([deltas, pooled], ignore_index=True)
    return deltas


def _compute_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SEF1%(method) / EF1%(GS-SP) for each method including GS-SP.

    Higher = better; 1.0 means strict EF fully matches optimistic EF.
    """
    baseline_ef = df[df["method"] == _BASELINE].set_index(
        ["dataset", "target"]
    )["EF1%"]
    nonzero = baseline_ef[baseline_ef > 0]
    n_dropped = len(baseline_ef) - len(nonzero)
    if n_dropped:
        typer.echo(
            f"  Recovery: dropped {n_dropped} targets with EF1%(GS-SP) = 0"
        )

    rows = []
    for method in _ALL_METHODS:
        mdf = df[df["method"] == method].set_index(["dataset", "target"])[
            "SEF1%"
        ]
        shared = nonzero.index.intersection(mdf.index)
        if shared.empty:
            continue
        rec = (mdf.loc[shared] / nonzero.loc[shared]).to_frame(name="recovery")
        rec = rec.reset_index()
        rec["method"] = method
        rows.append(rec)

    return pd.concat(rows, ignore_index=True), n_dropped


def _summary_table(
    deltas: pd.DataFrame,
    recovery: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    focus_delta = ["dAUROC", "dSEF1%"]
    full_delta = _DELTA_COLS

    def _agg_delta(group: pd.DataFrame, cols: list[str]) -> dict:
        row: dict = {"n_targets": len(group)}
        for c in cols:
            metric_label = c[1:]
            row[f"median Δ{metric_label}"] = group[c].median()
            row[f"mean Δ{metric_label}"] = group[c].mean()
            row[f"win rate {metric_label}"] = (group[c] > 0).mean()
        return row

    summary_rows = []
    full_rows = []
    for (ds, method), grp in deltas.groupby(["dataset", "method"]):
        base = {"dataset": ds, "method": method}
        summary_rows.append({**base, **_agg_delta(grp, focus_delta)})
        full_rows.append({**base, **_agg_delta(grp, full_delta)})

    rec_stats = []
    for (ds, method), grp in recovery.groupby(["dataset", "method"]):
        rec_stats.append(
            {
                "dataset": ds,
                "method": method,
                "median recovery": grp["recovery"].median(),
                "mean recovery": grp["recovery"].mean(),
            }
        )
    for method in _ALL_METHODS:
        grp = recovery[recovery["method"] == method]
        if not grp.empty:
            rec_stats.append(
                {
                    "dataset": "All",
                    "method": method,
                    "median recovery": grp["recovery"].median(),
                    "mean recovery": grp["recovery"].mean(),
                }
            )
    rec_df = pd.DataFrame(rec_stats)

    ds_order = {ds: i for i, ds in enumerate(_DATASET_ORDER)}
    summary = pd.DataFrame(summary_rows).sort_values(
        "dataset", key=lambda s: s.map(ds_order)
    )
    full = pd.DataFrame(full_rows).sort_values(
        "dataset", key=lambda s: s.map(ds_order)
    )
    rec_df = rec_df.sort_values("dataset", key=lambda s: s.map(ds_order))

    return summary, full, rec_df


def _median_iqr(vals):
    med = vals.median()
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    return med, q1, q3


def _plot(recovery: pd.DataFrame, out_dir: Path):
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(6, 4.2))

    n_methods = len(_ALL_METHODS)
    bar_width = 0.55

    for mi, method in enumerate(_ALL_METHODS):
        sub = recovery[recovery["method"] == method]
        if sub.empty:
            continue
        med, q1, q3 = _median_iqr(sub["recovery"])
        ax.bar(
            mi,
            med,
            bar_width,
            color=_METHOD_PALETTE[method],
            alpha=0.45,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.errorbar(
            mi,
            med,
            yerr=[[med - q1], [q3 - med]],
            fmt="none",
            capsize=3,
            ecolor="black",
            elinewidth=1.0,
            capthick=1.0,
            zorder=5,
        )

        for ds_name, sty in DATASET_STYLES.items():
            mask = sub["dataset"] == ds_name
            if not mask.any():
                continue
            jitter = rng.uniform(
                -bar_width * 0.35, bar_width * 0.35, mask.sum()
            )
            ax.scatter(
                mi + jitter,
                sub.loc[mask, "recovery"],
                s=14,
                marker=sty["marker"],
                color=sty["color"],
                edgecolors="white",
                linewidths=0.3,
                alpha=0.7,
                zorder=4,
                label=ds_name if mi == 0 else None,
            )

    ax.axhspan(ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)

    linthresh = 1
    ax.set_yscale("symlog", linthresh=linthresh, base=100)

    hi = ax.get_ylim()[1]
    upper = 10 ** np.ceil(np.log10(max(hi, 1.01)))
    ax.set_ylim(bottom=0, top=upper)

    lo, hi = ax.get_ylim()
    powers = [
        10**k
        for k in range(
            int(np.floor(np.log10(max(linthresh, 1)))),
            int(np.ceil(np.log10(hi))) + 1,
        )
    ]
    lin_major = np.arange(0, linthresh + 0.01, 0.5)
    all_major = np.unique(np.concatenate([lin_major, powers]))
    ax.yaxis.set_major_locator(FixedLocator(all_major))

    log_minor = SymmetricalLogLocator(
        base=10,
        linthresh=linthresh,
        subs=np.arange(2, 10),
    )
    log_ticks = log_minor.tick_values(lo, hi)
    lin_minor = np.arange(0, linthresh, 0.1)
    all_minor = np.unique(np.concatenate([log_ticks, lin_minor]))
    ax.yaxis.set_minor_locator(FixedLocator(all_minor))
    ax.tick_params(axis="y", which="minor", length=3, width=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(
        [
            "Without tie-breaking",
            "PharmaGist tie-breaking",
            "AutoDock Vina tie-breaking",
        ],
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        "Recovery relative to GS-SP EF1% (%)",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.legend(
        title="Dataset",
        fontsize=8,
        title_fontsize=9,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
    )
    fig.tight_layout()

    for ext in ("svg", "pdf"):
        fig.savefig(
            (out_dir / "tie_breaking_recovery").with_suffix(f".{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


@app.command()
def main(
    csv: Path,
    out_dir: Path = Path("tiebreaking-analysis"),
):
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Helvetica Neue",
            "ytick.left": True,
        },
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    typer.echo("Loading and validating data...")
    df = _load_and_validate(csv)

    typer.echo("Computing deltas vs GS-SP...")
    deltas = _compute_deltas(df)

    typer.echo("Computing recovery metrics...")
    recovery, n_excluded = _compute_recovery(df)

    typer.echo("Computing summary statistics...")
    summary, summary_full, rec_summary = _summary_table(deltas, recovery)
    summary.to_csv(out_dir / "tie_breaking_summary.csv", index=False)
    summary_full.to_csv(out_dir / "tie_breaking_summary_full.csv", index=False)
    rec_summary.to_csv(out_dir / "tie_breaking_recovery.csv", index=False)
    typer.echo(summary.to_string(index=False))
    typer.echo("\nRecovery (SEF1% / EF1%(GS-SP)):")
    typer.echo(rec_summary.to_string(index=False))

    typer.echo("\nPlotting...")
    _plot(recovery, out_dir)

    typer.echo("\n=== Overall Summary ===")
    overall = deltas[deltas["dataset"] == "All"]
    for method in _METHODS:
        m = overall[overall["method"] == method]
        med_sef = m["dSEF1%"].median()
        wr_sef = (m["dSEF1%"] > 0).mean()
        typer.echo(
            f"  {method}: median ΔSEF1%={med_sef:+.3f} (win {wr_sef:.1%})"
        )

    n_rec_targets = recovery[recovery["method"] == _BASELINE][
        "target"
    ].nunique()
    typer.echo(
        f"\n  Recovery (SEF1% / EF1%(GS-SP)), median "
        f"({n_rec_targets} targets, {n_excluded} excluded):"
    )
    for method in _ALL_METHODS:
        m = recovery[recovery["method"] == method]
        typer.echo(f"    {method}: {m['recovery'].median():.3f}")

    meds_sef = {
        m: overall[overall["method"] == m]["dSEF1%"].median() for m in _METHODS
    }
    best_sef = max(meds_sef, key=meds_sef.get)
    typer.echo(f"\n  Best median ΔSEF1%: {best_sef}")

    typer.echo(f"\nDone. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    app()
