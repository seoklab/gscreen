from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, SymmetricalLogLocator
from typer import Typer

from shared_metrics import DATASET_STYLES

app = Typer(pretty_exceptions_enable=False)

_BASELINE = "GS-SP"
_METHODS = ["GS-SP + PG", "GS-SP + Vina"]
_ALL_METHODS = [_BASELINE, *_METHODS]

_METRICS = ["AUROC", "EF0.1%", "EF1%", "EF5%"]
_DELTA_COLS = [f"d{m}" for m in _METRICS]

_DATASET_ORDER = ["DUD-E", "LIT-PCBA", "MUV", "All"]
_METHOD_PALETTE = {
    "GS-SP": "#ff7f0e",
    "GS-SP + PG": "#2ca02c",
    "GS-SP + Vina": "#d62728",
}


def _load_scores(bench_home: Path):
    df = pd.read_csv(bench_home / "scores.csv")
    df["dataset"] = bench_home.name.upper()
    df = (
        df.pivot_table(
            index=["method", "dataset", "target"],
            columns="metric",
            values="score",
        )
        .reset_index()
        .rename(
            columns={
                "aucroc": "AUROC",
                "ef 0.1%": "EF0.1%",
                "ef 1%": "EF1%",
                "ef 5%": "EF5%",
            }
        )
    )
    return df


def _compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    baseline: pd.DataFrame = df.loc[_BASELINE]
    for method in _METHODS:
        mdf: pd.DataFrame = df.loc[method]
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


def _summary_table(deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_delta = ["dAUROC", "dEF1%"]
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

    ds_order = {ds: i for i, ds in enumerate(_DATASET_ORDER)}
    summary = pd.DataFrame(summary_rows).sort_values(
        "dataset", key=lambda s: s.map(ds_order)
    )
    full = pd.DataFrame(full_rows).sort_values(
        "dataset", key=lambda s: s.map(ds_order)
    )
    return summary, full


def _plot_raw(
    df: pd.DataFrame,
    ax: plt.Axes,
    bar_width: float,
):
    for mi, method in enumerate(_ALL_METHODS):
        mdf = df.loc[method]
        mdf = mdf[mdf["EF1%"].notna()]
        if mdf.empty:
            continue

        sns.boxplot(
            x=np.full(len(mdf), mi),
            y=mdf["EF1%"],
            width=bar_width,
            color=_METHOD_PALETTE[method],
            fliersize=0,
            ax=ax,
            legend=False,
        )

        for ds_name, sty in DATASET_STYLES.items():
            dsub = mdf.loc[ds_name]
            if dsub.empty:
                continue

            ax.scatter(
                mi + dsub["jitter"],
                dsub["EF1%"],
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

    ax.set_xticks(range(len(_ALL_METHODS)))
    ax.set_xticklabels(
        [
            "No Tie Breaking",
            "PharmaGist\nTie Breaking",
            "AutoDock Vina\nTie Breaking",
        ],
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("EF 1%", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)


def _plot_ratio(
    df: pd.DataFrame,
    ax: plt.Axes,
    bar_width: float,
):
    baseline = df.loc[_BASELINE]
    baseline = baseline[baseline["EF1%"] > 0]

    xpos = np.arange(len(_METHODS)) * 0.7
    for x, method in zip(xpos, _METHODS):
        mdf = df.loc[method]
        mdf = mdf[mdf["EF1%"].notna()]
        shared = baseline.index.intersection(mdf.index)
        if shared.empty:
            continue

        ratio = mdf.loc[shared, "EF1%"] / baseline.loc[shared, "EF1%"]
        sns.boxplot(
            x=np.full(len(ratio), x),
            y=ratio,
            width=bar_width,
            color=_METHOD_PALETTE[method],
            fliersize=0,
            ax=ax,
            legend=False,
            native_scale=True,
        )

        for ds_name, sty in DATASET_STYLES.items():
            dsub = mdf.loc[shared].loc[ds_name]
            if dsub.empty:
                continue

            ax.scatter(
                x + dsub["jitter"],
                ratio.loc[:, dsub.index],
                s=14,
                marker=sty["marker"],
                color=sty["color"],
                edgecolors="white",
                linewidths=0.3,
                alpha=0.7,
                zorder=4,
            )

    ax.axhspan(ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)

    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [
            "PharmaGist\nTie Breaking",
            "AutoDock Vina\nTie Breaking",
        ],
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Relative EF 1%", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.grid(axis="x", visible=False)


def _plot(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(6, 4),
        sharex=False,
        sharey=False,
    )
    bar_width = 0.55

    rng = np.random.default_rng(42)

    df = df.copy()
    unique_targets = df.index.get_level_values("target").unique()
    target_jitters = rng.uniform(
        -bar_width * 0.35, bar_width * 0.35, size=len(unique_targets)
    )
    df["jitter"] = df.index.get_level_values("target").map(
        dict(zip(unique_targets, target_jitters))
    )

    _plot_raw(df, axes[0], bar_width)
    _plot_ratio(df, axes[1], bar_width)
    for ax, letter in zip(axes, "ab"):
        ax.annotate(
            letter,
            xy=(-0.2, 0.97),
            xycoords="axes fraction",
            fontsize=16,
            fontweight=700,
        )

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
            (out_dir / "tie-breaking").with_suffix(f".{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


@app.command()
def main(
    bench_home: Path,
    datasets: str = "dud-e,lit-pcba,muv",
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
    df = pd.concat(
        [_load_scores(bench_home / ds) for ds in datasets.split(",")],
        ignore_index=True,
    ).set_index(["method", "dataset", "target"])
    _plot(df, out_dir)

    typer.echo("Computing deltas vs GS-SP...")
    deltas = _compute_deltas(df)

    typer.echo("Computing summary statistics...")
    summary, summary_full = _summary_table(deltas)
    summary.to_csv(out_dir / "tie_breaking_summary.csv", index=False)
    summary_full.to_csv(out_dir / "tie_breaking_summary_full.csv", index=False)
    typer.echo(summary.to_string(index=False))

    typer.echo("\n=== Overall Summary ===")
    overall = deltas[deltas["dataset"] == "All"]
    for method in _METHODS:
        m = overall[overall["method"] == method]
        med_ef = m["dEF1%"].median()
        wr_ef = (m["dEF1%"] > 0).mean()
        typer.echo(f"  {method}: median ΔEF1%={med_ef:+.3f} (win {wr_ef:.1%})")

    meds_ef = {
        m: overall[overall["method"] == m]["dEF1%"].median() for m in _METHODS
    }
    best_ef = max(meds_ef, key=meds_ef.get)
    typer.echo(f"\n  Best median ΔEF1%: {best_ef}")

    typer.echo(f"\nDone. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    app()
