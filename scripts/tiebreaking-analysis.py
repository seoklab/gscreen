from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from typer import Typer

from shared_metrics import DATASET_STYLES

app = Typer(pretty_exceptions_enable=False)

_pairs = [
    ("GS-SP", "GS-SP + PG"),
    ("GS-SP", "GS-SP + Vina"),
    ("PharmaGist", "GS-SP + PG"),
    ("AutoDock Vina", "GS-SP + Vina"),
]

_METRICS = ["AUROC", "EF0.1%", "EF1%", "EF5%"]
_DELTA_COLS = [f"d{m}" for m in _METRICS]

_DATASET_ORDER = ["DUD-E", "LIT-PCBA", "MUV", "All"]
colors = (
    list(sns.color_palette("magma_r"))[:2]
    + list(sns.color_palette("viridis_r"))[1:3]
)
_METHOD_PALETTE = {
    "PharmaGist": colors[0],
    "GS-SP + PG": colors[1],
    "AutoDock Vina": colors[2],
    "GS-SP + Vina": colors[3],
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


def _load_speed_cache(bench_home: Path):
    df = pd.read_csv(bench_home / "bench/speed_cache.csv")
    df = (
        df[
            (df["nproc"] == 1)
            & (df["method"].isin(["gscreen", "pharmagist", "vina"]))
        ]
        .drop(columns=["nproc", "memkb", "key"])
        .reset_index(drop=True)
    )

    df["method"] = df["method"].map(
        {
            "gscreen": "GS-SP",
            "pharmagist": "PharmaGist",
            "vina": "AutoDock Vina",
        }
    )
    df = df.set_index(["method", "db", "target"])

    gssp_vina = df.loc["GS-SP"] + df.loc["AutoDock Vina"] * 0.01
    gssp_pg = df.loc["GS-SP"] + df.loc["PharmaGist"] * 0.01
    prefilter = (
        pd.concat(
            [gssp_vina, gssp_pg],
            keys=["AutoDock Vina", "PharmaGist"],
            names=["method"],
        )
        .dropna()
        .rename(columns={"time": "GS-SP time"})
    )

    df = df.rename(columns={"time": "Vanilla time"}).join(
        prefilter, how="right"
    )
    df["Speedup"] = df["Vanilla time"] / df["GS-SP time"]
    df = df.reset_index()
    return df


def _compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for baseline, method in _pairs:
        base: pd.DataFrame = df.loc[baseline]
        mdf: pd.DataFrame = df.loc[method]
        shared = base.index.intersection(mdf.index)
        if shared.empty:
            continue

        delta = mdf.loc[shared] - base.loc[shared]
        delta.columns = _DELTA_COLS
        delta = delta.reset_index()
        delta["method"] = method
        delta["baseline"] = baseline
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
    for (ds, baseline, method), grp in deltas.groupby(
        ["dataset", "baseline", "method"]
    ):
        base = {"dataset": ds, "baseline": baseline, "method": method}
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


def _shift_boxes(ax: plt.Axes, old_x: list[float], new_x: list[float]):
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


def _plot(df: pd.DataFrame, sdf: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(6, 4),
        sharex=False,
        sharey=False,
    )
    bar_width = 0.75

    rng = np.random.default_rng(42)

    df = df.copy()
    sdf = sdf.copy()

    unique_targets = df.index.get_level_values("target").unique()
    target_jitters = rng.uniform(
        -bar_width * 0.2, bar_width * 0.2, size=len(unique_targets)
    )
    target_jitters = dict(zip(unique_targets, target_jitters))
    df["jitter"] = df.index.get_level_values("target").map(target_jitters)
    sdf["jitter"] = sdf["target"].map(target_jitters)

    methods = ["PharmaGist", "GS-SP + PG", "AutoDock Vina", "GS-SP + Vina"]

    ax = axes[0]
    for mi, method in enumerate(methods):
        mdf = df.loc[method]
        mdf = mdf[mdf["EF1%"].notna()]
        if mdf.empty:
            continue

        sns.boxplot(
            x=np.full(len(mdf), mi),
            y=mdf["EF1%"],
            width=bar_width,
            color=_METHOD_PALETTE[method],
            linecolor="#555555",
            showfliers=False,
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
                s=10,
                marker=sty["marker"],
                color=sty["color"],
                edgecolors="white",
                linewidths=0.3,
                alpha=0.7,
                zorder=4,
            )

    xs = [0, bar_width, 0.25 + bar_width * 2, 0.25 + bar_width * 3]
    _shift_boxes(ax, old_x=list(range(len(methods))), new_x=xs)

    ax.axhspan(ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)

    ax.set_xticks(
        [bar_width * 0.5, 1 + bar_width * 1.5],
        ["\nPharmaGist", "\nAutoDock Vina"],
        fontsize=9,
    )
    ax.set_xticks(
        xs,
        ["(All)", "(GS-SP Top 1%)"] * 2,
        minor=True,
        fontsize=6,
    )

    ax.set_xlabel("")
    ax.set_xlim(-0.25 - bar_width * 0.5, 1.25 + bar_width * 2.5)
    ax.set_ylabel("EF1%", fontsize=10)
    ax.set_ylim(bottom=0, top=100)
    ax.set_yticks(
        [1, 5, 10, 25, 50, 100],
        ["1x", "5x", "10x", "25x", "50x", "100x"],
        fontsize=8,
    )
    ax.grid(axis="y", ls=":", alpha=0.4)

    ax = axes[1]
    sns.boxplot(
        sdf,
        x="method",
        y="Speedup",
        hue="method",
        order=["PharmaGist", "AutoDock Vina"],
        hue_order=["PharmaGist", "AutoDock Vina"],
        palette=_METHOD_PALETTE,
        linecolor="#555555",
        showfliers=False,
        ax=ax,
    )

    method_order = {
        m: i for i, m in enumerate(["PharmaGist", "AutoDock Vina"])
    }
    for (ds, method), group in sdf.groupby(["db", "method"]):
        sty = DATASET_STYLES[ds.upper()]
        ax.scatter(
            np.full(len(group), method_order[method]) + group["jitter"] * 2,
            group["Speedup"],
            marker=sty["marker"],
            color=sty["color"],
            edgecolors="white",
            s=12,
            linewidths=0.3,
            alpha=0.7,
            zorder=4,
            label=ds.upper() if method == "PharmaGist" else None,
        )

    ax.axhspan(ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)
    ax.grid(axis="y", ls=":", alpha=0.4)

    ax.set_xlabel("")
    ax.set_ylim(bottom=0, top=100)
    ax.set_ylabel(
        "Estimated speedup (GS-SP filtering, Top 1%)",
        fontsize=10,
    )
    ax.set_yticks(
        [1, 5, 10, 25, 50, 100],
        ["1x", "5x", "10x", "25x", "50x", "100x"],
        fontsize=8,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        fontsize=8,
        title="Dataset",
        title_fontsize=9,
        bbox_to_anchor=(1.14, 0.5),
    )

    for ax, letter in zip(axes, "ab"):
        ax.annotate(
            letter,
            xy=(-0.25, 0.97),
            xycoords="axes fraction",
            fontsize=14,
            fontweight=700,
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
    sdf = _load_speed_cache(bench_home)
    _plot(df, sdf, out_dir)

    typer.echo("Computing deltas vs GS-SP...")
    deltas = _compute_deltas(df)

    typer.echo("Computing summary statistics...")
    summary, summary_full = _summary_table(deltas)
    summary.to_csv(out_dir / "tie_breaking_summary.csv", index=False)
    summary_full.to_csv(out_dir / "tie_breaking_summary_full.csv", index=False)
    typer.echo(summary.to_string(index=False))

    typer.echo(f"\nDone. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    app()
