from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
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

_METHOD_KEYS = ["PharmaGist", "GS-SP + PG", "AutoDock Vina", "GS-SP + Vina"]
_TOPS = [100, 30, 25]
_DATASET_ORDER = ["DUD-E", "LIT-PCBA", "MUV"]
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


def _load_extra(
    sdf: pd.DataFrame,
    nligs_csv: Path,
    proj_csv: Path,
) -> pd.DataFrame:
    nligs = pd.read_csv(nligs_csv)
    proj = pd.read_csv(proj_csv)

    memory = proj.merge(nligs, on=["dataset", "target"], how="left")
    memory["estimate"] = 10 ** (
        memory["slope"] * np.log10(memory["n"]) + memory["intercept"]
    )
    memory["method"] = memory["method"].map(
        lambda m: m.replace("GS-P/SP", "GS-SP")
    )
    memory = (
        memory.loc[
            memory["method"].isin(["GS-SP", "PharmaGist"]),
            ["method", "dataset", "target", "estimate"],
        ]
        .set_index(["method", "dataset", "target"])
        .sort_index()
    )
    gssp_pg = memory.loc["GS-SP"] + memory.loc["PharmaGist"] * 0.01
    gssp_pg = (
        pd.concat([gssp_pg], keys=["PharmaGist"], names=["method"])
        .dropna()
        .rename(columns={"estimate": "GS-SP memory"})
    )
    memory = memory.rename(columns={"estimate": "Vanilla memory"}).join(
        gssp_pg, how="right"
    )
    memory["Memory reduction"] = (
        memory["Vanilla memory"] / memory["GS-SP memory"]
    )
    resource = (
        memory[["Memory reduction"]]
        .reset_index()
        .merge(
            sdf[["method", "db", "target", "Speedup"]].rename(
                columns={"db": "dataset"}
            ),
            on=["method", "dataset", "target"],
            how="left",
        )
        .dropna()
    )
    resource["dataset"] = resource["dataset"].str.upper()
    return resource


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
        for path in coll.get_paths():
            verts = path.vertices
            cx = round(np.mean([verts[:, 0].min(), verts[:, 0].max()]))
            dx = shift.get(cx, 0)
            if dx:
                path.vertices[:, 0] += dx

        offs = coll.get_offsets()
        if len(offs) == 0:
            continue
        cx = round(offs[:, 0].mean())
        dx = shift.get(cx, 0)
        if dx:
            offs[:, 0] += dx
            coll.set_offsets(offs)


def _plot(
    df: pd.DataFrame,
    sdf: pd.DataFrame,
    rsrc: pd.DataFrame,
    out_dir: Path,
):
    rng = np.random.default_rng(42)
    bar_width = 0.75

    plot_df = df.copy().sort_index()

    unique_targets = plot_df.index.get_level_values("target").unique()
    target_jitters = rng.uniform(
        -bar_width * 0.2, bar_width * 0.2, size=len(unique_targets)
    )
    target_jitters = dict(zip(unique_targets, target_jitters))
    plot_df["jitter"] = plot_df.index.get_level_values("target").map(
        target_jitters
    )

    sdf["jitter"] = sdf["target"].map(target_jitters)

    fig = plt.figure(figsize=(9, 8), dpi=300)
    gs = fig.add_gridspec(2, 6)

    axes = []
    for i in range(3):
        axes.append(fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]))
    axes.append(fig.add_subplot(gs[1, :3]))
    axes.append(fig.add_subplot(gs[1, 3:]))

    for ax, db, top in zip(axes[:3], _DATASET_ORDER, _TOPS):
        for mi, method in enumerate(_METHOD_KEYS):
            try:
                mdf: pd.DataFrame = plot_df.loc[method, db]
            except KeyError:
                print(method, db)
                raise

            mdf = mdf[mdf["EF1%"].notna()]
            if mdf.empty:
                continue

            sns.violinplot(
                x=np.full(len(mdf), mi),
                y=mdf["EF1%"],
                width=bar_width,
                color=_METHOD_PALETTE[method],
                linecolor="#555555",
                ax=ax,
                legend=False,
            )

        xs = [0, bar_width, 0.25 + bar_width * 2, 0.25 + bar_width * 3]
        _shift_boxes(ax, old_x=list(range(len(_METHOD_KEYS))), new_x=xs)

        ax.axhspan(
            ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0
        )
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

        ax.set_title(db, fontsize=11, fontweight=500)
        ax.set_xlabel("")
        ax.set_xlim(-0.25 - bar_width * 0.5, 1.25 + bar_width * 2.5)
        ax.set_ylabel("Enrichment factor (1%)", fontsize=10)
        ax.set_yticks(
            [1, 5, 10, 25, 50, 75, 100],
            ["1x", "5x", "10x", "25x", "50x", "75x", "100x"],
            fontsize=8,
        )
        ax.set_ylim(bottom=0, top=top)
        ax.grid(axis="y", ls=":", alpha=0.4)

    ax = axes[3]
    sns.boxplot(
        sdf,
        x="method",
        y="Speedup",
        hue="method",
        order=["PharmaGist", "AutoDock Vina"],
        hue_order=["PharmaGist", "AutoDock Vina"],
        palette=_METHOD_PALETTE,
        linecolor="#555555",
        ax=ax,
    )
    ax.axhspan(ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)
    ax.grid(axis="y", ls=":", alpha=0.4)

    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(bottom=0, top=100)
    ax.set_ylabel(
        "Estimated runtime speedup",
        fontsize=10,
    )
    ax.set_yticks(
        [1, 5, 10, 25, 50, 75, 100],
        ["1x", "5x", "10x", "25x", "50x", "75x", "100x"],
        fontsize=8,
    )

    pg = sdf[sdf["method"] == "PharmaGist"]
    pg_max = 8

    inset = ax.inset_axes((0.3, 0.15, 0.2, 0.3))
    sns.boxplot(
        pg,
        x="method",
        y="Speedup",
        order=["PharmaGist"],
        hue="method",
        hue_order=["PharmaGist"],
        palette=_METHOD_PALETTE,
        linecolor="#555555",
        ax=inset,
        legend=False,
    )
    inset.axhspan(
        inset.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0
    )
    inset.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)
    inset.set_xlabel("")
    inset.set_xticks([])
    inset.set_xlim(-0.5, 0.5)
    inset.set_ylabel("")
    inset.set_yticks([1, 2, 5], ["1x", "2x", "5x"], fontsize=6)
    inset.tick_params(axis="y", length=3, pad=1)
    inset.set_ylim(0, pg_max)
    inset.grid(axis="y", ls=":", alpha=0.4)
    for spine in inset.spines.values():
        spine.set_linewidth(0.75)

    ax.indicate_inset(
        bounds=(-0.45, 0.0, 0.9, pg_max),
        edgecolor="#666666",
        linewidth=0.6,
        alpha=0.8,
        zorder=3,
    )
    cp1 = ConnectionPatch(
        xyA=(0.02, pg_max / 100),
        axesA=ax,
        coordsA="axes fraction",
        xyB=(0, 0),
        axesB=inset,
        coordsB="axes fraction",
        color="#666666",
        linestyle="--",
        linewidth=0.6,
        alpha=0.8,
        zorder=3,
    )
    cp2 = ConnectionPatch(
        xyA=(0.475, pg_max / 100),
        axesA=ax,
        coordsA="axes fraction",
        xyB=(1, 0),
        axesB=inset,
        coordsB="axes fraction",
        color="#666666",
        linestyle="--",
        linewidth=0.6,
        alpha=0.8,
        zorder=3,
    )
    fig.add_artist(cp1)
    fig.add_artist(cp2)

    ax = axes[4]
    for ds_name, sty in DATASET_STYLES.items():
        sns.scatterplot(
            data=rsrc[rsrc["dataset"] == ds_name],
            x="Speedup",
            y="Memory reduction",
            marker=sty["marker"],
            color=sty["color"],
            ax=ax,
            legend=False,
            label=ds_name,
        )
    ax.legend(title="Dataset", title_fontsize=9, fontsize=8, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xticks([0.1, 1, 10, 100], ["0.1x", "1x", "10x", "100x"], fontsize=8)
    ax.set_xlabel("Estimated runtime speedup", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_yticks([0.1, 1, 10, 100], ["0.1x", "1x", "10x", "100x"], fontsize=8)
    ax.set_ylabel("Estimated memory reduction", fontsize=10)
    ax.tick_params(axis="x", which="both", bottom=True)
    ax.set_aspect("equal")

    ax.grid(axis="both", ls=":", alpha=0.5)
    ax.axhspan(ax.get_ylim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axhline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)
    ax.axvspan(ax.get_xlim()[0], 1.0, color="#000000", alpha=0.04, zorder=0)
    ax.axvline(1.0, ls="--", color="#999999", lw=0.8, zorder=1)

    fig.tight_layout()

    for ax, letter in zip(axes[:3], "abc"):
        ax.annotate(
            letter,
            xy=(-0.27, 1.02),
            xycoords="axes fraction",
            fontsize=14,
            fontweight=700,
        )
    axes[3].annotate(
        "d",
        xy=(-0.16, 0.97),
        xycoords="axes fraction",
        fontsize=14,
        fontweight=700,
    )
    axes[4].annotate(
        "e",
        xy=(-0.18, 0.97),
        xycoords="axes fraction",
        fontsize=14,
        fontweight=700,
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"tiebreaking_analysis.{ext}", dpi=300)


@app.command()
def main(
    bench_home: Path,
    nligs_csv: Path = typer.Option(),
    proj_csv: Path = typer.Option(),
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
    rsrc = _load_extra(sdf, nligs_csv, proj_csv)

    _plot(df, sdf, rsrc, out_dir)

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
