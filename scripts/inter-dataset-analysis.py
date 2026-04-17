import functools
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from scipy import stats
from typer import Typer

app = Typer(pretty_exceptions_enable=False)

_DATASET_STYLE: dict[str, dict] = {
    "DUD-E": {"marker": "o", "color": "#4c72b0"},
    "LIT-PCBA": {"marker": "s", "color": "#dd8452"},
    "MUV": {"marker": "D", "color": "#55a868"},
}


def _plot_enrichment_diagram(bench_home: Path, out_dir: Path):
    cache = out_dir / "enrichment-pertarget.data.parquet"
    if cache.exists():
        data = pd.read_parquet(cache)
    else:
        enrichments = [
            pd.read_csv(bench_home / "dud-e/enrichment.csv"),
            pd.read_csv(bench_home / "lit-pcba/enrichment.csv"),
            pd.read_csv(bench_home / "muv/enrichment.csv"),
        ]

        enrichments[0]["dataset"] = "DUD-E"
        enrichments[1]["dataset"] = "LIT-PCBA"
        enrichments[2]["dataset"] = "MUV"
        enrichments = pd.concat(enrichments, ignore_index=True)

        sp = (
            enrichments[enrichments["method"] == "GS-SP"]
            .drop(columns=["method"])
            .rename(columns={"ratio": "GS-SP"})
            .set_index(["dataset", "target"])
        )
        pg = (
            enrichments[enrichments["method"] == "PG"]
            .drop(columns=["method"])
            .rename(columns={"ratio": "pharma"})
            .set_index(["dataset", "target"])
        )
        data = sp.join(pg).reset_index()
        data.to_parquet(cache)

    lo, hi, rand = 0.5, 10.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 6))

    # Both worse than random
    ax.fill_between(
        [lo, rand],
        lo,
        rand,
        color="#cccccc",
        alpha=0.5,
        lw=0,
        zorder=0,
    )
    # Only PharmaGist worse than random (left strip)
    ax.fill_between(
        [lo, rand],
        rand,
        hi,
        color="#cccccc",
        alpha=0.2,
        lw=0,
        zorder=0,
    )
    # Only GS-SP worse than random (bottom strip)
    ax.fill_between(
        [rand, hi],
        lo,
        rand,
        color="#cccccc",
        alpha=0.2,
        lw=0,
        zorder=0,
    )
    ax.axvline(rand, ls="--", color="#999999", lw=0.7, zorder=1)
    ax.axhline(rand, ls="--", color="#999999", lw=0.7, zorder=1)

    # Winner shading: above diagonal = GS-SP wins
    ax.fill_between(
        [lo, hi],
        [lo, hi],
        hi,
        color="#4c72b0",
        alpha=0.05,
        lw=0,
    )
    # Below diagonal = PharmaGist wins
    ax.fill_between(
        [lo, hi],
        lo,
        [lo, hi],
        color="#dd8452",
        alpha=0.05,
        lw=0,
    )

    line = np.linspace(lo, hi, 100)
    ax.plot(line, line, color="gray", linestyle="--", lw=0.8, label="$y = x$")

    for ds, sty in _DATASET_STYLE.items():
        mask = data["dataset"] == ds
        if not mask.any():
            continue
        ax.scatter(
            data.loc[mask, "pharma"],
            data.loc[mask, "GS-SP"],
            marker=sty["marker"],
            color=sty["color"],
            edgecolors="white",
            linewidths=0.3,
            label=ds,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks([0.5, 1.0, 10.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_yticks([0.5, 1.0, 10.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.legend(loc="lower right")

    ax.set(
        xlabel="PharmaGist Similarity Enrichment (Top 1%)",
        ylabel="GS-SP Similarity Enrichment (Top 1%)",
    )

    for ext in ("svg", "pdf"):
        fig.savefig(
            (out_dir / "enrichment-pertarget").with_suffix(f".{ext}"),
            bbox_inches="tight",
            dpi=500,
        )
    plt.close(fig)


@functools.cache
def _load_data(bench_home: Path):
    typer.echo("Loading scores and gscreen data...")

    scores = [
        pd.read_csv(bench_home / "dud-e/scores.csv"),
        pd.read_csv(bench_home / "lit-pcba/scores.csv"),
        pd.read_csv(bench_home / "muv/scores.csv"),
    ]
    scores[0]["dataset"] = "DUD-E"
    scores[1]["dataset"] = "LIT-PCBA"
    scores[2]["dataset"] = "MUV"
    scores = pd.concat(scores, ignore_index=True)

    gscreen_all = [
        pd.read_csv(bench_home / "dud-e/gscreen.csv"),
        pd.read_csv(bench_home / "lit-pcba/gscreen.csv"),
        pd.read_csv(bench_home / "muv/gscreen.csv"),
    ]
    gscreen_all[0]["dataset"] = "DUD-E"
    gscreen_all[1]["dataset"] = "LIT-PCBA"
    gscreen_all[2]["dataset"] = "MUV"
    gscreen_all = pd.concat(gscreen_all, ignore_index=True)

    return scores, gscreen_all


def _plot_dataset_analysis(bench_home: Path, out_dir: Path):
    cache = out_dir / "dude-others-compare.data.parquet"
    if cache.exists():
        target_sim = pd.read_parquet(cache)
    else:
        _, gscreen_all = _load_data(bench_home)
        target_sim = gscreen_all.groupby(["dataset", "target", "is_active"])[
            "ecfp4"
        ].mean()
        target_sim = target_sim.reset_index()
        target_sim["type"] = np.where(
            target_sim["is_active"], "Active", "Decoy"
        )
        target_sim.to_parquet(cache)

    fig, ax = plt.subplots()
    sns.boxplot(
        data=target_sim,
        x="dataset",
        y="ecfp4",
        hue="type",
        hue_order=["Active", "Decoy"],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average Tanimoto Similarity to Reference")
    ax.set_ylim(0, 0.5)
    ax.get_legend().set_title("")
    for ext in ("svg", "pdf"):
        fig.savefig(
            (out_dir / "dude-others-compare").with_suffix(f".{ext}"),
            dpi=500,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close(fig)


def _load_active_ratio(all_scores: pd.DataFrame, method: str):
    scores_active = (
        all_scores[all_scores["is_active"]]
        .groupby(["dataset", "target"])[method]
        .mean()
    )
    scores_active = scores_active.to_frame()

    scores_all = all_scores.groupby(["dataset", "target"])[method].mean()
    ratio = (scores_active[method] / scores_all).to_frame()

    return scores_active.reset_index(), ratio.reset_index()


def _plot_gscreen_analysis(bench_home: Path, out_dir: Path):
    cache = out_dir / "shape_vs_auroc.data.parquet"
    if cache.exists():
        plot_data = pd.read_parquet(cache)
    else:
        scores, gscreen_all = _load_data(bench_home)
        sim_active, _ = _load_active_ratio(gscreen_all, "ecfp4")
        _, shape_ratio = _load_active_ratio(gscreen_all, "shape")

        auroc = scores[
            (scores["method"] == "GS-SP") & (scores["metric"] == "aucroc")
        ][["dataset", "target", "score"]].set_index(["dataset", "target"])
        plot_data = (
            sim_active.set_index(["dataset", "target"])
            .join(
                [auroc, shape_ratio.set_index(["dataset", "target"])["shape"]],
                how="inner",
            )
            .reset_index()
        )
        plot_data = plot_data.rename(
            columns={"score": "GS-SP", "ecfp4": "tanimoto"}
        )
        plot_data.to_parquet(cache)

    rho_a, p_a = stats.spearmanr(plot_data["tanimoto"], plot_data["shape"])
    typer.echo(
        f"Tanimoto vs Shape: Spearman's rho = {rho_a:.3f}, p-value = {p_a:.3e}"
    )

    rho_b, p_b = stats.spearmanr(plot_data["shape"], plot_data["GS-SP"])
    typer.echo(
        f"Shape vs GS-SP: Spearman's rho = {rho_b:.3f}, p-value = {p_b:.3e}"
    )

    fig, axes = plt.subplots(
        1,
        2,
        sharey=False,
        gridspec_kw={"wspace": 0.2},
        figsize=(11.23625, 5),
    )
    key_label = "Baseline-normalized Active Alignment"

    # Panel a
    ax = axes[0]
    x_lo, x_hi = 0, 0.5
    y_lo_a, y_hi_a = 0.8, 1.6
    ax.fill_between(
        [x_lo, x_hi],
        y_lo_a,
        1.0,
        color="#cccccc",
        alpha=0.3,
        lw=0,
        zorder=0,
    )
    ax.axhline(1.0, ls="--", color="#999999", lw=0.7, zorder=1)

    for ds, sty in _DATASET_STYLE.items():
        mask = plot_data["dataset"] == ds
        if not mask.any():
            continue
        ax.scatter(
            plot_data.loc[mask, "tanimoto"],
            plot_data.loc[mask, "shape"],
            marker=sty["marker"],
            color=sty["color"],
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    sns.regplot(
        data=plot_data,
        x="tanimoto",
        y="shape",
        scatter=False,
        ax=ax,
        color="grey",
    )
    ax.set_xlabel("Mean ECFP4 Similarity (Active)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel(key_label)
    ax.set_ylim(y_lo_a, y_hi_a)
    stars_a = (
        "***"
        if p_a < 0.001
        else "**"
        if p_a < 0.01
        else "*"
        if p_a < 0.05
        else ""
    )
    ax.annotate(
        f"$\\rho$={rho_a:+.2f}{stars_a}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white", alpha=0.7),
    )
    ax.annotate(
        "a",
        xy=(-0.13, 0.97),
        xycoords="axes fraction",
        fontsize=16,
        fontweight=700,
    )

    # Panel b
    ax = axes[1]
    x_lo_b, x_hi_b = 0.8, 1.6
    y_lo_b, y_hi_b = 0.4, 1.1
    rand_x, rand_y = 1.0, 0.5

    # Enrichment worse (left strip) — grey
    ax.fill_between(
        [x_lo_b, rand_x],
        y_lo_b,
        y_hi_b,
        color="#bbbbbb",
        alpha=0.2,
        lw=0,
        zorder=0,
    )
    # AUROC good (top strip) — amber
    ax.fill_between(
        [x_lo_b, x_hi_b],
        rand_y,
        y_hi_b,
        color="#e8b058",
        alpha=0.05,
        lw=0,
        zorder=0,
    )
    # AUROC bad (bottom strip) — cool blue
    ax.fill_between(
        [x_lo_b, x_hi_b],
        y_lo_b,
        rand_y,
        color="#6ca0d8",
        alpha=0.08,
        lw=0,
        zorder=0,
    )

    ax.axvline(rand_x, ls="--", color="#999999", lw=0.7, zorder=1)
    ax.axhline(rand_y, ls="--", color="#999999", lw=0.7, zorder=1)

    for ds, sty in _DATASET_STYLE.items():
        mask = plot_data["dataset"] == ds
        if not mask.any():
            continue
        ax.scatter(
            plot_data.loc[mask, "shape"],
            plot_data.loc[mask, "GS-SP"],
            marker=sty["marker"],
            color=sty["color"],
            edgecolors="white",
            linewidths=0.3,
            label=ds,
            zorder=3,
        )
    sns.regplot(
        data=plot_data,
        x="shape",
        y="GS-SP",
        scatter=False,
        ax=ax,
        color="grey",
    )
    ax.set_xlabel(key_label)
    ax.set_xlim(x_lo_b, x_hi_b)
    ax.set_ylabel("GS-SP AUROC")
    ax.set_ylim(y_lo_b, y_hi_b)
    stars_b = (
        "***"
        if p_b < 0.001
        else "**"
        if p_b < 0.01
        else "*"
        if p_b < 0.05
        else ""
    )
    ax.annotate(
        f"$\\rho$={rho_b:+.2f}{stars_b}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white", alpha=0.7),
    )
    ax.annotate(
        "b",
        xy=(-0.13, 0.97),
        xycoords="axes fraction",
        fontsize=16,
        fontweight=700,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        title="Dataset",
        loc="right",
        bbox_to_anchor=(1.03, 0.5),
    )
    try:
        ax.get_legend().remove()
    except AttributeError:
        pass

    for ext in ("svg", "pdf"):
        fig.savefig(
            (out_dir / "shape_vs_auroc").with_suffix(f".{ext}"),
            dpi=500,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close(fig)


@app.command()
def main(
    bench_home: Path = Path.home() / "repo/seoklab/gscreen-data/benchmark",
    out_dir: Path = Path("inter-dataset-analysis"),
    plot_enrichment: bool = False,
    plot_dataset_analysis: bool = False,
    plot_gscreen_analysis: bool = True,
):
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Helvetica Neue",
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )

    out_dir.mkdir(exist_ok=True, parents=True)

    if plot_enrichment:
        typer.echo("Plotting enrichment diagram...")
        _plot_enrichment_diagram(bench_home, out_dir)

    if plot_dataset_analysis:
        typer.echo("Plotting dataset analysis...")
        _plot_dataset_analysis(bench_home, out_dir)

    if plot_gscreen_analysis:
        typer.echo("Plotting gscreen analysis...")
        _plot_gscreen_analysis(bench_home, out_dir)

    typer.echo(f"Done. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    app()
