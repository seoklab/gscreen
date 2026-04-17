from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from scipy import stats
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


def _plot_enrichment_diagram(bench_home: Path, out_dir: Path):
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

    fig = plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(
        data=data,
        x="GS-SP",
        y="pharma",
        hue="dataset",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.5, 10.0)
    ax.set_xticks([0.5, 1.0, 10.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_yscale("log")
    ax.set_ylim(0.5, 10.0)
    ax.set_yticks([0.5, 1.0, 10.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.legend(
        handles=[
            *ax.legend_.legend_handles,
            plt.Line2D([0], [0], color="gray", linestyle="--"),
        ],
        labels=["DUD-E", "LIT-PCBA", "MUV", "$y = x$"],
        loc="lower right",
        bbox_to_anchor=(1.0, 0),
    )
    ax.set(
        xlabel="GS-SP Similarity Enrichment (Top 1%)",
        ylabel="PharmaGist Similarity Enrichment (Top 1%)",
    )
    line = np.linspace(0.5, 10.0, 100)
    ax.plot(line, line, color="gray", linestyle="--")

    fig.savefig(
        out_dir / "enrichment-pertarget-pt.svg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.close(fig)


def _load_data(bench_home: Path):
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


def _plot_dataset_analysis(gscreen_all: pd.DataFrame, out_dir: Path):
    target_sim = gscreen_all.groupby(["dataset", "target", "is_active"])[
        "ecfp4"
    ].mean()
    target_sim = target_sim.reset_index()
    target_sim["type"] = np.where(target_sim["is_active"], "Active", "Decoy")

    ax = sns.boxplot(
        data=target_sim,
        x="dataset",
        y="ecfp4",
        hue="type",
        hue_order=["Active", "Decoy"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average Tanimoto Similarity to Reference")
    ax.set_ylim(0, 0.5)
    ax.get_legend().set_title("")
    plt.savefig(
        out_dir / "dude-others-compare.svg",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


def _load_active_ratio(all_scores: pd.DataFrame, method: str):
    scores_active = (
        all_scores[all_scores["is_active"]]
        .groupby(["dataset", "target"])[method]
        .mean()
    )
    scores_active = scores_active.to_frame()

    scores_decoy = all_scores.groupby(["dataset", "target"])[method].mean()
    ratio = (scores_active[method] / scores_decoy).to_frame()

    return scores_active.reset_index(), ratio.reset_index()


def _plot_gscreen_analysis(
    scores: pd.DataFrame,
    gscreen_all: pd.DataFrame,
    out_dir: Path,
):
    tani_active, _ = _load_active_ratio(gscreen_all, "ecfp4")
    _, shape_ratio = _load_active_ratio(gscreen_all, "shape")

    auroc = scores[scores["method"] == "galign + gscreen"][
        ["target", "score"]
    ].set_index("target")
    plot_data = (
        tani_active.set_index("target")
        .join([auroc, shape_ratio.set_index("target")["shape"]], how="inner")
        .reset_index()
    )

    plot_data = plot_data.rename(
        columns={"score": "GS-SP", "ecfp4": "tanimoto"}
    )

    rho, p = stats.spearmanr(plot_data["tanimoto"], plot_data["shape"])
    typer.echo(
        f"Tanimoto vs Shape: Spearman's rho = {rho:.3f}, p-value = {p:.3e}"
    )

    rho, p = stats.spearmanr(plot_data["shape"], plot_data["GS-SP"])
    typer.echo(
        f"Shape vs GS-SP: Spearman's rho = {rho:.3f}, p-value = {p:.3e}"
    )

    fig, axes = plt.subplots(
        1,
        2,
        sharey=False,
        gridspec_kw={"wspace": 0.2},
        figsize=(11.23625, 5),
    )

    ax = axes[0]
    sns.scatterplot(
        data=plot_data,
        x="tanimoto",
        y="shape",
        hue="dataset",
        ax=ax,
        legend=False,
    )
    sns.regplot(
        data=plot_data,
        x="tanimoto",
        y="shape",
        scatter=False,
        ax=ax,
    )
    ax.set_xlabel("Mean ECFP4 Similarity (Active)")
    ax.set_xlim(0, 0.5)
    ax.set_ylabel("Alignment-score enrichment (Active / All)")
    ax.set_ylim(0.8, 1.6)
    ax.annotate(
        "a",
        xy=(-0.17, 1),
        xycoords="axes fraction",
        fontsize=21,
        fontweight=700,
    )

    ax = axes[1]
    sns.scatterplot(
        data=plot_data,
        x="shape",
        y="GS-SP",
        hue="dataset",
        ax=ax,
        legend="full",
    )
    sns.regplot(
        data=plot_data,
        x="shape",
        y="GS-SP",
        scatter=False,
        ax=ax,
    )
    ax.set_xlabel("Alignment-score enrichment (Active / All)")
    ax.set_xlim(0.8, 1.6)
    ax.set_ylabel("GS-SP AUROC")
    ax.set_ylim(0.4, 1.1)
    ax.annotate(
        "b",
        xy=(-0.17, 1),
        xycoords="axes fraction",
        fontsize=21,
        fontweight=700,
    )

    fig.legend(
        handles=ax.get_legend().legend_handles,
        labels=["DUD-E", "LIT-PCBA", "MUV"],
        title="Dataset",
        loc="right",
        bbox_to_anchor=(1.03, 0.5),
    )
    ax.get_legend().remove()

    plt.savefig(
        out_dir / "shape_vs_auroc.svg",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )


@app.command()
def main(
    bench_home: Path,
    out_dir: Path = Path("inter-dataset-analysis"),
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

    _plot_enrichment_diagram(bench_home, out_dir)

    scores, gscreen_all = _load_data(bench_home)
    _plot_dataset_analysis(gscreen_all, out_dir)
    _plot_gscreen_analysis(scores, gscreen_all, out_dir)


if __name__ == "__main__":
    app()
