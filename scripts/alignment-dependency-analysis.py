"""Compare GS-P (G-align) vs GS-P (PharmaGist) alignment dependency."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.ticker import FixedLocator, SymmetricalLogLocator
from scipy.stats import wilcoxon

from shared_metrics import DATASET_STYLES

app = typer.Typer()

_G = "GS-P (G-align)"
_P = "GS-P (PharmaGist)"
_METRICS = ["aucroc", "ef 1%"]
_METRIC_LABELS = {"aucroc": "AUROC", "ef 1%": "EF1%"}


def _load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    ds_map = df.drop_duplicates(subset=["target"])[["target", "dataset"]]
    wide = df.pivot_table(
        index=["target", "metric"],
        columns="method",
        values="score",
    ).reset_index()
    wide.columns.name = None
    wide = wide.merge(ds_map, on="target")
    wide["dataset"] = wide["dataset"].str.upper()
    return wide


def _paired_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-target paired comparison for each metric."""
    rows = []
    for metric in _METRICS:
        sub = wide[wide["metric"] == metric].sort_values("target")
        for _, r in sub.iterrows():
            rows.append(
                {
                    "metric": _METRIC_LABELS.get(metric, metric),
                    "target": r["target"],
                    "G-align": r[_G],
                    "PharmaGist": r[_P],
                    "diff (G − P)": r[_G] - r[_P],
                }
            )
    return pd.DataFrame(rows)


def _summary_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in _METRICS:
        sub = wide[wide["metric"] == metric]
        g = sub[_G].values
        p = sub[_P].values
        diff = g - p

        _, pval = wilcoxon(g, p, alternative="two-sided")

        rows.append(
            {
                "metric": _METRIC_LABELS.get(metric, metric),
                "mean G-align": np.mean(g),
                "mean PharmaGist": np.mean(p),
                "mean diff (G − P)": np.mean(diff),
                "median diff (G − P)": np.median(diff),
                "G-align wins": int(np.sum(diff > 0)),
                "PharmaGist wins": int(np.sum(diff < 0)),
                "ties": int(np.sum(diff == 0)),
                "n": len(diff),
                "Wilcoxon p": pval,
            }
        )
    return pd.DataFrame(rows)


_RANDOM_LEVEL = {"aucroc": 0.5, "ef 1%": 1.0}


def _draw_panel(ax, sub, metric, add_labels):
    g = sub[_G].values
    p = sub[_P].values
    rand = _RANDOM_LEVEL[metric]

    all_vals = np.concatenate([g, p])
    margin = (all_vals.max() - all_vals.min()) * 0.06
    lo = max(0, all_vals.min() - margin)
    hi = all_vals.max() + margin

    if metric == "ef 1%":
        hi = 10 ** np.ceil(np.log10(max(hi, 1.01)))

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
    # Only G-align worse than random (bottom strip)
    ax.fill_between(
        [rand, hi],
        lo,
        rand,
        color="#cccccc",
        alpha=0.2,
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
    ax.axvline(rand, ls="--", color="#999999", lw=0.7, zorder=1)
    ax.axhline(rand, ls="--", color="#999999", lw=0.7, zorder=1)

    # Diagonal y=x
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=0.8, zorder=1)
    # Winner shading (above/below diagonal)
    ax.fill_between(
        [lo, hi],
        [lo, hi],
        hi,
        color="#4c72b0",
        alpha=0.05,
        lw=0,
    )
    ax.fill_between(
        [lo, hi],
        lo,
        [lo, hi],
        color="#dd8452",
        alpha=0.05,
        lw=0,
    )

    for ds, sty in DATASET_STYLES.items():
        mask = sub["dataset"] == ds
        if not mask.any():
            continue
        ax.scatter(
            sub.loc[mask, _P],
            sub.loc[mask, _G],
            s=40,
            marker=sty["marker"],
            color=sty["color"],
            edgecolors="white",
            linewidths=0.4,
            label=ds if add_labels else None,
            zorder=3,
        )

    if metric == "ef 1%":
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        log_minor = SymmetricalLogLocator(
            base=10,
            linthresh=1,
            subs=np.arange(2, 10),
        )
        lin_minor = np.arange(0.2, 1.0, 0.2)
        for axis in (ax.xaxis, ax.yaxis):
            log_ticks = log_minor.tick_values(lo, hi)
            all_minor = np.unique(np.concatenate([log_ticks, lin_minor]))
            axis.set_minor_locator(FixedLocator(all_minor))
        ax.tick_params(which="minor", length=3, width=0.5)

    label = _METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(f"PharmaGist Aligned ({label})")
    ax.set_ylabel(f"G-align Aligned ({label})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)


def _plot(wide: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, len(_METRICS), figsize=(8.7, 4.2))

    for i, (ax, metric) in enumerate(zip(axes, _METRICS)):
        sub = wide[wide["metric"] == metric]
        _draw_panel(ax, sub, metric, add_labels=(i == 0))

    for ax, letter in zip(axes, "ab"):
        ax.annotate(
            letter,
            xy=(-0.18, 0.97),
            xycoords="axes fraction",
            fontsize=14,
            fontweight=700,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        fontsize=8,
        title="Dataset",
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 0.90, 1))
    for ext in ("svg", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


@app.command()
def main(
    csv: Path,
    output_dir: Path = Path("alignment_dependency_results"),
):
    wide = _load(csv)

    typer.echo("=== Per-target comparison ===")
    paired = _paired_table(wide)
    typer.echo(paired.to_string(index=False))

    typer.echo("\n=== Summary ===")
    summary = _summary_table(wide)
    typer.echo(summary.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    _plot(wide, output_dir / "alignment_comparison")
    typer.echo(f"\nFigures saved to {output_dir}/")


if __name__ == "__main__":
    app()
