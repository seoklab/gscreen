"""Plot per-target ROC curves for all methods across benchmark datasets.

Reads gscreen and external method scores, computes ROC curves, and generates
multi-page SVG figures with consistent line styles across all panels.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn import metrics as skmetrics

from shared_metrics import (
    ALL_METHODS,
    METHOD_SLUG_MAP,
    METHOD_STYLES,
    load_gscreen_scores,
    load_method_scores,
)

app = typer.Typer(pretty_exceptions_enable=False)

NCOLS = 4
NROWS = 6
_PANELS_PER_PAGE = NCOLS * NROWS

# A4 in inches
# Text area: A4 minus 2.54 cm margins on each side
_FIG_W = (21.0 - 2 * 2.54) / 2.54  # 6.27 in
_FIG_H = (29.7 - 2 * 2.54) / 2.54  # 9.69 in


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------


def _compute_roc(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = skmetrics.roc_curve(labels, scores)
    return fpr, tpr


def _collect_rocs(
    gscreen: dict[str, pd.DataFrame],
    externals: dict[str, dict[str, pd.DataFrame]],
    db_label: str,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Returns {label: {method: (fpr, tpr)}} with label = 'DB / target'."""
    targets = sorted(gscreen.keys())
    result: dict[str, dict[str, tuple]] = {}

    gscreen_submethods = {
        "GS-S": "shape",
        "GS-P": "pharma",
        "GS-SP": "score",
    }
    for target in targets:
        label = f"{db_label} / {target}"
        result[label] = {}
        df = gscreen[target]
        for method, col in gscreen_submethods.items():
            result[label][method] = _compute_roc(df["is_active"], df[col])

    for slug, method_scores in externals.items():
        method_name = METHOD_SLUG_MAP.get(slug, slug)
        for target, df in method_scores.items():
            label = f"{db_label} / {target}"
            if label not in result:
                continue
            result[label][method_name] = _compute_roc(
                df["is_active"],
                df["score"],
            )

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_roc_panel(
    ax: plt.Axes,
    rocs: dict[str, tuple[np.ndarray, np.ndarray]],
    target: str,
):
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.5, linestyle="--")

    for method in ALL_METHODS:
        if method not in rocs:
            continue
        fpr, tpr = rocs[method]
        style = METHOD_STYLES[method]
        ax.plot(fpr, tpr, **style)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(target, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)


def _make_legend_handles():
    handles = []
    for method in ALL_METHODS:
        style = METHOD_STYLES[method]
        (h,) = plt.plot([], [], label=method, **style)
        handles.append(h)
    return handles


def _plot_pages(
    all_rocs: dict[str, dict[str, tuple]],
    output_dir: Path,
):
    targets = list(all_rocs.keys())
    n_pages = max(1, -(-len(targets) // _PANELS_PER_PAGE))

    handles = _make_legend_handles()

    for page in range(n_pages):
        start = page * _PANELS_PER_PAGE
        page_targets = targets[start : start + _PANELS_PER_PAGE]
        n = len(page_targets)
        nrows = min(NROWS, -(-n // NCOLS))

        _LEGEND_H = 0.35  # inches, fixed space for legend
        panel_h = (_FIG_H - _LEGEND_H) * nrows / NROWS
        fig_h = panel_h + _LEGEND_H
        legend_frac = _LEGEND_H / fig_h
        fig, axes = plt.subplots(
            nrows,
            NCOLS,
            figsize=(_FIG_W, fig_h),
            squeeze=False,
        )

        for i, target in enumerate(page_targets):
            r, c = divmod(i, NCOLS)
            _draw_roc_panel(axes[r, c], all_rocs[target], target)

        for i in range(n, nrows * NCOLS):
            r, c = divmod(i, NCOLS)
            axes[r, c].set_visible(False)

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(ALL_METHODS),
            fontsize=7,
            frameon=False,
        )

        fig.tight_layout(rect=[0, legend_frac, 1, 1], h_pad=0.8)
        out = output_dir / f"roc_curves_{page + 1}.svg"
        fig.savefig(out)
        plt.close(fig)

    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    results: Path,
    databases: str = "dud-e,lit-pcba,muv",
    db_home: Path = Path.home() / "db",
    bench_home: Path = Path.home() / "benchmark",
    fallback_home: Optional[Path] = None,
    skip_missing: bool = True,
):
    db_names = [s.strip() for s in databases.split(",")]

    all_rocs: dict[str, dict[str, tuple]] = {}
    for db in db_names:
        db_results = results / db
        db_db = db_home / db
        db_bench = bench_home / db
        db_fallback = fallback_home / db if fallback_home is not None else None

        typer.echo(f"Loading gscreen scores for {db} ...")
        gscreen = load_gscreen_scores(db_results, db_db, db_fallback)
        typer.echo(f"  {len(gscreen)} targets")

        typer.echo(f"Loading external method scores for {db} ...")
        externals: dict[str, dict[str, pd.DataFrame]] = {}
        for slug, name in METHOD_SLUG_MAP.items():
            externals[slug] = load_method_scores(
                db_bench,
                slug,
                skip_missing=skip_missing,
            )
            typer.echo(f"  {name}: {len(externals[slug])} targets")

        db_label = db.upper()
        typer.echo("Computing ROC curves ...")
        all_rocs.update(_collect_rocs(gscreen, externals, db_label))

    typer.echo(f"\nPlotting to {results}/ ...")
    _plot_pages(all_rocs, results)

    n_pages = -(-len(all_rocs) // _PANELS_PER_PAGE)
    typer.echo(f"  {len(all_rocs)} targets, {n_pages} SVG file(s)")


if __name__ == "__main__":
    app()
