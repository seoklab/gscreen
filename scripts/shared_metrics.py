import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openbabel import pybel
from sklearn import metrics as skmetrics

_logger = logging.getLogger(__name__)

GSCREEN_METHODS = ["GS-S", "GS-P", "GS-SP"]
BASELINE_METHODS = ["Flexi-LS-align", "PharmaGist", "AutoDock Vina"]
ALL_METHODS = GSCREEN_METHODS + BASELINE_METHODS

METHOD_SLUG_MAP = {
    "ls-align": "Flexi-LS-align",
    "pharmagist": "PharmaGist",
    "autodock-vina": "AutoDock Vina",
}

TICK_LABELS = {
    "Flexi-LS-align": "LA",
    "PharmaGist": "PG",
    "AutoDock Vina": "Vina",
}

DATASET_STYLES = {
    "DUD-E": {"marker": "o", "color": "#4c72b0"},
    "LIT-PCBA": {"marker": "s", "color": "#dd8452"},
    "MUV": {"marker": "D", "color": "#55a868"},
}

METHOD_STYLES = {
    "GS-S": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.2},
    "GS-P": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.2},
    "GS-SP": {"color": "#009E73", "linestyle": "-", "linewidth": 1.6},
    "Flexi-LS-align": {
        "color": "#CC79A7",
        "linestyle": "--",
        "linewidth": 1.0,
    },
    "PharmaGist": {"color": "#F0E442", "linestyle": "--", "linewidth": 1.0},
    "AutoDock Vina": {"color": "#56B4E9", "linestyle": ":", "linewidth": 1.0},
}


def tanimoto_distance_matrix(fps: list[pybel.Fingerprint]):
    """Compute condensed pairwise Tanimoto distance matrix."""
    n = len(fps)
    dists = np.empty(n * (n - 1) // 2, dtype=float)

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dists[k] = 1.0 - (fps[i] | fps[j])
            k += 1

    return dists


def ecfp4_weight(df: pd.DataFrame):
    ecfp4 = df["ecfp4"].to_numpy()
    return np.clip(6 * (ecfp4 - 0.1), 0, 0.9)


def enrichment_factor(
    labels,
    scores,
    ratio: float = 0.01,
    strict_mode: bool = False,
) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    total_len = len(scores)
    n_select = max(1, math.ceil(ratio * total_len))

    kth = total_len - n_select
    threshold = np.partition(scores, kth)[kth]

    above = scores > threshold
    tied = scores == threshold

    total_actives = labels.sum()
    assert total_actives > 0

    n_above = above.sum()
    actives_above = labels[above].sum()
    n_tied = max(tied.sum(), 1)
    actives_tied = labels[tied].sum()

    n_from_tied = n_select - n_above
    expected_actives = actives_above + actives_tied * (n_from_tied / n_tied)
    if strict_mode:
        n_select = n_above + n_tied
    return (expected_actives / n_select) / (total_actives / total_len)


def load_gscreen_scores(
    results: Path,
    db_home: Path,
    fallback: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    scores: dict[str, pd.DataFrame] = {}
    for db_target in sorted(db_home.iterdir()):
        if not db_target.is_dir():
            continue

        key = db_target.name
        score_csv = results / key / "scores.csv"
        if fallback is not None and not score_csv.is_file():
            score_csv = fallback / key / "scores.csv"

        df = pd.read_csv(score_csv)
        if "is_active" not in df.columns:
            df["is_active"] = df["type"] == "active"
            df = df.drop(columns=["type"])
            df = df.rename(
                columns={
                    "pharma_score": "pharma",
                    "shape_score": "shape",
                    "tani_sim": "ecfp4",
                }
            )

        weight = ecfp4_weight(df)
        df["score"] = df["shape"] * weight + df["pharma"] * (1 - weight)
        df["id"] = df["id"].astype(str).str.strip()
        df["target"] = key
        scores[key] = df

    return scores


def load_method_scores(
    bench_home: Path,
    method_slug: str,
    skip_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    scores: dict[str, pd.DataFrame] = {}
    outputs = bench_home / method_slug / "outputs"
    if not outputs.is_dir():
        if skip_missing:
            _logger.warning("No outputs directory for %s", method_slug)
            return scores
        raise FileNotFoundError(outputs)

    for target_dir in sorted(outputs.iterdir()):
        if not target_dir.is_dir():
            continue

        try:
            df = pd.read_csv(target_dir / "scores.csv", index_col=0)
        except FileNotFoundError:
            if skip_missing:
                _logger.warning(
                    "Missing scores for %s in %s",
                    target_dir.name,
                    method_slug,
                )
                continue
            raise

        df["id"] = df["id"].astype(str).str.strip()
        df["target"] = target_dir.name
        scores[target_dir.name] = df

    return scores


def compute_metrics(
    df: pd.DataFrame,
    score_col: str,
    active_col: str,
    ratios: list[float],
    metric_names: list[str],
) -> dict[str, float]:
    return {
        metric_names[0]: skmetrics.roc_auc_score(
            df[active_col], df[score_col]
        ),
        **{
            name: enrichment_factor(df[active_col], df[score_col], ratio=r)
            for name, r in zip(metric_names[1:], ratios)
        },
    }
