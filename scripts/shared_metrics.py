import math

import numpy as np
import pandas as pd


def tanimoto_distance_matrix(fps: list):
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


def enrichment_factor(labels, scores, ratio: float = 0.01):
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
    return (expected_actives / n_select) / (total_actives / total_len)
