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
