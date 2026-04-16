import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openbabel import openbabel as ob
from openbabel import pybel
from scipy.cluster import hierarchy as hier
from scipy.spatial.distance import squareform

from shared_metrics import tanimoto_distance_matrix

ob.obErrorLog.StopLogging()


def load_target_ligands(target_dir: Path):
    """Load *_ligand.mol2 files for one target, return (names, fingerprints)."""
    names: list[str] = []
    fps: list[pybel.Fingerprint] = []

    for mol2 in sorted(target_dir.glob("*_ligand.mol2")):
        if mol2.stem.startswith("crystal_"):
            continue

        mol = next(pybel.readfile("mol2", str(mol2)))
        names.append(mol2.stem)
        fps.append(mol.calcfp("ecfp4"))

    return names, fps


def read_crystal_pdbid(target_dir: Path) -> str:
    """Read the 'final: <pdbid>' line from selection.txt."""
    sel = target_dir / "selection.txt"
    for line in sel.read_text().splitlines():
        line = line.strip()
        if line.startswith("final:"):
            return line.split()[1].strip()

    raise ValueError("No pdb id")


def cluster_target(names, fps, method, max_clusters):
    """Cluster ligands for a single target.

    Returns (cluster_labels, threshold, dist_matrix) where threshold is the
    effective distance cutoff used (0.0 if no merging was needed) and
    dist_matrix is the square distance matrix (None if n < 2).
    """
    n = len(fps)
    if n <= max_clusters:
        dist_sq = squareform(tanimoto_distance_matrix(fps)) if n >= 2 else None
        return np.arange(1, n + 1), 0.0, dist_sq

    dists = tanimoto_distance_matrix(fps)
    lnk = hier.linkage(dists, method=method)
    labels = hier.fcluster(lnk, max_clusters, criterion="maxclust")

    threshold = float(lnk[n - max_clusters - 1, 2])
    return labels, threshold, squareform(dists)


def find_cluster_centers(names, labels, dist_matrix, crystal_idx=None):
    """Find the medoid (center) of each cluster.

    If crystal_idx is given, that ligand is forced as the center of its
    cluster (overriding the medoid).

    Returns dict mapping cluster_id -> (center_index, center_name) where
    center_name is names[idx].
    """
    crystal_cid = labels[crystal_idx] if crystal_idx is not None else None
    centers: dict[int, tuple[int, str]] = {}

    for cid in sorted(set(labels)):
        if cid == crystal_cid:
            centers[cid] = (crystal_idx, names[crystal_idx])
            continue

        members = [i for i, l in enumerate(labels) if l == cid]
        if len(members) == 1:
            centers[cid] = (members[0], names[members[0]])
            continue

        sub = dist_matrix[np.ix_(members, members)]
        medoid_local = sub.sum(axis=1).argmin()
        idx = members[medoid_local]
        centers[cid] = (idx, names[idx])
    return centers


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Per-target clustering of reference ligands by ECFP4.",
    )
    parser.add_argument(
        "db_home",
        type=Path,
        help="Database root (e.g. ~/db/lit-pcba)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <db_home>/ligand_clusters.csv)",
    )
    parser.add_argument(
        "-n",
        "--max-clusters",
        type=int,
        default=5,
        help="Maximum number of clusters per target (default: 5)",
    )
    parser.add_argument(
        "-m",
        "--method",
        default="average",
        help="Linkage method (default: average)",
    )
    return parser


def main():
    args = _get_parser().parse_args()
    db_home: Path = args.db_home.expanduser()
    output: Path = args.output or db_home / "ligand_clusters.csv"

    rows: list[dict] = []
    center_rows: list[dict] = []

    for target_dir in sorted(db_home.iterdir()):
        if not target_dir.is_dir():
            continue

        target = target_dir.name
        names, fps = load_target_ligands(target_dir)
        if not fps:
            continue

        clusters, threshold, dist_matrix = cluster_target(
            names,
            fps,
            args.method,
            args.max_clusters,
        )
        n_clusters = len(set(clusters))

        try:
            crystal_pdbid = read_crystal_pdbid(target_dir)
            crystal_name = f"{crystal_pdbid}_ligand"
            crystal_idx = names.index(crystal_name)
        except FileNotFoundError:
            assert len(names) == 1
            crystal_name = names[0]
            crystal_idx = 0

        if dist_matrix is not None:
            centers = find_cluster_centers(
                names,
                clusters,
                dist_matrix,
                crystal_idx,
            )
        else:
            centers = {1: (0, names[0])}

        print(
            f"{target}: {len(fps)} ligands -> {n_clusters} clusters "
            f"(threshold={threshold:.3f})"
        )
        for name, cid in zip(names, clusters):
            rows.append(
                {
                    "target": target,
                    "ligand": name,
                    "cluster": int(cid),
                    "threshold": threshold,
                }
            )

        for cid, (center_idx, center_name) in centers.items():
            center_rows.append(
                {
                    "target": target,
                    "cluster": int(cid),
                    "center": center_name,
                    "size": int(sum(clusters == cid)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nResults written to {output}")

    centers_output = output.with_name(
        output.stem + "_centers" + output.suffix,
    )
    centers_df = pd.DataFrame(center_rows)
    centers_df.to_csv(centers_output, index=False)
    print(f"Cluster centers written to {centers_output}")

    print("\nPer-target summary:")
    for target, group in df.groupby("target"):
        sizes = group["cluster"].value_counts()
        print(
            f"  {target}: {len(group)} ligands, {len(sizes)} clusters "
            f"(singletons={sum(sizes == 1)}, max={sizes.max()})"
        )
        for cid, cgroup in group.groupby("cluster"):
            members = ", ".join(cgroup["ligand"])
            print(f"    [{cid}] ({len(cgroup)}): {members}")


if __name__ == "__main__":
    main()
