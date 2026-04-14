import itertools
from pathlib import Path

import numpy as np
import typer
from openbabel import openbabel as ob
from openbabel import pybel
from scipy.cluster import hierarchy as hier
from scipy.spatial.distance import squareform
from tqdm import tqdm

from shared_metrics import tanimoto_distance_matrix
from gscreen import io as gio

ob.obErrorLog.StopLogging()

app = typer.Typer(pretty_exceptions_enable=False)


def _cluster_and_select(mols: list[gio.Mol2], max_clusters: int, method: str):
    """Cluster actives by ECFP4 Tanimoto and return medoid indices."""
    n = len(mols)
    if n <= max_clusters:
        return list(range(n))

    fps = [pybel.readstring("mol2", str(mol)).calcfp("ecfp4") for mol in mols]

    dists = tanimoto_distance_matrix(fps)
    lnk = hier.linkage(dists, method=method)
    labels = hier.fcluster(lnk, max_clusters, criterion="maxclust")
    dist_sq = squareform(dists)

    selected: list[int] = []
    for cid in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == cid]
        if len(members) == 1:
            selected.append(members[0])
        else:
            sub = dist_sq[np.ix_(members, members)]
            selected.append(members[sub.sum(axis=1).argmin()])

    return selected


@app.command()
def main(
    actives: Path,
    decoys: Path,
    output_dir: str = "subset",
    n_decoys: int = typer.Option(
        1000, "--n-decoys", "-d", help="Target number of decoys to sample"
    ),
    max_actives: int = typer.Option(
        10,
        "--max-actives",
        "-a",
        help="Maximum number of active clusters (medoids) to keep",
    ),
    linkage_method: str = typer.Option(
        "average",
        "--linkage",
        "-l",
        help="Linkage method for hierarchical clustering",
    ),
    seed: int = 42,
):
    amols = list(tqdm(gio.Mol2Reader(actives), desc="Reading actives"))
    active_sel = _cluster_and_select(amols, max_actives, linkage_method)
    print(f"Actives: {len(amols)} -> {len(active_sel)} (clustered)")

    rng = np.random.default_rng(seed)
    dmols = gio.Mol2Reader(decoys)
    n_decoy_sel = min(n_decoys, dmols.count)
    decoy_mask = np.zeros(dmols.count, dtype=bool)
    sel = rng.choice(dmols.count, size=n_decoy_sel, replace=False)
    decoy_mask[sel] = True
    print(f"Decoys: {dmols.count} -> {n_decoy_sel} (random)")

    (actives.parent / output_dir).mkdir(parents=True, exist_ok=True)
    with open(actives.parent / output_dir / actives.name, "wb") as f:
        for i in active_sel:
            f.write(bytes(amols[i]))

    (decoys.parent / output_dir).mkdir(parents=True, exist_ok=True)
    with open(decoys.parent / output_dir / decoys.name, "wb") as f:
        for mol in itertools.compress(tqdm(dmols), decoy_mask):
            f.write(bytes(mol))


if __name__ == "__main__":
    app()
