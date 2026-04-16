from collections import defaultdict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from openbabel import openbabel as ob
from openbabel import pybel
from tqdm import tqdm
from typer import Typer

app = Typer(pretty_exceptions_enable=False)

_input_type_suffix = defaultdict(lambda: "", {"dud-e": "_final"})


def _count_one_mol2(mol2: Path, active: bool):
    metrics = []
    for mol in pybel.readfile("mol2", str(mol2)):
        mol.removeh()
        metrics.append(
            (
                mol.title.strip(),
                mol.OBMol.NumHvyAtoms(),
                mol.OBMol.NumBonds(),
                mol.OBMol.NumRotors(),
            )
        )
    df = pd.DataFrame(
        metrics,
        columns=["id", "nheavy", "nbonds", "nrotors"],
    )
    df["is_active"] = active
    return df


def _count_one_target(target_home: Path, suffix: str):
    ob.obErrorLog.StopLogging()
    mols = [
        _count_one_mol2(
            target_home / f"actives{suffix}_corina_addchg.mol2", True
        ),
        _count_one_mol2(
            target_home / f"decoys{suffix}_corina_addchg.mol2", False
        ),
    ]
    df = pd.concat(mols, ignore_index=True)
    df["target"] = target_home.name
    return df


@app.command()
def main(db_home: Path, nproc: int = 8):
    targets = sorted(d for d in db_home.iterdir() if d.is_dir())

    dfs = list(
        tqdm(
            Parallel(n_jobs=nproc, return_as="generator_unordered")(
                delayed(_count_one_target)(
                    target_home,
                    _input_type_suffix[target_home.parent.name],
                )
                for target_home in targets
            ),
            total=len(targets),
        )
    )

    df = pd.concat(dfs, ignore_index=True)
    df["dataset"] = db_home.name
    df.to_csv(db_home / "ligand-flexibility-metrics.csv", index=False)


if __name__ == "__main__":
    app()
