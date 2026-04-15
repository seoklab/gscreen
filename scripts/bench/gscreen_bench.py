from pathlib import Path

import pandas as pd
import typer
from openbabel import pybel

from gscreen import pcdetect
from gscreen.cli.gscreen import _load_clusters
from gscreen.pipeline import Parallelizer, Pipeline, modulize
from gscreen.tools import Chimera, GAlign


def gscreen_main(
    input_file: Path,
    ref_mol: Path,
    report: Path,
    output: Path,
    nproc: int,
):
    clusters = _load_clusters(report)

    ref = next(pybel.readfile("mol2", str(ref_mol)))
    ref_fp = ref.calcfp("ecfp4")

    chimera = Chimera()
    addh_module = modulize(chimera.addh, modname="addh")

    modules = [
        GAlign(ref_mol, cutoff=0.0, nproc=1),
        addh_module,
        pcdetect.PCFilter(ref_fp, clusters, cutoff=0.0, nproc=1),
    ]

    result = output / "output.mol2"

    pipeline = Parallelizer(Pipeline(modules), nproc=nproc)
    pipeline(input_file, result)


def _load_single_target(results: Path):
    cols = ["name", "pharma_score", "shape_score", "tani_sim"]
    try:
        return pd.read_csv(results / "2_pcfilter/scores.csv")[cols]
    except FileNotFoundError:
        return pd.DataFrame(columns=cols)


def load_target(results: Path, nproc: int):
    if nproc < 2:
        df = _load_single_target(results)
    else:
        field_size = len(str(nproc - 1))

        df = pd.concat(
            [
                _load_single_target(results / f"split/part{i:0{field_size}d}")
                for i in range(nproc)
            ],
            ignore_index=True,
        )

    df = df.rename(columns={"name": "id"})
    df["pharma_score"] = df["pharma_score"].astype(float)
    df["shape_score"] = df["shape_score"].astype(float)
    df["tani_sim"] = df["tani_sim"].astype(float)
    return df


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run_target(
    ref_mol: Path,
    input_file: Path,
    target_home: Path,
    ganal_home: Path,
    result_home: Path,
    nproc: int = 1,
):
    db, target = target_home.parts[-2:]
    report = ganal_home / db / target / "ganal.json"
    gscreen_main(input_file, ref_mol, report, result_home, nproc)

    df = load_target(result_home, nproc)
    df.to_csv(result_home / "scores.csv", index=False)


if __name__ == "__main__":
    app()
