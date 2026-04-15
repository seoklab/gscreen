import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from openbabel import pybel

from gscreen.cli.ganal import analyze as ganal
from gscreen.io import Mol2Reader
from gscreen.pcdetect import PCFilter, load_reports
from gscreen.pipeline import Module, Parallelizer, Pipeline, modulize
from gscreen.tools import Chimera, GAlign


def run_ganal(receptor: Path, ref: Path):
    report = ganal(receptor, ref)
    return load_reports([report])


def run_pcfilter(screen: Module, ligands: Path, output: Path):
    screen(ligands, output)
    scores = pd.read_csv(output.parent / "2_pcfilter/scores.csv")
    return scores[["name", "score"]]


def final_scoring(
    ligands: Path, ref_fp: pybel.Fingerprint, scores: np.ndarray
):
    sims = np.array(
        [
            ref_fp | mol.calcfp("ecfp4")
            for mol in pybel.readfile("mol2", str(ligands))
        ]
    )
    weights = np.clip(6 * (sims - 0.1), 0, 0.9)

    shape_scores = np.array(
        [GAlign.model_score(mol) for mol in Mol2Reader(ligands)]
    )

    return weights * shape_scores + (1 - weights) * scores


def run_gscreen(
    ligands: Path,
    output: Path,
    screen: Module,
    ref_fp: pybel.Fingerprint,
):
    df = run_pcfilter(screen, ligands, output)
    scores = final_scoring(output, ref_fp, df["score"].to_numpy())
    df["score"] = scores
    return df


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("-r", "--ref", type=Path, required=True)
    parser.add_argument("receptor", type=Path)
    parser.add_argument("ligands", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ref = next(pybel.readfile("mol2", str(args.ref)))
    ref_fp = ref.calcfp("ecfp4")

    pcs = run_ganal(args.receptor, args.ref)

    screen = Pipeline(
        [
            GAlign(args.ref, cutoff=0, nproc=args.jobs),
            Parallelizer(
                modulize(Chimera(verbose=False).addh, modname="addh"),
                nproc=args.jobs,
            ),
            PCFilter(pcs, cutoff=0, nproc=args.jobs),
        ]
    )

    with TemporaryDirectory() as tmpd:
        result = run_gscreen(
            args.ligands,
            Path(tmpd, "result.mol2"),
            screen,
            ref_fp,
        )
    df = result.rename({"name": "id", "score": f"gscreen-{args.jobs}"})
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
