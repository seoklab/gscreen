import logging
from pathlib import Path
from typing import List, TypeVar

from tqdm import tqdm

from .. import api, utils
from ..pcdetect import *
from . import utils as cli_utils

_logger = cli_utils.get_main_logger()
_S = TypeVar("_S", bound=Site)
_T = TypeVar("_T", bound=Site)


def _filter_interaction(lpcs: List[_S], rpcs: List[_T]):
    l_mask = [False] * len(lpcs)
    r_mask = [False] * len(rpcs)
    for i, lpc in enumerate(lpcs):
        for j, rpc in enumerate(rpcs):
            if lpc.interact(rpc) > 0:
                l_mask[i] = True
                r_mask[j] = True

    lpcs = [lpc for lpc, interact in zip(lpcs, l_mask) if interact]
    rpcs = [rpc for rpc, interact in zip(rpcs, r_mask) if interact]
    return lpcs, rpcs


def analyze(rec: Path, lig: Path):
    _logger.info(f"Analyzing {rec.name} and {lig.name}")

    receptor = next(Mol.load(rec))
    ligand = next(Mol.load(lig))

    lhp = Hydrophobic.from_mol(ligand)

    lars, rars = _filter_interaction(
        PiStacking.from_mol(ligand), PiStacking.from_mol(receptor)
    )
    lhbs, rhbs = _filter_interaction(
        HydrogenBonding.from_mol(ligand), HydrogenBonding.from_mol(receptor)
    )
    lchg, rchg = _filter_interaction(
        Charged.from_mol(ligand), Charged.from_mol(receptor, protein=True)
    )

    return Report(
        {
            PiStacking: lars,
            Hydrophobic: lhp,
            HydrogenBonding: lhbs,
            Charged: lchg,
        },
        {PiStacking: rars, HydrogenBonding: rhbs, Charged: rchg},
        lig,
        rec,
    )


def get_parser():
    parser = cli_utils.GParser()
    parser.add_argument(
        "-r",
        "--receptor",
        type=utils.abspath,
        nargs="+",
        required=True,
        help=(
            "Protein structure files (pdb), each containing single known "
            "receptor structure."
        ),
    )
    parser.add_argument(
        "-l",
        "--ligand",
        type=utils.abspath,
        nargs="+",
        required=True,
        help=(
            "Ligand structure files (mol2), each containing single known "
            "ligand structure binding to the protein receptor."
        ),
    )
    return parser


class _NS(cli_utils.GNamespace):
    receptor: List[Path]
    ligand: List[Path]


@cli_utils.wrap_main
def main():
    args = get_parser().parse_args(namespace=_NS())
    receptors = args.receptor
    ligands = args.ligand

    if len(receptors) != len(ligands):
        _logger.critical(
            (
                "Number of receptor and ligand files must be equal."
                f"Got: {len(receptors)}, {len(ligands)}."
            )
        )
        return 1

    reports = []
    for rec, lig in tqdm(
        zip(receptors, ligands),
        total=len(receptors),
        disable=not _logger.isEnabledFor(logging.INFO),
    ):
        try:
            reports.append(analyze(rec, lig))
        except Exception:
            if utils.inside_debugger():
                raise

            _logger.critical(
                f"Failed to analyze {rec.name} and {lig.name}.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
            return 1

    out = args.output / "ganal.json"
    with out.open("w" if args.force else "x") as f:
        api.dump({Report: reports}, f, indent=4)


if __name__ == "__main__":
    exit(main())
