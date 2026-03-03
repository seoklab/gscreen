import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from .. import io, utils
from ..tools import Chimera
from . import utils as cli_utils

_logger = logging.getLogger(sys.argv[0])


def get_parser() -> ArgumentParser:
    parser = cli_utils.GParser()
    parser.add_argument(
        "complex",
        type=utils.abspath,
        nargs="+",
        help=(
            "Protein structure files (pdb), each containing single known "
            "protein-ligand complex structure."
        ),
    )
    return parser


def pdb_split(query: Path, rec_out: Path, lig_out: Path, force: bool):
    if (
        not force
        and utils.exists_and_newer(rec_out, query)
        and utils.exists_and_newer(lig_out, query)
    ):
        _logger.info(f"Skipping {query}...")
        return

    reader = io.PDBReader(query)
    recs, ligs = zip(*(pdb.split() for pdb in reader))
    rec_out.write_bytes(b"".join(map(bytes, recs)))

    lig_out_pdb = (
        lig_out.with_suffix(".pdb") if lig_out.suffix != ".pdb" else lig_out
    )
    lig_out_pdb.write_bytes(b"".join(map(bytes, ligs)))
    if lig_out_pdb is not lig_out:
        _logger.debug(f"converting to {lig_out.suffix}")
        Chimera().convert(lig_out_pdb, lig_out, force=force)


@cli_utils.wrap_main
def main():
    args = get_parser().parse_args()

    force: bool = args.force
    complex: List[Path] = args.complex

    cwd = Path.cwd()
    recs_dir = utils.mkdir_p(cwd / "receptor")
    ligs_dir = utils.mkdir_p(cwd / "ligand")
    recs = [recs_dir / f"{i}.pdb" for i in range(len(complex))]
    ligs = [ligs_dir / f"{i}.mol2" for i in range(len(complex))]

    for cplx, rec, lig in zip(complex, recs, ligs):
        pdb_split(cplx, rec, lig, force)
