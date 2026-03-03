import logging
import subprocess as sp
from functools import partial, wraps
from pathlib import Path
from typing import List, Optional

from .. import api, pcdetect, tools, utils
from ..pcdetect import *
from ..pipeline import Module, Parallelizer, Pipeline, modulize
from ..typing import StrPath
from . import utils as cli_utils

_logger = cli_utils.get_main_logger()


def get_parser():
    parser = cli_utils.GParser()
    parser.add_argument(
        "-p",
        "--report",
        type=utils.abspath,
        required=True,
        help="Path to the interaction report (json).",
    )
    parser.add_argument(
        "-n",
        "--need",
        type=utils.abspath,
        default=None,
        help="Path to the required interaction report (json).",
    )
    parser.add_argument(
        "-l",
        "--ligand",
        type=utils.abspath,
        required=True,
        help="Path to the reference ligand for alignment (mol2).",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=partial(cli_utils.parse_comm_list, tp=float),
        default=[0.0],
        help=(
            "Comma-separated cutoff value(s) for score. If single value is "
            "given, it is used for shape score and screening score. If two "
            "values are given, they are used for shape score and screening "
            "score respectively."
        ),
    )
    parser.add_argument(
        "-u",
        "--penalty",
        type=float,
        default=0.0,
        help="Penalty for unmatched pharmacophores.",
    )
    parser.add_argument(
        "-s",
        "--generate-stereo",
        action="store_true",
        dest="stereo",
        help="Whether to generate 3D conformers for the query molecules.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        dest="nproc",
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "query",
        metavar="FILE",
        type=utils.abspath,
        nargs="+",
        help=(
            "The query molecules for virtual screening. "
            "Supported formats: smi, sdf, mol2."
        ),
    )
    return parser


class _NS(cli_utils.GNamespace):
    report: Path
    need: Optional[Path]
    ligand: Path
    cutoff: list[float]
    penalty: float
    stereo: bool
    nproc: Optional[int]
    query: List[Path]


def _load_clusters(report: Path):
    with report.open("r") as f:
        reports: List[Report] = api.load(f)[Report]
    return load_reports(reports)


@modulize(modname="convert")
def _converter(query: Path, result: Path, force: bool):
    stdout = None if _logger.isEnabledFor(logging.INFO) else sp.DEVNULL
    sp.run(
        ["obabel", "-i", query.suffix[1:], query, "-o", "mol2", "-O", result],
        stdout=stdout,
        check=True,
    )


def _addchg_factory(chimera: tools.Chimera, **kwargs):
    if "force" in kwargs:
        raise ValueError("Cannot pre-configure overwrite mode.")

    @wraps(chimera.addchg)
    def wrapper(query: StrPath, result: StrPath, force: bool):
        return chimera.addchg(query, result, force=force, **kwargs)

    return wrapper


def _dispatch_cutoffs(cutoffs: list[float]):
    if len(cutoffs) > 2:
        _logger.warning("Too many cutoff values given. Only %d are used.", 2)

    if len(cutoffs) == 1:
        shape_cutoff = pc_cutoff = cutoffs[0]
    else:
        shape_cutoff, pc_cutoff, *_ = cutoffs

    return shape_cutoff, pc_cutoff


@cli_utils.wrap_main
def main():
    args = get_parser().parse_args(namespace=_NS())

    _logger.info("Loading clusters...")
    scoring_clusteres = _load_clusters(args.report)
    _logger.info("Loading required clusters...")
    required_clusteres = _load_clusters(args.need) if args.need else None

    # Define components
    chimera = tools.Chimera(verbose=_logger.isEnabledFor(logging.DEBUG))
    addh_module = modulize(chimera.addh, modname="addh")

    shape_cutoff, pc_cutoff = _dispatch_cutoffs(args.cutoff)

    ref_fp = next(Mol.load(args.ligand)).to_pybel().calcfp("ecfp4")

    # Assemble modules
    modules: List[Module]
    # Validate file type
    if not args.stereo:
        if any(query.suffix.lower() != ".mol2" for query in args.query):
            raise ValueError(
                "Valid 3D mol2 files are required to skip preprocessing step."
            )
        modules = []
    else:
        modules = [_converter, tools.Corina()]

    modules += [
        tools.align.GAlign(args.ligand, cutoff=shape_cutoff),
        addh_module,
        pcdetect.PCFilter(
            ref_fp,
            scoring_clusteres,
            required_clusteres,
            cutoff=pc_cutoff,
            strict=True,
        ),
    ]

    # Generate pipeline
    pipeline = Parallelizer(Pipeline(modules), nproc=args.nproc)

    ret = 0
    wd = args.output
    for query in args.query:
        result = wd / query.stem / "gscreen_result.mol2"
        try:
            pipeline(query, result, force=args.force)
        except Exception:
            _logger.error(
                f"Failed to process {query}",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
            ret += 1

    return ret


if __name__ == "__main__":
    exit(main())
