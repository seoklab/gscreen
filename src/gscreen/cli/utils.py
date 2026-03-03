import logging
import sys
from argparse import Action, ArgumentParser, Namespace
from functools import wraps
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from .. import io, utils

_logger = logging.getLogger(__name__)
_N = TypeVar("_N", bound=Namespace)


class GNamespace(Namespace):
    verbose: int
    quiet: bool
    force: bool
    output: Path


class GParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.defaults: Dict[str, Action] = {}

        self._add_default_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Increase verbosity.",
        )
        self._add_default_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Quiet mode; only print warnings and errors.",
        )
        self._add_default_argument(
            "-f",
            "--force",
            action="store_true",
            help="Whether to overwrite existing files.",
        )
        self._add_default_argument(
            "-o",
            "--output",
            type=utils.abspath(strict=False),
            default=Path.cwd(),
            help="Output directory.",
        )

    @wraps(ArgumentParser.add_argument)
    def _add_default_argument(self, *name_or_flags, **kwargs):
        action = self.add_argument(*name_or_flags, **kwargs)
        for key in name_or_flags:
            self.defaults[key] = action

    @overload
    def parse_args(self, args: Sequence[str] = ...) -> GNamespace: ...  # noqa: E501,E704
    @overload
    def parse_args(
        self, args: Sequence[str], namespace: None
    ) -> GNamespace: ...  # noqa: E501,E704
    @overload
    def parse_args(self, args: Sequence[str], namespace: _N) -> _N: ...  # noqa: E501,E704
    @overload
    def parse_args(self, *, namespace: None) -> GNamespace: ...  # noqa: E501,E704
    @overload
    def parse_args(self, *, namespace: _N) -> _N: ...  # noqa: E501,E704

    def parse_args(self, args=None, namespace=None):
        ns = GNamespace() if namespace is None else namespace
        ret = super().parse_args(args, ns)
        self._config_logging(ret)
        try:
            utils.mkdir_p(ret.output)
        except Exception:
            pass
        return ret

    @staticmethod
    def _config_logging(args):
        loglvl = logging.INFO
        if not args.quiet:
            loglvl = max(loglvl - 10 * args.verbose, logging.DEBUG)
        logging.basicConfig(level=loglvl)

        _logger.debug("args = %s", args)


def get_main_logger(appname: str = None):
    if appname is None:
        appname = sys.argv[0]
        appname = "__main__" if appname == "-c" else Path(appname).name

    return logging.getLogger(appname)


@utils.preconfigurable
def wrap_main(
    main: Callable[[], Union[None, int]], appname: Optional[str] = None
):
    @wraps(main)
    def main_wrapper():
        try:
            return main()
        except Exception:
            if utils.inside_debugger():
                raise

            get_main_logger(appname).critical(
                "Error is unrecoverable; aborting", exc_info=True
            )
        except SystemExit:
            raise
        except BaseException:
            sys.stderr.write("\n")

        return 1

    return main_wrapper


@wrap_main
def split_into():
    parser = GParser()
    parser.defaults["-o"].default = None
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Count of resulting files.",
    )
    parser.add_argument(
        "-s", "--suffix", default="_part", help="Suffix of resulting files."
    )
    parser.add_argument(
        "query",
        metavar="FILE",
        type=utils.abspath,
        help=(
            "The query molecules for splitting. "
            "Supported formats: smi, sdf, mol2, pdb."
        ),
    )

    args = parser.parse_args()
    query: Path = args.query
    output = args.output or query.parent / "split"

    io.get_reader(query.suffix[1:])(query).split(
        args.count, output, suffix=args.suffix
    )


_T = TypeVar("_T")


def parse_comm_list(s: str, tp: Type[_T]) -> List[_T]:
    return list(map(tp, s.split(",")))
