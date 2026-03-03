import contextlib
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Protocol

from joblib import Parallel, delayed

from . import io, utils
from .io import LoggerMixin
from .typing import StrPath

__all__ = [
    "Module",
    "ParallelModuleMixin",
    "ParallelModule",
    "Pipeline",
    "Parallelizer",
    "modulize",
]


class Module(LoggerMixin, ABC):
    nproc: int = 1

    @abstractmethod
    def run(self, query: Path, result: Path, force: bool) -> None:
        pass

    def __call__(
        self, query: StrPath, result: StrPath, force: bool = False
    ) -> None:
        query, result, force = utils.check_query_result(query, result, force)
        if result.exists():
            if not force:
                self.log_skip()
                return

            # Extra check for the "same" file
            if query.samefile(result):
                raise ValueError("Query and result cannot be the same file")

        utils.mkdir_p(result.parent)

        try:
            self.run(query, result, force)
        except BaseException:
            result.unlink(missing_ok=True)
            raise

    @classmethod
    @property
    def modname(cls):
        return cls.__name__.lower()

    def log_skip(self, lvl: int = logging.INFO):
        self.log(
            lvl, f"Skipping {self.modname} run; set force to True to overwrite"
        )


class ParallelModuleMixin:
    def __init__(self, *args, nproc: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nproc = nproc or utils.get_nproc()
        if self.nproc < 1:
            raise ValueError("nproc must be a positive integer")


class ParallelModule(ParallelModuleMixin, Module):
    pass


class Pipeline(Module):
    def __init__(
        self, modules: List[Module], intermediate: str = "ligands.mol2"
    ):
        super().__init__()
        self.mods = modules
        self.intermediate = intermediate

        self._nproc = super().nproc

    @property
    def nproc(self) -> int:
        return self._nproc

    @nproc.setter
    def nproc(self, nproc: int):  # type: ignore
        self._nproc = nproc
        for mod in self.mods:
            mod.nproc = nproc

    def run(self, query: Path, result: Path, force: bool):
        workdir = result.parent
        mod_result = query

        for i, mod in enumerate(self.mods):
            mod_query = mod_result

            mod_dir = workdir / f"{i}_{mod.modname}"
            mod_result = mod_dir / self.intermediate
            if force and mod_dir.exists():
                # No need to re-create the directory;
                # will be handled by Module.__call__
                shutil.rmtree(mod_dir, ignore_errors=True)

            try:
                mod(mod_query, mod_result, force=force)
            except Exception as e:
                self.log_error(f"Failed to run {mod.modname}", exc_info=e)
                mod_result.touch()
                break
            except BaseException:
                mod_result.unlink(missing_ok=True)
                raise

        shutil.copy(mod_result, result)


class Parallelizer(ParallelModule):
    def __init__(
        self,
        module: Module,
        workdir: str = "split",
        cleanup: bool = False,
        ifmt: str = None,
        ofmt: str = None,
        loglvl: int = None,
        nproc: int = None,
    ):
        super().__init__(nproc=nproc)
        self.mod = module
        self.workdir = workdir
        self.cleanup = cleanup
        self.ifmt = ifmt
        self.ofmt = ofmt

        self.loglvl = loglvl or logging.getLogger().getEffectiveLevel()

    @property
    def modname(self):
        return f"parallel_{self.mod.modname}"

    def run(self, query: Path, result: Path, force: bool):
        if self.nproc < 2:
            # Transparent execution if nproc < 2
            return self.mod(query, result, force)

        # Multiprocessing

        ifmt = self.ifmt or query.suffix[1:]
        if self.ofmt:
            result = result.with_suffix(f".{self.ofmt}")

        with contextlib.ExitStack() as stack:
            if self.cleanup:
                tempdir = tempfile.TemporaryDirectory()
                parent = Path(stack.enter_context(tempdir))
                self.log_info("Using temporary directory: %s\n", parent)
            else:
                parent = utils.mkdir_p(result.parent / self.workdir)

            querys = io.get_reader(ifmt)(query).split(
                self.nproc, parent / "querys"
            )
            for sq in querys:
                # Keep timestamps
                shutil.copystat(query, sq)

            field_size = len(str(self.nproc - 1))
            results = [
                (parent / f"part{i:0{field_size}d}" / result.name)
                for i in range(self.nproc)
            ]

            log_parent = utils.mkdir_p(parent / "logs")
            log_files = [
                log_parent / f"part{i:0{field_size}d}.log"
                for i in range(self.nproc)
            ]

            with utils.nproc(1):
                self.mod.nproc = 1
                Parallel(n_jobs=self.nproc)(
                    delayed(self._run_parallel)(q, r, force, log)
                    for q, r, log in zip(querys, results, log_files)
                )

            io.merge_files(results, result)

    def _run_parallel(
        self, query: Path, result: Path, force: bool, logfile: Path
    ):
        with io.redirect_all(logfile):
            logging.basicConfig(level=self.loglvl)
            return self.mod(query, result, force)


class _RunLike(Protocol):
    def __call__(self, query: Path, result: Path, force: bool) -> None: ...


@utils.preconfigurable
def modulize(func: _RunLike, modname: str = None) -> Module:
    if modname is None:
        modname = getattr(func, "__name__", None) or func.__class__.__name__

    try:
        # functools.partial support
        module = getattr(func, "__module__", None) or func.func.__module__
    except AttributeError:
        module = None

    attributes = {"run": staticmethod(func), "__module__": module}
    return type(f"module_{modname}", (Module,), attributes)()
