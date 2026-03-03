import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from ..typing import StrPath
from .misc import preconfigurable

__all__ = [
    "abspath",
    "mkdir_p",
    "with_stem",
    "with_stem_suffix",
    "relative_symlink_to",
    "is_older",
    "exists_and_older",
    "exists_and_newer",
    "check_query_result",
    "fdlock",
    "chdir",
]

_logger = logging.getLogger(".".join(__name__.split(".")[:-1]))


# Path.absoulte is undocumented and untested before 3.11
if sys.version_info >= (3, 11):

    @preconfigurable
    def abspath(path: StrPath, strict: bool = True, resolve: bool = False):
        """Make a path-like object absoulte.

        Supports pre-configuring strictness before actual call to
        Path.abosulte, for easier use in e.g. "type" argument in
        ArgumentParser.add_argument.

        Parameters
        ----------
        path : str | Path
            Path-like object to make absoulte. If not given, returns a callable
            that could be used for path resolving.
        strict : bool, optional
            Whether to raise error on nonexistent path, by default True
        resolve : bool, optional
            Whether to cannonicalize the path, by default False
        """
        if resolve:
            return Path(path).resolve(strict=strict)

        p = Path(path).absolute()
        if strict and not p.exists():
            raise FileNotFoundError(str(path))
        return p
else:

    @preconfigurable
    def abspath(path: StrPath, strict: bool = True, resolve: bool = False):
        """Make a path-like object absoulte.

        Supports pre-configuring strictness before actual path manipulation,
        for easier use in e.g. "type" argument in ArgumentParser.add_argument.

        Parameters
        ----------
        path : str | Path
            Path-like object to make absoulte. If not given, returns a callable
            that could be used for path resolving.
        strict : bool, optional
            Whether to raise error on nonexistent path, by default True
        resolve : bool, optional
            Whether to cannonicalize the path, by default False
        """
        if resolve:
            return Path(path).resolve(strict=strict)

        p = Path.cwd() / path  # Automatically handles absolute paths
        if strict and not p.exists():
            raise FileNotFoundError(str(path))
        return p


def mkdir_p(path: Path):
    _logger.debug(f"mkdir_p: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


if sys.version_info >= (3, 9):

    def with_stem(path: Path, stem: str):
        return path.with_stem(stem)
else:

    def with_stem(path: Path, stem: str):
        return path.with_name(f"{stem}{path.suffix}")


def with_stem_suffix(path: Path, suffix: str):
    return with_stem(path, f"{path.stem}{suffix}")


def relative_symlink_to(
    path: Path, target: Path, strict: bool = False, resolve: bool = False
):
    """Create a symlink to a target relative to the path."""
    if path.exists():
        raise FileExistsError(f"{path} already exists")

    path = abspath(path, strict=False, resolve=False)
    target = abspath(target, strict=strict, resolve=resolve)

    common_parent = os.path.commonpath([path, target])
    if common_parent == "/":
        path.symlink_to(target)
        return

    relpath = path.relative_to(common_parent)
    intermediate_count = len(relpath.parts) - 1

    reltarget = target.relative_to(common_parent)
    symlink_target = Path("../" * intermediate_count, reltarget)
    path.symlink_to(symlink_target)


def is_older(this: Path, that: Path) -> bool:
    return this.stat().st_mtime < that.stat().st_mtime


def exists_and_older(this: Path, that: Path) -> bool:
    return this.exists() and is_older(this, that)


def exists_and_newer(this: Path, that: Path) -> bool:
    return this.exists() and not is_older(this, that)


def check_query_result(query: StrPath, result: StrPath, force: bool):
    query = abspath(query)
    result = abspath(result, strict=False)

    result_exists = result.exists()
    if result_exists and query.samefile(result):
        raise ValueError("Query and result cannot be the same file")

    force = force or (result_exists and is_older(result, query))
    return query, result, force


@contextmanager
def fdlock(file_or_dir: Path):
    if file_or_dir.is_dir():
        lock = file_or_dir / "lock"
    else:
        lock = file_or_dir.with_suffix(".lock")

    while True:
        try:
            lock.touch(exist_ok=False)
        except Exception:
            pass
        else:
            break

    try:
        yield file_or_dir
    finally:
        lock.unlink()


def logging_chdir(path):
    _logger.debug(f"chdir: {path}")
    os.chdir(path)


@contextmanager
def chdir(dest_dir: Path):
    cwd = Path.cwd()
    try:
        mkdir_p(dest_dir)
        logging_chdir(dest_dir)
        yield dest_dir
    finally:
        logging_chdir(cwd)
