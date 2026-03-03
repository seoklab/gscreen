import ctypes
from typing import Iterable

from ..typing import StrPath

__all__ = ["RD_MAX", "fast_open", "merge_files"]

# c int max
RD_MAX: int = 2 ** (ctypes.sizeof(ctypes.c_int) * 8 - 1) - 1


def fast_open(path, mode: str = "rb", **kwargs):
    """Open a file for the fastest available I/O method on a best effort basis."""
    if "b" not in mode:
        return open(path, mode, **kwargs)
    return open(path, mode, buffering=RD_MAX)


def merge_files(infiles: Iterable[StrPath], outfile: StrPath):
    with open(outfile, "wb", buffering=0) as fo:
        fo_write = fo.write
        for file in infiles:
            with open(file, "rb", buffering=0) as fi:
                fi_read = fi.read
                while blk := fi_read(RD_MAX):
                    fo_write(blk)
