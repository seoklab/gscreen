import string
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import ExitStack
from functools import cached_property
from itertools import cycle
from typing import Dict, List, Type

import numpy as np
from file_read_backwards import FileReadBackwards

from .. import utils
from ..typing import SizedIterable, StrPath
from . import fastio

__all__ = [
    "Format",
    "Smi",
    "Sdf",
    "Mol2",
    "PDB",
    "FormatReader",
    "SmiReader",
    "SdfReader",
    "Mol2Reader",
    "PDBReader",
    "get_reader",
]


class Format(Sequence[bytes], ABC):
    """Abstract base class for formatted molecules.

    Subclasses must have cntr property.

    Parameters
    ----------
    data: list[bytes]
        Lines of the original file.
    """

    def __init__(self, data: List[bytes]):
        super().__init__()
        self._data = data

    @classmethod
    @property
    def format_name(cls) -> str:
        return cls.__name__.lower()

    @property
    @abstractmethod
    def cntr(self) -> np.ndarray:
        pass

    def __getitem__(self, idx: int) -> bytes:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __bytes__(self) -> bytes:
        return b"".join(self)

    def __str__(self) -> str:
        return bytes(self).decode("utf-8")


class Smi(Format):
    @cached_property
    def smi(self):
        return self._data[0].split(maxsplit=1)[0]

    @cached_property
    def name(self):
        return self._data[0].split(maxsplit=1)[1]

    @property
    def cntr(self) -> np.ndarray:
        return np.zeros(3)


class Sdf(Format):
    @property
    def cntr(self) -> np.ndarray:
        raise NotImplementedError


class Mol2(Format):
    @property
    def cntr(self) -> np.ndarray:
        tmp = []

        iter_self = iter(self)
        for line in iter_self:
            if line.startswith(b"@<TRIPOS>ATOM"):
                break

        for line in iter_self:
            if line.startswith(b"@<TRIPOS>BOND"):
                break

            args = line.split()
            if args[5] == b"H":
                continue
            tmp.append(np.array(args[2:5], dtype=float))

        return np.average(tmp, axis=0)

    @cached_property
    def name(self):
        mol2_iter = iter(self._data)
        for line in mol2_iter:
            if line.startswith(b"@<TRIPOS>MOLECULE"):
                return next(mol2_iter)

        raise ValueError("Invalid molecule")

    def replace_name(self, other: "Mol2"):
        name = other.name
        for i, line in enumerate(self._data):
            if line.startswith(b"@<TRIPOS>MOLECULE"):
                self._data[i + 1] = name
                break
        return self


class PDB(Format):
    @property
    def cntr(self) -> np.ndarray:
        raise NotImplementedError

    def residue(self, idx: int):
        res = []
        pdb_lines_iter = iter(self._data)
        for line in pdb_lines_iter:
            if line.startswith(b"ATOM") and int(line[22:26]) == idx:
                res.append(line)
                for line in pdb_lines_iter:
                    if line.startswith(b"ATOM") and int(line[22:26]) == idx:
                        res.append(line)
                    else:
                        break
                return PDB(res)

        raise KeyError(idx)

    def residues(self):
        pdb_lines_iter = iter(self._data)

        for line in pdb_lines_iter:
            if line.startswith(b"ATOM"):
                res = [line]
                idx = int(line[22:26])
                break
        else:
            return

        for line in pdb_lines_iter:
            if line.startswith(b"ATOM"):
                curr_idx = int(line[22:26])
                if curr_idx == idx:
                    res.append(line)
                else:
                    yield idx, PDB(res)
                    res = [line]
                    idx = curr_idx

        if res:
            yield idx, PDB(res)

    def split(self):
        rec = []
        lig = []
        for line in self:
            if line.startswith(b"ATOM"):
                rec.append(line)
            elif line.startswith((b"HETATM", b"CONECT")):
                lig.append(line)
            elif line.startswith((b"MODEL", b"ENDMDL")):
                rec.append(line)
                lig.append(line)
        return PDB(rec), PDB(lig)


_reader_factory: Dict[str, Type["FormatReader"]] = {}


class FormatReader(SizedIterable):  # type: ignore
    """Abstract base class for formatted readers.

    Subclasses must implement __iter__ and __len__ magic methods.

    Parameters
    ----------
    file: str or os.PathLike
        Path to the file to read.

    resolve: bool, optional
        If True, raise an exception if the file does not exist.
        Default to False.
    """

    def __init__(self, file: StrPath, resolve: bool = False):
        super().__init__()
        self.file = utils.abspath(file, strict=resolve)

    def __init_subclass__(cls, /, **kwargs) -> None:
        super().__init_subclass__()
        _reader_factory[cls.__name__.lower().removesuffix("reader")] = cls

    def __len__(self):
        return self.count

    @property
    @abstractmethod
    def count(self) -> int:
        pass

    @property
    def first(self):
        return next(iter(self))

    def split(self, n: int, out_dir: StrPath, suffix: str = "_part"):
        out_dir = utils.abspath(out_dir, strict=False)
        utils.mkdir_p(out_dir)

        basename = f"{self.file.stem}{suffix}"
        ext = self.file.suffix

        filepaths = [out_dir.joinpath(f"{basename}{i}{ext}") for i in range(n)]
        with ExitStack() as stack:
            files = [stack.enter_context(fp.open("wb")) for fp in filepaths]
            for m, f in zip(self, cycle(files)):
                f.write(bytes(m))
        return filepaths


class SmiReader(FormatReader):
    _whitespaces = frozenset(ws.encode("utf-8") for ws in string.whitespace)

    def __iter__(self):
        with fastio.fast_open(self.file) as fb:
            for line in fb:
                yield Smi([line])

    @cached_property
    def count(self):
        cnt = 0
        lastblk = b""
        with self.file.open("rb", buffering=0) as fb:
            while blk := fb.read(fastio.RD_MAX):
                cnt += blk.count(b"\n")
                lastblk = blk
        # Special case: nonempty file not ends with whitespace
        if cnt and lastblk[-1] not in self._whitespaces:
            cnt += 1
        return cnt


class SdfReader(FormatReader):
    def __iter__(self):
        with fastio.fast_open(self.file) as fb:
            lines = []
            for line in fb:
                lines.append(line)
                if line.startswith(b"$$$$"):
                    yield Sdf(lines)
                    lines = []

    @cached_property
    def count(self):
        cnt = 0
        blk = b""
        target = b"$$$$"
        with self.file.open("rb", buffering=0) as fb:
            while tmp_blk := fb.read(fastio.RD_MAX):
                blk = blk[-len(target) + 1 :] + tmp_blk
                cnt += blk.count(target)
        return cnt


class Mol2Reader(FormatReader):
    def __iter__(self):
        with fastio.fast_open(self.file) as fb:
            for line in fb:
                if line.startswith(b"@<TRIPOS>MOLECULE"):
                    lines = [line]
                    break
            else:
                return

            for line in fb:
                if line.startswith(b"@<TRIPOS>MOLECULE"):
                    yield Mol2(lines)
                    lines = []
                lines.append(line)

            yield Mol2(lines)

    @cached_property
    def count(self):
        cnt = 0
        blk = b""
        target = b"@<TRIPOS>MOLECULE"
        with self.file.open("rb", buffering=0) as fb:
            while tmp_blk := fb.read(fastio.RD_MAX):
                blk = blk[-len(target) + 1 :] + tmp_blk
                cnt += blk.count(target)
        return cnt


class PDBReader(FormatReader):
    def __iter__(self):
        with fastio.fast_open(self.file) as fb:
            lines = []
            for line in fb:
                lines.append(line)
                if line.startswith(b"ENDMDL"):
                    yield PDB(lines)
                    lines = []
            if lines:
                yield PDB(lines)

    @cached_property
    def count(self):
        with FileReadBackwards(self.file) as fr:
            line: str
            for line in fr:
                if line.startswith("MODEL"):
                    return int(line[6:])
            else:
                raise ValueError("Cannot find last model index")


def get_reader(fmt: str) -> Type[FormatReader]:
    return _reader_factory[fmt]
