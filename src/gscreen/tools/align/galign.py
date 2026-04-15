import subprocess as sp
from pathlib import Path

from ... import utils
from ...io import Mol2, Mol2Reader
from ...pipeline import ParallelModule
from ...typing import StrPath

__all__ = ["GAlign"]


class GAlign(ParallelModule):
    def __init__(
        self, reference: StrPath, cutoff: float = 0.5, nproc: int = None
    ):
        super().__init__(nproc=nproc)
        self.reference = utils.abspath(reference)
        self.cutoff = cutoff

    def run(self, query: Path, result: Path, force: bool):
        tmp_out = result.with_name(f"{result.stem}_all{result.suffix}")
        if query.stat().st_size == 0:
            result.touch()
            return

        with result.with_name("galign.log").open("wb", buffering=0) as flog:
            sp.run(
                [
                    "galign",
                    "-ln",
                    str(self.nproc),
                    "-s",
                    str(tmp_out),
                    str(query),
                    str(self.reference),
                ],
                check=True,
                stdout=flog,
                stderr=sp.STDOUT,
            )

        with result.open("wb") as f:
            for mol in Mol2Reader(tmp_out):
                if self.model_score(mol) >= self.cutoff:
                    f.write(bytes(mol))

    @staticmethod
    def model_score(mol: Mol2):
        return float(mol[6].split()[-1])
