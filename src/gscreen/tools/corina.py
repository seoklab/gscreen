import subprocess as sp
from pathlib import Path

from ..pipeline import Module

__all__ = ["Corina"]


class Corina(Module):
    def __init__(self, ifmt: str = "mol2", ofmt: str = "mol2"):
        super().__init__()
        self.ifmt = ifmt
        self.ofmt = ofmt

    def run(self, query: Path, result: Path, force: bool):
        sp.run(
            [
                "corina",
                "-t",
                "n",
                "-i",
                f"t={self.ifmt}",
                "-o",
                f"t={self.ofmt}",
                query,
                result,
            ],
            check=True,
            stdout=sp.DEVNULL,
            stderr=sp.STDOUT,
        )
