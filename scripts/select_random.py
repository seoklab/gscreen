import itertools
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm

from gscreen import io as gio

app = typer.Typer()


@app.command()
def main(
    actives: Path,
    decoys: Path,
    output: Path,
    n: int = 100,
    repeat: int = 50,
    seed: int = 42,
):
    amols = gio.Mol2Reader(actives)
    dmols = gio.Mol2Reader(decoys)

    rng = np.random.default_rng(seed)
    total = amols.count + dmols.count
    sel = rng.choice(
        total,
        size=min(n, total),
        replace=False,
    )
    mask = np.zeros(total, dtype=bool)
    mask[sel] = True

    selected = [
        bytes(mol)
        for mol in itertools.compress(itertools.chain(amols, dmols), mask)
    ]

    with (
        open(output, "wb") as f1,
        open(output.with_stem(output.stem + f"-{repeat}x"), "wb") as f2,
    ):
        for mol in tqdm(selected):
            f1.write(mol)
            f2.write(mol)

        for _ in tqdm(range(repeat - 1)):
            rng.shuffle(selected)
            for mol in selected:
                f2.write(mol)


if __name__ == "__main__":
    app()
