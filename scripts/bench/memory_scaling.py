"""Memory scaling benchmark driver.

Creates molecule subsets of increasing size and measures peak RSS
for each method, showing how memory scales with input size.
"""

import subprocess as sp
import sys
import tempfile
from pathlib import Path

import pandas as pd
import typer

from gscreen.io import Mol2Reader

app = typer.Typer(pretty_exceptions_enable=False)

_GSCREEN_SIZES = [1, 10, 100, 500, 1000, 5000]
_BASELINE_SIZES = [1, 10, 100]


def _make_subset(src: Path, n: int, dst: Path) -> int:
    """Write first `n` molecules from src to dst. Returns actual count."""
    count = 0
    with open(dst, "wb") as f:
        for mol in Mol2Reader(src):
            f.write(bytes(mol))
            count += 1
            if count >= n:
                break
    return count


def _parse_time_file(path: Path):
    """Parse /usr/bin/time -v output file for peak RSS and wall time."""
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "Maximum resident set size" in line:
            result["memkb"] = int(line.split(":")[-1].strip())
        elif "Elapsed (wall clock)" in line:
            ts = line.split(": ", 1)[-1].strip()
            secs = sum(
                float(t) * 60**i for i, t in enumerate(reversed(ts.split(":")))
            )
            result["time"] = secs
    return result


def _run_timed(
    gnu_time: Path,
    cmd: list,
    time_file: Path,
    stdout=sp.PIPE,
    allow_nonzero: bool = False,
    **kwargs,
):
    """Run cmd wrapped with /usr/bin/time -v, return parsed metrics."""
    proc = sp.run(
        [gnu_time, "-vo", time_file, *map(str, cmd)],
        encoding="utf-8",
        check=False,
        stdout=stdout,
        stderr=sp.STDOUT,
        **kwargs,
    )
    if not allow_nonzero and proc.returncode != 0:
        typer.echo(
            f"  FAILED (rc={proc.returncode}): "
            f"{' '.join(str(c) for c in cmd[:3])}...",
            err=True,
        )
        if proc.stdout:
            typer.echo(f"  stdout: {proc.stdout[:500]}", err=True)
        return {}

    return _parse_time_file(time_file)


def _run_galign(
    gnu_time: Path,
    exe: Path,
    ligands: Path,
    ref: Path,
    nproc: int,
    tmpdir: Path,
) -> dict:
    time_file = tmpdir / "galign.time"
    log = tmpdir / "galign.log"
    cmd = [exe, f"-ln{nproc}", ligands, ref]
    with open(log, "w") as f:
        return _run_timed(gnu_time, cmd, time_file, cwd=tmpdir, stdout=f)


def _run_gscreen(
    gnu_time: Path,
    script: Path,
    ligands: Path,
    ref: Path,
    receptor: Path,
    nproc: int,
    tmpdir: Path,
) -> dict:
    time_file = tmpdir / "gscreen.time"
    output_csv = tmpdir / "gscreen_scores.csv"
    cmd = [
        sys.executable,
        script,
        f"-j{nproc}",
        "-r",
        ref,
        receptor,
        ligands,
        output_csv,
    ]
    return _run_timed(gnu_time, cmd, time_file)


def _run_lsalign(
    gnu_time: Path,
    exe: Path,
    ligands: Path,
    ref: Path,
    tmpdir: Path,
) -> dict:
    time_file = tmpdir / "lsalign.time"
    log = tmpdir / "lsalign.log"
    cmd = [exe, ligands, ref, "-rf", "1"]
    with open(log, "w") as f:
        return _run_timed(
            gnu_time,
            cmd,
            time_file,
            allow_nonzero=True,
            cwd=tmpdir,
            stdout=f,
        )


def _run_pharmagist(
    gnu_time: Path,
    exe: Path,
    cfg: Path,
    ligands: Path,
    ref: Path,
    tmpdir: Path,
) -> dict:
    time_file = tmpdir / "pharmagist.time"
    out_dir = tmpdir / "pharmagist_out"
    out_dir.mkdir()
    cmd = [exe, "-c", cfg, "-o", out_dir, "-d", ligands, ref]
    return _run_timed(gnu_time, cmd, time_file, cwd=out_dir)


def _prepare_vina_inputs(
    script: Path,
    ligands_mol2: Path,
    receptor: Path,
    ref: Path,
    tmpdir: Path,
    n_jobs: int = 4,
) -> tuple[Path, Path]:
    """Convert mol2 ligands to pdbqt and create vina config.

    Returns (ligands_dir, config_file).
    """
    cmd = [
        sys.executable,
        script,
        f"-j{n_jobs}",
        "-r",
        ref,
        receptor,
        ligands_mol2,
        tmpdir / "vina_prep",
    ]
    sp.run([str(c) for c in cmd], check=True, capture_output=True)
    return (
        tmpdir / "vina_prep" / "ligands",
        tmpdir / "vina_prep" / "options.txt",
    )


def _run_vina(
    gnu_time: Path,
    exe: Path,
    ligands_dir: Path,
    config: Path,
    nproc: int,
    tmpdir: Path,
) -> dict:
    time_file = tmpdir / "vina.time"
    out_dir = tmpdir / "vina_out"
    out_dir.mkdir()
    cmd = [
        exe,
        "--config",
        config,
        "--batch",
        ligands_dir,
        "--dir",
        out_dir,
        "--cpu",
        nproc,
    ]
    return _run_timed(gnu_time, cmd, time_file)


def _choose_source(target_dir: Path, n: int) -> Path:
    """Pick the smallest source file that has at least n molecules."""
    big = target_dir / "subset-50x.mol2"
    small = target_dir / "subset.mol2"
    if n > 100:
        return big
    return small


@app.command()
def main(
    target_dir: Path,
    output: str = "memory_scaling.csv",
    nproc: int = 1,
    gscreen_bench_script: Path = (
        Path(__file__).resolve().parent / "gscreen_bench.py"
    ),
    vina_prepare_script: Path = (
        Path(__file__).resolve().parent / "vina_prepare.py"
    ),
    galign: Path = Path.home() / "opt/galign/bin/galign",
    lsalign: Path = Path.home() / "opt/lsalign/lsalign",
    pharmagist: Path = Path.home() / "opt/pharmagist/pharmagist64.linux",
    pharmagist_cfg: Path = Path.home() / "opt/pharmagist/pharmagist.config",
    vina: Path = Path.home() / "opt/vina/latest/bin/vina",
    gnu_time: Path = Path("/usr/bin/time"),
):
    """Measure peak memory for each method at increasing molecule counts.

    TARGET_DIR is a database target directory (e.g. ~/db/dud-e/aa2ar)
    containing subset.mol2 and crystal_ligand.mol2.
    """
    ref = target_dir / "crystal_ligand.mol2"
    receptor = target_dir / "receptor_dock.pdb"

    methods_gscreen = ["galign", "gscreen"]
    methods_baseline = ["lsalign", "pharmagist", "vina"]

    typer.echo(f"Target: {target_dir.name}  nproc={nproc}")
    typer.echo(f"G-screen methods: {methods_gscreen}  sizes: {_GSCREEN_SIZES}")
    typer.echo(
        f"Baseline methods: {methods_baseline}  sizes: {_BASELINE_SIZES}"
    )

    rows: list = []

    for method in methods_gscreen:
        for n in _GSCREEN_SIZES:
            src = _choose_source(target_dir, n)
            with tempfile.TemporaryDirectory() as tmpd:
                tmpdir = Path(tmpd)
                subset = tmpdir / "subset.mol2"
                actual = _make_subset(src, n, subset)
                if actual < n:
                    raise ValueError(
                        f"Only {actual} molecules available for n={n}, "
                        "cannot run gscreen methods",
                    )

                typer.echo(f"  {method} n={actual} ...", nl=False)

                if method == "galign":
                    metrics = _run_galign(
                        gnu_time,
                        galign,
                        subset,
                        ref,
                        nproc,
                        tmpdir,
                    )
                else:
                    metrics = _run_gscreen(
                        gnu_time,
                        gscreen_bench_script,
                        subset,
                        ref,
                        receptor,
                        nproc,
                        tmpdir,
                    )

            if metrics:
                typer.echo(
                    f"  {metrics['memkb'] / 1024:.0f} MB  "
                    f"{metrics['time']:.1f}s"
                )
                rows.append(
                    {
                        "method": method,
                        "nproc": nproc,
                        "n_mols": actual,
                        **metrics,
                    }
                )
            else:
                typer.echo("  FAILED")

    for method in methods_baseline:
        for n in _BASELINE_SIZES:
            src = _choose_source(target_dir, n)
            with tempfile.TemporaryDirectory() as tmpd:
                tmpdir = Path(tmpd)
                subset = tmpdir / "subset.mol2"
                actual = _make_subset(src, n, subset)
                if actual < n:
                    typer.echo(
                        f"  {method} n={n}: only {actual} "
                        "molecules available, skipping",
                    )
                    continue

                typer.echo(f"  {method} n={actual} ...", nl=False)

                if method == "lsalign":
                    metrics = _run_lsalign(
                        gnu_time,
                        lsalign,
                        subset,
                        ref,
                        tmpdir,
                    )
                elif method == "pharmagist":
                    metrics = _run_pharmagist(
                        gnu_time,
                        pharmagist,
                        pharmagist_cfg,
                        subset,
                        ref,
                        tmpdir,
                    )
                elif method == "vina":
                    typer.echo(" prep ...", nl=False)
                    lig_dir, config = _prepare_vina_inputs(
                        vina_prepare_script,
                        subset,
                        receptor,
                        ref,
                        tmpdir,
                    )
                    metrics = _run_vina(
                        gnu_time,
                        vina,
                        lig_dir,
                        config,
                        nproc,
                        tmpdir,
                    )
                else:
                    metrics = {}

            if metrics:
                typer.echo(
                    f"  {metrics['memkb'] / 1024:.0f} MB  "
                    f"{metrics['time']:.1f}s"
                )
                rows.append(
                    {
                        "method": method,
                        "nproc": nproc if method == "vina" else 1,
                        "n_mols": actual,
                        **metrics,
                    }
                )
            else:
                typer.echo("  FAILED")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    typer.echo(f"\nResults saved to {output}")

    if not df.empty:
        typer.echo("\nSummary (peak RSS in MB):")
        pivot = df.pivot_table(
            index="method", columns="n_mols", values="memkb"
        )
        pivot = pivot / 1024
        pd.options.display.float_format = "{:.0f}".format
        typer.echo(pivot.to_string())
        pd.options.display.float_format = None


if __name__ == "__main__":
    app()
