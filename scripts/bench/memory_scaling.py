"""Memory scaling benchmark driver.

Creates molecule subsets of increasing size and measures peak RSS
for each method, showing how memory scales with input size.
"""

import subprocess as sp
import sys
from pathlib import Path

import pandas as pd
import typer

from gscreen.io import Mol2Reader

app = typer.Typer(pretty_exceptions_enable=False)

_small_subsets = [1, 10, 50, 100]
_large_subsets = [50, 100, 500, 1000]


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
    allowed_retcode: int = 0,
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
    if proc.returncode == 9:
        # likely OOM
        typer.echo(
            f"  OOM (rc={proc.returncode}): "
            f"{' '.join(str(c) for c in cmd[:3])}...",
            err=True,
        )
        try:
            return _parse_time_file(time_file)
        except Exception:
            return {"memkb": float("nan"), "time": float("nan")}

    if proc.returncode != allowed_retcode:
        typer.echo(
            f"  FAILED (rc={proc.returncode}): "
            f"{' '.join(str(c) for c in cmd[:3])}...",
            err=True,
        )
        if proc.stdout:
            typer.echo(f"  stdout: {proc.stdout}", err=True)
        return {"memkb": float("nan"), "time": float("nan")}

    return _parse_time_file(time_file)


def _run_galign(
    gnu_time: Path,
    exe: Path,
    ligands: Path,
    ref: Path,
    nproc: int,
    cwd: Path,
) -> dict:
    time_file = cwd / "galign.time"
    log = cwd / "galign.log"
    cmd = [exe, f"-ln{nproc}", ligands, ref]
    with open(log, "w") as f:
        return _run_timed(gnu_time, cmd, time_file, cwd=cwd, stdout=f)


def _run_gscreen(
    gnu_time: Path,
    script: Path,
    ligands: Path,
    ref: Path,
    target_home: Path,
    ganal_home: Path,
    nproc: int,
    cwd: Path,
) -> dict:
    time_file = cwd / "gscreen.time"
    cmd = [
        sys.executable,
        script,
        "--nproc",
        nproc,
        ref,
        ligands,
        target_home,
        ganal_home,
        cwd,
    ]
    return _run_timed(gnu_time, cmd, time_file)


def _run_lsalign(
    gnu_time: Path,
    exe: Path,
    ligands: Path,
    ref: Path,
    cwd: Path,
) -> dict:
    time_file = cwd / "lsalign.time"
    log = cwd / "lsalign.log"
    cmd = [exe, ligands, ref, "-rf", "1"]
    with open(log, "w") as f:
        return _run_timed(
            gnu_time,
            cmd,
            time_file,
            allowed_retcode=1,
            cwd=cwd,
            stdout=f,
        )


def _run_pharmagist(
    gnu_time: Path,
    exe: Path,
    cfg: Path,
    ligands: Path,
    ref: Path,
    cwd: Path,
) -> dict:
    if not ref.is_file():
        typer.echo(
            f"  WARNING: pharmagist reference ligand {ref} not found",
            err=True,
        )
        return {}

    time_file = cwd / "pharmagist.time"
    out_dir = cwd / "pharmagist_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-c", cfg, "-o", out_dir, "-d", ligands, ref]
    return _run_timed(gnu_time, cmd, time_file, cwd=out_dir)


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
    output_dir: Path = Path.cwd(),
    gscreen_bench_script: Path = (
        Path(__file__).resolve().parent / "gscreen_bench.py"
    ),
    ganal_home: Path = Path.home() / "benchmark/scaling/ganal",
    galign: Path = Path.home() / "opt/galign/bin/galign",
    lsalign: Path = Path.home() / "opt/lsalign/lsalign",
    pharmagist: Path = Path.home() / "opt/pharmagist/pharmagist64.linux",
    pharmagist_cfg: Path = Path.home() / "opt/pharmagist/pharmagist.config",
    pharmagist_inputs: Path = Path.home() / "benchmark/scaling/pg-inputs",
    gnu_time: Path = Path("/usr/bin/time"),
):
    ref = target_dir / "crystal_ligand.mol2"

    db, target = target_dir.parts[-2:]
    output_dir = output_dir / db / target
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, int, list[int]]] = [
        # Galign
        ("galign", 1, _small_subsets),
        ("galign", 32, _large_subsets),
        ("galign", 128, _large_subsets),
        # Gscreen
        ("gscreen", 1, _small_subsets),
        ("gscreen", 32, _large_subsets),
        ("gscreen", 128, _large_subsets),
        # lsalign
        ("lsalign", 1, _small_subsets),
        # Pharmagist
        ("pharmagist", 1, _small_subsets),
    ]

    typer.echo(f"Target: {target_dir.name}")
    for method, nproc, sizes in runs:
        typer.echo(f"  {method}: nproc={nproc}  sizes={sizes}")

    rows = []
    for method, nproc, sizes in runs:
        for n in sizes:
            src = _choose_source(target_dir, n)

            tmpdir = output_dir / f"{method}/{nproc}/{n}"
            tmpdir.mkdir(parents=True, exist_ok=True)

            subset = tmpdir / "subset.mol2"
            actual = _make_subset(src, n, subset)
            if actual < n:
                raise ValueError(
                    f"Source {src} has only {actual} molecules, "
                    f"cannot create subset of size {n}"
                )

            typer.echo(f"  {method} nproc={nproc} n={actual} ...", nl=False)

            if method == "galign":
                metrics = _run_galign(
                    gnu_time,
                    galign,
                    subset,
                    ref,
                    nproc,
                    tmpdir,
                )
            elif method == "gscreen":
                metrics = _run_gscreen(
                    gnu_time,
                    gscreen_bench_script,
                    subset,
                    ref,
                    target_dir,
                    ganal_home,
                    nproc,
                    tmpdir,
                )
            elif method == "lsalign":
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
                    pharmagist_inputs / db / target / "crystal_ligand.mol2",
                    tmpdir,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            if metrics:
                typer.echo(
                    f"  {metrics['memkb'] / 1024:.0f} MB  "
                    f"{metrics['time']:.1f}s"
                )
            else:
                typer.echo("  FAILED")

            rows.append(
                {
                    "method": method,
                    "nproc": nproc,
                    "n_mols": actual,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "memory_scaling.csv"
    df.to_csv(csv_path, index=False)
    typer.echo(f"\nResults saved to {csv_path}")

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
