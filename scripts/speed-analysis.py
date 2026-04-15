"""Benchmark speed analysis.

Loads GNU time outputs for all methods, reports relative average runtime
(GS-SP 128T = 1x), and generates a grouped box plot.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from joblib import Parallel, delayed

app = typer.Typer(pretty_exceptions_enable=False)

_GROUPS = [
    ("GS-S", ["galign-1", "galign-128"], ["(Single)", "(128 Thr)"]),
    ("GS-P/SP", ["gscreen-1", "gscreen-128"], ["(Single)", "(128 Thr)"]),
    ("Flexi-LS-align", ["ls-align-1"], [""]),
    ("PharmaGist", ["pharmagist-1"], [""]),
    ("AutoDock Vina", ["vina-1", "vina-8"], ["(Single)", "(8 Thr)"]),
]

_INTRA_GAP = 0.75
_GROUP_SPACING = 1.75

_REFERENCE_KEY = "gscreen-128"


# ---------------------------------------------------------------------------
# GNU time parsing
# ---------------------------------------------------------------------------


def _parse_gnu_time(file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file, sep=": ", names=["key", "value"], index_col=0, engine="python"
    ).T
    df = df[
        [
            "Elapsed (wall clock) time (h:mm:ss or m:ss)",
            "Maximum resident set size (kbytes)",
        ]
    ].rename(
        columns={
            "Elapsed (wall clock) time (h:mm:ss or m:ss)": "time",
            "Maximum resident set size (kbytes)": "memkb",
        }
    )
    df["time"] = df["time"].apply(
        lambda x: sum(
            float(t) * 60**i for i, t in enumerate(reversed(x.split(":")))
        )
    )
    df["memkb"] = df["memkb"].astype(int)
    return df


def _load_gnu_time(
    data_dir: Path,
    method: str,
    nproc: int,
    name: Optional[str] = None,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for db in data_dir.joinpath(method, str(nproc)).iterdir():
        for target in db.iterdir():
            df = _parse_gnu_time(target / f"{name or method}.time")
            df["db"] = db.name if db.name != "muv-extra" else "muv"
            df["target"] = target.name
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.insert(0, "method", method)
    df.insert(1, "nproc", nproc)
    return df


def _load_target_vina(db: str, target: Path) -> pd.DataFrame:
    dfs = [
        _parse_gnu_time(target / f"{i}/receptor_dock/vina.time")
        for i in range(100)
    ]
    df = pd.concat(dfs, ignore_index=True)
    agg = pd.DataFrame(
        {"time": [df["time"].sum()], "memkb": [df["memkb"].max()]}
    )
    agg["db"] = db
    agg["target"] = target.name
    return agg


def _load_vina(data_dir: Path, nproc: int) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = Parallel(n_jobs=8)(
        delayed(_load_target_vina)(
            db.name if db.name != "muv-extra" else "muv", target
        )
        for db in data_dir.joinpath("autodock-vina", str(nproc)).iterdir()
        for target in db.iterdir()
    )
    df = pd.concat(dfs, ignore_index=True)
    df.insert(0, "method", "vina")
    df.insert(1, "nproc", nproc)
    return df


_CACHE_NAME = "speed_cache.csv"


def _load_all(bench_home: Path, cache: Path) -> pd.DataFrame:
    if cache.is_file():
        typer.echo(f"  Using cached {cache}")
        return pd.read_csv(cache)

    dfs = [
        _load_gnu_time(bench_home, method, n, name)
        for method, n, name in [
            ("galign", 1, None),
            ("galign", 128, None),
            ("gscreen", 1, None),
            ("gscreen", 128, None),
            ("ls-align", 1, "lsalign"),
            ("pharmagist", 1, None),
        ]
    ] + [_load_vina(bench_home, n) for n in [1, 8]]

    df = pd.concat(dfs, ignore_index=True)
    # 128-thread runs batch 50 jobs per invocation
    df.loc[df["nproc"] == 128, "time"] /= 50
    df["time"] /= 100
    df["key"] = df["method"] + "-" + df["nproc"].astype(str)

    df.to_csv(cache, index=False)
    typer.echo(f"  Cached to {cache}")
    return df


# ---------------------------------------------------------------------------
# Relative runtime report
# ---------------------------------------------------------------------------


def _fmt_method(grp_name: str, nproc: int) -> str:
    thr = f"{nproc} Thr" if nproc > 1 else "Single"
    return f"{grp_name:20s} ({thr:>8s})"


def _sig2(x: float) -> str:
    """Format a float to 2 significant figures, always in decimal."""
    if x == 0:
        return "0.0"
    import math

    digits = -int(math.floor(math.log10(abs(x)))) + 1
    return f"{x:.{max(digits, 0)}f}"


def _report_relative(df: pd.DataFrame):
    key_order = [k for _, keys, _ in _GROUPS for k in keys]
    databases = sorted(df["db"].unique())

    # --- Time ---
    ref_means = df.loc[df["key"] == _REFERENCE_KEY].groupby("db")["time"].mean()
    ref_all = df.loc[df["key"] == _REFERENCE_KEY, "time"].mean()

    time_by_db = (
        df.groupby(["db", "key"])["time"].mean().unstack(level="db")
    )
    time_by_db = time_by_db.loc[key_order]
    time_by_db["all"] = df.groupby("key")["time"].mean().loc[key_order]

    ratio_by_db = time_by_db.copy()
    for db in databases:
        ratio_by_db[db] = time_by_db[db] / ref_means[db]
    ratio_by_db["all"] = time_by_db["all"] / ref_all

    db_headers = "".join(f"{db:>12s}" for db in databases)
    typer.echo("\nRelative average runtime (GS-SP 1T = 1.0x):")
    typer.echo(f"  {'':32s} {db_headers}{'all':>12s}")
    for grp_name, keys, _ in _GROUPS:
        for key in keys:
            nproc = int(key.rsplit("-", 1)[1])
            label = _fmt_method(grp_name, nproc)
            vals = "".join(
                f"{_sig2(ratio_by_db.loc[key, db]):>11s}x"
                for db in databases
            )
            vals += f"{_sig2(ratio_by_db.loc[key, 'all']):>11s}x"
            typer.echo(f"  {label} {vals}")

    # --- Memory ---
    mem_by_db = (
        df.groupby(["db", "key"])["memkb"].max().unstack(level="db")
    )
    mem_by_db = mem_by_db.loc[key_order]
    mem_by_db["all"] = df.groupby("key")["memkb"].max().loc[key_order]

    typer.echo("\nPeak memory / MB (max RSS across targets):")
    typer.echo(f"  {'':32s} {db_headers}{'all':>12s}")
    for grp_name, keys, _ in _GROUPS:
        for key in keys:
            nproc = int(key.rsplit("-", 1)[1])
            label = _fmt_method(grp_name, nproc)
            vals = "".join(
                f"{mem_by_db.loc[key, db] / 1024:11.1f} " for db in databases
            )
            vals += f"{mem_by_db.loc[key, 'all'] / 1024:11.1f} "
            typer.echo(f"  {label} {vals}")

    typer.echo()


# ---------------------------------------------------------------------------
# Grouped box plot
# ---------------------------------------------------------------------------


def _compute_positions() -> dict[str, float]:
    """Map each key to its x position with grouped layout."""
    pos: dict[str, float] = {}
    for gi, (_, keys, _) in enumerate(_GROUPS):
        center = gi * _GROUP_SPACING
        n = len(keys)
        start = center - (n - 1) * _INTRA_GAP / 2
        for ki, key in enumerate(keys):
            pos[key] = start + ki * _INTRA_GAP
    return pos


def _shift_artists(ax: plt.Axes, old_x: list[float], new_x: list[float]):
    shift = {round(o): n - o for o, n in zip(old_x, new_x)}

    for patch in ax.patches:
        path = patch.get_path()
        verts = path.vertices.copy()
        cx = round(np.mean([verts[:, 0].min(), verts[:, 0].max()]))
        dx = shift.get(cx, 0)
        if dx:
            verts[:, 0] += dx
            patch._path = plt.matplotlib.path.Path(verts, path.codes)

    for line in ax.lines:
        xd = line.get_xdata()
        if len(xd) == 0:
            continue
        cx = round(np.mean(xd))
        dx = shift.get(cx, 0)
        if dx:
            line.set_xdata(np.asarray(xd, dtype=float) + dx)

    for coll in ax.collections:
        offs = coll.get_offsets()
        if len(offs) == 0:
            continue
        cx = round(offs[:, 0].mean())
        dx = shift.get(cx, 0)
        if dx:
            offs[:, 0] += dx
            coll.set_offsets(offs)


def _plot(df: pd.DataFrame, output: Path):
    key_order = [k for _, keys, _ in _GROUPS for k in keys]
    positions = _compute_positions()

    fig = plt.figure(figsize=(8, 4))
    ax = sns.boxplot(
        data=df,
        x="key",
        y="time",
        hue="db",
        order=key_order,
        hue_order=["dud-e", "lit-pcba", "muv"],
        log_scale=True,
    )

    old_x = list(range(len(key_order)))
    new_x = [positions[k] for k in key_order]
    _shift_artists(ax, old_x, new_x)

    ax.set_xlabel("")
    ax.set_ylabel("Wall Clock Time (s/molecule)")
    ax.legend(
        handles=ax.legend_.legend_handles,
        labels=["DUD-E", "LIT-PCBA", "MUV"],
    )

    ax.tick_params(axis="x", which="both", length=0)
    xmax = max(positions.values()) + 0.75
    ax.set_xlim(None, xmax)

    group_centers = [i * _GROUP_SPACING for i in range(len(_GROUPS))]
    ax.set_xticks(
        group_centers,
        labels=[f"\n{name}" for name, _, _ in _GROUPS],
        minor=False,
    )

    minor_ticks = []
    minor_labels = []
    for gi, (_, keys, sub_labels) in enumerate(_GROUPS):
        for ki, (key, sub) in enumerate(zip(keys, sub_labels)):
            if sub:
                minor_ticks.append(positions[key])
                minor_labels.append(sub)

    if minor_ticks:
        ax.set_xticks(minor_ticks, labels=minor_labels, minor=True, fontsize=8)

    fig.savefig(output, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(bench_home: Path = Path.home() / "benchmark/bench"):
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Helvetica Neue",
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )

    cache = bench_home / _CACHE_NAME
    output = bench_home / "speed.svg"

    typer.echo("Loading timing data ...")
    df = _load_all(bench_home, cache)
    _report_relative(df)

    typer.echo(f"Plotting to {output} ...")
    _plot(df, output)
    typer.echo("Done.")


if __name__ == "__main__":
    app()
