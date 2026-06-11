"""Randomization test for the LIT-PCBA cross-validation decoy subset.

For each target x method pair, this checks whether the 1,000-decoy subset that
was randomly selected with seed 42 (by ``select_random_crossval.py``) yields
screening metrics that are systematically favorable relative to other random
1,000-decoy draws from the same full decoy pool.

Both screening metrics are reported on a [0, 1] scale: AUROC, and NEF1% =
EF1% / EF_max, the enrichment factor normalized by its theoretical maximum for
the evaluated set.  Normalizing by EF_max divides out the mechanical dependence
of the EF ceiling on the decoy-set size, so NEF is comparable across set sizes.

Step A (observed): with the subset's active scores held fixed, compute AUROC and
NEF1% using the exact 1,000 seed-42 decoys.

Step B (null): repeatedly resample 1,000 decoys without replacement from the
full decoy score vector and recompute AUROC and NEF1% with the same fixed active
scores.

Step C (locate): per target x method x metric, report the observed value, the
null median, the 95% null interval, the percentile rank of the observed subset
within the null, and the centered deviation (observed - null mean).  A
cross-pair summary reports the mean deviation, the mean null SD, the count of
pairs inside the 95% null interval, the fraction of targets above the null
median (effective %, ties counted as 0.5), and a two-sided Wilcoxon
signed-rank test of the percentile ranks against 0.5 -- a fraction near 0.5
(and not systematically above it) with a non-significant Wilcoxon p is
evidence against favorable decoy bias.

Decoy-set-size test (concern i): in the same pass, the full-decoy-pool AUROC
and NEF1% (10 fixed actives + all decoys) are compared against the seed-42
1,000-decoy subset value.  The agreement delta ``agree_delta = observed -
full_observed`` (subset minus full pool) is summarized with Bland-Altman style
descriptives -- bias (mean delta), 95% limits of agreement (bias +/- 1.96 SD),
maximum absolute delta -- and Lin's concordance correlation coefficient between
the subset and full-pool values across the 78 (target x method) pairs per
metric.  A negative bias means the subset understates the full-set metric
(conservative).  Because the agreement set uses the realized seed-42 draw (not
the resampling null mean), there is no pseudoreplication.  A 1-Wasserstein
distance (normalized by the full pool's IQR) quantifies how faithfully the
subset reproduces the full decoy-score distribution, and the randomization-test
summary reports the count of targets whose metric is invariant to decoy
resampling (``n_degenerate_null``).
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from joblib import Parallel, delayed
from scipy import stats
from sklearn import metrics as skmetrics

from shared_metrics import (
    BASELINE_METHODS,
    GSCREEN_METHODS,
    METHOD_SLUG_MAP,
    METHOD_STYLES,
    enrichment_factor,
    load_gscreen_scores,
    load_method_scores,
)

app = typer.Typer(pretty_exceptions_enable=False)

# G-screen submethod -> score column in the per-target score table.
GSCREEN_SUBMETHODS = {"GS-S": "shape", "GS-P": "pharma", "GS-SP": "score"}
# External baselines to evaluate (slug -> display name via METHOD_SLUG_MAP).
BASELINE_SLUGS = ["ls-align", "pharmagist", "autodock-vina"]

ALL_METHODS = GSCREEN_METHODS + BASELINE_METHODS

# Columns kept in the (unchanged) randomization-test per-pair table.
RANDTEST_COLS = [
    "target",
    "method",
    "metric",
    "observed",
    "delta",
    "null_median",
    "null_mean",
    "null_std",
    "ci_lo",
    "ci_hi",
    "q25",
    "q75",
    "percentile",
    "n_null",
    "n_actives",
    "n_decoys",
    "n_pool",
]
# Columns written for the decoy-set-size test per-pair table.  Carries the null
# context (null_median/q25/q75/observed) needed to redraw the overlay panel.
FULLSET_COLS = [
    "target",
    "method",
    "metric",
    "observed",
    "null_median",
    "null_std",
    "q25",
    "q75",
    "full_observed",
    "full_delta",
    "agree_delta",
    "full_percentile",
    "full_within_ci",
    "w1",
    "w1_norm",
]


# ---------------------------------------------------------------------------
# Subset selection recovery
# ---------------------------------------------------------------------------


def _read_mol2_ids(path: Path) -> list[str]:
    """Return molecule names (PubChem CIDs) in a mol2 file, in file order.

    The molecule name is the first line after each ``@<TRIPOS>MOLECULE`` record.
    """
    ids: list[str] = []
    with open(path) as fh:
        lines = iter(fh)
        for line in lines:
            if line.startswith("@<TRIPOS>MOLECULE"):
                name = next(lines, "").strip()
                if name:
                    ids.append(name)
    return ids


def _discover_subset_targets(
    db_home: Path,
) -> dict[str, tuple[list[str], list[str]]]:
    """Map each target with a generated subset to its (active, decoy) id lists.

    Targets lacking ``subset/decoys_*.mol2`` (e.g. FEN1, OPRK1) are skipped.
    """
    targets: dict[str, tuple[list[str], list[str]]] = {}
    for target_dir in sorted(db_home.iterdir()):
        subset = target_dir / "subset"
        if not subset.is_dir():
            continue

        actives = sorted(subset.glob("actives*.mol2"))
        decoys = sorted(subset.glob("decoys*.mol2"))
        if not actives or not decoys:
            continue

        active_ids = _read_mol2_ids(actives[0])
        decoy_ids = _read_mol2_ids(decoys[0])
        if not active_ids or not decoy_ids:
            typer.echo(
                f"Warning: empty subset ids for {target_dir.name}, skipping",
                err=True,
            )
            continue

        targets[target_dir.name] = (active_ids, decoy_ids)

    return targets


# ---------------------------------------------------------------------------
# Slim per-(target, method) score lookups (parent process)
# ---------------------------------------------------------------------------


def _gscreen_lookups(
    df: pd.DataFrame,
    active_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """id-indexed [shape, pharma, score] frames for one target's G-screen table.

    Returns (active_df, decoy_df): active_df is restricted to the seed-42 active
    ids that are present; decoy_df holds every decoy with valid scores.
    """
    cols = list(GSCREEN_SUBMETHODS.values())
    g = df.drop_duplicates("id").set_index("id")
    is_active = g["is_active"].astype(bool)
    decoy_df = g.loc[~is_active, cols].dropna()
    active_df = g[cols].reindex(active_ids).dropna()
    return active_df, decoy_df


def _method_lookups(
    df: pd.DataFrame,
    active_ids: list[str],
) -> tuple[pd.Series, pd.Series]:
    """id-indexed score series for one target's baseline table.

    Returns (active_series, decoy_series): active_series is restricted to the
    seed-42 active ids that are present; decoy_series holds every decoy with a
    valid score.
    """
    m = df.drop_duplicates("id").set_index("id")
    is_active = m["is_active"].astype(bool)
    decoy_series = m.loc[~is_active, "score"].dropna()
    active_series = m["score"].reindex(active_ids).dropna()
    return active_series, decoy_series


# ---------------------------------------------------------------------------
# Metrics + per-pair randomization test (worker process)
# ---------------------------------------------------------------------------


def _max_ef(n_actives: int, n_total: int, ratio: float) -> float:
    """Theoretical maximum EF at ``ratio`` for ``n_actives`` of ``n_total``.

    Uses the same selection size ``n_select = ceil(ratio * n_total)`` as
    ``enrichment_factor``; the best case puts ``min(n_actives, n_select)``
    actives at the top, giving ``EF_max = min(n_act, n_select) * n_total /
    (n_select * n_act)`` (i.e. ``min(1/ratio, n_total/n_act)``).
    """
    n_select = ratio * n_total
    return min(n_actives, n_select) * n_total / (n_select * n_actives)


def _metrics(
    active_scores: np.ndarray,
    decoy_scores: np.ndarray,
    ratio: float,
) -> tuple[float, float]:
    """AUROC and NEF at ``ratio`` for fixed actives against given decoys.

    NEF (normalized EF) is ``EF / EF_max`` with ``EF_max`` evaluated for *this*
    set's size, so it lies in ``[0, 1]`` and is comparable across decoy-set
    sizes -- the mechanical dependence of the EF ceiling on the pool size (and
    hence ``n_select``) is divided out, isolating genuine enrichment quality.
    """
    n_act = active_scores.size
    n_total = n_act + decoy_scores.size
    labels = np.concatenate([np.ones(n_act), np.zeros(decoy_scores.size)])
    scores = np.concatenate([active_scores, decoy_scores])
    auroc = skmetrics.roc_auc_score(labels, scores)
    ef = enrichment_factor(labels, scores, ratio=ratio, strict_mode=False)
    nef = ef / _max_ef(n_act, n_total, ratio)
    return float(auroc), float(nef)


def _locate(
    target: str,
    method: str,
    metric: str,
    observed: float,
    full_observed: float,
    null: np.ndarray,
    n_actives: int,
    n_decoys: int,
    n_pool: int,
    w1: float,
    w1_norm: float,
) -> dict:
    """Locate the subset ``observed`` and the ``full_observed`` value in ``null``.

    The same decoy-resampling null (fixed actives/references; only decoy
    count/identity varies) serves both tests: ``observed`` is the seed-42
    1,000-decoy subset metric (randomization test, concern ii) and
    ``full_observed`` is the full-decoy-pool metric (size test, concern i).
    Placing the full-set value here isolates the effect of decoy-set size.
    """
    ci_lo, ci_hi = np.percentile(null, [2.5, 97.5])
    q25, q75 = np.percentile(null, [25, 75])
    null_std = null.std(ddof=1)
    null_mean = null.mean()

    # Metrics are on a [0, 1] scale (AUROC; NEF = EF / EF_max).  ``full_delta``
    # places the full-pool value relative to the null (full - null_mean), while
    # ``agree_delta = observed - full_observed`` (seed-42 subset minus full
    # pool) measures agreement of the realized draw with the full set directly,
    # so a negative value means the subset understates the full-set metric.
    full_delta = float(full_observed - null_mean)

    return {
        "target": target,
        "method": method,
        "metric": metric,
        "observed": observed,
        "delta": float(observed - null_mean),
        "null_median": float(np.median(null)),
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "q25": float(q25),
        "q75": float(q75),
        "percentile": stats.percentileofscore(null, observed, kind="mean")
        / 100,
        "n_null": int(null.size),
        "n_actives": n_actives,
        "n_decoys": n_decoys,
        "n_pool": n_pool,
        "full_observed": full_observed,
        "full_delta": full_delta,
        "agree_delta": float(observed - full_observed),
        "full_percentile": stats.percentileofscore(
            null, full_observed, kind="mean"
        )
        / 100,
        "full_within_ci": bool(ci_lo <= full_observed <= ci_hi),
        "w1": float(w1),
        "w1_norm": float(w1_norm),
    }


def _run_pair(
    target: str,
    method: str,
    active_scores: np.ndarray,
    obs_decoy_scores: np.ndarray,
    pool_scores: np.ndarray,
    null_index: np.ndarray,
    ratio: float,
    metric_names: tuple[str, str],
) -> list[dict]:
    """Steps A-C for one target x method pair against the shared null draws.

    ``null_index`` is the per-target (n_null, n_dec) matrix of decoy positions
    into ``pool_scores`` -- identical across methods of the same target, so only
    the scoring differs between methods.
    """
    n_dec = obs_decoy_scores.size
    n_act = active_scores.size
    n_null = null_index.shape[0]

    obs_auroc, obs_ef = _metrics(active_scores, obs_decoy_scores, ratio)
    full_auroc, full_ef = _metrics(active_scores, pool_scores, ratio)

    # Score-distribution fidelity of the 1,000-decoy subset relative to the
    # full pool, normalized by the pool's IQR (robust to LIT-PCBA's weak
    # active/decoy separation).  Metric-independent, so shared by both rows.
    w1 = stats.wasserstein_distance(obs_decoy_scores, pool_scores)
    pool_iqr = stats.iqr(pool_scores)
    w1_norm = w1 / pool_iqr if pool_iqr > 0 else float("nan")

    null_auroc = np.empty(n_null)
    null_ef = np.empty(n_null)
    for i in range(n_null):
        sample = pool_scores[null_index[i]]
        null_auroc[i], null_ef[i] = _metrics(active_scores, sample, ratio)

    auroc_name, ef_name = metric_names
    n_pool = pool_scores.size
    return [
        _locate(
            target,
            method,
            auroc_name,
            obs_auroc,
            full_auroc,
            null_auroc,
            n_act,
            n_dec,
            n_pool,
            w1,
            w1_norm,
        ),
        _locate(
            target,
            method,
            ef_name,
            obs_ef,
            full_ef,
            null_ef,
            n_act,
            n_dec,
            n_pool,
            w1,
            w1_norm,
        ),
    ]


# ---------------------------------------------------------------------------
# Cross-pair summary
# ---------------------------------------------------------------------------


def _lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient between ``x`` and ``y``.

    Uses population moments (``ddof=0``):
    ``ccc = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))**2)``,
    where ``2 * rho * sx * sy == 2 * cov``.  CCC penalizes both poor
    correlation and any systematic shift, so a value near 1 means the
    seed-42 subset and full-pool metrics agree on both location and scale.
    Returns ``nan`` when both vectors are constant (variance undefined).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    if vx == 0 and vy == 0:
        return float("nan")
    cov = float(((x - mx) * (y - my)).mean())
    return float(2 * cov / (vx + vy + (mx - my) ** 2))


def _wilcoxon_p(values: np.ndarray) -> float:
    if np.all(values == 0):
        return 1.0

    return float(
        stats.wilcoxon(
            values,
            zero_method="zsplit",
            alternative="two-sided",
        ).pvalue
    )


def _summarize(
    per_pair: pd.DataFrame,
    group_cols: list[str],
    metric_names: tuple[str, str],
) -> pd.DataFrame:
    """Per-group percentile summary of the realized-draw deviation.

    For each method x metric this reports the mean native deviation
    (``mean_delta = observed - null_mean``), the mean per-pair null SD
    (``mean_std``), the count of pairs whose metric is invariant to decoy
    resampling (``n_degenerate_null``), the count/fraction of pairs whose
    observed metric falls inside the per-pair 95% null interval, the
    count/fraction of pairs whose observed value exceeds the null median
    (``n_above_median``/``frac_above_median``, ties counted as 0.5), and a
    two-sided Wilcoxon signed-rank test of the observed percentile ranks
    against the no-bias expectation of 0.5.  Values near no bias (delta ~ 0,
    fraction ~ 0.5, non-significant Wilcoxon) indicate the seed-42 draw is not
    favorable.
    """
    rows: list[dict] = []
    for keys, grp in per_pair.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        pct = grp["percentile"].to_numpy()
        n = int(pct.size)

        n_above = int((pct > 0.5).sum())
        n_ties = int((pct == 0.5).sum())
        n_eff = n_above + n_ties / 2

        within = (grp["observed"] >= grp["ci_lo"]) & (
            grp["observed"] <= grp["ci_hi"]
        )
        n_within = int(within.sum())

        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_pairs": n,
                "mean_delta": float(grp["delta"].mean()),
                "mean_std": float(grp["null_std"].mean()),
                "n_degenerate_null": _n_degenerate_null(grp["null_std"]),
                "n_within_ci": n_within,
                "frac_within_ci": n_within / n,
                "n_above_median": n_eff,
                "frac_above_median": n_eff / n,
                "wilcoxon_percentile_p": _wilcoxon_p(pct - 0.5),
            }
        )

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(
        df["method"],
        categories=ALL_METHODS,
        ordered=True,
    )
    df["metric"] = pd.Categorical(
        df["metric"],
        categories=list(metric_names),
        ordered=True,
    )
    df = df.sort_values(group_cols).reset_index(drop=True)
    return df


def _n_degenerate_null(null_std: pd.Series) -> int:
    """Count targets whose metric is invariant to decoy resampling.

    A point-mass null (every resample yields the identical metric) has zero
    spread; this is a robustness descriptor -- the metric does not move when
    the decoys are redrawn -- not a caveat.
    """
    return int((null_std.to_numpy(dtype=float) <= 1e-9).sum())


def _agree_delta(grp: pd.DataFrame) -> np.ndarray:
    """``agree_delta = observed - full_observed`` for a per-pair group.

    Recomputed from ``observed``/``full_observed`` rather than read from the
    stored column so ``--summarize-only`` works on per-pair CSVs written before
    the ``agree_delta`` column existed.
    """
    return (grp["observed"] - grp["full_observed"]).to_numpy(dtype=float)


def _fullset_agreement_by_method(
    per_pair: pd.DataFrame,
    metric_names: tuple[str, str],
) -> pd.DataFrame:
    """Per-(metric, method) agreement summary for the decoy-set-size test.

    For each metric x method across the 13 targets, the agreement delta
    ``agree_delta = observed - full_observed`` (seed-42 1,000-decoy subset
    minus full pool) is described by its ``bias`` (mean), ``sd_diff``
    (std, ddof=1), and ``max_abs_delta`` (max absolute delta).  A negative bias
    means the subset understates the full-set metric (conservative).  The
    median normalized W1 (subset decoys vs. full pool; metric-independent, so
    identical across a method's two metric rows) rounds out the row as a
    robustness descriptor.
    """
    rows: list[dict] = []
    for keys, grp in per_pair.groupby(["metric", "method"], observed=True):
        metric, method = keys
        delta = _agree_delta(grp)

        rows.append(
            {
                "metric": metric,
                "method": method,
                "n": int(delta.size),
                "bias": float(np.mean(delta)),
                "sd_diff": float(np.std(delta, ddof=1)),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "median_w1_norm": float(grp["w1_norm"].median()),
            }
        )

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(
        df["method"], categories=ALL_METHODS, ordered=True
    )
    df["metric"] = pd.Categorical(
        df["metric"], categories=list(metric_names), ordered=True
    )
    df = df.sort_values(["metric", "method"]).reset_index(drop=True)
    return df


def _fullset_agreement_overall(
    per_pair: pd.DataFrame,
    metric_names: tuple[str, str],
) -> pd.DataFrame:
    """Per-metric agreement aggregate over all (target x method) pairs.

    Pools the 78 distinct (target x method) pairs per metric into a single
    Bland-Altman style row: ``bias`` (mean ``agree_delta``), the 95% limits of
    agreement ``loa_lower/upper = bias -/+ 1.96 * sd_diff``, ``max_abs_delta``,
    and Lin's concordance correlation coefficient (``lins_ccc``) between the
    subset and full-pool value vectors.  CCC near 1 means the 1,000-decoy
    subset reproduces the full-pool metric on both location and scale.
    """
    rows: list[dict] = []
    for metric, grp in per_pair.groupby("metric", observed=True):
        delta = _agree_delta(grp)
        bias = float(np.mean(delta))
        sd_diff = float(np.std(delta, ddof=1))

        rows.append(
            {
                "metric": metric,
                "n_pairs": int(delta.size),
                "bias": bias,
                "loa_lower": bias - 1.96 * sd_diff,
                "loa_upper": bias + 1.96 * sd_diff,
                "max_abs_delta": float(np.max(np.abs(delta))),
                "lins_ccc": _lins_ccc(
                    grp["observed"].to_numpy(dtype=float),
                    grp["full_observed"].to_numpy(dtype=float),
                ),
            }
        )

    df = pd.DataFrame(rows)
    df["metric"] = pd.Categorical(
        df["metric"], categories=list(metric_names), ordered=True
    )
    df = df.sort_values("metric").reset_index(drop=True)
    return df


def _report_fullset(
    fullset_pp: pd.DataFrame,
    output_dir: Path,
    metric_names: tuple[str, str],
):
    """Write the size-test agreement summaries + figures and echo the readout.

    Emits a per-(metric, method) frame and a per-metric Bland-Altman aggregate,
    frames the readout as agreement (bias, 95% limits of agreement, max |delta|,
    Lin's CCC, median normalized W1), and names the single largest-|delta| pair
    per metric for transparency.
    """
    bymethod = _fullset_agreement_by_method(fullset_pp, metric_names)
    bymethod_path = output_dir / "randtest_fullset_agreement_bymethod.csv"
    bymethod.to_csv(bymethod_path, index=False)
    typer.echo(f"  wrote {bymethod_path}")

    agg = _fullset_agreement_overall(fullset_pp, metric_names)
    agg_path = output_dir / "randtest_fullset_agreement.csv"
    agg.to_csv(agg_path, index=False)
    typer.echo(f"  wrote {agg_path}")

    fig_path = output_dir / "randtest_fullset_observed_vs_null.svg"
    _plot_observed_vs_null(fullset_pp, fig_path, metric_names, show_full=True)
    typer.echo(f"  wrote {fig_path}")

    ba_path = output_dir / "randtest_fullset_bland_altman.svg"
    _plot_bland_altman(fullset_pp, agg, ba_path, metric_names)
    typer.echo(f"  wrote {ba_path}")

    pd.options.display.float_format = "{:.4f}".format
    typer.echo("\nDecoy-set-size test (concern i) -- per-method agreement:")
    typer.echo(bymethod.to_string(index=False))
    typer.echo("\nDecoy-set-size test (concern i) -- per-metric agreement:")
    typer.echo(agg.to_string(index=False))

    typer.echo(
        "\nInterpretation: both metrics are on a [0, 1] scale (AUROC; "
        "NEF = EF / EF_max).  agree_delta = subset - full pool, so a negative "
        "bias means the 1,000-decoy subset understates the full-set metric "
        "(conservative).  Per metric (78 target x method pairs):"
    )
    fullset_pp = fullset_pp.assign(agree_delta=_agree_delta(fullset_pp))
    for _, row in agg.iterrows():
        sub = fullset_pp[fullset_pp["metric"] == row["metric"]]
        median_w1 = float(sub["w1_norm"].median())
        typer.echo(
            f"  {str(row['metric']).upper():>8s}: bias={row['bias']:+.4f}, "
            f"95% LoA=[{row['loa_lower']:+.4f}, {row['loa_upper']:+.4f}], "
            f"max|delta|={row['max_abs_delta']:.4f}, "
            f"CCC={row['lins_ccc']:.4f}, median W1_norm={median_w1:.4f}"
        )
        worst = sub.loc[sub["agree_delta"].abs().idxmax()]
        typer.echo(
            f"           largest |delta|: {worst['method']} / "
            f"{worst['target']} subset={worst['observed']:.3f} vs "
            f"full={worst['full_observed']:.3f} "
            f"(delta={worst['agree_delta']:+.4f})"
        )
    pd.options.display.float_format = None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _metric_label(metric: str) -> str:
    return "AUROC" if metric == "aucroc" else metric.upper()


def _plot_observed_vs_null(
    per_pair: pd.DataFrame,
    output: Path,
    metric_names: tuple[str, str],
    show_full: bool = False,
):
    """Per-target caterpillar of the observed metric against its null interval.

    Rows are metrics, columns are methods.  Within each panel every target shows
    its decoy-resampling null median with the interquartile range (25-75
    percentiles; grey error bar) and the observed seed-42 value (colored marker)
    drawn on top, so the reader can see the observed value lying within the bulk
    of the null per target.  When ``show_full`` is set, the full-decoy-pool
    value (size test, concern i) is overlaid as a star marker so the reader can
    see where the full-set metric lands relative to the same null.
    """
    metrics = [m for m in metric_names if m in per_pair["metric"].unique()]
    methods = [m for m in ALL_METHODS if m in per_pair["method"].unique()]
    targets = sorted(per_pair["target"].unique())
    y = np.arange(len(targets), dtype=float)

    fig = plt.figure(
        figsize=(2 * len(methods), 0.2 * len(targets) * len(metrics) + 2.0),
        layout="constrained",
    )
    subfigs = np.atleast_1d(fig.subfigures(len(metrics), 1))

    handles: list = []
    for r, (subfig, metric) in enumerate(zip(subfigs, metrics)):
        subfig.suptitle(_metric_label(metric), fontweight=600, fontsize=12)
        axs = subfig.subplots(
            1, len(methods), sharex=True, sharey=True, squeeze=False
        )[0]
        for c, (ax, method) in enumerate(zip(axs, methods)):
            sub = (
                per_pair[
                    (per_pair["metric"] == metric)
                    & (per_pair["method"] == method)
                ]
                .set_index("target")
                .reindex(targets)
            )

            null_med = sub["null_median"].to_numpy()
            lo = sub["q25"].to_numpy()
            hi = sub["q75"].to_numpy()
            obs = sub["observed"].to_numpy()
            xerr = np.vstack([null_med - lo, hi - null_med])
            color = METHOD_STYLES.get(method, {}).get("color", "#0072B2")

            h_null = ax.errorbar(
                null_med,
                y,
                xerr=xerr,
                fmt="o",
                ms=4,
                color="grey",
                ecolor="grey",
                elinewidth=1.0,
                capsize=4,
                capthick=1.0,
                zorder=2,
                label="null median (IQR)",
            )
            h_obs = ax.scatter(
                obs,
                y,
                marker="D",
                s=24,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
                label="observed (1,000-decoy subset)",
            )
            h_full = None
            if show_full:
                full = sub["full_observed"].to_numpy()
                h_full = ax.scatter(
                    full,
                    y,
                    marker="*",
                    s=70,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=4,
                    label="full decoy pool",
                )
            if r == 0 and c == 0:
                handles = [h_obs, h_null]
                if h_full is not None:
                    handles.insert(1, h_full)

            ax.set_title(method, fontsize=10, fontweight=500)
            if c == 0:
                ax.set_yticks(range(len(targets)))
                ax.set_yticklabels(targets, fontsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.margins(y=0.03)
        axs[0].invert_yaxis()

    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    for ext in [".svg", ".png"]:
        fig.savefig(
            output.with_suffix(ext),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def _plot_bland_altman(
    per_pair: pd.DataFrame,
    agg: pd.DataFrame,
    output: Path,
    metric_names: tuple[str, str],
):
    """Per-metric Bland-Altman agreement panel (subset vs. full decoy pool).

    One panel per metric: x = full-pool value, y = ``agree_delta`` (subset -
    full pool), points colored by method.  Horizontal reference lines mark the
    pooled bias and the 95% limits of agreement (bias +/- 1.96 SD) from
    ``agg``.  Descriptive only -- it visualizes the agreement quantities, not a
    test.
    """
    metrics = [m for m in metric_names if m in per_pair["metric"].unique()]
    methods = [m for m in ALL_METHODS if m in per_pair["method"].unique()]
    agg_by_metric = agg.set_index("metric")

    # NEF1% is exactly 0 for both subset and full pool on many pairs (no active
    # in the top 1%), so those points coincide at (0, 0).  A small seeded
    # x-jitter plus marker transparency reveals that stack as density without
    # perturbing the meaningful y = agree_delta value.
    rng = np.random.default_rng(0)

    fig, axs = plt.subplots(
        1,
        len(metrics),
        figsize=(4.5 * len(metrics), 4.0),
        squeeze=False,
        layout="constrained",
    )
    axs = axs[0]
    handles: list = []
    for ax, metric in zip(axs, metrics):
        sub = per_pair[per_pair["metric"] == metric]
        x_all = sub["full_observed"].to_numpy()
        x_span = float(np.ptp(x_all)) if x_all.size else 0.0
        jitter = 0.005 * x_span
        for method in methods:
            ms = sub[sub["method"] == method]
            if ms.empty:
                continue
            color = METHOD_STYLES.get(method, {}).get("color", "#0072B2")
            xs = ms["full_observed"].to_numpy()
            if jitter > 0:
                xs = xs + rng.normal(0.0, jitter, xs.shape)
            h = ax.scatter(
                xs,
                ms["agree_delta"].to_numpy(),
                s=26,
                color=color,
                alpha=0.6,
                edgecolor="black",
                linewidth=0.3,
                label=method,
                zorder=3,
            )
            if ax is axs[0]:
                handles.append(h)

        bias = float(agg_by_metric.loc[metric, "bias"])
        lo = float(agg_by_metric.loc[metric, "loa_lower"])
        hi = float(agg_by_metric.loc[metric, "loa_upper"])
        ax.axhline(0.0, color="grey", lw=0.8, ls=":", zorder=1)
        ax.axhline(bias, color="black", lw=1.2, zorder=2, label="bias")
        ax.axhline(lo, color="black", lw=1.0, ls="--", zorder=2)
        ax.axhline(
            hi, color="black", lw=1.0, ls="--", zorder=2, label="95% LoA"
        )

        n_pairs = int(agg_by_metric.loc[metric, "n_pairs"])
        ccc = float(agg_by_metric.loc[metric, "lins_ccc"])
        ax.text(
            0.97,
            0.97,
            f"$N = {n_pairs}$\nLin's CCC$= {ccc:.4f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            multialignment="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="grey",
                alpha=0.8,
            ),
        )

        ax.set_xlabel(_metric_label(metric) + " (full)")
        ax.set_ylabel(R"$\Delta(\mathrm{{subset}} - \mathrm{{full}})$")
        ax.tick_params(labelsize=8)

    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 6),
    )
    for ext in [".svg", ".png"]:
        fig.savefig(output.with_suffix(ext), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    results: Path = Path.home()
    / "repo/seoklab/gscreen-data/benchmark/lit-pcba",
    db_home: Path = Path.home() / "db",
    bench_home: Path = Path.home() / "benchmark",
    output_dir: Path = Path("tmp/lit-pcba-randtest"),
    n_null: int = 2000,
    ratio: float = 0.01,
    seed: int = 42,
    nproc: int = 8,
    summarize_only: bool = False,
):
    db = results.name
    db_home = db_home / db
    bench_home = bench_home / db
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names: tuple[str, str] = ("aucroc", f"nef {ratio * 100:g}%")
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "Helvetica Neue",
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )
    fig_path = output_dir / "randtest_observed_vs_null.svg"

    if summarize_only:
        per_pair = pd.read_csv(output_dir / "randtest_per_pair.csv")

        summary = _summarize(per_pair, ["metric", "method"], metric_names)
        summary_path = output_dir / "randtest_summary.csv"
        summary.to_csv(summary_path, index=False)
        typer.echo(f"  wrote {summary_path}")

        _plot_observed_vs_null(per_pair, fig_path, metric_names)
        typer.echo(f"  wrote {fig_path}")

        fullset_pp = pd.read_csv(output_dir / "randtest_fullset_per_pair.csv")
        _report_fullset(fullset_pp, output_dir, metric_names)
        return

    typer.echo(f"Discovering cross-validation subsets under {db_home} ...")
    subset_ids = _discover_subset_targets(db_home)
    targets = sorted(subset_ids)
    typer.echo(f"  {len(targets)} targets with subsets: {', '.join(targets)}")

    # ------------------------------------------------------------------
    # Load slim per-(target, method) score lookups for every method at once
    # (needed to intersect molecules across methods), discarding the big
    # DataFrames after slicing to keep peak memory bounded.
    typer.echo("Loading G-screen scores ...")
    gscreen_scores = load_gscreen_scores(results, db_home)
    gscreen_lookups: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target in targets:
        active_ids, _ = subset_ids[target]
        gscreen_lookups[target] = _gscreen_lookups(
            gscreen_scores[target], active_ids
        )
    del gscreen_scores

    baseline_lookups: dict[str, dict[str, tuple[pd.Series, pd.Series]]] = {}
    for slug in BASELINE_SLUGS:
        name = METHOD_SLUG_MAP.get(slug, slug)
        typer.echo(f"Loading {name} scores ...")
        method_scores = load_method_scores(bench_home, slug, skip_missing=True)
        baseline_lookups[slug] = {}
        for target in targets:
            mdf = method_scores.get(target)
            if mdf is None:
                typer.echo(f"Warning: no {name} scores for {target}", err=True)
                continue
            active_ids, _ = subset_ids[target]
            baseline_lookups[slug][target] = _method_lookups(mdf, active_ids)
        del method_scores

    # ------------------------------------------------------------------
    # Per target: intersect molecules across methods and pre-generate one
    # shared set of null decoy draws (identical for every method).
    typer.echo(
        "\nBuilding shared per-target null (intersection across methods) ..."
    )
    child_seeds = np.random.SeedSequence(seed).spawn(len(targets))
    tasks: list[tuple] = []
    for target, child_seed in zip(targets, child_seeds):
        seed42_active_ids, seed42_decoy_ids = subset_ids[target]
        active_df, decoy_df = gscreen_lookups[target]

        present = {
            slug: baseline_lookups[slug][target]
            for slug in BASELINE_SLUGS
            if target in baseline_lookups[slug]
        }

        active_idx = [set(active_df.index)] + [
            set(a.index) for a, _ in present.values()
        ]
        decoy_idx = [set(decoy_df.index)] + [
            set(d.index) for _, d in present.values()
        ]
        common_active = sorted(set.intersection(*active_idx))
        pool_set = set.intersection(*decoy_idx)
        pool_ids = sorted(pool_set)
        obs_decoy_ids = [d for d in seed42_decoy_ids if d in pool_set]

        n_act, n_dec, n_pool = (
            len(common_active),
            len(obs_decoy_ids),
            len(pool_ids),
        )
        typer.echo(
            f"  {target}: common actives {n_act}/{len(seed42_active_ids)}, "
            f"decoy pool {n_pool}, observed decoys "
            f"{n_dec}/{len(seed42_decoy_ids)}"
        )
        if n_act == 0 or n_dec == 0:
            typer.echo(
                f"Warning: {target} has no usable common molecules, skipping",
                err=True,
            )
            continue

        rng = np.random.default_rng(child_seed)
        null_index = np.stack(
            [
                rng.choice(n_pool, n_dec, replace=False, shuffle=False)
                for _ in range(n_null)
            ]
        ).astype(np.int32)

        method_cols: list[tuple[str, pd.Series, pd.Series]] = [
            (method, active_df[col], decoy_df[col])
            for method, col in GSCREEN_SUBMETHODS.items()
        ]
        method_cols += [
            (METHOD_SLUG_MAP.get(slug, slug), active_s, decoy_s)
            for slug, (active_s, decoy_s) in present.items()
        ]

        for method, active_s, decoy_s in method_cols:
            active_scores = active_s.reindex(common_active).to_numpy(float)
            obs_scores = decoy_s.reindex(obs_decoy_ids).to_numpy(float)
            pool_scores = decoy_s.reindex(pool_ids).to_numpy(float)
            missing = (
                np.isnan(active_scores).any()
                or np.isnan(obs_scores).any()
                or np.isnan(pool_scores).any()
            )
            assert not missing, (
                f"unexpected missing scores for {method} on {target}"
            )
            tasks.append(
                (
                    target,
                    method,
                    active_scores,
                    obs_scores,
                    pool_scores,
                    null_index,
                )
            )

    typer.echo(
        f"\nRunning randomization test on {len(tasks)} target x method "
        f"pairs ({n_null} null draws each, nproc={nproc}) ..."
    )

    results_nested = Parallel(n_jobs=nproc)(
        delayed(_run_pair)(
            target,
            method,
            active_scores,
            obs_scores,
            pool_scores,
            null_index,
            ratio,
            metric_names,
        )
        for (
            target,
            method,
            active_scores,
            obs_scores,
            pool_scores,
            null_index,
        ) in tasks
    )
    per_pair = pd.DataFrame(
        list(itertools.chain.from_iterable(results_nested))
    )

    per_pair["method"] = pd.Categorical(
        per_pair["method"], categories=ALL_METHODS, ordered=True
    )
    per_pair["metric"] = pd.Categorical(
        per_pair["metric"], categories=list(metric_names), ordered=True
    )
    per_pair = per_pair.sort_values(
        ["metric", "method", "target"]
    ).reset_index(drop=True)

    per_pair_path = output_dir / "randtest_per_pair.csv"
    per_pair[RANDTEST_COLS].to_csv(per_pair_path, index=False)
    typer.echo(f"  wrote {per_pair_path}")

    fullset_pp = per_pair[FULLSET_COLS]
    fullset_pp_path = output_dir / "randtest_fullset_per_pair.csv"
    fullset_pp.to_csv(fullset_pp_path, index=False)
    typer.echo(f"  wrote {fullset_pp_path}")

    # ------------------------------------------------------------------
    summary = _summarize(per_pair, ["metric", "method"], metric_names)
    summary_path = output_dir / "randtest_summary.csv"
    summary.to_csv(summary_path, index=False)
    typer.echo(f"  wrote {summary_path}")

    # ------------------------------------------------------------------
    _plot_observed_vs_null(per_pair, fig_path, metric_names)
    typer.echo(f"  wrote {fig_path}")

    # ------------------------------------------------------------------
    pd.options.display.float_format = "{:.3f}".format

    typer.echo("\nPer-method summary (randomization test, concern ii):")
    typer.echo(summary.to_string(index=False))

    typer.echo(
        "\nInterpretation: a fraction of targets above the null median near "
        "0.5 (not systematically above) and non-significant Wilcoxon tests "
        "indicate the seed-42 decoy subset is not biased toward favorable "
        "metrics."
    )
    pd.options.display.float_format = None

    # ------------------------------------------------------------------
    _report_fullset(fullset_pp, output_dir, metric_names)

    typer.echo(f"\nAll outputs written to {output_dir}/")


if __name__ == "__main__":
    app()
