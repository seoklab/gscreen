import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from openbabel import openbabel as ob
from openbabel import pybel
from tqdm.auto import tqdm

from .. import api
from ..io import Mol2, Mol2Reader
from ..pipeline import ParallelModule
from ..tools import GAlign
from .chemistry import Mol
from .pharmacophore import *

WPc = Dict[Type[Site], List[Tuple[Site, float]]]
_logger = logging.getLogger(__name__)


class _StopProcessingException(Exception):
    pass


class _Callbacks:
    def on_begin(self):
        pass

    def on_success(self, weight: float):
        pass

    def on_end(self):
        pass


class _RequiredCallbacks(_Callbacks):
    def __init__(self) -> None:
        super().__init__()
        self.has_interaction = False

    def on_begin(self):
        self.has_interaction = False

    def on_success(self, weight: float):
        self.has_interaction = True

    def on_end(self):
        if not self.has_interaction:
            raise _StopProcessingException


class _ScoringCallbacks(_Callbacks):
    def __init__(self) -> None:
        super().__init__()
        self.score = 0.0

    def on_success(self, weight: float):
        self.score += weight


class PCFilter(ParallelModule):
    def __init__(
        self,
        ref_fp: pybel.Fingerprint,
        pcs: WPc,
        required_pcs: WPc = None,
        cutoff: float = 0.2,
        penalty: float = 0.0,
        strict: bool = True,
        nproc: Optional[int] = None,
        score_file: str = "scores.csv",
        pc_file: str = "pharmacophore.json",
    ):
        super().__init__(nproc=nproc)
        self.pcs = pcs
        self.required_pcs = required_pcs or {}
        self.ref_fp = np.array(ref_fp.bits)
        self.cutoff = cutoff
        self.penalty = penalty
        self.strict = strict
        self.score_file = score_file
        self.pc_file = pc_file

    def run(self, query: Path, result: Path, force: bool):
        results: List[
            Tuple[float, float, Optional[float], List[int], Optional[Mol2]]
        ]
        results = Parallel(n_jobs=self.nproc)(
            delayed(self._run_single)(mol)
            for mol in tqdm(
                Mol2Reader(query),
                disable=not _logger.isEnabledFor(logging.INFO),
            )
        )

        filtered: List[
            Tuple[float, float, Optional[float], List[int], Mol2]
        ] = [res for res in results if res[-1] is not None]
        if not filtered:
            self.log_warning("No molecules survived the filter")
            result.touch()
            return

        filtered.sort(key=lambda x: x[0], reverse=True)
        maxscore = filtered[0][0]

        scores = pd.DataFrame.from_records(
            [
                (
                    mol.name.rstrip().decode(),
                    *counts,
                    pharma_score,
                    shape_score,
                    tani_sim,
                )
                for (
                    pharma_score,
                    shape_score,
                    tani_sim,
                    counts,
                    mol,
                ) in filtered
            ],
            columns=[
                "name",
                *(
                    f"{site.type}{i}"
                    for sites in self.pcs.values()
                    for i, (site, _) in enumerate(sites)
                ),
                "pharma_score",
                "shape_score",
                "tani_sim",
            ],
        )

        mode = "wb" if force else "xb"

        with open(result.with_name(self.score_file), mode[0]) as fs:
            scores.to_csv(fs, index=False)

        with open(result.with_name(self.pc_file), mode[0]) as fp:
            api.dump(self.pcs, fp)

        if maxscore < self.cutoff:
            self.log_warning(f"No molecules above cutoff ({self.cutoff})")
            result.touch()
            return

        with open(result, mode) as fm:
            fm.write(
                b"".join(
                    bytes(mol)
                    for score, *_, mol in filtered
                    if score >= self.cutoff
                )
            )

    def _run_single(self, mol_bytes: Mol2):
        ob.obErrorLog.StopLogging()

        mol = Mol.loadf(mol_bytes)
        all_sites: Dict[Type[Site], List[Site]] = {
            cls: cls.from_mol(mol)
            for cls in [Hydrophobic, PiStacking, HydrogenBonding]
        }
        all_masks: Dict[Type[Site], List[bool]] = {
            cls: [True] * len(sites) for cls, sites in all_sites.items()
        }

        # TODO: get rid of this meaningless list
        ref_counts = [0] * sum(
            len(sites) for sites in self.required_pcs.values()
        )

        callbacks = _RequiredCallbacks()
        try:
            _eval_interaction(
                self.required_pcs,
                ref_counts,
                all_sites,
                all_masks,
                callbacks,
                self.strict,
            )
        except _StopProcessingException:
            return 0.0, None

        callbacks = _ScoringCallbacks()
        ref_counts = [0] * sum(len(sites) for sites in self.pcs.values())
        _eval_interaction(
            self.pcs, ref_counts, all_sites, all_masks, callbacks, self.strict
        )

        pharma_score = callbacks.score - self.penalty * sum(
            v for masks in all_masks.values() for v in masks
        )
        shape_score = GAlign.model_score(mol_bytes)

        tani_sim = _tani_smiliarity(
            self.ref_fp,
            np.array(mol.to_pybel().calcfp("ecfp4").bits),
        )

        return pharma_score, shape_score, tani_sim, ref_counts, mol_bytes


def _eval_interaction(
    all_refs: WPc,
    ref_counts: List[int],
    all_sites: Pharmacophore,
    all_masks: Dict[Type[Site], List[bool]],
    callbacks: _Callbacks,
    strict: bool,
):
    i = 0
    for t, pc in all_refs.items():
        if not pc or t not in all_sites:
            i += len(pc)
            continue

        sites = all_sites[t]
        no_interaction = all_masks[t]
        for ref, weight in pc:
            callbacks.on_begin()
            for j, site in enumerate(sites):
                score = t.evaluate(ref, site, strict=strict)
                if score != 0:
                    callbacks.on_success(weight * score)
                    ref_counts[i] += score
                if score > 0:
                    no_interaction[j] = False
            callbacks.on_end()
            i += 1


def _tani_smiliarity(b1: np.ndarray, b2: np.ndarray) -> float:
    bmax = max(b1[-1], b2[-1])

    fp1 = np.zeros(bmax + 1, dtype=np.bool_)
    fp1[b1] = True

    fp2 = np.zeros(bmax + 1, dtype=np.bool_)
    fp2[b2] = True

    return np.sum(fp1 & fp2) / np.sum(fp1 | fp2)
