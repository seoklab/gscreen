import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Type

from ..api import Serializable, SerializableDic, SerializableObj
from .pharmacophore import (
    HydrogenBonding,
    Hydrophobic,
    Pharmacophore,
    PiStacking,
    Site,
)

__all__ = ["Report", "load_reports"]

_logger = logging.getLogger(__name__)


class Report(Serializable):
    def __init__(
        self,
        lig_pc: Pharmacophore,
        rec_pc: Pharmacophore,
        lig_file: Path = None,
        rec_file: Path = None,
    ):
        super().__init__()
        self.lig_pc: Pharmacophore = lig_pc
        self.lig_file: Optional[Path] = lig_file
        self.rec_pc: Pharmacophore = rec_pc
        self.rec_file: Optional[Path] = rec_file

    @staticmethod
    def _make_record(pc: Pharmacophore, file: Optional[Path]):
        ret: SerializableDic = {"pharmacophore": pc}
        if file is not None:
            ret["file"] = str(file)
        return ret

    @staticmethod
    def _from_record(record: dict):
        try:
            file = Path(record["file"])
        except KeyError:
            file = None
        return record.get("pharmacophore", {}), file

    def __getstate__(self) -> SerializableObj:
        return {
            "ligand": self._make_record(self.lig_pc, self.lig_file),
            "receptor": self._make_record(self.rec_pc, self.rec_file),
        }

    def __setstate__(self, state: SerializableDic):
        self.lig_pc, self.lig_file = self._from_record(state.get("ligand", {}))
        self.rec_pc, self.rec_file = self._from_record(
            state.get("receptor", {})
        )


def _merge_pcs(pcs: Iterable[Pharmacophore]):
    merged: Pharmacophore = defaultdict(list)
    for pc in pcs:
        for k, v in pc.items():
            merged[k].extend(v)
    return merged


def _cluster_pcs(cls: Type[Site], pcs: Pharmacophore):
    try:
        pc = pcs[cls]
    except KeyError:
        clustered = []
    else:
        clustered = cls.cluster(pc)
    return clustered


def load_reports(
    reports: Iterable[Report],
) -> Dict[Type[Site], List[Tuple[Site, float]]]:
    clustered: Dict[Type[Site], List[Tuple[Site, int]]] = defaultdict(list)

    lig_pcs = _merge_pcs(report.lig_pc for report in reports)
    if lig_pcs:
        clustered[Hydrophobic] = _cluster_pcs(Hydrophobic, lig_pcs)

    rec_pcs = _merge_pcs(report.rec_pc for report in reports)
    if rec_pcs:
        for cls in (PiStacking, HydrogenBonding):
            clustered[cls] = _cluster_pcs(cls, rec_pcs)

    total_cnt = sum(cnt for pcs in clustered.values() for _, cnt in pcs)
    ret = {
        cls: [(site, cnt / total_cnt) for site, cnt in pcs]
        for cls, pcs in clustered.items()
    }
    _logger.debug(ret)
    return ret
