import functools
import logging
import math
from abc import abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import networkx as nx
import numpy as np
from openbabel import openbabel as ob
from scipy.cluster import hierarchy as hier
from scipy.spatial import distance as D

from .. import utils
from ..api.serialize import Serializable, SerializableDic
from ..typing import Points, Vector
from . import geometry as geom
from . import topology as topo
from .chemistry import Mol

__all__ = [
    "Site",
    "PiStacking",
    "Hydrophobic",
    "HydrogenBonding",
    "Pharmacophore",
    "Charged",
]

_logger = logging.getLogger(__name__)

_S = TypeVar("_S", bound="Site")
Pharmacophore = Dict[Type["Site"], List["Site"]]

# Pre-calculated constants

_cos_15 = math.cos(math.radians(15))
_cos_30 = math.cos(math.radians(30))
_cos_45 = math.cos(math.radians(45))
_cos_60 = 0.5
_cos_90 = 0
_cos_120 = -_cos_60
_cos_135 = -_cos_45
_cos_150 = -_cos_30
_cos_165 = -_cos_15


class Site(Serializable, register=False):
    def __init__(self, center: Vector, idxs: Optional[List[int]] = None):
        super().__init__()
        self.center: Vector = center
        self.idxs: Optional[List[int]] = idxs

    def __getstate__(self) -> SerializableDic:
        state = {
            "center": self.center.tolist(),
            "command": self.to_chimera_command(),
        }
        if self.idxs is not None:
            state["idxs"] = list(map(int, self.idxs))
        return state  # type: ignore

    def __setstate__(self, state: SerializableDic) -> None:
        self.center = np.array(state["center"])
        self.idxs = state.get("idxs")  # type: ignore

    @classmethod
    @abstractmethod
    def cluster(
        cls: Type[_S], sites: List[_S], cutoff: float = 2.5
    ) -> List[Tuple[_S, int]]:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(reference: _S, query: _S, strict: bool = True) -> int:
        pass

    @classmethod
    @abstractmethod
    def from_mol(cls: Type[_S], mol: Mol, **kwargs) -> List[_S]:
        pass

    def interact(self, other: "Site", strict: bool = True) -> int:
        """Check interaction between self and other.

        Parameters
        ----------
        other : Site
            The other site.

        Returns
        -------
        int
            Score of interaction between self and other.
            Returns `0` if `not isinstance(self, type(other))` and
            `not isinstance(other, type(self))`.
        """
        if not isinstance(other, type(self)):
            if isinstance(self, type(other)):
                return other.interact_impl(self, strict)
            return 0
        return self.interact_impl(other, strict)

    def overlap(self, other: "Site", strict: bool = True) -> int:
        """Check overlap between self and other.

        Parameters
        ----------
        other : Site
            The other site.

        Returns
        -------
        int
            Score of overlap between self and other.
            Returns `0` if `not isinstance(self, type(other))` and
            `not isinstance(other, type(self))`.
        """
        if not isinstance(other, type(self)):
            if isinstance(self, type(other)):
                return other.overlap_impl(self, strict)
            return 0
        return self.overlap_impl(other, strict)

    @abstractmethod
    def interact_impl(self: _S, other: _S, strict: bool) -> int:
        """Check interaction between self and other.

        `isinstance(other, type(self))` is guaranteed to be `True`.
        """
        pass

    @abstractmethod
    def overlap_impl(self: _S, other: _S, strict: bool) -> int:
        """Check overlap between self and other.

        `isinstance(other, type(self))` is guaranteed to be `True`.
        """
        return int(D.euclidean(self.center, other.center) <= 1.0)

    @abstractmethod
    def to_chimera_command(self, color: str = None) -> str:
        pass


class PiStacking(Site):
    """Pi-stacking site.

    Created from the coordinates of the aromatic ring in the receptor protein.
    """

    def __init__(
        self,
        center: Vector,
        plane: Vector,
        radius: float,
        idxs: List[int] = None,
        residue: int = None,
    ):
        super().__init__(center, idxs=idxs)
        self.plane: Vector = plane
        self.radius: float = radius
        self.residue: Optional[int] = residue

    def __getstate__(self) -> SerializableDic:
        state = super().__getstate__() | {
            "plane": self.plane.tolist(),
            "radius": self.radius,
        }
        if self.residue is not None:
            state["residue"] = self.residue
        return state

    def __setstate__(self, state: SerializableDic) -> None:
        super().__setstate__(state)
        self.plane = np.array(state["plane"])
        self.radius = state["radius"]  # type: ignore
        self.residue = state.get("residue", None)  # type: ignore

    @classmethod
    def cluster(
        cls, sites: List["PiStacking"], cutoff: float = 2.5
    ) -> List[Tuple["PiStacking", int]]:
        """
        Cluster each type of PiStacking sites hierarchically by their
        center coordinates and normal vectors.

        Parameters
        ----------
        sites : List[PiStacking]
            PiStacking sites to cluster.

        Returns
        -------
        List[Tuple[PiStacking, int]]
            List of PiStacking sites and their occurences.
        """
        if len(sites) < 2:
            return [(site, 1) for site in sites]

        cntrs = np.stack([site.center for site in sites])
        cntr_clust = _cluster_sites_by(sites, cntrs, cutoff)
        _logger.info(
            "Clustered %d pi-stacking site(s) by center coordinates.",
            len(cntr_clust),
        )
        _logger.debug(cntr_clust)

        # Align directions of normal vectors
        normals = [site.plane for site in sites]
        for site, aligned in zip(sites, geom.align_parity(normals)):
            site.plane = aligned

        cos_cutoff = 1 - _cos_15
        vect_clust = _cluster_clusters_by(
            cntr_clust,
            lambda s: s.plane,
            cos_cutoff,
            method="complete",
            metric="cosine",
        )

        result = []
        for sites in vect_clust:
            cntr = np.mean([site.center for site in sites], axis=0)
            plane = np.mean([site.plane for site in sites], axis=0)
            radius = np.mean([site.radius for site in sites])
            result.append((cls(cntr, plane, radius), len(sites)))
        return result

    @staticmethod
    def evaluate(
        reference: "PiStacking", query: "PiStacking", strict: bool = True
    ) -> int:
        return reference.interact(query, strict=strict)

    @classmethod
    def from_coords(cls, coords: Points, idxs: List[int], residue: int = None):
        center = coords.mean(axis=0)
        n, _ = geom.fit_plane(coords)
        radius = float(np.linalg.norm(coords[0] - center))
        return cls(center, n, radius, idxs=idxs, residue=residue)

    @classmethod
    def from_mol(cls, mol: Mol, **_):
        """Find all aromatic rings in the molecule.
        Only 5-8 membered rings will be considered as aromatic.
        Use the `rings_planar` attribute to get all planar rings.

        Returns
        -------
        List[List[int]]:
            List of aromatic rings, each ring is a list of atom indices.
        """
        return [
            cls.from_coords(mol.coords[ring], ring) for ring in mol.aromatics
        ]

    def interact_impl(self, other: "PiStacking", strict: bool) -> int:
        cntr_diff = other.center - self.center

        # cos(angle) = -cos(180 - angle)
        cos_angle = abs(self.plane.dot(other.plane))
        if cos_angle >= _cos_45:
            return self._interact_displaced(other, cntr_diff, strict)

        return self._interact_tshaped(other, cntr_diff, strict)

    def _interact_displaced(
        self, other: "PiStacking", cntr_diff: np.ndarray, strict: bool
    ) -> int:
        def _check_offset(this: "PiStacking", offset):
            return this.radius - 1.5 <= offset < this.radius + 1.5

        cntr_diff_proj = self.plane.dot(cntr_diff) * self.plane
        cntr_distance = np.linalg.norm(cntr_diff_proj)
        interact = cntr_distance <= 5.0
        if strict and interact:
            cntr_diff_plane = cntr_diff - cntr_diff_proj
            cntr_offset = np.linalg.norm(cntr_diff_plane)
            interact = _check_offset(self, cntr_offset) or _check_offset(
                other, cntr_offset
            )
        return int(interact)

    def _interact_tshaped(
        self, other: "PiStacking", cntr_diff: np.ndarray, strict: bool
    ) -> int:
        diff_normalized = geom.normalized(cntr_diff)
        this_cos_angle = abs(self.plane.dot(diff_normalized))
        other_cos_angle = abs(other.plane.dot(diff_normalized))
        return (
            _interact_tshaped(self, cntr_diff, strict)
            if this_cos_angle >= other_cos_angle
            else _interact_tshaped(other, cntr_diff, strict)
        )

    def overlap_impl(self, other: "PiStacking", strict: bool):
        return int(
            super().overlap_impl(other, strict)
            and (abs(self.plane.dot(other.plane)) >= _cos_30)
        )

    def to_chimera_command(self, color: str = None):
        rotation = geom.solve_rotation(np.array([0, 0, 1]), self.plane)
        cmd = (
            f"shape cylinder radius {self.radius:.3f} height 1.0 caps true "
            f"qrotation {','.join(f'{x:.3f}' for x in rotation.as_quat())} "
            f"center {','.join(f'{x:.3f}' for x in self.center)} mesh true"
        )
        if color is not None:
            cmd += f" color {color}"
        return cmd


class Hydrophobic(Site):
    """Hydrophobic site.

    Created from the center coordinates and the radius of the hydrophobic core.
    Based on the following paper:
         J. Chem. Inf. Comput. Sci. 1994, 34, 1297—1308.
         doi: https://doi.org/10.1021/ci00022a012
    """

    _cutoff = 1.0

    def __init__(
        self, center: np.ndarray, radius: float, idxs: List[int] = None
    ):
        super().__init__(center, idxs=idxs)
        self.radius: float = radius

    def __getstate__(self) -> SerializableDic:
        return super().__getstate__() | {
            "radius": self.radius,
        }

    def __setstate__(self, state: SerializableDic) -> None:
        super().__setstate__(state)
        self.radius = state["radius"]  # type: ignore

    @classmethod
    def cluster(
        cls, sites: List["Hydrophobic"], cutoff: float = 2.5
    ) -> List[Tuple["Hydrophobic", int]]:
        """
        Cluster each type of hydrophobic sites hierarchically by their
        center coordinates.

        Parameters
        ----------
        sites : List[Hydrophobic]
            Hydrophobic sites to cluster.

        Returns
        -------
        List[Tuple[Hydrophobic, int]]
            List of hydrophobic sites and their occurences.
        """
        if len(sites) < 2:
            return [(site, 1) for site in sites]

        cntrs = np.stack([site.center for site in sites])
        clustered = _cluster_sites_by(sites, cntrs, cutoff)

        result = []
        for sites in clustered:
            cntrs = np.stack([site.center for site in sites])
            radius = np.array([site.radius for site in sites])

            center = np.average(cntrs, axis=0, weights=radius)
            result.append((cls(center, radius.mean()), len(sites)))
        return result

    @staticmethod
    def evaluate(
        reference: "Hydrophobic", query: "Hydrophobic", strict: bool = True
    ) -> int:
        return reference.overlap(query, strict=strict)

    @classmethod
    def from_coords(
        cls,
        coords: np.ndarray,
        idxs: List[int],
        weights: np.ndarray = None,
        radius_weights: np.ndarray = None,
    ):
        """Create hydrophobic interaction from the coordinates of the core.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the core. It may be coordinates of the receptor
            or the ligand. Shape: (N, 3).

        idxs : list[int]
            Zero-based atom indices of the core.

        weights : np.ndarray, optional
            The weights of the atoms in the core. The default is 1.
            Shape: (N,).

        radius_weights : np.ndarray, optional
            The weights of the atoms in the core, used for calculating radius.
            The default is 1. Shape: (N,).

        ligand : bool, optional
            Whether the provided coordinates are coordinates of the ligands.

        Returns
        -------
        Hydrophobic
            An instance representing hydrophobic core.
        """
        center = np.average(coords, axis=0, weights=weights)
        distances = np.linalg.norm(coords - center[np.newaxis, :], axis=1)

        # Exclude points with weights == 0
        if weights is not None:
            radius = np.mean(distances[weights > 0])
        else:
            radius = np.mean(distances)

        # Shrink radius
        if radius_weights is not None:
            radius *= np.mean(radius_weights)

        return cls(center.squeeze(), float(radius), idxs=idxs)

    @classmethod
    def from_mol(
        cls, mol: Mol, min_cnt: int = 3, min_radius: float = 0.5, **_
    ):
        def subst_atomcnt_gt(node: int, visited: FrozenSet[int], cutoff: int):
            def subst_atomcnt_gt_impl(node: int) -> bool:
                nonlocal cutoff
                if cutoff < 0:
                    return True

                n: int
                for n in mol.graph[node]:
                    if mol.atoms[n].atomicnum != 1 and n not in visited_curr:
                        visited_curr.add(n)
                        cutoff -= 1
                        if subst_atomcnt_gt_impl(n):
                            return True

                return False

            visited_curr = set(visited)
            return subst_atomcnt_gt_impl(node)

        def include_substs(
            begin: int, exclude: FrozenSet[int], maxdepth=2
        ) -> Set[int]:
            small_substs = set()
            if maxdepth <= 0:
                return small_substs

            for n in mol.graph[begin]:
                if (
                    n not in exclude
                    and mol.atoms[n].atomicnum != 1
                    and h[n] > 0
                ):
                    small_substs.add(n)
                    small_substs |= include_substs(
                        n, exclude | small_substs, maxdepth=maxdepth - 1
                    )
            return small_substs

        candidates = {
            i for i, atom in enumerate(mol.atoms) if atom.atomicnum != 1
        }

        hydrophobics: List[List[int]] = []
        t = mol.tdhf()
        s = mol.sasa()
        h = t * s
        maybe_hydrophobic = set(np.flatnonzero(h > 0))

        hydrophobic_smallsubst = maybe_hydrophobic.copy()
        for ring in mol.rings:
            if len(ring) < 8:
                ring_set = frozenset(ring)
                smallsubst_hydrophobic_curr = set()
                small_substs = set()
                for atom in ring_set:
                    if subst_atomcnt_gt(atom, ring_set, 2):
                        try:
                            hydrophobic_smallsubst.remove(atom)
                        except KeyError:
                            pass
                    else:
                        if atom in hydrophobic_smallsubst:
                            smallsubst_hydrophobic_curr.add(atom)
                        small_substs |= include_substs(atom, ring_set)

                # Include small substituents in the ring
                ring_smallsubst_set = ring_set | small_substs
                # Exclude rings for further considerations
                candidates -= ring_smallsubst_set

                # Skip for not hydrophobic rings + substituents
                ring_smallsubst = list(ring_smallsubst_set)
                h_ring_subst_sum = (
                    h[ring_smallsubst].sum() * len(ring) / len(ring_smallsubst)
                )
                if h_ring_subst_sum < mol.h_min:
                    continue

                substituents: Set[int] = {
                    n
                    for i in ring
                    for n in mol.graph[i]
                    if mol.atoms[n].atomicnum != 1
                } - ring_set

                sides = geom.determine_side(
                    mol.coords[ring],
                    mol.coords[list(substituents)],
                    tol=mol.tol,
                )
                if np.all(sides >= 0) or np.all(sides <= 0):
                    hydrophobics.append(ring_smallsubst)
                else:
                    is_hydrophobic = False
                    for atom in smallsubst_hydrophobic_curr:
                        for n in mol.graph[atom]:
                            if n in smallsubst_hydrophobic_curr:
                                hydrophobics.append(ring_smallsubst)
                                is_hydrophobic = True
                                break
                        if is_hydrophobic:
                            break

        candidates &= maybe_hydrophobic
        subg = mol.graph.subgraph(candidates)
        hydrophobics.extend(map(list, nx.connected_components(subg)))

        # Try to merge single-atom hydrophobic groups
        large, small = topo.try_merge_small(mol.graph, hydrophobics, min_cnt=2)
        if min_cnt > 1:
            large = [group for group in large if len(group) >= min_cnt]
        else:
            large += small

        ret: List["Hydrophobic"] = []
        for core in large:
            site = cls.from_coords(
                mol.coords[core], core, weights=h[core], radius_weights=t[core]
            )
            if site.radius > min_radius:
                ret.append(site)
        return ret

    def to_chimera_command(self, color: str = None):
        cmd = (
            f"shape sphere radius {self.radius} center "
            f"{','.join(f'{x:.3f}' for x in self.center)} mesh true"
        )
        if color is not None:
            cmd += f" color {color}"
        return cmd

    def interact_impl(self, other: "Hydrophobic", strict: bool):
        distance = D.euclidean(self.center, other.center)
        allowed = self.radius + other.radius
        if strict:
            allowed += self._cutoff
        return int(distance <= allowed)

    def overlap_impl(self, other: "Hydrophobic", strict: bool):
        distance = D.euclidean(self.center, other.center)
        if strict:
            return int(
                distance <= self._cutoff
                or distance <= self.radius + other.radius
            )

        return int(distance <= self.radius + other.radius + self._cutoff)


class HBCategory(IntEnum):
    PSI_PHI = 0
    NU_TAU = 1
    THETA_TAU = 2
    WATER = 3


class HydrogenBonding(Site):
    """Base class for hydrogen bonding sites.

    Hydrogen bonding detection is based on the following paper:
        J. Comput. Aided Mol. Des. 1996, 10, 607-622.
        doi: https://doi.org/10.1007/BF00134183
    """

    def __init__(
        self,
        center: Vector,
        reference: Vector,
        category: HBCategory,
        is_donor: bool,
        idxs: List[int] = None,
        *,
        _normal: Optional[Vector] = None,
    ):
        super().__init__(center, idxs)
        self.reference: Vector = reference
        self.is_donor: bool = is_donor
        self.category: HBCategory = category
        self._normal: Optional[Vector] = _normal

    def __getstate__(self) -> SerializableDic:
        return super().__getstate__() | {
            "reference": utils.tolist_optional(self.reference),
            "is_donor": self.is_donor,
            "category": int(self.category),
            "_normal": utils.tolist_optional(self._normal),
        }

    def __setstate__(self, state: SerializableDic):
        super().__setstate__(state)
        self.reference = np.array(state["reference"])
        self.is_donor = state["is_donor"]  # type: ignore
        self.category = HBCategory(state["category"])
        self._normal = utils.toarray_optional(state["_normal"])  # type: ignore

    @classmethod
    def from_mol(cls, mol: Mol, **kwargs):
        hbds: List[cls] = []

        for i, atom in enumerate(mol.atoms):
            is_donor = atom.OBAtom.IsHbondDonor()
            is_acceptor = atom.OBAtom.IsHbondAcceptor()
            if not (is_donor or is_acceptor):
                continue

            hyb = atom.hyb
            center = mol.coords[i]
            neighbors = list(mol.graph[i])
            # Altlocs, ...
            if not neighbors:
                continue

            idxs = [i] + neighbors
            heavy = [nei for nei in neighbors if mol.atoms[nei].atomicnum != 1]
            n_heavy = len(heavy)

            if is_acceptor:
                n_neigh = len(neighbors)
                neighbor_coords: Points = mol.coords[neighbors]

                reference = neighbor_coords.mean(axis=0)

                # psi-phi or theta-tau
                if n_neigh == 2 or hyb == 2:
                    category = HBCategory.PSI_PHI

                    inplane_neighbors = set(idxs)
                    if len(inplane_neighbors) < 3:
                        # need at least 3 atoms to determine the plane
                        for j in topo.k_neighbors(mol.graph, i, 2):
                            btom = mol.atoms[j]
                            if btom.atomicnum == 1 or btom.hyb < 3:
                                inplane_neighbors.add(j)
                    _normal = geom.fit_plane(
                        mol.coords[list(inplane_neighbors)]
                    )[0]
                else:
                    category = (
                        HBCategory.THETA_TAU if n_heavy else HBCategory.WATER
                    )
                    _normal = None

                hbds.append(
                    cls(
                        center,
                        reference,
                        category,
                        not is_acceptor,
                        idxs,
                        _normal=_normal,
                    )
                )

            if is_donor:
                heavy_coords: Points = mol.coords[heavy]

                if n_heavy > 1 or any(
                    mol.graph.edges[i, h]["order"] > 1 for h in heavy
                ):
                    # theata-tau
                    category = HBCategory.THETA_TAU
                    reference = center

                    hydrogens = [
                        h for h in mol.graph[i] if mol.atoms[h].atomicnum == 1
                    ]
                    hydrogen_coords = mol.coords[hydrogens]
                    hbds.extend(
                        cls(center, reference, category, is_donor, idxs)
                        for center in hydrogen_coords
                    )
                    continue

                if n_heavy == 0:
                    # water molecule, no angle restrictions
                    category = HBCategory.WATER
                    reference = None  # type: ignore
                else:
                    # terminal & rotable, nu-tau
                    category = HBCategory.NU_TAU
                    reference: Vector = heavy_coords[0]
                hbds.append(cls(center, reference, category, is_donor, idxs))
        return hbds

    @classmethod
    def cluster(
        cls, sites: List["HydrogenBonding"], cutoff: float = 2.5
    ) -> List[Tuple["HydrogenBonding", int]]:
        if len(sites) < 2:
            return [(site, 1) for site in sites]

        result = []
        classified = defaultdict(list)
        for site in sites:
            classified[(site.category, site.is_donor)].append(site)

        for key, sites in classified.items():
            if len(sites) < 2:
                result.append((sites[0], 1))
                continue
            result += cls._cluster_same_atoms(sites, *key, cutoff)

        return result

    @classmethod
    def _cluster_same_atoms(
        cls,
        sites: List["HydrogenBonding"],
        category: HBCategory,
        is_donor: bool,
        cutoff: float,
    ) -> List[Tuple["HydrogenBonding", int]]:
        cntrs = np.stack([site.center for site in sites])
        cntr_clust = _cluster_sites_by(sites, cntrs, cutoff)
        _logger.info(
            "Clustered %d hbond site(s) by center coordinates.",
            len(cntr_clust),
        )
        _logger.debug(cntr_clust)

        if category == HBCategory.WATER:
            return [(cluster[0], len(cluster)) for cluster in cntr_clust]

        cos_cutoff = 1 - _cos_15
        if is_donor and category == HBCategory.THETA_TAU:
            clustered = _cluster_clusters_by(
                cntr_clust, lambda s: s.reference, cutoff
            )

            def _mean_ref(cluster: List["HydrogenBonding"], center) -> Vector:
                return np.mean([s.reference for s in cluster], axis=0)
        else:
            clustered = _cluster_clusters_by(
                cntr_clust,
                lambda s: s.reference - s.center,
                cos_cutoff,
                method="complete",
                metric="cosine",
            )

            def _mean_ref(cluster: List["HydrogenBonding"], center) -> Vector:
                return (
                    geom.normalized(
                        np.stack(
                            [site.reference - site.center for site in cluster]
                        ),
                        axis=1,
                    ).mean(axis=0)
                    + center
                )

        if category == HBCategory.PSI_PHI:
            clustered = _cluster_clusters_by(
                clustered,
                lambda s: s._normal,
                cos_cutoff,  # type: ignore
                method="complete",
                metric="cosine",
            )
            normals = geom.normalized(
                np.stack(
                    [
                        np.mean(
                            [s._normal for s in cluster],  # type: ignore
                            axis=0,
                        )
                        for cluster in clustered
                    ]
                ),
                axis=1,
            )
        else:
            normals = [None] * len(clustered)

        result = []
        for cluster, normal in zip(clustered, normals):
            center = np.mean([site.center for site in cluster], axis=0)
            reference = _mean_ref(cluster, center)
            result.append(
                (
                    cls(center, reference, category, is_donor, _normal=normal),
                    len(cluster),
                )
            )
        return result

    @staticmethod
    def evaluate(
        reference: "HydrogenBonding",
        query: "HydrogenBonding",
        strict: bool = True,
    ) -> int:
        return reference.interact(query, strict=strict)

    def interact_impl(self, other: "HydrogenBonding", strict: bool) -> int:
        if self.is_donor == other.is_donor:
            return 0

        if self.is_donor:
            return _hbond_interact(self, other, strict)
        else:
            return _hbond_interact(other, self, strict)

    def overlap_impl(self, other: "HydrogenBonding", strict: bool) -> int:
        if not (
            self.is_donor == other.is_donor
            and self.category == other.category
            and super().overlap_impl(other, strict)
        ):
            return 0

        if not strict or self.category == HBCategory.WATER:
            return 1

        if self.is_donor and self.category == HBCategory.THETA_TAU:
            ref_close = D.sqeuclidean(self.reference, other.reference) < 0.25
        else:
            ref_close = (
                D.cosine(
                    self.reference - self.center,
                    other.reference - other.center,
                )
                < 1 - _cos_15
            )

        if not ref_close:
            return 0

        if self.category != HBCategory.PSI_PHI:
            return 1

        normal_dist = D.cosine(self._normal, other._normal)  # type: ignore
        # Consider flipped normal vectors
        normal_dist = min(normal_dist, 2 - normal_dist)
        return int(normal_dist < 1 - _cos_15)

    def to_chimera_command(self, color: str = None) -> str:
        if not self.idxs:
            return ""

        sel_arg = " or ".join(f"serialNumber={i + 1}" for i in self.idxs)
        cmd = f"sel @/{sel_arg}; di sel; color {color or 'red'} sel"
        return cmd


class Charged(Site):
    def __init__(
        self,
        center: Vector,
        charge: int,
        idxs: Optional[List[int]] = None,
    ):
        super().__init__(center, idxs)
        self.charge: int = charge

    def __getstate__(self) -> SerializableDic:
        return super().__getstate__() | {
            "charge": self.charge,
        }

    def __setstate__(self, state: SerializableDic):
        super().__setstate__(state)
        self.charge = state["charge"]  # type: ignore

    @classmethod
    def from_mol(
        cls,
        mol: Mol,
        protein: bool = False,
        **kwargs,
    ):
        if protein:
            return _protein_charged_residues(mol)
        return _ligand_charged_sites(mol)

    @classmethod
    def cluster(
        cls,
        sites: List["Charged"],
        cutoff: float = 2.5,
    ) -> List[Tuple["Charged", int]]:
        if len(sites) < 2:
            return [(site, 1) for site in sites]

        classified: Dict[int, List["Charged"]] = defaultdict(list)
        for site in sites:
            classified[site.charge].append(site)

        result = []
        for charge, sites in classified.items():
            cntrs = np.stack([site.center for site in sites])
            clustered = _cluster_sites_by(sites, cntrs, cutoff)

            for sites in clustered:
                cntrs = np.stack([site.center for site in sites])
                result.append((cls(cntrs.mean(axis=0), charge), len(sites)))
        return result

    @staticmethod
    def evaluate(
        reference: "Charged",
        query: "Charged",
        strict: bool = True,
    ) -> int:
        return reference.interact(query, strict=strict)

    def interact_impl(self, other: "Charged", strict: bool) -> int:
        if self.charge * other.charge > 0:
            return 0

        dist = D.euclidean(self.center, other.center)
        return int(1.5 <= dist <= 5.6)

    def overlap_impl(self, other: "Charged", strict: bool) -> int:
        distance = D.euclidean(self.center, other.center)
        return int(self.charge * other.charge > 0 and distance <= 2.5)

    def to_chimera_command(self, color: str = "") -> str:
        assert not color, (
            "Color is determined by charge, cannot be set manually."
        )

        cmd = (
            "shape sphere center "
            f"{','.join(f'{x:.3f}' for x in self.center)} mesh true"
        )
        if self.charge > 0:
            cmd += " color red"
        else:
            cmd += " color blue"
        return cmd


def _cluster_sites_by(
    sites: List[_S],
    y: np.ndarray,
    cutoff: float,
    method: str = "ward",
    metric="euclidean",
    criterion: str = "distance",
) -> List[List[_S]]:
    lnk = hier.linkage(y, method=method, metric=metric)
    fcl = hier.fcluster(lnk, cutoff, criterion=criterion)

    clustered = [[] for _ in range(fcl.max())]
    for i, site in zip(fcl, sites):
        clustered[i - 1].append(site)
    return clustered


def _cluster_clusters_by(
    clusters: List[List[_S]],
    y_from: Callable[[_S], np.ndarray],
    cutoff: float,
    **kwargs,
) -> List[List[_S]]:
    result = []
    for cluster in clusters:
        if len(cluster) < 2:
            result.append(cluster)
            continue

        y = np.stack([y_from(site) for site in cluster])
        result += _cluster_sites_by(cluster, y, cutoff, **kwargs)
    return result


def _interact_tshaped(
    base: PiStacking, cntr_diff: np.ndarray, strict: bool
) -> int:
    cntr_distance = np.linalg.norm(cntr_diff)
    interact = cntr_distance <= 8.0
    if strict and interact:
        cntr_diff_perpendicular = base.plane.dot(cntr_diff) * base.plane
        cntr_offset = np.linalg.norm(cntr_diff - cntr_diff_perpendicular)
        interact = cntr_offset < base.radius + 1.0
    return int(interact)


def _hbond_nu_tau_angle_check(dn_vec: Vector, da_vec: Vector):
    # 90deg < ∠NDA < 150deg
    return _cos_90 >= 1 - D.cosine(dn_vec, da_vec) >= _cos_150


def _hbond_don_theta_tau_angle_check(hd_vec: Vector, ha_vec: Vector):
    # ∠DHA > 135deg
    return D.cosine(hd_vec, ha_vec) >= 1 - _cos_135


def _hbond_psi_phi_angle_check(ar_vec: Vector, ad_vec: Vector, normal: Vector):
    # phi >= 90deg
    ad_vec_proj = geom.project_vecs_onto(
        ad_vec[np.newaxis, :], normal
    ).squeeze()
    return D.cosine(ar_vec, ad_vec_proj) >= 1 - _cos_90


def _hbond_acc_theta_tau_angle_check(ar_vec: Vector, ad_vec: Vector, _):
    # nu >= 90deg
    return D.cosine(ar_vec, ad_vec) >= 1 - _cos_90


def _factory_warn_invalid(kind: str):
    def warn_invalid(*_):
        _logger.debug(f"Invalid category for {kind}")
        return False

    return warn_invalid


_don_angle_checkers = (
    _factory_warn_invalid("donor"),
    _hbond_nu_tau_angle_check,
    _hbond_don_theta_tau_angle_check,
)

_acc_angle_checkers = (
    _hbond_psi_phi_angle_check,
    _factory_warn_invalid("acceptor"),
    _hbond_acc_theta_tau_angle_check,
)


def _hbond_interact(
    don: HydrogenBonding, acc: HydrogenBonding, strict: bool
) -> int:
    # H: donor hydrogen, D: donor, A: acceptor
    if don.category == HBCategory.THETA_TAU:
        distsq = 2.5 * 2.5  # H - A distance
        ad_vec = don.reference - acc.center
    else:
        distsq = 3.5 * 3.5  # D - A distance
        ad_vec = don.center - acc.center

    # Check distance
    is_close = D.sqeuclidean(don.center, acc.center) <= distsq
    if not (strict and is_close):
        return int(is_close)

    # Donor geometry
    if don.category != HBCategory.WATER:
        don_cr_vec = don.reference - don.center
        dc_ac_vec = acc.center - don.center
        don_angle_ok = _don_angle_checkers[don.category](don_cr_vec, dc_ac_vec)
        if not don_angle_ok:
            return 0

    # Acceptor geometry
    if acc.category != HBCategory.WATER:
        acc_cr_vec = acc.reference - acc.center
        acc_angle_ok = _acc_angle_checkers[acc.category](
            acc_cr_vec, ad_vec, acc._normal
        )
        return int(acc_angle_ok)

    return 1


_charged_sidechain = {
    "ARG": (+1, frozenset({"NE", "CZ", "NH1", "NH2"})),
    "HIS": (+1, frozenset({"CG", "ND1", "CD2", "CE1", "NE2"})),
    "LYS": (+1, frozenset({"NZ"})),
    "ASP": (-1, frozenset({"CG", "OD1", "OD2"})),
    "GLU": (-1, frozenset({"CD", "OE1", "OE2"})),
}


def _protein_charged_residues(mol: Mol) -> List[Charged]:
    """Find charged sidechain centers for protein residues.

    Positive: ARG (guanidinium), HIS (imidazole), LYS (amine)
    Negative: ASP (carboxylate), GLU (carboxylate)

    Parameters
    ----------
    mol : Mol
        Protein molecule with residue information.

    Returns
    -------
    List[Charged]
        Charged sites, one per charged residue.
    """
    charged: List[Charged] = []

    for res_idx in range(mol.obmol.NumResidues()):
        residue: ob.OBResidue = mol.obmol.GetResidue(res_idx)
        resname = residue.GetName().strip()

        entry = _charged_sidechain.get(resname)
        if entry is None:
            continue

        charge, target_names = entry
        idxs: List[int] = []
        for atom in ob.OBResidueAtomIter(residue):
            if residue.GetAtomID(atom).strip() in target_names:
                idxs.append(atom.GetIdx() - 1)

        if not idxs:
            continue

        center = mol.coords[idxs].mean(axis=0)
        charged.append(Charged(center, charge, idxs=idxs))

    return charged


# SMARTS patterns for ligand ionizable groups.
# Each entry: (SMARTS, center_atom_indices_in_match or empty for all).
# Patterns within each list are ordered by specificity (most specific first);
# atoms claimed by earlier matches are excluded from later ones.
_ChargedPattern = Tuple[str, List[int]]

# Some patterns taken from openbabel repository
# https://github.com/openbabel/openbabel/blob/889c350feb179b43aa43985799910149d4eaa2bc/data/SMARTS_InteLigand.txt
_positive_smarts: List[_ChargedPattern] = [
    # Guanidine
    ("[N;v3X3,v4X4+][CX3](=[N;v3X2,v4X3+])[N;v3X3,v4X4+]", []),
    # Amidine
    ("[NX3;!$(NC=[O,S])][CX3;$([CH]),$([C][#6])]=[NX2;!$(NC=[O,S])]", []),
    # Amine
    ("[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]", []),
    # Imidazole, etc.
    ("[cX3]1[nX2&H0,nX3+&H1][cX3][nX3][cX3]1", []),
    # Pyridine, etc.
    ("[nX2,nX3+&H1]", []),
    # Generic positive charge not adjacent to a negative charge
    ("[+1,+2,+3;!$([+1,+2,+3]~[-1,-2,-3])]", []),
]

_negative_smarts: List[_ChargedPattern] = [
    # Tetrazole: 5-membered ring with 4 N and 1 C
    ("[#7]1~[#7]~[#7]~[#7]~[#6]1", []),
    # Trifluoromethyl sulfonamide: center at NH
    ("[$([NX-]),$([NX2;H1,H2])]S(=O)(=O)C(F)(F)F", [0]),
    # Sulfonic acid / sulfonate
    ("[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[$([OX2H]),$([OX1-])]", []),
    # Phosphonic acid / phosphonate
    (
        "[PX4;$([H1]),$([H0][#6])](=[OX1])([$([OX2H]),$([OX1-])])[$([OX2H]),$([OX1-])]",
        [],
    ),
    # Sulfinic acid / sulfinate
    ("[SX3;$([H1]),$([H0][#6])](=[OX1])[$([OX2H]),$([OX1-])]", []),
    # Carboxylic acid / carboxylate
    ("[CX3;$([R0][#6]),$([H1R0])](=[OX1])[$([OX2H]),$([OX1-])]", []),
    # Phosphinic acid / phosphinate
    (
        "[PX4;$([H2]),$([H1][#6]),$([H0]([#6])[#6])](=[OX1])[$([OX2H]),$([OX1-])]",
        [],
    ),
    # Generic negative charge not adjacent to a positive charge
    ("[-1,-2,-3;!$([-1,-2,-3]~[+1,+2,+3])]", []),
]


@functools.cache
def _compile_smarts_pattern(pattern: str) -> ob.OBSmartsPattern:
    sp = ob.OBSmartsPattern()
    if not sp.Init(pattern):
        raise ValueError(f"Invalid SMARTS pattern: {pattern}")
    return sp


def _match_smarts_patterns(
    mol: Mol,
    patterns: List[_ChargedPattern],
    claimed: Set[int],
    default_charge: int,
):
    seen: Set[FrozenSet[int]] = set()

    for smarts, center_idxs in patterns:
        sp = _compile_smarts_pattern(smarts)
        sp.Match(mol.obmol)

        for match in sp.GetUMapList():
            idxs: List[int] = [idx - 1 for idx in match]
            idxs = [idxs[i] for i in center_idxs] or idxs

            key = frozenset(idxs)
            if key in seen or claimed.issuperset(idxs):
                continue

            seen.add(key)
            claimed.update(idxs)

            center = mol.coords[idxs].mean(axis=0)
            charge = (
                sum(mol.atoms[i].formalcharge for i in idxs) or default_charge
            )
            yield Charged(center, charge, idxs=idxs)


def _ligand_charged_sites(mol: Mol) -> List[Charged]:
    """Detect ionizable groups in a small-molecule ligand.

    Positive ionizable: basic amines, amidines, guanidines, and formal
    positive charges not adjacent to a negative charge.

    Negative ionizable: trifluoromethyl sulfonamides, sulfonic/sulfinic/
    carboxylic/phosphonic/phosphinic acids, tetrazoles, and formal negative
    charges not adjacent to a positive charge.

    Parameters
    ----------
    mol : Mol
        Small-molecule ligand.

    Returns
    -------
    List[Charged]
        Charged sites for all detected ionizable groups.
    """
    claimed: Set[int] = set()
    sites = [
        *_match_smarts_patterns(mol, _positive_smarts, claimed, +1),
        *_match_smarts_patterns(mol, _negative_smarts, claimed, -1),
    ]
    return sites
