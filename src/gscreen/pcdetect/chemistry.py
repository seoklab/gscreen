import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx
import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel
from scipy.spatial import distance as D

from .. import io
from .geometry import fibonacci_sphere, fit_plane, inplane
from .topology import k_neighbors, kth_neighbors

__all__ = ["Mol"]

_logger = logging.getLogger(__name__)

# vdW radii of elements.
# Values taken from https://en.wikipedia.org/wiki/Van_der_Waals_radius.
vdw_radii = {
    1: 1.10,
    5: 1.92,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    14: 2.10,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}

# Approximate vdW radius of water.
# Value taken from https://en.wikipedia.org/wiki/Accessible_surface_area.
solvent_radius = 1.4


def get_vdwr(atoms: Iterable[int], default: float = 2.0):
    return np.array([vdw_radii.get(atom, default) for atom in atoms])


class Mol:
    def __init__(
        self,
        obmol: ob.OBMol,
        atoms: List[pybel.Atom],
        graph: nx.Graph,
        coords: np.ndarray,
        rings: List[List[int]],
        rings_planar: List[List[int]],
        aromatics: List[List[int]],
        tol: float = 0.1,
    ):
        super().__init__()
        self.obmol = obmol
        self.atoms = atoms

        self.graph = graph
        self.coords = coords

        self.rings = rings
        self.rings_planar = rings_planar
        self.aromatics = aromatics

        self.tol = tol
        self._normals = _fix_hyb(self)

    @classmethod
    def from_pybel(cls, mol: pybel.Molecule, tol: float = 0.1):
        atoms = mol.atoms
        graph = graph_from_pybel(mol)
        coords = np.array([atom.coords for atom in atoms], dtype=float)

        rings = nx.cycle_basis(graph)
        rings_planar = planar_rings(graph, coords, rings, tol)
        rings_planar_set = set(map(tuple, rings_planar))
        aromatics = []
        for ring in rings:
            if all(atoms[atom].OBAtom.IsAromatic() for atom in ring) or (
                tuple(ring) in rings_planar_set and 5 <= len(ring) <= 6
            ):
                aromatics.append(ring)

        return cls(
            mol.OBMol,
            atoms,
            graph,
            coords,
            rings,
            rings_planar,
            aromatics,
            tol=tol,
        )

    def to_pybel(self):
        return pybel.Molecule(self.obmol)

    @classmethod
    def load(cls, path: Path, fmt: Optional[str] = None, tol=0.1):
        ob.obErrorLog.StopLogging()
        _logger.info("Loading molecule from %s", path)

        if fmt is None:
            fmt = path.suffix.lower().removeprefix(".")

        for mol in pybel.readfile(fmt, str(path)):
            _logger.debug(f"{mol = }")
            yield cls.from_pybel(mol, tol=tol)

    @classmethod
    def loadf(cls, data: io.Format, tol: float = 0.1):
        return cls.from_pybel(
            pybel.readstring(data.format_name, str(data)), tol=tol
        )

    @classmethod
    @property
    def h_min(cls) -> float:
        try:
            return cls._h_min
        except AttributeError:
            ethane = pybel.readstring(
                "mol2",
                """\
@<TRIPOS>MOLECULE
ethane
8 7 1 0 0
SMALL
AMBER ff14SB


@<TRIPOS>ATOM
      1 C          -0.0144    1.0833    0.0048 C.3       1 UNK   -0.0682
      2 C          -0.7277    1.5785    1.2710 C.3       1 UNK   -0.0682
      3 HC1        -0.3132    2.5425    1.5902 H         1 UNK    0.0227
      4 HC2        -1.7989    1.7091    1.0807 H         1 UNK    0.0227
      5 HC3        -0.6056    0.8614    2.0901 H         1 UNK    0.0227
      6 HC4        -0.1452    1.8026   -0.8130 H         1 UNK    0.0227
      7 HC5         1.0588    0.9644    0.1916 H         1 UNK    0.0227
      8 HC6        -0.4199    0.1176   -0.3160 H         1 UNK    0.0227
@<TRIPOS>BOND
     1    1    2 1
     2    3    2 1
     3    4    2 1
     4    5    2 1
     5    6    1 1
     6    7    1 1
     7    8    1 1
""",
            )
            ethane = cls.from_pybel(ethane)
            cls._h_min = (ethane.sasa() * ethane.tdhf())[0] / 2
            return cls._h_min

    def sasa(self, ignore_noh=True, exclude_hydrogen=True, samples=250):
        """Calculate solvent accessible surface area of each atom, based on
        evenly sampled points on the van der Waals surface.

        Parameters
        ----------
        ignore_noh : bool, optional
            Whether to ignore N, O, H atoms.

        samples : int, optional
            Number of evenly sampled points on the surface.
            N = 250 will be a good compromise. (max error < 1.0, rmse ~ 0.2).

        Returns
        ------
        np.ndarray:
            Solvent accessible surface area for each atom. Shape: (N, )
        """
        # Shape: (N, )
        vdwrs = get_vdwr(atom.atomicnum for atom in self.atoms)
        vdwrs_solv = vdwrs + solvent_radius
        # Shape: (N, samples, 3)
        spheres = fibonacci_sphere(vdwrs_solv, samples=samples)
        # Shape: (N, samples, 3)
        solvent_cntr = self.coords[:, np.newaxis, :] + spheres

        # min value for not very accessible atoms
        f_min = 0.1 / samples
        f = np.zeros(len(self))
        mask = np.ones(len(self), dtype=bool)
        vdwrs_solv = vdwrs_solv[np.newaxis, :]
        if exclude_hydrogen:
            mask[[atom.atomicnum == 1 for atom in self.atoms]] = False

        for i, (solv_peratom, atom) in enumerate(
            zip(solvent_cntr, self.atoms)
        ):
            if ignore_noh and atom.atomicnum in {1, 7, 8}:
                continue

            mask[i] = False
            # Shape: (samples, N - 1)
            dist = D.cdist(solv_peratom, self.coords[mask])
            # Shape: (samples, )
            is_accessible = np.all(dist >= vdwrs_solv[:, mask], axis=1)
            # make accessible surface area > 0
            f[i] = max(np.mean(is_accessible, keepdims=True), f_min)
            if not exclude_hydrogen or atom.atomicnum != 1:
                mask[i] = True

        return 4 * math.pi * np.square(vdwrs) * f

    def tdhf(self):
        """The topology-dependent hydrophobicity factor per atom.
        Algorithm and values taken from:
            J. Chem. Inf. Comput. Sci. 1994, 34, 6, 1297-1308.
            doi: https://doi.org/10.1021/ci00022a012

        Returns
        -------
        np.ndarray:
            Topology-dependent hydrophobicity factor for each atom.
            Shape: (N, )
        """
        tdhf = np.ones(len(self), dtype=float)
        dot6_cnt = np.zeros(len(self), dtype=int)
        no_cnt = np.zeros(len(self), dtype=int)
        conjugated = _find_conjugated(self)

        for i, atom in enumerate(self.atoms):
            atnum = atom.atomicnum
            if atnum == 1:
                # 1) H
                tdhf[i] = 0.0
                continue

            le_two_zero = False
            if atom.formalcharge:
                # 3) <= 2 bonds away from charged atom
                le_two_zero = True

            if atnum in (7, 8):
                # 1) N, O
                tdhf[i] = 0.0
                if i not in conjugated:
                    # 13) 1 neighboring O or N with no delocalized electrons
                    one_away = list(kth_neighbors(self.graph, i))
                    tdhf[one_away] *= 0.25
                    no_cnt[one_away] += 1

                    if not le_two_zero and self._contains_adjof(i):
                        # 4) <= 2 bonds away from OH / NH with no conjugation
                        le_two_zero = True

                if atnum == 8 and any(bo == 2 for bo in self._bond_data_of(i)):
                    # 6) <= 2 bonds away from =O
                    le_two_zero = True

                    # 9) 3 bonds away from =O
                    three_away = list(kth_neighbors(self.graph, i, kth=3))
                    tdhf[three_away] *= 0.6
                    dot6_cnt[three_away] += 1
            elif atnum == 16:
                if self._contains_adjof(i):
                    # 2) S in SH
                    tdhf[i] = 0.0
                    if i not in conjugated:
                        # 5) <= 1 bond away from SH with no conjugation
                        tdhf[list(self.graph[i])] = 0.0

                if sum(self._bond_data_of(i)) > 2:
                    # 7) <= 1 bond away from S with valence > 2
                    tdhf[list(k_neighbors(self.graph, i))] = 0.0
                    # 10) 2 bonds away from S with valence > 2
                    two_away = list(kth_neighbors(self.graph, i, kth=2))
                    tdhf[two_away] *= 0.6
                    dot6_cnt[two_away] += 1

                if any(bo == 2 for bo in self._bond_data_of(i)):
                    # 8) =S
                    tdhf[i] = 0.0
                    # 11) 1 bond away from =S
                    one_away = list(kth_neighbors(self.graph, i))
                    tdhf[one_away] *= 0.6
                    dot6_cnt[one_away] += 1

            if le_two_zero:
                tdhf[list(k_neighbors(self.graph, i, cutoff=2))] = 0.0

        # 12) > 1 category of 9-11 is true
        tdhf[dot6_cnt > 1] = 0.0
        # 14) > 1 neighboring N, O with no conjudation
        tdhf[no_cnt > 1] = 0.0
        return tdhf

    def _contains_adjof(self, atom: int, atomicnum: int = 1):
        n: int
        for n in self.graph[atom]:
            if self.atoms[n].atomicnum == atomicnum:
                return True
        return False

    def _bond_data_of(self, atom: int, data: str = "order") -> Iterable[int]:
        for *_, d in self.graph.edges(atom, data=data):
            yield d

    def __len__(self):
        return len(self.atoms)

    def __repr__(self):
        return (
            "Mol("
            + "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())
            + ")"
        )


def hyb_from_angle(angle: float) -> int:
    angle = math.degrees(abs(angle))
    if angle <= 90:
        # > sp3 hybridization
        return 4
    if angle <= (109.5 + 120) / 2:
        # 90 < angle <= 114.75 degrees -> sp3 hybridization
        return 3
    if angle <= 135:
        # 114.75 < angle <= 135 degrees -> sp2 hybridization
        return 2
    if angle <= 170:
        # 135 < angle <= 170 degrees -> > sp3 hybridization
        return 4
    # 170 < angle  -> sp hybridization
    return 1


def graph_from_pybel(mol: pybel.Molecule):
    bonds = []
    for i in range(mol.OBMol.NumBonds()):
        bond: ob.OBBond = mol.OBMol.GetBond(i)
        bi, ei = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        bonds.append((bi, ei, bond.GetBondOrder()))

    graph = nx.Graph()
    graph.add_nodes_from(range(mol.OBMol.NumAtoms()))
    graph.add_weighted_edges_from(bonds, weight="order")
    return graph


def planar_rings(
    graph: nx.Graph, coords: np.ndarray, rings: List[List[int]], tol: float
):
    planars: List[List[int]] = []
    for ring in rings:
        neighbors = set()
        for atom in ring:
            # Already has more than 3 neighbors, not planar
            if len(graph[atom]) > 3:
                break
            neighbors |= k_neighbors(graph, atom)
        else:
            if inplane(coords[list(neighbors)], tol=tol):
                planars.append(ring)
    return planars


def _fix_hyb(mol: Mol):
    atom_normals = np.zeros_like(mol.coords)
    atom_planar_rings = frozenset(n for r in mol.rings_planar for n in r)

    for i, (atom, hyb) in enumerate(
        zip(mol.atoms, (atom.hyb for atom in mol.atoms))
    ):
        # Neighbors + node itself
        adjacents = k_neighbors(mol.graph, i)
        # Skip terminal or non-planar atoms
        if not (3 <= len(adjacents) <= 4):
            continue

        # Coordinates including atom i
        adj_coords = mol.coords[list(adjacents)]
        normal, d = fit_plane(adj_coords)

        # Fix hybridization
        hyb_fixed = hyb

        if i in atom_planar_rings:
            # Assume sp2 for planar rings
            hyb_fixed = 2
        elif len(adjacents) == 3:
            adj_coords = mol.coords[list(mol.graph[i])]
            bond_vectors = adj_coords - mol.coords[np.newaxis, i]
            bondangle = math.acos(
                1 - D.cosine(bond_vectors[0], bond_vectors[1])
            )
            hyb_fixed = hyb_from_angle(bondangle)
        elif np.allclose(adj_coords @ normal + d, 0, atol=mol.tol):
            # Always sp2 for 3 neighbors (len(adjacents) == 4)
            # and planar atoms
            hyb_fixed = 2

        if hyb_fixed == 2:
            atom_normals[i] = normal
        if hyb_fixed != hyb:
            atom.OBAtom.SetHyb(hyb_fixed)

    return atom_normals


def _find_conjugated(mol):
    conjugated = set()
    atom_neighbors = [k_neighbors(mol.graph, i) for i in range(len(mol))]
    for i, (atom, normal, neighbors) in enumerate(
        zip(mol.atoms, mol._normals, atom_neighbors)
    ):
        # Consider only B, C, N, O, P, S
        # First check for in-plane neighbors, excluding terminal atoms
        if (
            atom.atomicnum not in {5, 6, 7, 8, 15, 16}
            or atom.hyb > 2
            or len(neighbors) < 3
        ):
            continue

        # len(neighbors) == 3 or (len(neighbors) == 4 and inplane)
        for n in mol.graph[i]:
            # Assume conjugation if neighbor is P or S
            neighbor = mol.atoms[n]
            if neighbor.atomicnum in {15, 16}:
                conjugated |= {i, n}
                continue

            # Exclude terminal atoms and hyb > sp2 atoms
            if not (3 <= len(atom_neighbors[n]) <= 4):
                continue

            # arccos(0.99) ~ 8 degrees
            if neighbor.hyb < 3 and (
                abs(np.dot(normal, mol._normals[n])) > 1 - mol.tol
            ):
                conjugated |= {i, n}
                for m in (set(mol.graph[n]) | neighbors) - conjugated:
                    # Consider only "terminals"; will be condidered
                    # in the outer loop otherwise
                    if len(mol.graph[m]) > 1:
                        continue

                    atom_m = mol.atoms[m]
                    # Assume conjugation if neighbor is P or S
                    if atom_m.atomicnum in {15, 16} or (
                        atom_m.atomicnum in {5, 6, 7, 8} and atom_m.hyb < 3
                    ):
                        conjugated.add(m)
                break
    return conjugated
