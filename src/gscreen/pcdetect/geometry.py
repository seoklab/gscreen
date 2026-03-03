import math
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import distance as D
from scipy.spatial.transform import Rotation as R

from ..typing import Points, Vector

phi = math.pi * (3.0 - math.sqrt(5.0))


def normalized(
    vec: np.ndarray, axis: Optional[int] = None, eps: float = 1e-12
) -> np.ndarray:
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + eps)


def plane_from_norm_pt(
    norm: Vector, pt: Vector, normalize: bool = True
) -> Tuple[Vector, float]:
    """Find the plane equation given a normal vector and a point on the plane.

    Shapes: (3, ), (3, ) -> ((3, ), (scalar))
    """
    if normalize:
        norm = normalized(norm)
    return norm, -np.dot(norm, pt)


def project_pts_onto(pts: Points, n: Vector, d: float) -> Points:
    """Project points pts onto plane (n · x + d = 0). n must be normalized.

    Shapes: (N, 3), (3, ), (scalar) -> (N, 3)
    """
    coeffs = pts @ n + d
    return pts - coeffs[:, np.newaxis] * n


def project_vecs_onto(vecs: Points, n: Vector) -> Points:
    """Project vectors vecs onto plane (n · x = 0). n must be normalized.

    Shapes: (N, 3), (3, ) -> (N, 3)
    """
    return project_pts_onto(vecs, n, 0.0)


def solve_rotation(src: np.ndarray, dst: np.ndarray):
    # cos(1 deg) ~ 0.99985
    if D.cosine(src, dst) < 1e-4:
        rot = R.identity()
    else:
        w = np.cross(src, dst)
        A = np.stack([src, w, np.cross(src, w)], axis=1)
        B = np.stack([dst, w, np.cross(dst, w)], axis=1)
        rot = R.from_matrix(B @ np.linalg.inv(A))
    return rot


def align_parity(vectors: List[Vector]) -> List[Vector]:
    if len(vectors) < 2:
        return vectors

    v = vectors[0]
    W = np.stack(vectors[1:])
    W[W @ v < 0] *= -1
    return [v, *W]


def fit_plane(points: np.ndarray, normalize=True) -> Tuple[np.ndarray, float]:
    """Fit a plane to a set of points.

    Parameters
    ----------
    points : np.ndarray
        The points to fit the plane to. Shape: (N, 3)
    normalize : bool
        Whether to normalize the plane normal vector.

    Returns
    -------
    Tuple[np.ndarray, float]:
        A tuple of the normal vector of the plane (normalized),
        and the distance of the plane to the origin.
    """
    centroid = points.mean(axis=0, keepdims=True)
    *_, Vh = np.linalg.svd(points - centroid)
    normal = Vh[2]
    if normalize:
        normal = normalized(normal)
    d = np.dot(normal, centroid.squeeze())
    return normal, -d


def fibonacci_sphere(r: Union[np.ndarray, float], samples=50):
    """Implements fibonacci sphere sampling algorithm.
    Code taken from https://stackoverflow.com/a/26127012 and then optimized.

    Parameters
    ----------
    r : np.ndarray | float
        The radii of the spheres. Shape: (N, ) if np.ndarray.
    samples : int
        The number of points to generate per sphere.

    Returns
    -------
    np.ndarray:
        The points on the sphere. Shape: (N, samples, 3)
    """
    y = np.linspace(-r, r, samples, axis=-1)
    theta = phi * np.arange(samples, dtype=float)
    if isinstance(r, np.ndarray):
        theta = theta[np.newaxis, :]
        r = r[:, np.newaxis]

    xz_radius = np.sqrt(r**2 - np.square(y))
    x = xz_radius * np.cos(theta)
    z = xz_radius * np.sin(theta)
    return np.stack((x, y, z), axis=-1)


def inplane(coords: Points, tol=0.1) -> bool:
    """Shape: (N, 3)"""
    n, d = fit_plane(coords)
    dists = coords @ n + d
    return np.allclose(dists, 0.0, atol=tol)


def determine_side(
    coords: np.ndarray, points: np.ndarray, tol=0.1
) -> np.ndarray:
    """Shape: (N, 3), (M, 3) -> (M, )"""
    n, d = fit_plane(coords)
    projection = n[np.newaxis, :] * (points + np.array([[0.0, 0.0, d]]))
    # Shape: (M, 3) @ (3, ) -> (M, )
    vec_dot = projection @ n
    # Zero out small values
    vec_dot[vec_dot < tol] = 0.0
    return vec_dot


def sphere_intersection(
    c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, return_minmax=True
):
    vol = 0.0
    r_min, r_max = (r1, r2) if r1 <= r2 else (r2, r1)

    d = D.euclidean(c1, c2)
    if d + r_min <= r_max:
        vol = 4 / 3 * math.pi * r_min**3
    elif d < (r_sum := r1 + r2):
        vol = math.pi / (
            12
            * d
            * (r_sum - d) ** 2
            * (d**2 + 2 * d * r_sum - 3 * (r1 - r2) ** 2)
        )

    if return_minmax:
        return vol, (r_min, r_max)
    return vol


def farthest_from(pts: Points, cntr: Vector) -> Vector:
    """Find the point farthest from pts on the same sphere.
    Returns unit directon vector.

    Shapes: (N, 3), (3, ) -> (3, )
    """
    vecs = pts - cntr
    units = normalized(vecs, axis=1)
    centroid = units.mean(axis=0)
    return -normalized(centroid)


def orthogonal_to(vec: Vector) -> Vector:
    """Find the vector orthogonal to vec.
    Returns unit directon vector.
    """
    return normalized(np.cross(vec, vec + np.array([1, 0, 0])))
