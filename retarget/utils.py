"""Shared math helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


FINGER_SLUGS = {
    "大拇指指尖": "thumb",
    "食指指尖": "index",
    "中指指尖": "middle",
    "无名指指尖": "ring",
    "小拇指指尖": "pinky",
}


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def rotation_x(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_y(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = [safe_float(v) for v in rpy]
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def compose_homogeneous(
    rotation: np.ndarray | None = None,
    translation: np.ndarray | list[float] | tuple[float, float, float] | None = None,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -(rotation.T @ translation)
    return inv


def transform_points(transform: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    ones = np.ones((points_m.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points_m, ones], axis=1)
    result = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T
    return result[:, :3]


def transform_point(transform: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    return transform_points(transform, np.asarray(point_m, dtype=np.float64).reshape(1, 3))[0]


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()


def relative_rotation_log(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = np.asarray(reference, dtype=np.float64).T @ np.asarray(current, dtype=np.float64)
    return Rotation.from_matrix(delta).as_rotvec()


def rotation_log_so3(rotation: np.ndarray) -> np.ndarray:
    """Fast SO(3) logarithm for matrices produced by our own FK.

    ``Rotation.from_matrix`` defensively projects arbitrary matrices onto SO(3).
    FK outputs are already rotations, so the closed-form logarithm avoids that
    projection.  The near-pi case retains SciPy's robust implementation.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float((matrix[0, 0] + matrix[1, 1] + matrix[2, 2] - 1.0) * 0.5)
    cosine = max(-1.0, min(1.0, cosine))
    theta = float(np.arccos(cosine))
    vee = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1.0e-7:
        return 0.5 * vee
    if np.pi - theta < 1.0e-4:
        return Rotation.from_matrix(matrix).as_rotvec()
    return (theta / (2.0 * np.sin(theta))) * vee


def relative_rotation_log_so3(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    return rotation_log_so3(
        np.asarray(reference, dtype=np.float64).T
        @ np.asarray(current, dtype=np.float64)
    )


def axis_angle_transform(axis: np.ndarray, value: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    axis = axis / norm
    rotation = Rotation.from_rotvec(axis * float(value)).as_matrix()
    return compose_homogeneous(rotation=rotation)


def css_rgb(color: tuple[float, float, float]) -> str:
    r, g, b = [max(0, min(255, int(round(float(v) * 255)))) for v in color]
    return f"rgb({r},{g},{b})"


def yellow_to_red_css(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 1.0
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    return f"rgb(255,{int(round(229 * (1.0 - t)))},{int(round(92 * (1.0 - t)))})"


def frame_name(index: int) -> str:
    return f"frame_{index:05d}"


def ensure_array(transform: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(transform, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    return array


def is_valid_transform(transform: np.ndarray) -> bool:
    array = np.asarray(transform, dtype=np.float64)
    return array.shape == (4, 4) and np.all(np.isfinite(array))


def median_translation(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.asarray([item[:3, 3] for item in transforms], dtype=np.float64)
    return np.median(translations, axis=0)


def mean_rotation(transforms: list[np.ndarray]) -> np.ndarray:
    rotations = Rotation.from_matrix([item[:3, :3] for item in transforms])
    return rotations.mean().as_matrix()


def path_to_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def clip_norm(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def pairwise(iterable: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(iterable):
        for right in iterable[i + 1 :]:
            pairs.append((left, right))
    return pairs


def homogeneous_from_rotvec_translation(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return compose_homogeneous(rotation=rotation, translation=np.asarray(translation, dtype=np.float64))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def finite_or_zeros(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array[~np.isfinite(array)] = 0.0
    return array


def norm3(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def scale_transform_translation(transform: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.asarray(transform, dtype=np.float64).copy()
    scaled[:3, 3] *= float(scale)
    return scaled


def pretty_matrix(matrix: np.ndarray) -> str:
    return np.array2string(np.asarray(matrix, dtype=np.float64), precision=5, suppress_small=True)


PI = math.pi
