"""Build a metric camera trajectory from a fixed AprilTag in an input Zarr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp

from .io_utils import ensure_dir, frame_timestamps_from_dataset, write_json


@dataclass(frozen=True)
class AprilTagCameraResult:
    """Cleaned camera trajectory and serializable quality diagnostics."""

    camera_trajectory_cam0: np.ndarray
    cleaned_tag_to_camera: np.ndarray
    diagnostics: dict[str, Any]


def _root_array(root: Any, name: str, frame_count: int, dtype: Any) -> np.ndarray:
    if name not in root:
        raise KeyError(
            f"Input Zarr is missing root array {name!r}; run AprilTag detection/alignment first"
        )
    value = np.asarray(root[name][...], dtype=dtype)
    if value.shape[0] != frame_count:
        raise ValueError(
            f"{name} has {value.shape[0]} frames but the Zarr data group has {frame_count}"
        )
    return value


def _tag_timestamps(dataset: Any, frame_count: int) -> np.ndarray:
    root = dataset.root
    if "apriltag_10_timestamps" in root:
        timestamps = np.asarray(root["apriltag_10_timestamps"][...], dtype=np.float64)
        if timestamps.shape == (frame_count,) and np.all(np.isfinite(timestamps)):
            if np.all(np.diff(timestamps) > 0.0):
                return timestamps
    timestamps = frame_timestamps_from_dataset(dataset)
    if timestamps.shape != (frame_count,) or not np.all(np.isfinite(timestamps)):
        raise ValueError("Could not obtain one finite timestamp for every camera frame")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Camera timestamps must be strictly increasing")
    return timestamps


def _validate_transforms(tag_to_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(tag_to_cam)
    valid = np.all(np.isfinite(tag_to_cam), axis=(1, 2))
    valid &= np.max(
        np.abs(tag_to_cam[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])), axis=1
    ) < 1e-5
    normalized = tag_to_cam.copy()
    for index in range(frame_count):
        if not valid[index]:
            continue
        rotation = tag_to_cam[index, :3, :3]
        orthogonal_error = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        determinant = np.linalg.det(rotation)
        if orthogonal_error > 0.1 or determinant < 0.5:
            valid[index] = False
            continue
        normalized[index, :3, :3] = Rotation.from_matrix(rotation).as_matrix()
    # The detected tag must be in front of the RGB camera at a plausible table distance.
    valid &= normalized[:, 2, 3] > 0.05
    valid &= np.linalg.norm(normalized[:, :3, 3], axis=1) < 5.0
    return normalized, valid


def _camera_pose_vectors(tag_to_cam: np.ndarray, anchor: Rotation) -> tuple[np.ndarray, np.ndarray]:
    cam_to_tag = np.linalg.inv(tag_to_cam)
    position = cam_to_tag[:, :3, 3]
    tangent = (anchor.inv() * Rotation.from_matrix(cam_to_tag[:, :3, :3])).as_rotvec()
    return position, tangent


def _local_median_residual(values: np.ndarray, valid: np.ndarray, half_window: int) -> np.ndarray:
    residual = np.full(len(values), np.nan, dtype=np.float64)
    for index in np.flatnonzero(valid):
        start = max(0, index - half_window)
        stop = min(len(values), index + half_window + 1)
        neighbors = np.flatnonzero(valid[start:stop]) + start
        if len(neighbors) >= 3:
            residual[index] = np.linalg.norm(
                values[index] - np.median(values[neighbors], axis=0)
            )
    return residual


def _robust_gate_threshold(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    mad_scale: float,
    floor: float,
    ceiling: float,
) -> float:
    values = residual[valid & np.isfinite(residual)]
    if values.size == 0:
        return float(ceiling)
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    adaptive = median + mad_scale * max(mad, np.finfo(np.float64).eps)
    return float(np.clip(adaptive, floor, ceiling))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    result = mask.copy()
    for index in np.flatnonzero(mask):
        result[max(0, index - radius) : min(len(mask), index + radius + 1)] = True
    return result


def _missing_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if not is_valid and start is None:
            start = index
        if is_valid and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(valid) - 1))
    return runs


def _interpolate_pose_vectors(
    position: np.ndarray,
    tangent: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid direct AprilTag poses are required")
    result_position = position.copy()
    result_tangent = tangent.copy()
    interpolated = np.zeros(len(valid), dtype=bool)
    run_diagnostics: list[dict[str, Any]] = []
    median_dt = float(np.median(np.diff(timestamps)))

    for start, stop in _missing_runs(valid):
        duration = float(timestamps[stop] - timestamps[start] + median_dt)
        if duration > max_gap_sec + 1e-9:
            raise ValueError(
                "AprilTag trajectory contains an unrepairable missing/outlier run "
                f"at frames {start}..{stop} ({duration:.3f}s > {max_gap_sec:.3f}s)"
            )
        left = start - 1
        right = stop + 1
        if left >= 0 and right < len(valid) and valid[left] and valid[right]:
            query = timestamps[start : stop + 1]
            alpha = (query - timestamps[left]) / (timestamps[right] - timestamps[left])
            result_position[start : stop + 1] = (
                (1.0 - alpha[:, None]) * position[left]
                + alpha[:, None] * position[right]
            )
            rotations = Rotation.from_rotvec(np.stack([tangent[left], tangent[right]]))
            result_tangent[start : stop + 1] = Slerp(
                [timestamps[left], timestamps[right]], rotations
            )(query).as_rotvec()
            mode = "se3_interpolation"
        else:
            nearest = right if left < 0 else left
            if nearest < 0 or nearest >= len(valid) or not valid[nearest]:
                raise ValueError(f"Cannot repair edge AprilTag gap at frames {start}..{stop}")
            result_position[start : stop + 1] = position[nearest]
            result_tangent[start : stop + 1] = tangent[nearest]
            mode = "short_edge_hold"
        interpolated[start : stop + 1] = True
        run_diagnostics.append(
            {"start": start, "stop": stop, "duration_sec": duration, "mode": mode}
        )
    return result_position, result_tangent, interpolated, run_diagnostics


def _lowpass_initialization(values: np.ndarray, timestamps: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if len(values) < 10:
        return values.copy()
    sample_rate = 1.0 / float(np.median(np.diff(timestamps)))
    nyquist = 0.5 * sample_rate
    effective_cutoff = min(float(cutoff_hz), 0.8 * nyquist)
    if effective_cutoff <= 0.0:
        return values.copy()
    sos = butter(3, effective_cutoff, btype="lowpass", fs=sample_rate, output="sos")
    try:
        return np.asarray(sosfiltfilt(sos, values, axis=0), dtype=np.float64)
    except ValueError:
        return values.copy()


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    half = 0.5 * float(tag_size_m)
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _project_tag(
    position: np.ndarray,
    tangent: np.ndarray,
    anchor_matrix: np.ndarray,
    intrinsic: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    rotation = anchor_matrix @ Rotation.from_rotvec(tangent).as_matrix()
    tag_in_camera = np.einsum(
        "nij,nkj->nki",
        np.swapaxes(rotation, 1, 2),
        object_points[None, :, :] - position[:, None, :],
    )
    projected_h = np.einsum("ij,nkj->nki", intrinsic, tag_in_camera)
    depth = projected_h[:, :, 2:3]
    safe_depth = np.where(np.abs(depth) > 1e-9, depth, np.nan)
    return projected_h[:, :, :2] / safe_depth


def _jacobian_sparsity(frame_count: int, measured_indices: np.ndarray) -> Any:
    measurement_rows = 8 * len(measured_indices)
    acceleration_rows = 6 * max(0, frame_count - 2)
    sparsity = lil_matrix((measurement_rows + acceleration_rows, 6 * frame_count), dtype=int)
    row = 0
    for frame_index in measured_indices:
        sparsity[row : row + 8, 6 * frame_index : 6 * frame_index + 6] = 1
        row += 8
    for center in range(1, frame_count - 1):
        sparsity[row : row + 6, 6 * (center - 1) : 6 * (center + 2)] = 1
        row += 6
    return sparsity.tocsr()


def _smooth_with_corner_constraints(
    initial_position: np.ndarray,
    initial_tangent: np.ndarray,
    timestamps: np.ndarray,
    corners: np.ndarray,
    measured: np.ndarray,
    anchor: Rotation,
    intrinsic: np.ndarray,
    tag_size_m: float,
    pixel_sigma: float,
    translation_accel_sigma: float,
    rotation_accel_sigma: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_count = len(timestamps)
    measured_indices = np.flatnonzero(measured)
    if frame_count < 3 or len(measured_indices) < 2:
        return initial_position, initial_tangent, {
            "used": False,
            "reason": "too_few_frames_or_measurements",
        }
    anchor_matrix = anchor.as_matrix()
    object_points = _tag_object_points(tag_size_m)
    dt_left = timestamps[1:-1] - timestamps[:-2]
    dt_right = timestamps[2:] - timestamps[1:-1]
    if np.any(dt_left <= 0.0) or np.any(dt_right <= 0.0):
        raise ValueError("AprilTag timestamps must be strictly increasing")

    def acceleration(values: np.ndarray) -> np.ndarray:
        left_velocity = (values[1:-1] - values[:-2]) / dt_left[:, None]
        right_velocity = (values[2:] - values[1:-1]) / dt_right[:, None]
        return 2.0 * (right_velocity - left_velocity) / (
            dt_left[:, None] + dt_right[:, None]
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        poses = parameters.reshape(frame_count, 6)
        position = poses[:, :3]
        tangent = poses[:, 3:]
        projected = _project_tag(
            position[measured_indices],
            tangent[measured_indices],
            anchor_matrix,
            intrinsic,
            object_points,
        )
        pixel = (projected - corners[measured_indices]).reshape(-1) / pixel_sigma
        translation_accel = acceleration(position).reshape(-1) / translation_accel_sigma
        rotation_accel = acceleration(tangent).reshape(-1) / rotation_accel_sigma
        temporal = np.column_stack(
            [
                translation_accel.reshape(-1, 3),
                rotation_accel.reshape(-1, 3),
            ]
        ).reshape(-1)
        return np.concatenate([pixel, temporal])

    initial = np.column_stack([initial_position, initial_tangent]).reshape(-1)
    result = least_squares(
        residual,
        initial,
        jac_sparsity=_jacobian_sparsity(frame_count, measured_indices),
        loss="huber",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        # Pixel detections are only sub-pixel accurate; tighter tolerances spend many
        # evaluations chasing numerical changes with no observable trajectory benefit.
        xtol=1e-4,
        ftol=1e-4,
        gtol=1e-4,
        verbose=0,
    )
    optimized = result.x.reshape(frame_count, 6)
    return optimized[:, :3], optimized[:, 3:], {
        "used": True,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "cost": float(result.cost),
    }


def _poses_from_vectors(position: np.ndarray, tangent: np.ndarray, anchor: Rotation) -> np.ndarray:
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(position), axis=0)
    result[:, :3, :3] = anchor.as_matrix() @ Rotation.from_rotvec(tangent).as_matrix()
    result[:, :3, 3] = position
    return result


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _reprojection_error(
    position: np.ndarray,
    tangent: np.ndarray,
    anchor: Rotation,
    intrinsic: np.ndarray,
    tag_size_m: float,
    corners: np.ndarray,
) -> np.ndarray:
    projected = _project_tag(
        position,
        tangent,
        anchor.as_matrix(),
        intrinsic,
        _tag_object_points(tag_size_m),
    )
    return np.mean(np.linalg.norm(projected - corners, axis=2), axis=1)


def run_apriltag_camera_estimation(
    dataset: Any,
    indices: list[int],
    output_dir: str | Path,
    *,
    max_reprojection_error_px: float = 1.0,
    max_gap_sec: float = 0.4,
    gate_half_window: int = 5,
    gate_mad_scale: float = 6.0,
    gate_translation_floor_m: float = 0.02,
    gate_translation_ceiling_m: float = 0.06,
    gate_rotation_floor_deg: float = 1.0,
    gate_rotation_ceiling_deg: float = 5.0,
    gate_dilation_frames: int = 1,
    lowpass_cutoff_hz: float = 3.0,
    pixel_sigma: float = 0.25,
    translation_accel_sigma: float = 2.0,
    rotation_accel_sigma: float = 8.0,
    max_nfev: int = 120,
) -> AprilTagCameraResult:
    """Clean all source poses, then return selected ``T_cam0_cam_t`` poses."""
    frame_count = len(dataset)
    if not indices:
        raise ValueError("At least one selected frame is required")
    selected = np.asarray(indices, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= frame_count):
        raise IndexError("Selected camera frame index is outside the input Zarr")
    root = dataset.root
    raw_tag_to_cam = _root_array(root, "T_apriltag_10_to_cam", frame_count, np.float64)
    corners = _root_array(root, "apriltag_10_corners_px", frame_count, np.float64)
    pose_valid = _root_array(root, "apriltag_10_pose_valid", frame_count, bool).reshape(-1)
    pose_available = _root_array(root, "apriltag_10_pose_available", frame_count, bool).reshape(-1)
    pose_filled = _root_array(root, "apriltag_10_pose_filled", frame_count, bool).reshape(-1)
    saved_reprojection = _root_array(
        root, "apriltag_10_reprojection_error_px", frame_count, np.float64
    ).reshape(-1)
    timestamps = _tag_timestamps(dataset, frame_count)

    tag_to_cam, matrix_valid = _validate_transforms(raw_tag_to_cam)
    flag_valid = pose_valid & pose_available & ~pose_filled
    reprojection_valid = np.isfinite(saved_reprojection) & (
        saved_reprojection <= max_reprojection_error_px
    )
    corners_valid = np.all(np.isfinite(corners), axis=(1, 2))
    initial_valid = matrix_valid & flag_valid & reprojection_valid & corners_valid
    if np.count_nonzero(initial_valid) < 2:
        raise ValueError("Fewer than two direct, valid AprilTag poses remain after validity checks")

    first_direct = int(np.flatnonzero(initial_valid)[0])
    direct_cam_to_tag = np.linalg.inv(tag_to_cam[first_direct])
    anchor = Rotation.from_matrix(direct_cam_to_tag[:3, :3])
    safe_tag_to_cam = tag_to_cam.copy()
    safe_tag_to_cam[~matrix_valid] = tag_to_cam[first_direct]
    position, tangent = _camera_pose_vectors(safe_tag_to_cam, anchor)
    translation_residual = _local_median_residual(
        position, initial_valid, int(gate_half_window)
    )
    rotation_residual = _local_median_residual(
        tangent, initial_valid, int(gate_half_window)
    )
    translation_threshold = _robust_gate_threshold(
        translation_residual,
        initial_valid,
        mad_scale=gate_mad_scale,
        floor=gate_translation_floor_m,
        ceiling=gate_translation_ceiling_m,
    )
    rotation_threshold = _robust_gate_threshold(
        rotation_residual,
        initial_valid,
        mad_scale=gate_mad_scale,
        floor=np.deg2rad(gate_rotation_floor_deg),
        ceiling=np.deg2rad(gate_rotation_ceiling_deg),
    )
    temporal_outlier = initial_valid & (
        (translation_residual > translation_threshold)
        | (rotation_residual > rotation_threshold)
    )
    rejected_temporal = _dilate(temporal_outlier, int(gate_dilation_frames)) & initial_valid
    accepted = initial_valid & ~rejected_temporal

    position_filled, tangent_filled, interpolated, gap_diagnostics = _interpolate_pose_vectors(
        position,
        tangent,
        timestamps,
        accepted,
        float(max_gap_sec),
    )
    position_initial = _lowpass_initialization(
        position_filled, timestamps, float(lowpass_cutoff_hz)
    )
    tangent_initial = _lowpass_initialization(
        tangent_filled, timestamps, float(lowpass_cutoff_hz)
    )
    position_clean, tangent_clean, optimizer_diagnostics = _smooth_with_corner_constraints(
        position_initial,
        tangent_initial,
        timestamps,
        corners,
        accepted,
        anchor,
        np.asarray(dataset.intrinsic, dtype=np.float64),
        float(root.attrs.get("table_apriltag_size_m", 0.08)),
        float(pixel_sigma),
        float(translation_accel_sigma),
        float(rotation_accel_sigma),
        int(max_nfev),
    )
    clean_cam_to_tag = _poses_from_vectors(position_clean, tangent_clean, anchor)
    cleaned_tag_to_camera = np.linalg.inv(clean_cam_to_tag[selected])
    cam0_from_tag = np.linalg.inv(clean_cam_to_tag[selected[0]])
    trajectory = cam0_from_tag[None, :, :] @ clean_cam_to_tag[selected]
    trajectory[0] = np.eye(4, dtype=np.float64)

    tag_size_m = float(root.attrs.get("table_apriltag_size_m", 0.08))
    raw_reprojection = _reprojection_error(
        position[accepted],
        tangent[accepted],
        anchor,
        np.asarray(dataset.intrinsic, dtype=np.float64),
        tag_size_m,
        corners[accepted],
    )
    clean_reprojection = _reprojection_error(
        position_clean[accepted],
        tangent_clean[accepted],
        anchor,
        np.asarray(dataset.intrinsic, dtype=np.float64),
        tag_size_m,
        corners[accepted],
    )
    step_translation = np.linalg.norm(np.diff(position_clean, axis=0), axis=1)
    step_rotation_deg = np.rad2deg(np.linalg.norm(np.diff(tangent_clean, axis=0), axis=1))
    diagnostics: dict[str, Any] = {
        "method": "fixed_apriltag_10_se3_trajectory",
        "source_pose_semantics": "T_apriltag_10_to_cam = T_cam_t_tag",
        "camera_trajectory_semantics": "T_cam0_cam_t",
        "formula": "T_cam0_cam_t = T_cam0_tag @ T_tag_cam_t",
        "input_zarr": str(dataset.zarr_path),
        "source_frame_count": frame_count,
        "selected_frame_count": len(indices),
        "selected_indices": [int(index) for index in indices],
        "tag_size_m": tag_size_m,
        "validity": {
            "matrix_valid_count": int(np.count_nonzero(matrix_valid)),
            "direct_flag_valid_count": int(np.count_nonzero(flag_valid)),
            "reprojection_valid_count": int(np.count_nonzero(reprojection_valid)),
            "initial_valid_count": int(np.count_nonzero(initial_valid)),
            "source_filled_count": int(np.count_nonzero(pose_filled)),
            "accepted_direct_count": int(np.count_nonzero(accepted)),
        },
        "temporal_gate": {
            "translation_threshold_m": translation_threshold,
            "rotation_threshold_deg": float(np.rad2deg(rotation_threshold)),
            "outlier_core_indices": [int(index) for index in np.flatnonzero(temporal_outlier)],
            "rejected_indices": [int(index) for index in np.flatnonzero(rejected_temporal)],
        },
        "interpolation": {
            "max_gap_sec": float(max_gap_sec),
            "interpolated_frame_count": int(np.count_nonzero(interpolated)),
            "runs": gap_diagnostics,
        },
        "smoothing": {
            "initial_lowpass_cutoff_hz": float(lowpass_cutoff_hz),
            "pixel_sigma": float(pixel_sigma),
            "translation_accel_sigma_m_s2": float(translation_accel_sigma),
            "rotation_accel_sigma_rad_s2": float(rotation_accel_sigma),
            "optimizer": optimizer_diagnostics,
        },
        "quality": {
            "raw_direct_reprojection_px": _percentiles(raw_reprojection),
            "clean_direct_reprojection_px": _percentiles(clean_reprojection),
            "clean_step_translation_m": _percentiles(step_translation),
            "clean_step_rotation_deg": _percentiles(step_rotation_deg),
        },
    }

    output_path = ensure_dir(output_dir)
    np.save(output_path / "camera_trajectory_cam0.npy", trajectory)
    write_json(output_path / "apriltag_camera_diagnostics.json", diagnostics)
    return AprilTagCameraResult(
        camera_trajectory_cam0=trajectory,
        cleaned_tag_to_camera=cleaned_tag_to_camera,
        diagnostics=diagnostics,
    )
