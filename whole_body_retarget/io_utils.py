"""I/O helpers for whole-body retargeting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_URDF = (
    PROJECT_ROOT
    / "assets" / "robot"
    / "urdf"
    / "moz_wuji_7_3_dual_hands.urdf"
)
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs/robot_camera.json"
DEFAULT_CALIBRATION_KEY = "calib_cam_to_base"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/retarget_free_wrist.yaml"
DEFAULT_LAYOUT = PROJECT_ROOT / "assets/tactile/layout.json"
DEFAULT_BASE_IN_TAG = PROJECT_ROOT / "configs/base_in_scene_tag.json"


def ensure_project_imports() -> None:
    for path in (PROJECT_ROOT, WORKSPACE_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_transform_from_json(path: str | Path, key: str) -> np.ndarray:
    payload = read_json(path)
    if key not in payload:
        available = ", ".join(
            sorted(
                name
                for name, value in payload.items()
                if np.asarray(value).shape == (4, 4)
            )
        )
        raise KeyError(f"{key!r} not found in {path}. Available transform keys: {available}")
    transform = np.asarray(payload[key], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{key} must be a 4x4 matrix, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{key} contains non-finite values")
    return transform


def load_base_in_tag(path: str | Path) -> np.ndarray:
    """T_tag_base maps robot-base coordinates to the stationary scene-tag frame."""
    source = Path(path).expanduser().resolve()
    if source.suffix == ".npy":
        value = np.load(source, allow_pickle=False)
    else:
        value = read_json(source)["T_tag_base"]
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_tag_base must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("T_tag_base has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-5
    ):
        raise ValueError("T_tag_base rotation must be in SO(3)")
    return transform


def frame_timestamp(frame: Any) -> float:
    key = "realsense_timestamp" if "realsense_timestamp" in frame else "timestamp"
    return float(np.asarray(frame[key][...]).item())


def frame_timestamps_from_data_group(data_group: Any, frame_keys: list[str] | None = None) -> np.ndarray:
    keys = sorted(data_group.keys()) if frame_keys is None else frame_keys
    return np.asarray([frame_timestamp(data_group[key]) for key in keys], dtype=np.float64)


def frame_timestamps_from_dataset(dataset: Any) -> np.ndarray:
    return np.asarray(
        [frame_timestamp(dataset.frame_group(index)) for index in range(len(dataset))],
        dtype=np.float64,
    )


def _nearest_timestamp_indices(
    timestamps: np.ndarray,
    *,
    start_frame: int,
    max_frames: int | None,
    target_fps: float,
) -> list[int]:
    if target_fps <= 0.0 or not np.isfinite(target_fps):
        raise ValueError("target_fps must be a finite positive value")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if start_frame >= len(timestamps):
        raise ValueError("No frames selected")

    candidate_indices = np.arange(start_frame, len(timestamps), dtype=np.int64)
    candidate_timestamps = np.asarray(timestamps[start_frame:], dtype=np.float64)
    finite = np.isfinite(candidate_timestamps)
    candidate_indices = candidate_indices[finite]
    candidate_timestamps = candidate_timestamps[finite]
    if candidate_indices.size == 0:
        raise ValueError("No finite frame timestamps selected")
    if np.any(np.diff(candidate_timestamps) < 0.0):
        raise ValueError("Frame timestamps must be non-decreasing for --target-fps selection")

    period = 1.0 / float(target_fps)
    if candidate_timestamps.size > 2:
        diffs = np.diff(candidate_timestamps)
        positive_diffs = diffs[diffs > 0.0]
        median_dt = float(np.median(positive_diffs)) if positive_diffs.size else period
        leading_gap_threshold = max(3.0 * median_dt, 2.0 * period)
        first_valid = 0
        while (
            first_valid < diffs.size
            and float(diffs[first_valid]) > leading_gap_threshold
        ):
            first_valid += 1
        if first_valid > 0:
            candidate_indices = candidate_indices[first_valid:]
            candidate_timestamps = candidate_timestamps[first_valid:]

    start_ts = float(candidate_timestamps[0])
    end_ts = float(candidate_timestamps[-1])
    if end_ts <= start_ts:
        selected = candidate_indices[:1].astype(int).tolist()
        return selected if max_frames is None else selected[:max_frames]

    grid_count = int(np.floor((end_ts - start_ts) / period)) + 1
    grid = start_ts + period * np.arange(grid_count, dtype=np.float64)
    positions = np.searchsorted(candidate_timestamps, grid, side="left")
    selected: list[int] = []
    last_index = -1
    for target_ts, position in zip(grid, positions, strict=True):
        choices: list[int] = []
        if position > 0:
            choices.append(int(position - 1))
        if position < candidate_timestamps.size:
            choices.append(int(position))
        if not choices:
            continue
        best_local = min(
            choices,
            key=lambda item: (abs(float(candidate_timestamps[item]) - float(target_ts)), item),
        )
        frame_index = int(candidate_indices[best_local])
        if frame_index != last_index:
            selected.append(frame_index)
            last_index = frame_index
            if max_frames is not None and len(selected) >= max_frames:
                break
    if not selected:
        raise ValueError("No frames selected")
    return selected


def selected_indices(
    total_frames: int,
    start_frame: int,
    max_frames: int | None,
    frame_stride: int,
    *,
    target_fps: float | None = None,
    timestamps: np.ndarray | None = None,
) -> list[int]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if target_fps is not None:
        if frame_stride != 1:
            raise ValueError("--target-fps cannot be combined with frame_stride != 1")
        if timestamps is None:
            raise ValueError("timestamps are required when target_fps is set")
        if len(timestamps) != total_frames:
            raise ValueError(
                f"timestamp count {len(timestamps)} does not match frame count {total_frames}"
            )
        return _nearest_timestamp_indices(
            np.asarray(timestamps, dtype=np.float64),
            start_frame=start_frame,
            max_frames=max_frames,
            target_fps=float(target_fps),
        )
    indices = list(range(start_frame, total_frames, frame_stride))
    if max_frames is not None:
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        indices = indices[:max_frames]
    if not indices:
        raise ValueError("No frames selected")
    return indices


def frame_selection_metadata(
    *,
    total_frames: int,
    indices: list[int],
    start_frame: int,
    max_frames: int | None,
    frame_stride: int,
    target_fps: float | None,
    timestamps: np.ndarray | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "original_num_frames": int(total_frames),
        "selected_num_frames": int(len(indices)),
        "start_frame": int(start_frame),
        "max_frames": None if max_frames is None else int(max_frames),
        "frame_stride": int(frame_stride),
        "source_frame_index_preserved": True,
    }
    if target_fps is None:
        payload.update(
            {
                "downsample_source": None,
                "downsample_method": "frame_stride",
                "target_fps": None,
            }
        )
    else:
        payload.update(
            {
                "downsample_source": "data/realsense_timestamp",
                "downsample_method": "nearest_source_frame_to_uniform_timestamp_grid",
                "target_fps": float(target_fps),
            }
        )
    if indices:
        payload["first_source_frame_index"] = int(indices[0])
        payload["last_source_frame_index"] = int(indices[-1])
        payload["leading_source_frames_skipped"] = max(0, int(indices[0]) - int(start_frame))
    if timestamps is not None and indices:
        selected_timestamps = np.asarray(timestamps, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
        payload["first_realsense_timestamp"] = float(selected_timestamps[0])
        payload["last_realsense_timestamp"] = float(selected_timestamps[-1])
        if len(selected_timestamps) > 1:
            diffs = np.diff(selected_timestamps)
            duration = float(selected_timestamps[-1] - selected_timestamps[0])
            payload["effective_fps"] = (
                float((len(selected_timestamps) - 1) / duration)
                if duration > 0.0
                else None
            )
            payload["selected_timestamp_diff_min"] = float(np.min(diffs))
            payload["selected_timestamp_diff_median"] = float(np.median(diffs))
            payload["selected_timestamp_diff_max"] = float(np.max(diffs))
        else:
            payload["effective_fps"] = None
    return payload


def complete_wrist_free_output(output_dir: str | Path) -> bool:
    path = Path(output_dir)
    required = (
        "joints_left.npy",
        "joints_right.npy",
        "wrist_left.npy",
        "wrist_right.npy",
        "joint_names_left.json",
        "joint_names_right.json",
        "metadata.json",
    )
    return all((path / name).exists() for name in required)


def load_wrist_free_output(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir)
    if not complete_wrist_free_output(path):
        raise FileNotFoundError(f"Incomplete wrist-free output in {path}")
    return {
        "joints_left": np.load(path / "joints_left.npy"),
        "joints_right": np.load(path / "joints_right.npy"),
        "wrist_left": np.load(path / "wrist_left.npy"),
        "wrist_right": np.load(path / "wrist_right.npy"),
        "joint_names_left": json.loads((path / "joint_names_left.json").read_text(encoding="utf-8")),
        "joint_names_right": json.loads((path / "joint_names_right.json").read_text(encoding="utf-8")),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
    }


def assert_same_length(name_to_array: dict[str, np.ndarray], expected: int) -> None:
    bad = {name: int(array.shape[0]) for name, array in name_to_array.items() if int(array.shape[0]) != expected}
    if bad:
        raise ValueError(f"Expected {expected} frames, got mismatched arrays: {bad}")
