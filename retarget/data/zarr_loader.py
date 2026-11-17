"""Zarr dataset access for dual-hand tactile retargeting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retarget.utils import FINGER_ORDER, FINGER_SLUGS, frame_name, safe_float

if TYPE_CHECKING:
    import zarr


@dataclass(frozen=True)
class HumanFrameData:
    frame_index: int
    frame_key: str
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    T_left_finger_to_wrist: np.ndarray
    T_right_finger_to_wrist: np.ndarray
    T_left_finger_to_cam: np.ndarray
    T_right_finger_to_cam: np.ndarray
    T_left_sensor_to_cam: np.ndarray
    T_right_sensor_to_cam: np.ndarray
    left_tactile_distribution_force_json: dict[str, Any]
    right_tactile_distribution_force_json: dict[str, Any]
    left_tactile_wrench: np.ndarray
    right_tactile_wrench: np.ndarray
    left_tactile_valid: bool = False
    right_tactile_valid: bool = False
    left_tactile_alignment_mode: str = "missing"
    right_tactile_alignment_mode: str = "missing"
    left_tactile_bracket_gap_sec: float = float("nan")
    right_tactile_bracket_gap_sec: float = float("nan")


@dataclass(frozen=True)
class _AlignedTactileSample:
    payload: dict[str, Any]
    wrench: np.ndarray
    valid: bool
    mode: str
    bracket_gap_sec: float


class _TimestampTactileStream:
    """Interpolate an asynchronous tactile stream at camera timestamps."""

    def __init__(
        self,
        group: Any,
        sensor_names: list[str],
        read_json_scalar: Any,
        max_gap_sec: float,
    ) -> None:
        self.group = group
        self.sensor_names = sensor_names
        self.read_json_scalar = read_json_scalar
        self.max_gap_sec = float(max_gap_sec)
        timestamp_key = (
            "timestamps_realsense_aligned"
            if "timestamps_realsense_aligned" in group
            else "timestamps"
        )
        raw_timestamps = np.asarray(group[timestamp_key][...], dtype=np.float64).reshape(-1)
        if raw_timestamps.size == 0:
            self.timestamps = raw_timestamps
            self.groups: list[np.ndarray] = []
            return
        order = np.argsort(raw_timestamps, kind="stable")
        sorted_timestamps = raw_timestamps[order]
        unique_timestamps, starts = np.unique(sorted_timestamps, return_index=True)
        stops = np.r_[starts[1:], sorted_timestamps.size]
        self.timestamps = unique_timestamps
        self.groups = [order[start:stop] for start, stop in zip(starts, stops, strict=True)]

    def _payload_array(self, raw_index: int) -> np.ndarray:
        payload = self.read_json_scalar(self.group["distribution_force_json"][int(raw_index)])
        rows: list[np.ndarray] = []
        for label in self.sensor_names:
            cells = payload.get(label, [])
            if not isinstance(cells, list):
                cells = []
            values = np.zeros((len(cells), 3), dtype=np.float64)
            for index, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    continue
                for axis, key in enumerate(("X", "Y", "Z")):
                    try:
                        values[index, axis] = float(cell.get(key, 0.0))
                    except (TypeError, ValueError):
                        values[index, axis] = 0.0
            rows.append(values)
        width = max((row.shape[0] for row in rows), default=0)
        out = np.zeros((len(self.sensor_names), width, 3), dtype=np.float64)
        for index, row in enumerate(rows):
            out[index, : row.shape[0]] = row
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _merged_sample(self, unique_index: int) -> tuple[np.ndarray, np.ndarray]:
        raw_indices = self.groups[int(unique_index)]
        payloads = [self._payload_array(int(index)) for index in raw_indices]
        width = max((item.shape[1] for item in payloads), default=0)
        padded = np.zeros((len(payloads), len(self.sensor_names), width, 3), dtype=np.float64)
        for index, item in enumerate(payloads):
            padded[index, :, : item.shape[1]] = item
        payload = np.mean(padded, axis=0)
        wrench = np.mean(
            np.stack(
                [
                    np.asarray(self.group["wrench"][int(index)], dtype=np.float64)
                    for index in raw_indices
                ],
                axis=0,
            ),
            axis=0,
        )
        return payload, np.nan_to_num(wrench, nan=0.0, posinf=0.0, neginf=0.0)

    def _payload_dict(self, values: np.ndarray) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for finger_index, label in enumerate(self.sensor_names):
            payload[label] = [
                {"X": float(vector[0]), "Y": float(vector[1]), "Z": float(vector[2])}
                for vector in values[finger_index]
            ]
        return payload

    def query(self, timestamp: float) -> _AlignedTactileSample:
        if self.timestamps.size == 0:
            return _AlignedTactileSample({}, np.zeros((5, 6)), False, "missing", float("nan"))
        position = int(np.searchsorted(self.timestamps, float(timestamp), side="left"))
        if position < len(self.timestamps) and self.timestamps[position] == float(timestamp):
            payload, wrench = self._merged_sample(position)
            return _AlignedTactileSample(self._payload_dict(payload), wrench, True, "exact", 0.0)

        if position == 0 or position == len(self.timestamps):
            nearest = 0 if position == 0 else len(self.timestamps) - 1
            distance = abs(float(self.timestamps[nearest]) - float(timestamp))
            if distance <= self.max_gap_sec:
                payload, wrench = self._merged_sample(nearest)
                return _AlignedTactileSample(
                    self._payload_dict(payload), wrench, True, "nearest_endpoint", distance
                )
            return _AlignedTactileSample({}, np.zeros((5, 6)), False, "out_of_range", distance)

        left = position - 1
        right = position
        t0 = float(self.timestamps[left])
        t1 = float(self.timestamps[right])
        gap = t1 - t0
        if gap <= 0.0 or gap > self.max_gap_sec:
            return _AlignedTactileSample({}, np.zeros((5, 6)), False, "gap_too_large", gap)
        alpha = float(np.clip((float(timestamp) - t0) / gap, 0.0, 1.0))
        payload0, wrench0 = self._merged_sample(left)
        payload1, wrench1 = self._merged_sample(right)
        width = max(payload0.shape[1], payload1.shape[1])
        padded0 = np.zeros((len(self.sensor_names), width, 3), dtype=np.float64)
        padded1 = np.zeros_like(padded0)
        padded0[:, : payload0.shape[1]] = payload0
        padded1[:, : payload1.shape[1]] = payload1
        payload = (1.0 - alpha) * padded0 + alpha * padded1
        wrench = (1.0 - alpha) * wrench0 + alpha * wrench1
        return _AlignedTactileSample(self._payload_dict(payload), wrench, True, "linear", gap)

    def alignment_status(self, timestamp: float) -> tuple[bool, str, float]:
        """Classify alignment without decoding the relatively large tactile JSON payload."""
        if self.timestamps.size == 0:
            return False, "missing", float("nan")
        position = int(np.searchsorted(self.timestamps, float(timestamp), side="left"))
        if position < len(self.timestamps) and self.timestamps[position] == float(timestamp):
            return True, "exact", 0.0
        if position == 0 or position == len(self.timestamps):
            nearest = 0 if position == 0 else len(self.timestamps) - 1
            distance = abs(float(self.timestamps[nearest]) - float(timestamp))
            return distance <= self.max_gap_sec, (
                "nearest_endpoint" if distance <= self.max_gap_sec else "out_of_range"
            ), distance
        gap = float(self.timestamps[position] - self.timestamps[position - 1])
        return gap <= self.max_gap_sec, ("linear" if gap <= self.max_gap_sec else "gap_too_large"), gap


class ZarrHumanDataset:
    def __init__(self, zarr_path: str | Path, tactile_max_gap_sec: float = 0.03) -> None:
        try:
            import zarr  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
            raise ModuleNotFoundError(
                "zarr is required to read the tactile dataset. Install zarr in the target environment."
            ) from exc
        self._zarr = zarr
        self.zarr_path = Path(zarr_path).resolve()
        self.root = self._zarr.open(str(self.zarr_path), mode="r")
        self.frames = self.root["data"]
        self.frame_keys = sorted(self.frames.keys())
        self.intrinsic = np.asarray(self.root["intrinsic"][...], dtype=np.float64)
        sensor_names_json = self.root.attrs.get("left_tactile_sensor_names_json")
        if sensor_names_json is None and "left_tactile" in self.root:
            sensor_names_json = self.root["left_tactile"].attrs.get("sensor_names_json")
        if sensor_names_json is None:
            raise KeyError("Could not find tactile sensor_names_json in the root or left_tactile group")
        self.sensor_names = json.loads(sensor_names_json)
        self.tactile_max_gap_sec = float(tactile_max_gap_sec)
        self._timestamp_tactile: dict[str, _TimestampTactileStream] = {}
        for side in ("left", "right"):
            group_key = f"{side}_tactile"
            if group_key in self.root:
                self._timestamp_tactile[side] = _TimestampTactileStream(
                    group=self.root[group_key],
                    sensor_names=self.sensor_names,
                    read_json_scalar=self._read_json_scalar,
                    max_gap_sec=self.tactile_max_gap_sec,
                )
        self._nonzero_wrench_stats_cache: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.frame_keys)

    def frame_key(self, index: int) -> str:
        index = max(0, min(len(self.frame_keys) - 1, int(index)))
        return self.frame_keys[index]

    def frame_group(self, index: int) -> Any:
        return self.frames[self.frame_key(index)]

    def _read_json_scalar(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def tactile_json_for_side(self, frame: Any, frame_index: int, side: str) -> dict[str, Any]:
        key = f"{side}_tactile_distribution_force_json"
        if key in frame:
            return self._read_json_scalar(frame[key][...])
        aligned_key = f"{side}_tactile_distribution_force_json_aligned"
        if aligned_key in self.root:
            return self._read_json_scalar(self.root[aligned_key][frame_index])
        return {}

    def _aligned_tactile_for_side(
        self, frame: Any, frame_index: int, side: str, timestamp: float
    ) -> _AlignedTactileSample:
        key = f"{side}_tactile_distribution_force_json"
        wrench_key = f"{side}_tactile_wrench"
        if key in frame:
            wrench = (
                np.asarray(frame[wrench_key][...], dtype=np.float64)
                if wrench_key in frame
                else np.zeros((5, 6), dtype=np.float64)
            )
            return _AlignedTactileSample(
                self._read_json_scalar(frame[key][...]), wrench, True, "per_frame", 0.0
            )
        aligned_key = f"{side}_tactile_distribution_force_json_aligned"
        aligned_wrench_key = f"{side}_tactile_wrench_aligned"
        if aligned_key in self.root:
            wrench = (
                np.asarray(self.root[aligned_wrench_key][frame_index], dtype=np.float64)
                if aligned_wrench_key in self.root
                else np.zeros((5, 6), dtype=np.float64)
            )
            return _AlignedTactileSample(
                self._read_json_scalar(self.root[aligned_key][frame_index]),
                wrench,
                True,
                "root_aligned",
                0.0,
            )
        stream = self._timestamp_tactile.get(side)
        if stream is None:
            return _AlignedTactileSample({}, np.zeros((5, 6)), False, "missing", float("nan"))
        return stream.query(timestamp)

    def get_frame(self, index: int) -> HumanFrameData:
        key = self.frame_key(index)
        frame = self.frames[key]
        timestamp = float(
            frame["realsense_timestamp"][...]
            if "realsense_timestamp" in frame
            else frame["timestamp"][...]
        )
        left_tactile = self._aligned_tactile_for_side(frame, index, "left", timestamp)
        right_tactile = self._aligned_tactile_for_side(frame, index, "right", timestamp)
        return HumanFrameData(
            frame_index=index,
            frame_key=key,
            timestamp=timestamp,
            rgb=np.asarray(frame["rgb"][...], dtype=np.uint8),
            depth=np.asarray(frame["depth"][...], dtype=np.float32),
            intrinsic=self.intrinsic.copy(),
            T_left_finger_to_wrist=np.asarray(frame["T_left_finger_to_wrist"][...], dtype=np.float64),
            T_right_finger_to_wrist=np.asarray(frame["T_right_finger_to_wrist"][...], dtype=np.float64),
            T_left_finger_to_cam=np.asarray(frame["T_left_finger_to_cam"][...], dtype=np.float64),
            T_right_finger_to_cam=np.asarray(frame["T_right_finger_to_cam"][...], dtype=np.float64),
            T_left_sensor_to_cam=np.asarray(frame["T_left_sensor_to_cam"][...], dtype=np.float64),
            T_right_sensor_to_cam=np.asarray(frame["T_right_sensor_to_cam"][...], dtype=np.float64),
            left_tactile_distribution_force_json=left_tactile.payload,
            right_tactile_distribution_force_json=right_tactile.payload,
            left_tactile_wrench=left_tactile.wrench,
            right_tactile_wrench=right_tactile.wrench,
            left_tactile_valid=left_tactile.valid,
            right_tactile_valid=right_tactile.valid,
            left_tactile_alignment_mode=left_tactile.mode,
            right_tactile_alignment_mode=right_tactile.mode,
            left_tactile_bracket_gap_sec=left_tactile.bracket_gap_sec,
            right_tactile_bracket_gap_sec=right_tactile.bracket_gap_sec,
        )

    def tactile_alignment_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"max_gap_sec": self.tactile_max_gap_sec, "sides": {}}
        timestamps = [
            float(
                self.frames[key]["realsense_timestamp"][...]
                if "realsense_timestamp" in self.frames[key]
                else self.frames[key]["timestamp"][...]
            )
            for key in self.frame_keys
        ]
        for side in ("left", "right"):
            stream = self._timestamp_tactile.get(side)
            counts: dict[str, int] = {}
            gaps: list[float] = []
            if stream is not None:
                for timestamp in timestamps:
                    _, mode, gap = stream.alignment_status(timestamp)
                    counts[mode] = counts.get(mode, 0) + 1
                    if np.isfinite(gap):
                        gaps.append(float(gap))
            summary["sides"][side] = {
                "counts": counts,
                "valid_count": int(sum(value for key, value in counts.items() if key in {"exact", "linear", "nearest_endpoint"})),
                "invalid_count": int(sum(value for key, value in counts.items() if key not in {"exact", "linear", "nearest_endpoint"})),
                "max_observed_gap_sec": max(gaps) if gaps else None,
            }
        return summary

    def reconstruct_pointcloud(
        self,
        frame: HumanFrameData,
        sample_step: int,
        max_depth: float,
    ) -> dict[str, Any]:
        depth = frame.depth
        rgb = frame.rgb
        fy = float(frame.intrinsic[1, 1])
        fx = float(frame.intrinsic[0, 0])
        cx = float(frame.intrinsic[0, 2])
        cy = float(frame.intrinsic[1, 2])
        yy, xx = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
        mask = depth > 0
        if max_depth > 0:
            mask &= depth < max_depth
        mask &= (yy % sample_step == 0) & (xx % sample_step == 0)
        z = depth[mask].astype(np.float64)
        x = ((xx[mask] - cx) * z) / fx
        y = ((yy[mask] - cy) * z) / fy
        colors = rgb[mask]
        return {
            "count": int(len(z)),
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
            "color": [f"rgb({r},{g},{b})" for r, g, b in colors.tolist()],
        }

    def _nonzero_wrench_stats(self) -> tuple[int, int]:
        if self._nonzero_wrench_stats_cache is not None:
            return self._nonzero_wrench_stats_cache
        left = 0
        right = 0
        for key in self.frame_keys:
            frame = self.frames[key]
            if "left_tactile_wrench" in frame:
                if np.any(np.asarray(frame["left_tactile_wrench"][...], dtype=np.float64) != 0):
                    left += 1
            if "right_tactile_wrench" in frame:
                if np.any(np.asarray(frame["right_tactile_wrench"][...], dtype=np.float64) != 0):
                    right += 1
        self._nonzero_wrench_stats_cache = (left, right)
        return self._nonzero_wrench_stats_cache

    def inspect_summary(self) -> dict[str, Any]:
        first_frame = self.frames[self.frame_keys[0]]
        start = float(self.root.attrs["segment_start_ts"])
        end = float(self.root.attrs["segment_end_ts"])
        approx_fps = (len(self.frame_keys) - 1) / max(1e-12, end - start)
        left_first = self._find_first_nonzero_frame("left_tactile_wrench")
        right_first = self._find_first_nonzero_frame("right_tactile_wrench")
        left_nonzero_wrench_frames, right_nonzero_wrench_frames = self._nonzero_wrench_stats()
        return {
            "zarr_path": str(self.zarr_path),
            "total_frames": len(self.frame_keys),
            "segment_id": int(self.root.attrs["segment_id"]),
            "intrinsic": self.intrinsic.tolist(),
            "frame_fields": sorted(first_frame.keys()),
            "sensor_names": self.sensor_names,
            "approx_fps": approx_fps,
            "left_tactile_json_exists": any(
                "left_tactile_distribution_force_json" in self.frames[key] for key in self.frame_keys[:3]
            )
            or "left_tactile_distribution_force_json_aligned" in self.root,
            "right_tactile_json_exists": any(
                "right_tactile_distribution_force_json" in self.frames[key] for key in self.frame_keys[:3]
            )
            or "right_tactile_distribution_force_json_aligned" in self.root,
            "left_nonzero_wrench_frames": left_nonzero_wrench_frames,
            "right_nonzero_wrench_frames": right_nonzero_wrench_frames,
            "left_first_nonzero_wrench_frame": left_first,
            "right_first_nonzero_wrench_frame": right_first,
        }

    def _find_first_nonzero_frame(self, key: str) -> str | None:
        for index, frame_key_value in enumerate(self.frame_keys):
            frame = self.frames[frame_key_value]
            if key not in frame:
                continue
            if np.any(np.asarray(frame[key][...], dtype=np.float64) != 0):
                return frame_name(index)
        return None

    def tactile_force_norm_stats(self, frame_index: int) -> dict[str, dict[str, Any]]:
        frame = self.get_frame(frame_index)
        out: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
        for side, payload in (
            ("left", frame.left_tactile_distribution_force_json),
            ("right", frame.right_tactile_distribution_force_json),
        ):
            for label in self.sensor_names:
                finger = FINGER_SLUGS.get(label, label)
                cells = payload.get(label, [])
                norms = []
                if isinstance(cells, list):
                    for cell in cells:
                        if not isinstance(cell, dict):
                            norms.append(0.0)
                            continue
                        fx = safe_float(cell.get("X"))
                        fy = safe_float(cell.get("Y"))
                        fz = safe_float(cell.get("Z"))
                        norms.append(float(np.sqrt(fx * fx + fy * fy + fz * fz)))
                out[side][finger] = {
                    "taxel_count": len(norms),
                    "max_force_norm": max(norms) if norms else 0.0,
                    "mean_force_norm": float(np.mean(norms)) if norms else 0.0,
                }
        return out
