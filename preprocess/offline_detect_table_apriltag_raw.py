#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect the fixed table AprilTag in raw RGB and copy poses to aligned Zarrs.

Pose convention:
    T_apriltag_10_to_cam maps a point from the AprilTag coordinate system to
    the RealSense color-camera coordinate system (p_cam = R @ p_tag + t).
    Translation is stored in metres.

Every RGB frame receives one 4x4 matrix.  A direct detection has pose_valid=True.
If a frame misses the tag but another frame in the same raw sequence detects it,
the nearest direct observation is copied and pose_filled=True.  If the entire
sequence misses the tag, the matrix is identity and pose_available=False.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import zarr


TAG_FAMILY = "tag36h11"
TAG_ID = 10
TAG_SIZE_M = 0.080
DETECT_SCALES = (1.0, 1.5, 2.0)
RAW_GROUP_NAME = "table_apriltag_to_cam"
POSE_DATASET_NAME = "T_apriltag_10_to_cam"


def make_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV missing cv2.aruco; install opencv-contrib-python")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def detect_markers(gray, detector):
    if hasattr(cv2.aruco, "ArucoDetector") and isinstance(
        detector, cv2.aruco.ArucoDetector
    ):
        return detector.detectMarkers(gray)
    dictionary, params = detector
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def tag_object_points(tag_size_m=None):
    if tag_size_m is None:
        tag_size_m = TAG_SIZE_M
    half = float(tag_size_m) / 2.0
    # OpenCV AprilTag corner order: top-left, top-right, bottom-right,
    # bottom-left in the image.
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def _pose_candidates(image_points, K, dist, object_points):
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        ok = bool(result[0])
        rvecs = result[1] if ok and len(result) > 1 else []
        tvecs = result[2] if ok and len(result) > 2 else []
    except Exception:
        rvecs, tvecs = [], []

    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(translation)) or translation[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        error_px = float(
            np.mean(
                np.linalg.norm(
                    projected.reshape(4, 2) - image_points.reshape(4, 2), axis=1
                )
            )
        )
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        T_tag_to_cam = np.eye(4, dtype=np.float64)
        T_tag_to_cam[:3, :3] = rotation
        T_tag_to_cam[:3, 3] = translation
        candidates.append((error_px, T_tag_to_cam))
    return candidates


def detect_tag_pose(rgb, detector, K, dist):
    """Return (T_tag_to_cam, reprojection_error_px, corners_px), or Nones."""
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    object_points = tag_object_points()
    candidates = []

    for scale in DETECT_SCALES:
        if scale == 1.0:
            detect_gray = gray
        else:
            detect_gray = cv2.resize(
                gray,
                None,
                fx=float(scale),
                fy=float(scale),
                interpolation=cv2.INTER_CUBIC,
            )
        corners, ids, _ = detect_markers(detect_gray, detector)
        if ids is None:
            continue
        for corner, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) != TAG_ID:
                continue
            image_points = (
                np.asarray(corner, dtype=np.float32).reshape(4, 2) / float(scale)
            )
            for error_px, T_tag_to_cam in _pose_candidates(
                image_points, K, dist, object_points
            ):
                candidates.append((error_px, T_tag_to_cam, image_points))

    if not candidates:
        return None, None, None
    error_px, T_tag_to_cam, image_points = min(
        candidates, key=lambda item: item[0]
    )
    return T_tag_to_cam, error_px, image_points


def _replace_dataset(group, name, data, dtype=None):
    if name in group:
        del group[name]
    kwargs = {"data": data}
    if dtype is not None:
        kwargs["dtype"] = dtype
    group.create_dataset(name, **kwargs)


def _distortion_coeffs(root):
    if "distortion_coeffs" in root:
        return np.asarray(root["distortion_coeffs"][:], dtype=np.float64).reshape(-1, 1)
    return np.zeros((5, 1), dtype=np.float64)


def _cache_is_current(root, raw_group, n_frames):
    if not bool(root.attrs.get("table_apriltag_offline_detected", False)):
        return False
    if int(root.attrs.get("table_apriltag_id", -1)) != TAG_ID:
        return False
    if not np.isclose(
        float(root.attrs.get("table_apriltag_size_m", -1.0)), TAG_SIZE_M
    ):
        return False
    if RAW_GROUP_NAME not in raw_group:
        return False
    group = raw_group[RAW_GROUP_NAME]
    required = (
        "timestamps",
        POSE_DATASET_NAME,
        "pose_valid",
        "pose_filled",
        "pose_available",
        "reprojection_error_px",
        "corners_px",
    )
    return all(name in group and len(group[name]) == n_frames for name in required)


def _fill_missing_with_nearest(T_all, valid):
    filled = np.zeros(valid.shape, dtype=bool)
    available = np.zeros(valid.shape, dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return T_all, filled, available

    available[:] = True
    missing_indices = np.flatnonzero(~valid)
    for index in missing_indices:
        nearest = int(valid_indices[np.argmin(np.abs(valid_indices - index))])
        T_all[index] = T_all[nearest]
        filled[index] = True
    return T_all, filled, available


def detect_table_apriltag_for_raw_zarr(
    raw_zarr_path, overwrite=False, progress_every=50
):
    root = zarr.open(raw_zarr_path, mode="a")
    if "raw" not in root or "realsense" not in root["raw"]:
        raise KeyError(f"{raw_zarr_path}: raw/realsense group not found")
    if "intrinsic" not in root:
        raise KeyError(f"{raw_zarr_path}: intrinsic dataset not found")

    raw = root["raw"]
    realsense = raw["realsense"]
    timestamps = np.asarray(realsense["timestamps"][:], dtype=np.float64)
    rgb_array = realsense["rgb"]
    n_frames = len(timestamps)
    if len(rgb_array) != n_frames:
        raise ValueError(
            f"{raw_zarr_path}: RGB/timestamp length mismatch "
            f"({len(rgb_array)} != {n_frames})"
        )
    if n_frames == 0:
        print(f"[TableTagOffline] {raw_zarr_path}: no RealSense RGB frames")
        return False

    if not overwrite and _cache_is_current(root, raw, n_frames):
        valid_count = int(root.attrs.get("table_apriltag_direct_valid", 0))
        print(
            f"[TableTagOffline] cached {os.path.basename(raw_zarr_path)}: "
            f"direct={valid_count}/{n_frames}"
        )
        return True

    K = np.asarray(root["intrinsic"][:], dtype=np.float64).reshape(3, 3)
    dist = _distortion_coeffs(root)
    detector = make_detector()
    T_all = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n_frames, axis=0)
    valid = np.zeros((n_frames,), dtype=bool)
    errors = np.full((n_frames,), np.nan, dtype=np.float32)
    corners_all = np.full((n_frames, 4, 2), np.nan, dtype=np.float32)

    print(
        f"[TableTagOffline] detecting {os.path.basename(raw_zarr_path)}: "
        f"family={TAG_FAMILY}, id={TAG_ID}, size={TAG_SIZE_M * 1000.0:.1f} mm, "
        f"frames={n_frames}"
    )
    for index in range(n_frames):
        T_tag_to_cam, error_px, corners_px = detect_tag_pose(
            rgb_array[index], detector, K, dist
        )
        if T_tag_to_cam is not None:
            T_all[index] = T_tag_to_cam
            valid[index] = True
            errors[index] = np.float32(error_px)
            corners_all[index] = corners_px
        if progress_every > 0 and (
            (index + 1) % progress_every == 0 or index + 1 == n_frames
        ):
            print(
                f"  [{index + 1}/{n_frames}] direct={int(np.sum(valid))}"
            )

    T_all, filled, available = _fill_missing_with_nearest(T_all, valid)
    group = raw.require_group(RAW_GROUP_NAME)
    _replace_dataset(group, "timestamps", timestamps, dtype=np.float64)
    _replace_dataset(group, POSE_DATASET_NAME, T_all, dtype=np.float64)
    _replace_dataset(group, "pose_valid", valid, dtype=bool)
    _replace_dataset(group, "pose_filled", filled, dtype=bool)
    _replace_dataset(group, "pose_available", available, dtype=bool)
    _replace_dataset(group, "reprojection_error_px", errors, dtype=np.float32)
    _replace_dataset(group, "corners_px", corners_all, dtype=np.float32)
    group.attrs["tag_family"] = TAG_FAMILY
    group.attrs["tag_id"] = TAG_ID
    group.attrs["tag_size_m"] = TAG_SIZE_M
    group.attrs["pose_convention"] = (
        "T_apriltag_10_to_cam; p_cam = R @ p_tag + t; translation metres"
    )
    group.attrs["missing_pose_policy"] = "nearest direct observation; identity if none"

    direct_count = int(np.sum(valid))
    root.attrs["table_apriltag_offline_detected"] = True
    root.attrs["table_apriltag_family"] = TAG_FAMILY
    root.attrs["table_apriltag_id"] = TAG_ID
    root.attrs["table_apriltag_size_m"] = TAG_SIZE_M
    root.attrs["table_apriltag_direct_valid"] = direct_count
    print(
        f"[TableTagOffline] done {os.path.basename(raw_zarr_path)}: "
        f"direct={direct_count}/{n_frames}, filled={int(np.sum(filled))}/{n_frames}"
    )
    return True


def _nearest_timestamp_indices(source_ts, target_ts):
    if len(source_ts) == 0:
        return np.zeros((len(target_ts),), dtype=np.int64), np.zeros(
            (len(target_ts),), dtype=bool
        )
    order = np.argsort(source_ts)
    sorted_ts = source_ts[order]
    right = np.searchsorted(sorted_ts, target_ts, side="left")
    right = np.clip(right, 0, len(sorted_ts) - 1)
    left = np.clip(right - 1, 0, len(sorted_ts) - 1)
    choose_left = np.abs(target_ts - sorted_ts[left]) <= np.abs(
        target_ts - sorted_ts[right]
    )
    nearest_sorted = np.where(choose_left, left, right)
    indices = order[nearest_sorted]

    if len(sorted_ts) > 1:
        positive_steps = np.diff(sorted_ts)
        positive_steps = positive_steps[positive_steps > 0]
        median_step = float(np.median(positive_steps)) if len(positive_steps) else 0.0
    else:
        median_step = 0.0
    tolerance = max(1e-6, median_step * 0.25)
    matched = np.abs(source_ts[indices] - target_ts) <= tolerance
    return indices.astype(np.int64), matched


def _frame_keys(data_group):
    return sorted(
        key for key in data_group.group_keys() if str(key).startswith("frame_")
    )


def _output_timestamps(out_root, frame_keys):
    if "realsense_timestamps" in out_root:
        timestamps = np.asarray(
            out_root["realsense_timestamps"][:], dtype=np.float64
        )
        if len(timestamps) == len(frame_keys):
            return timestamps
    return np.asarray(
        [float(out_root["data"][key]["realsense_timestamp"][()]) for key in frame_keys],
        dtype=np.float64,
    )


def sync_raw_tag_to_data_zarr(raw_zarr_path, data_zarr_path):
    raw_root = zarr.open(raw_zarr_path, mode="r")
    raw_group = raw_root["raw"][RAW_GROUP_NAME]
    source_ts = np.asarray(raw_group["timestamps"][:], dtype=np.float64)
    source_T = np.asarray(raw_group[POSE_DATASET_NAME][:], dtype=np.float64)
    source_valid = np.asarray(raw_group["pose_valid"][:], dtype=bool)
    source_filled = np.asarray(raw_group["pose_filled"][:], dtype=bool)
    source_available = np.asarray(raw_group["pose_available"][:], dtype=bool)
    source_error = np.asarray(raw_group["reprojection_error_px"][:], dtype=np.float32)
    source_corners = np.asarray(raw_group["corners_px"][:], dtype=np.float32)

    out_root = zarr.open(data_zarr_path, mode="a")
    if "data" not in out_root:
        raise KeyError(f"{data_zarr_path}: data group not found")
    data_group = out_root["data"]
    frame_keys = _frame_keys(data_group)
    target_ts = _output_timestamps(out_root, frame_keys)
    indices, matched = _nearest_timestamp_indices(source_ts, target_ts)

    n_frames = len(frame_keys)
    T_out = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n_frames, axis=0)
    valid_out = np.zeros((n_frames,), dtype=bool)
    filled_out = np.zeros((n_frames,), dtype=bool)
    available_out = np.zeros((n_frames,), dtype=bool)
    error_out = np.full((n_frames,), np.nan, dtype=np.float32)
    corners_out = np.full((n_frames, 4, 2), np.nan, dtype=np.float32)
    if np.any(matched):
        dst = np.flatnonzero(matched)
        src = indices[matched]
        T_out[dst] = source_T[src]
        valid_out[dst] = source_valid[src]
        filled_out[dst] = source_filled[src]
        available_out[dst] = source_available[src]
        error_out[dst] = source_error[src]
        corners_out[dst] = source_corners[src]

    for index, key in enumerate(frame_keys):
        frame = data_group[key]
        _replace_dataset(frame, POSE_DATASET_NAME, T_out[index], dtype=np.float64)
        _replace_dataset(frame, "apriltag_10_pose_valid", valid_out[index], dtype=bool)
        _replace_dataset(frame, "apriltag_10_pose_filled", filled_out[index], dtype=bool)
        _replace_dataset(
            frame, "apriltag_10_pose_available", available_out[index], dtype=bool
        )
        _replace_dataset(
            frame,
            "apriltag_10_reprojection_error_px",
            error_out[index],
            dtype=np.float32,
        )
        _replace_dataset(
            frame, "apriltag_10_corners_px", corners_out[index], dtype=np.float32
        )

    _replace_dataset(
        out_root, "apriltag_10_timestamps", target_ts, dtype=np.float64
    )
    _replace_dataset(out_root, POSE_DATASET_NAME, T_out, dtype=np.float64)
    _replace_dataset(out_root, "apriltag_10_pose_valid", valid_out, dtype=bool)
    _replace_dataset(out_root, "apriltag_10_pose_filled", filled_out, dtype=bool)
    _replace_dataset(
        out_root, "apriltag_10_pose_available", available_out, dtype=bool
    )
    _replace_dataset(
        out_root, "apriltag_10_reprojection_error_px", error_out, dtype=np.float32
    )
    _replace_dataset(
        out_root, "apriltag_10_corners_px", corners_out, dtype=np.float32
    )
    out_root.attrs["table_apriltag_family"] = TAG_FAMILY
    out_root.attrs["table_apriltag_id"] = TAG_ID
    out_root.attrs["table_apriltag_size_m"] = TAG_SIZE_M
    out_root.attrs["table_apriltag_pose_convention"] = (
        "T_apriltag_10_to_cam; p_cam = R @ p_tag + t; translation metres"
    )
    out_root.attrs["table_apriltag_direct_valid"] = int(np.sum(valid_out))
    out_root.attrs["table_apriltag_timestamp_matched"] = int(np.sum(matched))
    print(
        f"[TableTagSync] {os.path.basename(data_zarr_path)}: "
        f"matched={int(np.sum(matched))}/{n_frames}, "
        f"direct={int(np.sum(valid_out))}/{n_frames}, "
        f"filled={int(np.sum(filled_out))}/{n_frames}"
    )
    return int(np.sum(matched)) == n_frames


def matching_data_zarrs(raw_zarr_path, data_dir):
    stem = os.path.splitext(os.path.basename(raw_zarr_path))[0]
    pattern = os.path.join(data_dir, f"{stem}_aligned_seg*.zarr")
    return sorted(path for path in glob.glob(pattern) if os.path.isdir(path))


def sync_raw_tag_to_data_dir(raw_zarr_path, data_dir):
    data_paths = matching_data_zarrs(raw_zarr_path, data_dir)
    if not data_paths:
        print(
            f"[TableTagSync] no aligned output for "
            f"{os.path.basename(raw_zarr_path)} in {data_dir}; skip"
        )
        return True
    ok = True
    for data_path in data_paths:
        ok = sync_raw_tag_to_data_zarr(raw_zarr_path, data_path) and ok
    return ok


def main():
    global TAG_ID, TAG_SIZE_M
    parser = argparse.ArgumentParser(
        description="Detect table AprilTag 10 in raw RGB and sync poses to data_N Zarrs"
    )
    parser.add_argument("--raw-zarr", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--tag-id", type=int, default=TAG_ID)
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_M, help="Tag edge in metres")
    args = parser.parse_args()
    if args.tag_size <= 0:
        parser.error("--tag-size must be positive")
    TAG_ID, TAG_SIZE_M = args.tag_id, args.tag_size

    ok = True
    try:
        if not args.sync_only:
            ok = detect_table_apriltag_for_raw_zarr(
                args.raw_zarr,
                overwrite=args.overwrite,
                progress_every=args.progress_every,
            )
        if args.data_dir:
            ok = sync_raw_tag_to_data_dir(args.raw_zarr, args.data_dir) and ok
    except Exception as exc:
        print(f"[TableTagOffline][ERROR] {args.raw_zarr}: {exc}")
        ok = False
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
