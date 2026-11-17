#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline cube-pose detection for raw zarr files.

Uses the standalone cube detector without acquisition hardware or ROS dependencies.
"""

import argparse
import os
import sys

import numpy as np
import zarr

def _load_detector_module():
    if __package__:
        from . import cube_pose_detector
    else:
        import cube_pose_detector
    return cube_pose_detector


def _replace_dataset(group, name, data, dtype=None):
    if name in group:
        del group[name]
    if dtype is None:
        group.create_dataset(name, data=data)
    else:
        group.create_dataset(name, data=data, dtype=dtype)


def _get_raw_group(root):
    if "raw" not in root:
        raise KeyError("raw group not found")
    return root["raw"]


def _dist_coeffs_for_root(root):
    if "distortion_coeffs" in root:
        return np.asarray(root["distortion_coeffs"][:], dtype=np.float64).reshape(-1, 1)
    return np.zeros((5, 1), dtype=np.float64)


def detect_cube_pose_for_raw_zarr(raw_zarr_path, overwrite=False, progress_every=50):
    detector_mod = _load_detector_module()
    root = zarr.open(raw_zarr_path, mode="a")
    raw = _get_raw_group(root)
    rs_group = raw["realsense"]
    rs_ts = np.asarray(rs_group["timestamps"][:], dtype=np.float64)
    rgb_arr = rs_group["rgb"]
    depth_arr = rs_group["depth"] if "depth" in rs_group else None
    n_frames = int(len(rs_ts))

    if n_frames == 0:
        print(f"[CubePoseOffline] {raw_zarr_path}: no RealSense frames")
        return False

    if (
        not overwrite
        and bool(root.attrs.get("cube_pose_offline_detected", False))
        and "right_cube_to_cam" in raw
        and "left_cube_to_cam" in raw
    ):
        print(f"[CubePoseOffline] skip already detected: {raw_zarr_path}")
        return True

    K = np.asarray(root["intrinsic"][:], dtype=np.float64)
    dist = _dist_coeffs_for_root(root)
    detector = detector_mod.make_apriltag_detector()

    T_right = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n_frames, axis=0)
    T_left = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n_frames, axis=0)
    right_valid = np.zeros((n_frames,), dtype=bool)
    left_valid = np.zeros((n_frames,), dtype=bool)
    last_right_T = None
    last_left_T = None

    print(f"[CubePoseOffline] detecting {raw_zarr_path} ({n_frames} frames)")
    for i in range(n_frames):
        rgb = np.asarray(rgb_arr[i], dtype=np.uint8)
        depth = np.asarray(depth_arr[i]) if depth_arr is not None else None
        right_T, left_T = detector_mod.detect_both_cube_poses(
            rgb,
            detector,
            K,
            dist,
            depth,
            prev_right_T=last_right_T,
            prev_left_T=last_left_T,
        )
        if right_T is not None:
            T_right[i] = np.asarray(right_T, dtype=np.float64)
            right_valid[i] = True
            last_right_T = T_right[i].copy()
        if left_T is not None:
            T_left[i] = np.asarray(left_T, dtype=np.float64)
            left_valid[i] = True
            last_left_T = T_left[i].copy()
        if progress_every > 0 and ((i + 1) % progress_every == 0 or i + 1 == n_frames):
            print(
                f"  [{i + 1}/{n_frames}] "
                f"right={int(np.sum(right_valid))}, left={int(np.sum(left_valid))}"
            )

    right_group = raw.require_group("right_cube_to_cam")
    left_group = raw.require_group("left_cube_to_cam")
    _replace_dataset(right_group, "timestamps", rs_ts, dtype=np.float64)
    _replace_dataset(right_group, "pose_valid", right_valid, dtype=bool)
    _replace_dataset(right_group, "T_right_cube_to_cam", T_right)
    _replace_dataset(left_group, "timestamps", rs_ts, dtype=np.float64)
    _replace_dataset(left_group, "pose_valid", left_valid, dtype=bool)
    _replace_dataset(left_group, "T_left_cube_to_cam", T_left)

    root.attrs["cube_pose_offline_detected"] = True
    root.attrs["cube_pose_offline_detector"] = "cube_pose_detector.detect_both_cube_poses"
    root.attrs["cube_pose_offline_right_valid"] = int(np.sum(right_valid))
    root.attrs["cube_pose_offline_left_valid"] = int(np.sum(left_valid))
    print(
        f"[CubePoseOffline] done {os.path.basename(raw_zarr_path)}: "
        f"right={int(np.sum(right_valid))}/{n_frames}, "
        f"left={int(np.sum(left_valid))}/{n_frames}"
    )
    return True


def find_zarr_files(directory):
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".zarr") and os.path.isdir(os.path.join(directory, name))
    )


def main():
    parser = argparse.ArgumentParser(description="Offline detect left/right cube pose for raw zarr.")
    parser.add_argument("--raw-zarr", default=None, help="单个 raw zarr 路径")
    parser.add_argument("--batch-dir", default=None, help="包含多个 raw zarr 的 raw_data_N 目录")
    parser.add_argument("--overwrite", action="store_true", help="即使已经离线检测过也重新检测")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    paths = []
    if args.raw_zarr:
        paths.append(args.raw_zarr)
    if args.batch_dir:
        paths.extend(find_zarr_files(args.batch_dir))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("No raw zarr path provided")

    ok = True
    for path in paths:
        try:
            ok = detect_cube_pose_for_raw_zarr(
                path,
                overwrite=args.overwrite,
                progress_every=args.progress_every,
            ) and ok
        except Exception as exc:
            ok = False
            print(f"[CubePoseOffline][ERROR] {path}: {exc}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
