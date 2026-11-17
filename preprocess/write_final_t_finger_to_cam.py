#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将最终 T_finger_to_cam / T_sensor_to_cam 写回对齐后的 zarr（支持双手）。

链式计算（AprilTag 版本）：
  T_{hand}_finger_to_cam = T_{hand}_cube_to_cam @ T_{hand}_wrist_to_cube @ T_{hand}_finger_to_wrist
  T_{hand}_sensor_to_cam = T_{hand}_finger_to_cam @ T_sensor_to_finger

T_{hand}_wrist_to_cube 是运动学腕部到 AprilTag 正方体中心坐标系的小偏移，默认使用论文附录式 (10)。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


# Appendix Eqs. (10) and (11), translations in metres.
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "configs/human_calibration.json"


def load_calibration(path=DEFAULT_CALIBRATION):
    payload = json.loads(Path(path).read_text())
    result = []
    for key in ("T_right_wrist_to_cube", "T_left_wrist_to_cube", "T_sensor_to_finger"):
        matrix = np.asarray(payload[key], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{key} must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"{key}: invalid homogeneous last row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
            raise ValueError(f"{key}: rotation must be in SO(3)")
        result.append(matrix)
    return tuple(result)

DEFAULT_T_WRIST_TO_CUBE_RIGHT, DEFAULT_T_WRIST_TO_CUBE_LEFT, DEFAULT_T_SENSOR_TO_FINGER = load_calibration()


def parse_matrix_16(vals):
    arr = np.array(vals, dtype=np.float64)
    if arr.size != 16:
        raise ValueError("需要 16 个数字（按行展开）")
    return arr.reshape(4, 4)


def _find_key(grp, candidates):
    """按优先级在 zarr group 中查找第一个存在的 key"""
    for k in candidates:
        if k in grp:
            return k
    return None


def _write_dataset(grp, name, data):
    """写入或覆盖 zarr dataset"""
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data.astype(np.float32))


def write_one_zarr(zarr_path, t_wrist_to_cube_right, t_wrist_to_cube_left, t_sensor_to_finger):
    root = zarr.open(str(zarr_path), mode="r+")
    if "data" not in root:
        raise RuntimeError(f"{zarr_path} 不包含 data 组")

    data_group = root["data"]
    n_frames = int(root.attrs.get("n_frames", len(data_group)))

    right_written = 0
    left_written = 0
    skipped = 0

    for i in range(n_frames):
        frame_key = f"frame_{i:05d}"
        if frame_key not in data_group:
            skipped += 1
            continue
        grp = data_group[frame_key]

        # ---- 右手 ----
        rt_key = _find_key(
            grp,
            ["T_right_cube_to_cam", "T_right_wrist_to_cam"],
        )
        rf_key = _find_key(grp, ["T_right_finger_to_wrist", "T_finger_to_wrist"])
        if rt_key and rf_key:
            t_cube_to_cam = np.array(grp[rt_key][:], dtype=np.float64)
            t_finger_to_wrist = np.array(grp[rf_key][:], dtype=np.float64)
            t_finger_to_cam = t_cube_to_cam @ t_wrist_to_cube_right @ t_finger_to_wrist
            t_sensor_to_cam = t_finger_to_cam @ t_sensor_to_finger
            _write_dataset(grp, "T_right_finger_to_cam", t_finger_to_cam)
            _write_dataset(grp, "T_right_sensor_to_cam", t_sensor_to_cam)
            # 向后兼容：也写无前缀版本
            _write_dataset(grp, "T_finger_to_cam", t_finger_to_cam)
            _write_dataset(grp, "T_sensor_to_cam", t_sensor_to_cam)
            right_written += 1

        # ---- 左手 ----
        lt_key = _find_key(grp, ["T_left_cube_to_cam", "T_left_wrist_to_cam"])
        lf_key = _find_key(grp, ["T_left_finger_to_wrist"])
        if lt_key and lf_key:
            t_cube_to_cam_l = np.array(grp[lt_key][:], dtype=np.float64)
            t_finger_to_wrist_l = np.array(grp[lf_key][:], dtype=np.float64)
            t_finger_to_cam_l = t_cube_to_cam_l @ t_wrist_to_cube_left @ t_finger_to_wrist_l
            t_sensor_to_cam_l = t_finger_to_cam_l @ t_sensor_to_finger
            _write_dataset(grp, "T_left_finger_to_cam", t_finger_to_cam_l)
            _write_dataset(grp, "T_left_sensor_to_cam", t_sensor_to_cam_l)
            left_written += 1

        if not (rt_key and rf_key) and not (lt_key and lf_key):
            skipped += 1

    expected_frames = n_frames - skipped
    if expected_frames <= 0:
        raise RuntimeError("未找到任何可写入的 frame")
    if right_written != expected_frames:
        raise RuntimeError(f"右手 final 写入帧数不完整: {right_written}/{expected_frames}")
    if left_written != expected_frames:
        raise RuntimeError(f"左手 final 写入帧数不完整: {left_written}/{expected_frames}")

    root.attrs["has_final_T_finger_to_cam"] = True
    root.attrs["has_final_T_sensor_to_cam"] = True
    root.attrs["T_wrist_to_cube_right_json"] = json.dumps(
        t_wrist_to_cube_right.tolist(), ensure_ascii=False
    )
    root.attrs["T_wrist_to_cube_left_json"] = json.dumps(
        t_wrist_to_cube_left.tolist(), ensure_ascii=False
    )
    root.attrs["T_sensor_to_finger_json"] = json.dumps(
        t_sensor_to_finger.tolist(), ensure_ascii=False
    )

    print(f"[Write] {zarr_path}")
    print(f"  右手写入帧数: {right_written}")
    print(f"  左手写入帧数: {left_written}")
    print(f"  跳过帧数: {skipped}")


def collect_zarr_paths(input_path, pattern, recursive):
    p = Path(input_path).expanduser().resolve()
    if p.name.endswith(".zarr") and p.is_dir():
        return [p]
    if p.is_dir():
        paths = sorted(p.rglob(pattern) if recursive else p.glob(pattern))
        return [x for x in paths if x.is_dir()]
    raise RuntimeError(f"输入路径不存在: {p}")


def main():
    parser = argparse.ArgumentParser(
        description="将最终 T_finger_to_cam / T_sensor_to_cam 写回对齐后的 zarr（双手）"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 zarr 路径或目录",
    )
    parser.add_argument(
        "--pattern",
        default="*_aligned_seg*.zarr",
        help="当 --input 为目录时的匹配模式",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="当 --input 为目录时递归搜索",
    )
    parser.add_argument(
        "--right-wrist-to-cube",
        type=float,
        nargs=16,
        default=None,
        help="右手 T_wrist_to_cube（运动学腕部→AprilTag 立方体中心，16 个数按行展开）",
    )
    parser.add_argument(
        "--left-wrist-to-cube",
        type=float,
        nargs=16,
        default=None,
        help="左手 T_wrist_to_cube（16 个数按行展开）",
    )
    parser.add_argument(
        "--sensor-matrix",
        type=float,
        nargs=16,
        default=None,
        help="覆盖 T_SENSOR_TO_FINGER（16个数字，按行展开）",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    default_right, default_left, default_sensor = load_calibration(args.calibration)

    if args.right_wrist_to_cube is not None:
        t_wrist_to_cube_right = parse_matrix_16(args.right_wrist_to_cube)
    else:
        t_wrist_to_cube_right = default_right

    if args.left_wrist_to_cube is not None:
        t_wrist_to_cube_left = parse_matrix_16(args.left_wrist_to_cube)
    else:
        t_wrist_to_cube_left = default_left

    if args.sensor_matrix is not None:
        t_sensor_to_finger = parse_matrix_16(args.sensor_matrix)
    else:
        t_sensor_to_finger = default_sensor

    zarr_paths = collect_zarr_paths(args.input, args.pattern, args.recursive)
    if len(zarr_paths) == 0:
        print("未找到可处理的 zarr")
        return

    ok = 0
    fail = 0
    for zarr_path in zarr_paths:
        try:
            write_one_zarr(zarr_path, t_wrist_to_cube_right, t_wrist_to_cube_left, t_sensor_to_finger)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[Write] 失败: {zarr_path}, 错误: {e}")

    print(f"完成：成功 {ok} 个，失败 {fail} 个")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
