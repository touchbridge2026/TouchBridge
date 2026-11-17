#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 T_cube_to_cam（或旧版 wrist）和手部关节 pose 对齐到 RGBD 时间戳上。

- 新版 raw：逐帧保存 cube pose，缺失帧保留 pose_valid=False
- 对 cube pose：仅对缺口帧做平移线性插值 + 旋转 Slerp；区间外用端点外推
- tactile：不插值、不降采样，按原始帧率和原始时间戳保存到 root 级 tactile group
- 兼容旧版稀疏流：raw/right_wrist_to_cam
"""

import argparse
import json
import os
import ast
from pathlib import Path
import numpy as np
import zarr
import numcodecs
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

T_r_l = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
# ===============================
# 插值工具
# ===============================
def slerp_quaternions(t_query, t_key, q_key):
    """
    对四元数做球面插值。
    q_key: (N, 4) 按 (w,x,y,z) 或 (x,y,z,w) - 需与 Rotation 一致
    scipy Rotation 使用 (x,y,z,w)
    """
    # 转为 (x,y,z,w) 格式
    if q_key.shape[1] == 4:
        # 假设输入是 (w,x,y,z)，转为 (x,y,z,w)
        q_xyzw = np.column_stack([q_key[:, 1], q_key[:, 2], q_key[:, 3], q_key[:, 0]])
    else:
        q_xyzw = q_key
    rots = Rotation.from_quat(q_xyzw)
    slerp = Slerp(t_key, rots)
    interp_rots = slerp(t_query)
    quat_xyzw = interp_rots.as_quat()  # (x,y,z,w)
    # 转回 (w,x,y,z)
    quat_wxyz = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])
    return quat_wxyz


def interpolate_T_4x4(t_target, t_wrist, T_wrist_list):
    """
    对 4x4 变换矩阵插值：平移线性，旋转 Slerp。
    t_wrist: (N,) 时间戳
    T_wrist_list: (N, 4, 4)
    t_target: 目标时间戳 (标量或数组)
    """
    t_target = np.atleast_1d(t_target)
    nq = len(t_target)
    T_out = np.zeros((nq, 4, 4), dtype=np.float32)
    T_out[:, 3, 3] = 1.0

    tw = np.atleast_1d(np.asarray(t_wrist, dtype=np.float64))
    Tlist = np.asarray(T_wrist_list, dtype=np.float64)
    if tw.size == 0 or Tlist.shape[0] == 0:
        T_out[:, :3, :3] = np.eye(3, dtype=np.float32)
        return T_out[0] if nq == 1 else T_out
    if tw.size == 1:
        T0 = Tlist[0].astype(np.float32)
        for i in range(nq):
            T_out[i] = T0
        return T_out[0] if nq == 1 else T_out

    # 平移：线性插值
    trans = np.array([T[:3, 3] for T in T_wrist_list])  # (N, 3)
    for i in range(3):
        f = interp1d(t_wrist, trans[:, i], kind='linear',
                     bounds_error=False, fill_value=(trans[0, i], trans[-1, i]))
        T_out[:, i, 3] = f(t_target)

    # 旋转：Slerp
    quats = []
    for T in T_wrist_list:
        R = Rotation.from_matrix(T[:3, :3])
        q = R.as_quat()  # (x,y,z,w)
        quats.append(q)
    quats = np.array(quats)  # (N, 4) xyzw

    slerp = Slerp(t_wrist, Rotation.from_quat(quats))
    t_rot_query = np.clip(t_target, t_wrist[0], t_wrist[-1])
    interp_rots = slerp(t_rot_query)
    n_out = len(t_target)
    for i in range(n_out):
        R = interp_rots[i] if n_out > 1 else interp_rots
        T_out[i, :3, :3] = R.as_matrix()

    if len(t_target) == 1:
        return T_out[0]
    return T_out

def pose_7_to_matrix(p):
    """(7,) -> 4x4: x,y,z,qw,qx,qy,qz"""
    T = np.eye(4)
    T[:3, 3] = p[:3]
    w, x, y, z = p[3], p[4], p[5], p[6]
    T[0, 0] = 1 - 2 * (y*y + z*z)
    T[0, 1] = 2 * (x*y - z*w)
    T[0, 2] = 2 * (x*z + y*w)
    T[1, 0] = 2 * (x*y + z*w)
    T[1, 1] = 1 - 2 * (x*x + z*z)
    T[1, 2] = 2 * (y*z - x*w)
    T[2, 0] = 2 * (x*z - y*w)
    T[2, 1] = 2 * (y*z + x*w)
    T[2, 2] = 1 - 2 * (x*x + y*y)
    return T

def interpolate_hand_poses(t_target, t_hand, poses_list):
    """
    对手部关节 pose 插值。每个关节: position(3) + quat(4) = 7
    poses_list: list of (n_joints, 7) arrays
    t_hand: (N,) 时间戳
    """
    t_target = np.atleast_1d(t_target)
    t_hand = np.atleast_1d(np.asarray(t_hand, dtype=np.float64))
    if len(poses_list) == 0 or t_hand.size == 0:
        raise ValueError("hand_pose 序列为空，无法对齐")
    n_joints = poses_list[0].shape[0]
    poses_stack = np.array(poses_list)  # (N, n_joints, 7)

    out = np.zeros((len(t_target), n_joints, 7), dtype=np.float32)

    if t_hand.size == 1:
        p0 = poses_stack[0].astype(np.float32)
        for i in range(len(t_target)):
            out[i] = p0
        return out[0] if len(t_target) == 1 else out

    for j in range(n_joints):
        pos = poses_stack[:, j, :3]   # (N, 3)
        quat = poses_stack[:, j, 3:7] # (N, 4) w,x,y,z

        # position: 线性插值
        for k in range(3):
            f = interp1d(t_hand, pos[:, k], kind='linear',
                         bounds_error=False, fill_value=(pos[0, k], pos[-1, k]))
            out[:, j, k] = f(t_target)

        # quaternion: Slerp，需要 (x,y,z,w) 给 Rotation
        q_xyzw = np.column_stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]])
        rots = Rotation.from_quat(q_xyzw)
        slerp = Slerp(t_hand, rots)
        t_rot_query = np.clip(t_target, t_hand[0], t_hand[-1])
        interp_rots = slerp(t_rot_query)
        n_out = len(t_target)
        for i in range(n_out):
            R = interp_rots[i] if n_out > 1 else interp_rots
            q = R.as_quat()  # xyzw
            out[i, j, 3:7] = [q[3], q[0], q[1], q[2]]  # 转回 w,x,y,z

    if len(t_target) == 1:
        return out[0]
    return out


def _load_side_cube_stream(raw, side: str, rs_ts: np.ndarray):
    """
    读取单手相机系正方体位姿。

    返回:
        ts_all: 与返回位姿对应的时间戳
        T_all: (N,4,4)
        pose_valid: (N,) bool
        source: 'cube' | 'wrist' | 'none'
    """
    if side not in ("right", "left"):
        raise ValueError(side)
    cube_grp, cube_key = f"{side}_cube_to_cam", f"T_{side}_cube_to_cam"
    if cube_grp in raw:
        g = raw[cube_grp]
        if "timestamps" in g and cube_key in g:
            ts = np.array(g["timestamps"], dtype=np.float64)
            T = np.array(g[cube_key], dtype=np.float64)
            if "pose_valid" in g:
                pose_valid = np.array(g["pose_valid"], dtype=bool)
            else:
                pose_valid = np.ones((len(ts),), dtype=bool)
            return ts, T, pose_valid, "cube"

    w_grp, w_key = f"{side}_wrist_to_cam", f"T_{side}_wrist_to_cam"
    for grp, dname, src in ((w_grp, w_key, "wrist"),):
        if grp not in raw:
            continue
        g = raw[grp]
        if "timestamps" not in g or dname not in g:
            continue
        ts = np.array(g["timestamps"], dtype=np.float64)
        T = np.array(g[dname], dtype=np.float64)
        pose_valid = np.ones((len(ts),), dtype=bool)
        return ts, T, pose_valid, src
    z = np.zeros((0,), dtype=np.float64)
    Z = np.zeros((0, 4, 4), dtype=np.float64)
    valid = np.zeros((0,), dtype=bool)
    return z, Z, valid, "none"


def fill_pose_gaps(ts_all, T_all, pose_valid, max_interp_gap_frames=5):
    """
    对逐帧 cube pose 填补缺口帧。

    规则：
    - 短缺口（连续缺失帧数 <= max_interp_gap_frames）且前后都有有效观测时：插值
    - 长缺口（连续缺失帧数 > max_interp_gap_frames）：认为物体已离开视野，继承最近历史有效位姿
    - 起始缺口因没有历史帧，使用首个有效位姿常值填充

    返回:
        T_filled: (N,4,4)
        pose_filled: (N,) bool，True 表示该帧由插值/外推得到
        pose_available: (N,) bool，True 表示最终该帧存在可用位姿
    """
    ts_all = np.asarray(ts_all, dtype=np.float64)
    T_all = np.asarray(T_all, dtype=np.float64)
    pose_valid = np.asarray(pose_valid, dtype=bool)
    n = len(ts_all)
    if n == 0:
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
        )

    if len(T_all) != n:
        raise ValueError("T_all 与 ts_all 长度不一致")

    if not np.any(pose_valid):
        T_out = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n, axis=0)
        return T_out, np.zeros((n,), dtype=bool), np.zeros((n,), dtype=bool)

    T_out = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n, axis=0)
    pose_filled = np.zeros((n,), dtype=bool)
    pose_available = np.zeros((n,), dtype=bool)

    valid_idx = np.flatnonzero(pose_valid)
    T_out[valid_idx] = T_all[valid_idx].astype(np.float32)
    pose_available[valid_idx] = True

    first_valid = int(valid_idx[0])
    last_valid = int(valid_idx[-1])

    # 起始缺口：没有历史帧时，只能用首个有效位姿常值填充
    if first_valid > 0:
        T_out[:first_valid] = T_all[first_valid].astype(np.float32)
        pose_filled[:first_valid] = True
        pose_available[:first_valid] = True

    # 末尾缺口：继承最近历史有效位姿
    if last_valid < n - 1:
        T_out[last_valid + 1:] = T_all[last_valid].astype(np.float32)
        pose_filled[last_valid + 1:] = True
        pose_available[last_valid + 1:] = True

    for prev_idx, next_idx in zip(valid_idx[:-1], valid_idx[1:]):
        gap_len = int(next_idx - prev_idx - 1)
        if gap_len <= 0:
            continue

        gap_slice = slice(prev_idx + 1, next_idx)
        pose_filled[gap_slice] = True
        pose_available[gap_slice] = True

        if gap_len <= int(max_interp_gap_frames):
            t_gap = ts_all[gap_slice]
            T_gap = interpolate_T_4x4(
                t_gap,
                np.array([ts_all[prev_idx], ts_all[next_idx]], dtype=np.float64),
                np.stack([T_all[prev_idx], T_all[next_idx]], axis=0),
            )
            if T_gap.ndim == 2:
                T_gap = T_gap[None, :, :]
            T_out[gap_slice] = np.asarray(T_gap, dtype=np.float32)
        else:
            # 长缺口：直接保持上一帧，避免跨出视野区间做不合理插值
            T_out[gap_slice] = T_all[prev_idx].astype(np.float32)

    return T_out, pose_filled, pose_available


def collect_sensor_names(sensor_data_list):
    """收集所有出现过的传感器名，保持首次出现顺序"""
    sensor_names = []
    for d in sensor_data_list:
        for name in d:
            if name not in sensor_names:
                sensor_names.append(name)
    return sensor_names



DEFAULT_SENSOR_PARSE_ORDER = [
    ["大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲"],
    ["食指近节", "食指中节", "食指指尖", "食指指甲"],
    ["中指近节", "中指中节", "中指指尖", "中指指甲"],
    ["无名指近节", "无名指中节", "无名指指尖", "无名指指甲"],
    ["小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲"],
    ["掌心1", "掌心2", "掌心3", "掌心4", "掌心5", "掌心6", "掌心7", "掌心8"],
]
DEFAULT_CONNECTED_SENSORS = [s for g in DEFAULT_SENSOR_PARSE_ORDER for s in g]


def _load_rotation_matrices(rotation_matrix_path):
    if not rotation_matrix_path or not os.path.exists(rotation_matrix_path):
        return {}
    try:
        with open(rotation_matrix_path, "r", encoding="utf-8") as f:
            content = f.read().strip().replace(";", "")
        matrix_list = ast.literal_eval(content)
        mats = {}
        for idx, flat_values in enumerate(matrix_list, start=1):
            if len(flat_values) == 9:
                mats[idx] = np.array(flat_values, dtype=np.float64).reshape(3, 3)
        return mats
    except Exception:
        return {}


def _parse_coord_file(coord_file):
    if not coord_file or not os.path.exists(coord_file):
        return {}
    coords = {}
    try:
        if coord_file.endswith(".xlsx") or coord_file.endswith(".xls"):
            df = pd.read_excel(coord_file, sheet_name="Sheet1")
            for _, row in df.iterrows():
                idx = int(row.iloc[0])
                coords[idx] = {"X": float(row["X"]), "Y": float(row["Y"]), "Z": float(row["Z"])}
            return coords
        if coord_file.endswith(".json"):
            with open(coord_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for k, v in payload.items():
                idx = int(k)
                coords[idx] = {"X": float(v["X"]), "Y": float(v["Y"]), "Z": float(v["Z"])}
            return coords
        with open(coord_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            coords[idx] = {"X": float(parts[1]), "Y": float(parts[2]), "Z": float(parts[3])}
    except Exception:
        return {}
    return coords


def _load_sensor_points_map(sensor_points_file):
    if not sensor_points_file or not os.path.exists(sensor_points_file):
        return {}
    try:
        with open(sensor_points_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out = {}
        for k, v in payload.items():
            out[str(k)] = int(v)
        return out
    except Exception:
        return {}


def _to_signed(byte_val):
    return byte_val if byte_val <= 127 else byte_val - 256


def _parse_total_force_3bytes(force_data):
    fx = _to_signed(force_data[0])
    fy = _to_signed(force_data[2])
    fz = force_data[4]
    return {"Fx": round(fx * 0.1, 1), "Fy": round(fy * 0.1, 1), "Fz": round(fz * 0.1, 1)}


def _parse_distribution_force(dist_data):
    out = []
    for i in range(0, len(dist_data), 3):
        if i + 2 >= len(dist_data):
            break
        out.append({
            "X": round(_to_signed(dist_data[i]) * 0.1, 1),
            "Y": round(_to_signed(dist_data[i + 1]) * 0.1, 1),
            "Z": round(dist_data[i + 2] * 0.1, 1),
        })
    return out


def _torque_from_distribution(distribution, coord_map, rotation_matrices):
    tx, ty, tz = 0.0, 0.0, 0.0
    for i, local_point in enumerate(distribution):
        point_idx = i + 1
        point = local_point
        if point_idx in rotation_matrices:
            vec = np.array([point["X"], point["Y"], point["Z"]], dtype=np.float64)
            gvec = rotation_matrices[point_idx] @ vec
            point = {"X": float(gvec[0]), "Y": float(gvec[1]), "Z": float(gvec[2])}
        coord = coord_map.get(point_idx)
        if coord is None:
            continue
        x, y, z = coord["X"], coord["Y"], coord["Z"]
        fx, fy, fz = point["X"], point["Y"], point["Z"]
        tx += y * fz - z * fy
        ty += z * fx - x * fz
        tz += x * fy - y * fx
    return {"Tx": round(tx, 5), "Ty": round(ty, 5), "Tz": round(tz, 5)}


def _decode_tactile_frame(raw_frame_hex, connected_sensors, points_map, coord_map, rotation_matrices):
    try:
        frame = bytes.fromhex(raw_frame_hex)
    except Exception:
        return {}
    if len(frame) < 8 or frame[0:2] != b"\xAA\x56":
        return {}
    frame_len = int.from_bytes(frame[3:5], "little")
    expected_len = 2 + 1 + 2 + frame_len + 1
    if len(frame) < expected_len:
        return {}
    payload = frame[6:expected_len - 1]

    offset = 0
    result = {}
    for group in DEFAULT_SENSOR_PARSE_ORDER:
        for sensor in group:
            if sensor not in connected_sensors:
                continue
            if offset + 6 > len(payload):
                return result
            force_data = payload[offset:offset + 6]
            offset += 6
            total_force = _parse_total_force_3bytes(force_data)

            points = int(points_map.get(sensor, 0))
            points = min(points, 9) if sensor.startswith("掌心") else points
            distribution = []
            if points > 0 and offset + points * 3 <= len(payload):
                dist_data = payload[offset:offset + points * 3]
                offset += points * 3
                distribution = _parse_distribution_force(dist_data)
            torque = _torque_from_distribution(distribution, coord_map, rotation_matrices)
            result[sensor] = {
                "total_force": total_force,
                "distribution_force": distribution,
                "torque": torque,
            }
    return result


def decode_sensor_entries(sensor_data_json, coord_file=None, rotation_matrix_file=None, sensor_points_file=None):
    default_connected = DEFAULT_CONNECTED_SENSORS
    default_points = {name: 0 for name in default_connected}
    default_points.update(_load_sensor_points_map(sensor_points_file))
    coord_map = _parse_coord_file(coord_file)
    rotation_matrices = _load_rotation_matrices(rotation_matrix_file)

    out = []
    for s in sensor_data_json:
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict) and "raw_frame_hex" in parsed:
            connected_sensors = parsed.get("sensors", default_connected)
            if not isinstance(connected_sensors, list):
                connected_sensors = default_connected
            points_map = dict(default_points)
            points_from_msg = parsed.get("points", {})
            if isinstance(points_from_msg, dict):
                for k, v in points_from_msg.items():
                    try:
                        points_map[str(k)] = int(v)
                    except Exception:
                        pass
            decoded = _decode_tactile_frame(
                parsed.get("raw_frame_hex", ""),
                connected_sensors,
                points_map,
                coord_map,
                rotation_matrices,
            )
            out.append(decoded if isinstance(decoded, dict) else {})
        elif isinstance(parsed, dict):
            # 兼容 ros_pub_3threads.py 中 key 为 raw/sensors/points 的 JSON
            if "raw" in parsed and isinstance(parsed["raw"], str):
                wrapped = {
                    "raw_frame_hex": parsed.get("raw", ""),
                    "sensors": parsed.get("sensors", default_connected),
                    "points": parsed.get("points", {}),
                }
                connected_sensors = wrapped.get("sensors", default_connected)
                if not isinstance(connected_sensors, list):
                    connected_sensors = default_connected
                points_map = dict(default_points)
                points_from_msg = wrapped.get("points", {})
                if isinstance(points_from_msg, dict):
                    for k, v in points_from_msg.items():
                        try:
                            points_map[str(k)] = int(v)
                        except Exception:
                            pass
                decoded = _decode_tactile_frame(
                    wrapped.get("raw_frame_hex", ""),
                    connected_sensors,
                    points_map,
                    coord_map,
                    rotation_matrices,
                )
                out.append(decoded if isinstance(decoded, dict) else {})
                continue
            out.append(parsed)
        else:
            out.append({})
    return out


def align_raw_tactile_frames_nearest(t_target, t_source, tactile_data_json):
    t_target = np.atleast_1d(t_target)
    if len(t_source) == 0 or len(tactile_data_json) == 0:
        return []
    idx = nearest_indices(t_target, t_source)
    aligned_json = np.array(tactile_data_json, dtype=object)[idx]
    return [str(x) for x in aligned_json]


def parse_aligned_tactile_frames(aligned_tactile_json, coord_file=None, rotation_matrix_file=None, sensor_points_file=None):
    parsed = decode_sensor_entries(
        aligned_tactile_json,
        coord_file=coord_file,
        rotation_matrix_file=rotation_matrix_file,
        sensor_points_file=sensor_points_file
    )
    sensor_names = collect_sensor_names(parsed)
    n_frames = len(parsed)
    n_sensors = len(sensor_names)
    wrench = np.zeros((n_frames, n_sensors, 6), dtype=np.float32)
    dist_json = []
    for i, frame in enumerate(parsed):
        dist_only = {}
        for j, name in enumerate(sensor_names):
            entry = frame.get(name, {})
            tf = entry.get("total_force", {})
            tq = entry.get("torque", {})
            wrench[i, j, 0] = tf.get("Fx", 0.0)
            wrench[i, j, 1] = tf.get("Fy", 0.0)
            wrench[i, j, 2] = tf.get("Fz", 0.0)
            wrench[i, j, 3] = tq.get("Tx", 0.0)
            wrench[i, j, 4] = tq.get("Ty", 0.0)
            wrench[i, j, 5] = tq.get("Tz", 0.0)
            dist_only[name] = entry.get("distribution_force", [])
        dist_json.append(json.dumps(dist_only, ensure_ascii=False))
    return wrench, sensor_names, dist_json


def build_raw_tactile_bundle(
    timestamps,
    raw_json,
    source_key,
    coord_file=None,
    rotation_file=None,
    sensor_points_file=None,
    time_offset_ms=0.0,
):
    """
    Preserve tactile samples at their native frame rate.

    `timestamps` stays exactly as recorded in raw data.  A separate
    `timestamps_realsense_aligned` is stored when a time offset is supplied so
    segment slicing can use the same compensated time base without destroying
    the original timestamps.
    """
    if timestamps is None or raw_json is None or len(timestamps) == 0 or len(raw_json) == 0:
        return None
    ts_raw = np.asarray(timestamps, dtype=np.float64)
    raw_list = [str(x) for x in np.asarray(raw_json, dtype=object)]
    n = min(len(ts_raw), len(raw_list))
    ts_raw = ts_raw[:n]
    raw_list = raw_list[:n]
    wrench, sensor_names, dist_json = parse_aligned_tactile_frames(
        raw_list,
        coord_file=coord_file,
        rotation_matrix_file=rotation_file,
        sensor_points_file=sensor_points_file,
    )
    ts_aligned = ts_raw + float(time_offset_ms) / 1000.0
    return {
        "timestamps": ts_raw,
        "timestamps_realsense_aligned": ts_aligned,
        "raw_json": raw_list,
        "wrench": wrench,
        "distribution_json": dist_json,
        "sensor_names": sensor_names,
        "source_key": str(source_key),
    }


def slice_tactile_bundle(bundle, start_ts, end_ts):
    if bundle is None:
        return None
    ts_for_slice = np.asarray(bundle["timestamps_realsense_aligned"], dtype=np.float64)
    mask = (ts_for_slice >= float(start_ts)) & (ts_for_slice <= float(end_ts))
    idx = np.flatnonzero(mask)
    return {
        "timestamps": np.asarray(bundle["timestamps"], dtype=np.float64)[idx],
        "timestamps_realsense_aligned": ts_for_slice[idx],
        "raw_json": [bundle["raw_json"][int(i)] for i in idx],
        "wrench": np.asarray(bundle["wrench"])[idx],
        "distribution_json": [bundle["distribution_json"][int(i)] for i in idx],
        "sensor_names": list(bundle["sensor_names"]),
        "source_key": bundle.get("source_key", ""),
    }


def nearest_indices(t_target, t_source):
    """
    为每个目标时间戳找到最近的源时间戳索引。
    要求 t_source 单调递增（采集时间戳一般满足）。
    """
    t_target = np.asarray(t_target, dtype=np.float64)
    t_source = np.asarray(t_source, dtype=np.float64)
    if len(t_source) == 0:
        return np.array([], dtype=np.int64)
    idx_right = np.searchsorted(t_source, t_target, side='left')
    idx_right = np.clip(idx_right, 0, len(t_source) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(t_source) - 1)
    choose_left = np.abs(t_target - t_source[idx_left]) <= np.abs(t_target - t_source[idx_right])
    return np.where(choose_left, idx_left, idx_right).astype(np.int64)


def align_sensor_wrench_nearest(t_target, t_sensor, sensor_data_list):
    """
    对传感器合力和力矩做最近邻对齐（不插值）。
    sensor_data_list: list of dicts
        {sensor_name: {"total_force": {"Fx","Fy","Fz"}, "torque": {"Tx","Ty","Tz"}}}
    Returns:
        aligned_data : (n_target, n_sensors, 6) float32, 顺序 [Fx,Fy,Fz,Tx,Ty,Tz]
        sensor_names: list[str]，与第二维对应
    """
    sensor_names = collect_sensor_names(sensor_data_list)

    n = len(t_sensor)
    n_sensors = len(sensor_names)

    # 构建原始时间序列 (N, n_sensors, 6)
    raw_values = np.zeros((n, n_sensors, 6), dtype=np.float32)
    for i, d in enumerate(sensor_data_list):
        for j, name in enumerate(sensor_names):
            if name not in d:
                continue
            tf = d[name].get("total_force", {})
            tq = d[name].get("torque", {})
            raw_values[i, j, 0] = tf.get("Fx", 0.0)
            raw_values[i, j, 1] = tf.get("Fy", 0.0)
            raw_values[i, j, 2] = tf.get("Fz", 0.0)
            raw_values[i, j, 3] = tq.get("Tx", 0.0)
            raw_values[i, j, 4] = tq.get("Ty", 0.0)
            raw_values[i, j, 5] = tq.get("Tz", 0.0)

    t_target = np.atleast_1d(t_target)
    idx = nearest_indices(t_target, t_sensor)
    aligned_data = raw_values[idx]
    return aligned_data, sensor_names


def align_sensor_distribution_nearest(t_target, t_sensor, sensor_data_list, sensor_names):
    """
    将 distribution_force 用最近邻时间戳对齐到 t_target。
    返回:
        aligned_distribution_json: list[str]，每帧一个 json 字符串
        格式: {sensor_name: [{"X":..,"Y":..,"Z":..}, ...], ...}
    """
    t_target = np.atleast_1d(t_target)
    idx_all = nearest_indices(t_target, t_sensor)
    aligned_distribution_json = []
    for idx in idx_all:
        src = sensor_data_list[int(idx)]
        frame_dist = {}
        for name in sensor_names:
            sensor_val = src.get(name, {})
            frame_dist[name] = sensor_val.get("distribution_force", [])
        aligned_distribution_json.append(json.dumps(frame_dist, ensure_ascii=False))
    return aligned_distribution_json


# ===============================
# 主处理流程
# ===============================
def save_tactile_bundle(out_root, side, bundle):
    if bundle is None:
        return
    grp = out_root.create_group(f"{side}_tactile")
    timestamps = np.asarray(bundle["timestamps"], dtype=np.float64)
    timestamps_aligned = np.asarray(bundle["timestamps_realsense_aligned"], dtype=np.float64)
    wrench = np.asarray(bundle["wrench"], dtype=np.float32)
    grp.create_dataset("timestamps", data=timestamps, dtype=np.float64)
    grp.create_dataset("timestamps_realsense_aligned", data=timestamps_aligned, dtype=np.float64)
    grp.create_dataset(
        "raw_json",
        data=np.array(bundle["raw_json"], dtype=object),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
    )
    grp.create_dataset("wrench", data=wrench)
    if wrench.ndim == 3 and wrench.shape[-1] >= 6:
        grp.create_dataset("total_force", data=wrench[:, :, :3])
        grp.create_dataset("torque", data=wrench[:, :, 3:6])
    grp.create_dataset(
        "distribution_force_json",
        data=np.array(bundle["distribution_json"], dtype=object),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
    )
    grp.attrs["sensor_names_json"] = json.dumps(bundle["sensor_names"], ensure_ascii=False)
    grp.attrs["source_key"] = bundle.get("source_key", "")
    grp.attrs["aligned_to_realsense_frames"] = False
    grp.attrs["wrench_order"] = "Fx,Fy,Fz,Tx,Ty,Tz"


def save_aligned_zarr(
    output_path,
    root,
    rs_ts,
    valid_rgb,
    valid_depth,
    T_right_cube_to_cam_interp,
    right_cube_pose_valid,
    right_cube_pose_filled,
    right_cube_pose_available,
    T_left_cube_to_cam_interp,
    left_cube_pose_valid,
    left_cube_pose_filled,
    left_cube_pose_available,
    right_hand_poses_in_wrist_interp,
    left_hand_poses_in_wrist_interp,
    left_tactile_bundle,
    right_tactile_bundle,
    mvs_left_aligned_rgb,
    mvs_right_aligned_rgb,
    segment_info=None
):
    out_root = zarr.open(output_path, mode='w')
    data = out_root.create_group('data')

    for i in range(len(rs_ts)):
        key = f'frame_{i:05d}'
        grp = data.create_group(key)
        grp.create_dataset('timestamp', data=np.float64(rs_ts[i]))
        grp.create_dataset('realsense_timestamp', data=np.float64(rs_ts[i]))
        grp.create_dataset('rgb', data=valid_rgb[i])
        grp.create_dataset('depth', data=valid_depth[i])
        grp.create_dataset('T_right_cube_to_cam', data=T_right_cube_to_cam_interp[i])
        grp.create_dataset('right_cube_pose_valid', data=bool(right_cube_pose_valid[i]), dtype=bool)
        grp.create_dataset('right_cube_pose_filled', data=bool(right_cube_pose_filled[i]), dtype=bool)
        grp.create_dataset('right_cube_pose_available', data=bool(right_cube_pose_available[i]), dtype=bool)
        grp.create_dataset('T_left_cube_to_cam', data=T_left_cube_to_cam_interp[i])
        grp.create_dataset('left_cube_pose_valid', data=bool(left_cube_pose_valid[i]), dtype=bool)
        grp.create_dataset('left_cube_pose_filled', data=bool(left_cube_pose_filled[i]), dtype=bool)
        grp.create_dataset('left_cube_pose_available', data=bool(left_cube_pose_available[i]), dtype=bool)

        right_finger_to_wrist_list = []
        for j in range(5):
            T_f_to_w = pose_7_to_matrix(right_hand_poses_in_wrist_interp[i][j])
            T_f_to_w = T_r_l @ T_f_to_w @ T_r_l
            right_finger_to_wrist_list.append(T_f_to_w)
        grp.create_dataset('T_right_finger_to_wrist', data=np.array(right_finger_to_wrist_list))

        left_finger_to_wrist_list = []
        for j in range(5):
            T_f_to_w_l = pose_7_to_matrix(left_hand_poses_in_wrist_interp[i][j])
            T_f_to_w_l = T_r_l @ T_f_to_w_l @ T_r_l
            left_finger_to_wrist_list.append(T_f_to_w_l)
        grp.create_dataset('T_left_finger_to_wrist', data=np.array(left_finger_to_wrist_list))
        if mvs_left_aligned_rgb is not None:
            grp.create_dataset('mvs_left_rgb', data=mvs_left_aligned_rgb[i])
        if mvs_right_aligned_rgb is not None:
            grp.create_dataset('mvs_right_rgb', data=mvs_right_aligned_rgb[i])

    out_root.create_dataset('intrinsic', data=root['intrinsic'][:])
    out_root.create_dataset('realsense_timestamps', data=np.asarray(rs_ts, dtype=np.float64))
    out_root.attrs['depth_scale'] = float(root.attrs.get('depth_scale', 1.0))
    out_root.attrs['n_frames'] = len(rs_ts)
    out_root.attrs['tactile_wrench_order'] = "Fx,Fy,Fz,Tx,Ty,Tz"
    save_tactile_bundle(out_root, "left", left_tactile_bundle)
    save_tactile_bundle(out_root, "right", right_tactile_bundle)
    if segment_info is not None:
        out_root.attrs['segment_id'] = int(segment_info['segment_id'])
        out_root.attrs['segment_start_ts'] = float(segment_info['start_ts'])
        out_root.attrs['segment_end_ts'] = float(segment_info['end_ts'])


def build_segment_ranges(raw, t_min, t_max):
    """
    从 raw/segments 构建有效分段范围。
    若没有 segments，则返回单段覆盖整个有效区间。
    """
    segments = []
    if 'segments' in raw:
        seg_group = raw['segments']
        seg_ids = np.array(seg_group['segment_id'])
        seg_starts = np.array(seg_group['start_timestamps'], dtype=np.float64)
        seg_ends = np.array(seg_group['end_timestamps'], dtype=np.float64)
        for sid, st, ed in zip(seg_ids, seg_starts, seg_ends):
            seg_end = t_max if np.isnan(ed) else float(ed)
            start = max(float(st), float(t_min))
            end = min(float(seg_end), float(t_max))
            if end >= start:
                segments.append({
                    'segment_id': int(sid),
                    'start_ts': start,
                    'end_ts': end,
                })
    if not segments:
        segments.append({
            'segment_id': 1,
            'start_ts': float(t_min),
            'end_ts': float(t_max),
        })
    return segments


def process(
    raw_zarr_path,
    output_zarr_path,
    time_offset_ms=0.0,
    coord_file=None,
    rotation_file=None,
    sensor_points_file=None,
    cube_hold_threshold_frames=5
):

    """
    读取原始 zarr，以 RGBD 时间戳为基准，对齐 T_cube_to_cam、hand_pose 和 sensor 合力/合力矩，输出对齐数据。
    time_offset_ms: 可选的全局时间偏移（毫秒），用于补偿设备间时钟差。
    """

    root = zarr.open(raw_zarr_path, mode='r')
    raw = root['raw']

    rs_ts = np.array(raw['realsense']['timestamps'], dtype=np.float64)
    rs_rgb = np.array(raw['realsense']['rgb'])
    rs_depth = np.array(raw['realsense']['depth'])

    rt_ts, rt_T, rt_valid, rt_src = _load_side_cube_stream(raw, "right", rs_ts)
    lt_ts, lt_T, lt_valid, lt_src = _load_side_cube_stream(raw, "left", rs_ts)
    if rt_src == "wrist":
        print("[Info] 右手使用旧版 raw/right_wrist_to_cam（语义上视为 cube pose）")
    if lt_src == "wrist":
        print("[Info] 左手使用旧版 raw/left_wrist_to_cam（语义上视为 cube pose）")

    rhp_ts = np.array(raw['right_hand_pose_in_wrist']['timestamps'], dtype=np.float64)
    rhp_poses = np.array(raw['right_hand_pose_in_wrist']['poses_in_wrist'])
    lhp_ts = np.array(raw['left_hand_pose_in_wrist']['timestamps'], dtype=np.float64)
    lhp_poses = np.array(raw['left_hand_pose_in_wrist']['poses_in_wrist'])

    # MVS 双相机（若存在）
    has_mvs_left = 'mvs_left_camera' in raw
    has_mvs_right = 'mvs_right_camera' in raw
    mvs_left_ts = mvs_left_rgb = None
    mvs_right_ts = mvs_right_rgb = None
    if has_mvs_left:
        mvs_left_ts = np.array(raw['mvs_left_camera']['timestamps'], dtype=np.float64)
        mvs_left_rgb = np.array(raw['mvs_left_camera']['rgb'])
        print(f"MVS Left 采样数: {len(mvs_left_ts)}")
    if has_mvs_right:
        mvs_right_ts = np.array(raw['mvs_right_camera']['timestamps'], dtype=np.float64)
        mvs_right_rgb = np.array(raw['mvs_right_camera']['rgb'])
        print(f"MVS Right 采样数: {len(mvs_right_ts)}")

    # 触觉双手（若存在）
    has_left_tactile = ('left_tactile' in raw) or ('sensor_0' in raw)
    has_right_tactile = ('right_tactile' in raw) or ('sensor_1' in raw)
    left_tactile_ts = left_tactile_ts_raw = left_tactile_raw_json = None
    right_tactile_ts = right_tactile_ts_raw = right_tactile_raw_json = None
    if coord_file is None:
        coord_file = str(Path(__file__).resolve().parents[1] / 'assets/tactile/sensor.xlsx')
    if rotation_file is None:
        rotation_file = os.path.join(os.path.dirname(__file__), '..', 'assets', 'tactile', 'S2015-Elite+Rotation+Matrix.h')
    if sensor_points_file is None:
        sensor_points_file = None
    if not os.path.exists(coord_file):
        print(f"[Warn] 未找到坐标文件: {coord_file}，sensor torque 可能为 0")
    if sensor_points_file and (not os.path.exists(sensor_points_file)):
        print(f"[Warn] 未找到点数文件: {sensor_points_file}，将优先使用 ROS 消息中的 points")
    if has_left_tactile:
        left_key = 'left_tactile' if 'left_tactile' in raw else 'sensor_0'
        left_tactile_ts_raw = np.array(raw[left_key]['timestamps'], dtype=np.float64)
        left_tactile_ts = left_tactile_ts_raw.copy()
        left_tactile_raw_json = np.array(raw[left_key]['data'], dtype=object)
    if has_right_tactile:
        right_key = 'right_tactile' if 'right_tactile' in raw else 'sensor_1'
        right_tactile_ts_raw = np.array(raw[right_key]['timestamps'], dtype=np.float64)
        right_tactile_ts = right_tactile_ts_raw.copy()
        right_tactile_raw_json = np.array(raw[right_key]['data'], dtype=object)

    # 时间偏移（秒）
    offset = time_offset_ms / 1000.0
    if offset != 0:
        rt_ts = rt_ts + offset
        lt_ts = lt_ts + offset
        rhp_ts = rhp_ts + offset
        lhp_ts = lhp_ts + offset
        if has_left_tactile:
            left_tactile_ts = left_tactile_ts + offset
        if has_right_tactile:
            right_tactile_ts = right_tactile_ts + offset

    n_frames = len(rs_ts)
    print(f"RGBD 帧数: {n_frames}")
    print(f"右手 cube pose 采样数: {int(np.sum(rt_valid))}/{len(rt_ts)}  (source={rt_src})")
    print(f"左手 cube pose 采样数: {int(np.sum(lt_valid))}/{len(lt_ts)}  (source={lt_src})")
    print(f"Right_hand_pose 采样数: {len(rhp_ts)}")
    print(f"Left_hand_pose 采样数: {len(lhp_ts)}")
    if has_left_tactile:
        print(f"Left_tactile 采样数: {len(left_tactile_ts)}")
    if has_right_tactile:
        print(f"Right_tactile 采样数: {len(right_tactile_ts)}")

    # 使用完整 RealSense RGB 时间轴作为全局目标时间轴。tactile 不参与
    # 裁剪、不重采样；后面按 segment 直接保存原始 tactile 帧。
    t_min = float(rs_ts.min())
    t_max = float(rs_ts.max())
    mask = np.ones_like(rs_ts, dtype=bool)
    valid_rs_ts = rs_ts[mask]
    valid_rgb = rs_rgb[mask]
    valid_depth = rs_depth[mask]

    if len(rhp_ts) < 1 or len(lhp_ts) < 1:
        print("Hand_pose 无样本，无法对齐")
        return
    if len(rhp_ts) < 2 or len(lhp_ts) < 2:
        print("[Warn] Hand_pose 某一侧少于 2 个样本，该侧将用常值填充整段 RGBD")
    if not np.any(rt_valid):
        print("[Warn] 右手 cube pose 整段均无检测，最终写入单位阵并标记 unavailable")
    if not np.any(lt_valid):
        print("[Warn] 左手 cube pose 整段均无检测，最终写入单位阵并标记 unavailable")

    # 新版 raw 为逐帧保存：直接按 RGBD 帧填补缺口；旧版稀疏流则退化为按时间重采样。
    if rt_src == "cube" and len(rt_ts) == len(rs_ts):
        rt_masked_valid = rt_valid[mask]
        rt_masked_T = rt_T[mask]
        T_right_cube_to_cam_interp, right_cube_pose_filled, right_cube_pose_available = fill_pose_gaps(
            valid_rs_ts, rt_masked_T, rt_masked_valid, max_interp_gap_frames=cube_hold_threshold_frames
        )
        right_cube_pose_valid = rt_masked_valid.astype(bool)
    else:
        T_right_cube_to_cam_interp = interpolate_T_4x4(valid_rs_ts, rt_ts[rt_valid], rt_T[rt_valid])
        if T_right_cube_to_cam_interp.ndim == 2:
            T_right_cube_to_cam_interp = T_right_cube_to_cam_interp[None, :, :]
        right_cube_pose_valid = np.zeros((len(valid_rs_ts),), dtype=bool)
        right_cube_pose_filled = np.ones((len(valid_rs_ts),), dtype=bool)
        right_cube_pose_available = np.ones((len(valid_rs_ts),), dtype=bool) if np.any(rt_valid) else np.zeros((len(valid_rs_ts),), dtype=bool)

    if lt_src == "cube" and len(lt_ts) == len(rs_ts):
        lt_masked_valid = lt_valid[mask]
        lt_masked_T = lt_T[mask]
        T_left_cube_to_cam_interp, left_cube_pose_filled, left_cube_pose_available = fill_pose_gaps(
            valid_rs_ts, lt_masked_T, lt_masked_valid, max_interp_gap_frames=cube_hold_threshold_frames
        )
        left_cube_pose_valid = lt_masked_valid.astype(bool)
    else:
        T_left_cube_to_cam_interp = interpolate_T_4x4(valid_rs_ts, lt_ts[lt_valid], lt_T[lt_valid])
        if T_left_cube_to_cam_interp.ndim == 2:
            T_left_cube_to_cam_interp = T_left_cube_to_cam_interp[None, :, :]
        left_cube_pose_valid = np.zeros((len(valid_rs_ts),), dtype=bool)
        left_cube_pose_filled = np.ones((len(valid_rs_ts),), dtype=bool)
        left_cube_pose_available = np.ones((len(valid_rs_ts),), dtype=bool) if np.any(lt_valid) else np.zeros((len(valid_rs_ts),), dtype=bool)

    right_hand_poses_in_wrist_interp = interpolate_hand_poses(valid_rs_ts, rhp_ts, list(rhp_poses))
    left_hand_poses_in_wrist_interp = interpolate_hand_poses(valid_rs_ts, lhp_ts, list(lhp_poses))

    # Tactile 不对齐、不插值、不降采样；保留原始帧率和原始时间戳。
    left_tactile_bundle = None
    if has_left_tactile:
        left_tactile_bundle = build_raw_tactile_bundle(
            left_tactile_ts_raw,
            left_tactile_raw_json,
            source_key=left_key,
            coord_file=coord_file,
            rotation_file=rotation_file,
            sensor_points_file=sensor_points_file,
            time_offset_ms=time_offset_ms,
        )
        if left_tactile_bundle is not None:
            print(
                f"Left_tactile 保留原始帧: {len(left_tactile_bundle['timestamps'])} 帧, "
                f"{len(left_tactile_bundle['sensor_names'])} 个传感器"
            )

    right_tactile_bundle = None
    if has_right_tactile:
        right_tactile_bundle = build_raw_tactile_bundle(
            right_tactile_ts_raw,
            right_tactile_raw_json,
            source_key=right_key,
            coord_file=coord_file,
            rotation_file=rotation_file,
            sensor_points_file=sensor_points_file,
            time_offset_ms=time_offset_ms,
        )
        if right_tactile_bundle is not None:
            print(
                f"Right_tactile 保留原始帧: {len(right_tactile_bundle['timestamps'])} 帧, "
                f"{len(right_tactile_bundle['sensor_names'])} 个传感器"
            )

    # MVS 双相机：最近邻对齐到 RGBD 时间戳
    mvs_left_aligned_rgb = None
    if has_mvs_left and mvs_left_ts is not None and len(mvs_left_ts) > 0:
        idx = np.array([int(np.argmin(np.abs(mvs_left_ts - t))) for t in valid_rs_ts])
        mvs_left_aligned_rgb = mvs_left_rgb[idx]
    mvs_right_aligned_rgb = None
    if has_mvs_right and mvs_right_ts is not None and len(mvs_right_ts) > 0:
        idx = np.array([int(np.argmin(np.abs(mvs_right_ts - t))) for t in valid_rs_ts])
        mvs_right_aligned_rgb = mvs_right_rgb[idx]

    # 按 segments 拆分保存（每段一个文件）
    seg_ranges = build_segment_ranges(raw, t_min, t_max)
    base, ext = os.path.splitext(output_zarr_path)
    if ext != '.zarr':
        base = output_zarr_path
        ext = '.zarr'

    total_saved = 0
    for seg in seg_ranges:
        seg_mask = (valid_rs_ts >= seg['start_ts']) & (valid_rs_ts <= seg['end_ts'])
        if not np.any(seg_mask):
            print(f"Segment {seg['segment_id']} 在有效重叠区间内无帧，跳过")
            continue
        out_path = f"{base}_seg{seg['segment_id']:02d}{ext}"
        left_tactile_seg = slice_tactile_bundle(
            left_tactile_bundle, seg['start_ts'], seg['end_ts']
        )
        right_tactile_seg = slice_tactile_bundle(
            right_tactile_bundle, seg['start_ts'], seg['end_ts']
        )
        save_aligned_zarr(
            out_path,
            root,
            valid_rs_ts[seg_mask],
            valid_rgb[seg_mask],
            valid_depth[seg_mask],
            T_right_cube_to_cam_interp[seg_mask],
            right_cube_pose_valid[seg_mask],
            right_cube_pose_filled[seg_mask],
            right_cube_pose_available[seg_mask],
            T_left_cube_to_cam_interp[seg_mask],
            left_cube_pose_valid[seg_mask],
            left_cube_pose_filled[seg_mask],
            left_cube_pose_available[seg_mask],
            right_hand_poses_in_wrist_interp[seg_mask],
            left_hand_poses_in_wrist_interp[seg_mask],
            left_tactile_seg,
            right_tactile_seg,
            mvs_left_aligned_rgb[seg_mask] if mvs_left_aligned_rgb is not None else None,
            mvs_right_aligned_rgb[seg_mask] if mvs_right_aligned_rgb is not None else None,
            segment_info=seg
        )
        n_seg_frames = int(np.sum(seg_mask))
        total_saved += n_seg_frames
        print(f"已保存第 {seg['segment_id']} 段: {out_path} (帧数: {n_seg_frames})")

    print(f"分段保存完成，共保存帧数: {total_saved}")


def run_batch(
    input_dir,
    output_dir,
    input_pattern="collected_raw*.zarr",
    time_offset_ms=0.0,
    recursive=False,
    coord_file=None,
    rotation_file=None,
    sensor_points_file=None,
    cube_hold_threshold_frames=10
):
    """批量处理目录内多个原始 zarr 文件"""
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        input_files = sorted(in_dir.rglob(input_pattern))
    else:
        input_files = sorted(in_dir.glob(input_pattern))

    # 只保留目录形式的 zarr（zarr 通常表现为目录）
    input_files = [p for p in input_files if p.is_dir()]

    if len(input_files) == 0:
        print(f"[Batch] 未找到匹配文件: dir={in_dir}, pattern={input_pattern}")
        return

    print(f"[Batch] 发现 {len(input_files)} 个输入，开始批量处理")
    success = 0
    failed = 0

    for idx, in_path in enumerate(input_files, start=1):
        try:
            # 例如：collected_raw_seg03.zarr -> collected_raw_seg03_aligned.zarr
            out_base = out_dir / f"{in_path.stem}_aligned.zarr"
            print(f"[Batch] ({idx}/{len(input_files)}) 处理: {in_path}")
            process(
                str(in_path),
                str(out_base),
                time_offset_ms=time_offset_ms,
                coord_file=coord_file,
                rotation_file=rotation_file,
                sensor_points_file=sensor_points_file,
                cube_hold_threshold_frames=cube_hold_threshold_frames
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"[Batch] 处理失败: {in_path}，错误: {e}")

    print(f"[Batch] 完成: 成功 {success} 个, 失败 {failed} 个")


def main():
    parser = argparse.ArgumentParser(
        description='将 T_cube_to_cam 和手部 pose 对齐到 RGBD 时间戳'
    )
    parser.add_argument(
        '--input',
        default=None,
        nargs='?',
        help='原始采集的 zarr 路径（提供时处理单文件）'
    )
    parser.add_argument(
        '--output',
        default='data/aligned_data.zarr',
        help='单文件模式输出 zarr 路径'
    )
    parser.add_argument(
        '--time-offset-ms',
        type=float,
        default=0.0,
        help='时间偏移(毫秒)，用于补偿设备间时钟差，正数表示 wrist/hand 时间往后移'
    )
    parser.add_argument(
        '--batch-dir',
        default=None,
        help='批量模式：输入目录（目录下匹配多个原始 zarr）'
    )
    parser.add_argument(
        '--batch-output-dir',
        default='data',
        help='批量模式：输出目录'
    )
    parser.add_argument(
        '--batch-pattern',
        default='collected_raw*.zarr',
        help='批量模式：输入匹配模式'
    )
    parser.add_argument(
        '--batch-recursive',
        action='store_true',
        help='批量模式：递归搜索输入目录'
    )
    parser.add_argument(
        '--coord-file',
        default=None,
        help='触觉坐标文件（csv/json），用于力矩计算'
    )
    parser.add_argument(
        '--rotation-file',
        default=None,
        help='触觉旋转矩阵文件（默认 utils/S2015-Elite+Rotation+Matrix.h）'
    )
    parser.add_argument(
        '--sensor-points-file',
        default=None,
        help='每个传感器分布点数 json 文件'
    )
    parser.add_argument(
        '--cube-hold-threshold-frames',
        type=int,
        default=5,
        help='cube pose 连续缺失超过该帧数时，不做插值而是保持上一有效帧（默认 5）'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='强制单文件模式（需同时提供 --input）'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='自动模式：扫描当前脚本目录下所有 raw_data_N/，对未处理的逐一插值到对应 data_N/'
    )
    args = parser.parse_args()

    if args.single:
        if not args.input:
            raise ValueError("--single 模式下必须提供 --input")
        process(
            args.input,
            args.output,
            args.time_offset_ms,
            coord_file=args.coord_file,
            rotation_file=args.rotation_file,
            sensor_points_file=args.sensor_points_file,
            cube_hold_threshold_frames=args.cube_hold_threshold_frames
        )
        return

    if args.input is not None:
        process(
            args.input,
            args.output,
            args.time_offset_ms,
            coord_file=args.coord_file,
            rotation_file=args.rotation_file,
            sensor_points_file=args.sensor_points_file,
            cube_hold_threshold_frames=args.cube_hold_threshold_frames
        )
        return

    if args.batch_dir:
        batch_out_dir = args.batch_output_dir
        if batch_out_dir is None:
            batch_out_dir = str(Path(args.output).expanduser().resolve().parent)
        run_batch(
            input_dir=args.batch_dir,
            output_dir=batch_out_dir,
            input_pattern=args.batch_pattern,
            time_offset_ms=args.time_offset_ms,
            recursive=args.batch_recursive,
            coord_file=args.coord_file,
            rotation_file=args.rotation_file,
            sensor_points_file=args.sensor_points_file,
            cube_hold_threshold_frames=args.cube_hold_threshold_frames
        )
    elif args.auto or args.batch_dir is None:
        # 自动模式：扫描 raw_data_N/ → data_N/ 一一对应
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        i = 0
        processed = 0
        while True:
            raw_dir = os.path.join(_script_dir, f'raw_data_{i}')
            if not os.path.isdir(raw_dir):
                break  # raw_data_N 不连续时停止
            data_dir = os.path.join(_script_dir, f'data_{i}')
            if os.path.isdir(data_dir):
                print(f'[Auto] data_{i}/ 已存在，跳过 raw_data_{i}/')
            else:
                print(f'[Auto] 处理 raw_data_{i}/ → data_{i}/')
                run_batch(
                    input_dir=raw_dir,
                    output_dir=data_dir,
                    input_pattern=args.batch_pattern,
                    time_offset_ms=args.time_offset_ms,
                    recursive=args.batch_recursive,
                    coord_file=args.coord_file,
                    rotation_file=args.rotation_file,
                    sensor_points_file=args.sensor_points_file,
                    cube_hold_threshold_frames=args.cube_hold_threshold_frames
                )
                processed += 1
            i += 1
        if i == 0:
            print('[Auto] 未找到任何 raw_data_N/ 目录，请先运行 data_collect_v2 采集数据')
        else:
            print(f'[Auto] 共扫描 {i} 个 raw_data 目录，新处理 {processed} 个')
    else:
        print("[Batch] 未指定输入目录，跳过")


if __name__ == "__main__":
    main()
