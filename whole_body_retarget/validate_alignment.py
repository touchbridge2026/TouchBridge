#!/usr/bin/env python3
"""Validate whole-body fingertip alignment against wrist-free outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .full_body_kinematics import FullBodyKinematics
from .hand_models import dual_hand_submodels_from_full
from .io_utils import DEFAULT_URDF, ensure_project_imports, load_wrist_free_output

ensure_project_imports()

from retarget.data.zarr_loader import ZarrHumanDataset  # noqa: E402
from retarget.robot.fk import HandKinematics  # noqa: E402
from retarget.robot.urdf_model import parse_urdf_model  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _translation_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)


def _summarize(name: str, values: list[float]) -> None:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: mean={data.mean():.6f} m, median={np.median(data):.6f} m, "
        f"p95={np.percentile(data, 95):.6f} m, max={data.max():.6f} m, n={data.size}"
    )


def _hand_tip_positions_cam0(
    cam0_from_cam: np.ndarray,
    wrist_cam: np.ndarray,
    hand_fk: HandKinematics,
    hand_q: np.ndarray,
    tip_links: list[str],
) -> dict[str, np.ndarray]:
    local_tips = hand_fk.fk_tips(hand_q)
    return {
        link_name: (cam0_from_cam @ wrist_cam @ local_tips[link_name])[:3, 3].copy()
        for link_name in tip_links
        if link_name in local_tips
    }


def validate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    metadata = _load_json(output_dir / "metadata.json")
    zarr_path = Path(args.zarr_path or metadata["zarr_path"]).resolve()
    frames = [int(value) for value in metadata["frames"]]

    full_model = parse_urdf_model(args.urdf, expected_dof=args.expected_dof)
    full_kin = FullBodyKinematics(full_model)
    left_model, right_model = dual_hand_submodels_from_full(full_model)
    left_fk = HandKinematics(left_model)
    right_fk = HandKinematics(right_model)

    q = np.load(output_dir / "whole_body_joints.npy")
    camera = np.load(output_dir / "camera_trajectory_cam0.npy")
    cam0_from_base = np.load(output_dir / "base_pose_cam0.npy")
    joint_names = _load_json(output_dir / "whole_body_joint_names.json")
    if joint_names != full_model.joint_names:
        raise ValueError("whole_body_joint_names.json does not match the provided full-body URDF.")
    if q.shape[0] != len(frames):
        raise ValueError(f"Output frame count {q.shape[0]} does not match metadata frames {len(frames)}")

    wrist_dir = Path(args.wrist_free_dir).resolve() if args.wrist_free_dir else output_dir / "wrist_free"
    wrist = load_wrist_free_output(wrist_dir)
    dataset = ZarrHumanDataset(zarr_path) if args.compare_human else None

    full_vs_wrist_tip: list[float] = []
    full_vs_human_tip: list[float] = []
    wrist_vs_human_tip: list[float] = []
    full_vs_wrist_palm: list[float] = []
    worst: list[tuple[float, int, str, str]] = []

    frame_indices = list(range(q.shape[0]))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(0, int(args.max_frames))]

    for out_i in frame_indices:
        cam0_from_cam = camera[out_i]
        full_links = full_kin.fk_all_links(q[out_i])
        left_wrist_tips = _hand_tip_positions_cam0(
            cam0_from_cam,
            wrist["wrist_left"][out_i],
            left_fk,
            wrist["joints_left"][out_i],
            left_model.tip_links,
        )
        right_wrist_tips = _hand_tip_positions_cam0(
            cam0_from_cam,
            wrist["wrist_right"][out_i],
            right_fk,
            wrist["joints_right"][out_i],
            right_model.tip_links,
        )
        full_left_palm = (cam0_from_base @ full_links["left_palm_link"])[:3, 3]
        full_right_palm = (cam0_from_base @ full_links["right_palm_link"])[:3, 3]
        wrist_left_palm = (cam0_from_cam @ wrist["wrist_left"][out_i])[:3, 3]
        wrist_right_palm = (cam0_from_cam @ wrist["wrist_right"][out_i])[:3, 3]
        full_vs_wrist_palm.extend(
            [
                float(np.linalg.norm(full_left_palm - wrist_left_palm)),
                float(np.linalg.norm(full_right_palm - wrist_right_palm)),
            ]
        )

        human_left: list[np.ndarray] = []
        human_right: list[np.ndarray] = []
        if dataset is not None:
            frame = dataset.get_frame(frames[out_i])
            human_left = [(cam0_from_cam @ transform)[:3, 3] for transform in frame.T_left_sensor_to_cam]
            human_right = [(cam0_from_cam @ transform)[:3, 3] for transform in frame.T_right_sensor_to_cam]

        for side, hand_tips, hand_tip_links, human_points in (
            ("left", left_wrist_tips, left_model.tip_links, human_left),
            ("right", right_wrist_tips, right_model.tip_links, human_right),
        ):
            for tip_i, link_name in enumerate(hand_tip_links):
                if link_name not in full_links or link_name not in hand_tips:
                    continue
                full_tip = (cam0_from_base @ full_links[link_name])[:3, 3]
                wrist_tip = hand_tips[link_name]
                err = float(np.linalg.norm(full_tip - wrist_tip))
                full_vs_wrist_tip.append(err)
                worst.append((err, int(frames[out_i]), side, link_name))
                if tip_i < len(human_points):
                    human_tip = np.asarray(human_points[tip_i], dtype=np.float64)
                    full_vs_human_tip.append(float(np.linalg.norm(full_tip - human_tip)))
                    wrist_vs_human_tip.append(float(np.linalg.norm(wrist_tip - human_tip)))

    worst.sort(reverse=True, key=lambda item: item[0])
    print(f"output_dir={output_dir}")
    print(f"zarr_path={zarr_path}")
    print(f"checked_output_frames={len(frame_indices)}")
    print(f"wrist_free_dir={wrist_dir}")
    print("Expected semantics: wrist_left/right.npy are palm-link poses in the current camera frame.")
    _summarize("full palm vs wrist-free palm", full_vs_wrist_palm)
    _summarize("full fingertips vs wrist-free fingertips", full_vs_wrist_tip)
    if args.compare_human:
        _summarize("wrist-free finger_tip_link vs sensor targets", wrist_vs_human_tip)
        _summarize("full-URDF finger_tip_link vs sensor targets", full_vs_human_tip)
    if worst:
        print("worst full-vs-wrist fingertip samples:")
        for err, frame_index, side, link_name in worst[: args.topk]:
            print(f"  frame={frame_index:04d} side={side} link={link_name} err={err:.6f} m")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--zarr_path", default=None)
    parser.add_argument("--wrist_free_dir", default=None)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--expected_dof", type=int, default=66)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--compare_human", action="store_true")
    return parser


def main() -> int:
    return validate(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
