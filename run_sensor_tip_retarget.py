#!/usr/bin/env python3
"""Batch-capable AprilTag + tactile whole-body retarget entry point.

Examples
--------
One sequence::

  python run_sensor_tip_retarget.py \
    --zarr-path /path/to/session/data_0/collected_raw_seg01_aligned_seg01.zarr \
    --output-root /data/retarget_sequences --max-frames 5 --overwrite

Every aligned Zarr directly inside one ``data_N`` directory::

  python run_sensor_tip_retarget.py \
    --data-dir /path/to/session/data_0 \
    --output-root /data/retarget_sequences --resume

Every aligned Zarr in all numeric ``data_*`` directories::

  python run_sensor_tip_retarget.py \
    --scripts-root /path/to/session \
    --output-root /data/retarget_sequences --resume

Add ``--with-no-compliance-branch`` only when the counterfactual full solve is
needed for inward-vector visualization.  ``--dry-run`` performs discovery and
preflight, prints commands, and writes no files.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from whole_body_retarget.io_utils import frame_timestamps_from_data_group, selected_indices


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "whole_body_retarget" / "outputs" / "sequences"
DEFAULT_URDF = (
    PROJECT_ROOT
    / "assets" / "robot"
    / "urdf"
    / "moz_wuji_7_3_dual_hands.urdf"
)
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs/robot_camera.json"
DEFAULT_CALIBRATION_KEY = "calib_cam_to_base"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retarget_free_wrist.yaml"
DEFAULT_LAYOUT = PROJECT_ROOT / "assets/tactile/layout.json"
DEFAULT_BASE_IN_TAG = PROJECT_ROOT / "configs/base_in_scene_tag.json"
DEFAULT_INITIAL_LEGWAIST = [0.001396, 0.357096, -0.576134, 0.383624, 0.0, 0.0]
DEFAULT_FIXED_HEAD = [0.007854, 0.739495]


def _numeric_key(path: Path) -> tuple:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", str(path)):
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def _require(path: Path, label: str, *, directory: bool) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if directory != path.is_dir():
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} must be a {kind}: {path}")
    return path


def _discover(args: argparse.Namespace) -> list[Path]:
    if args.zarr_path is not None:
        return [_require(args.zarr_path, "input Zarr", directory=True)]
    if args.data_dir is not None:
        data_dir = _require(args.data_dir, "data directory", directory=True)
        return sorted(
            [path.resolve() for path in data_dir.glob("*.zarr") if path.is_dir()],
            key=_numeric_key,
        )
    scripts_root = _require(args.scripts_root, "scripts root", directory=True)
    data_dirs = [
        path for path in scripts_root.iterdir()
        if path.is_dir() and re.fullmatch(r"data_\d+", path.name)
    ]
    sequences: list[Path] = []
    for data_dir in sorted(data_dirs, key=_numeric_key):
        sequences.extend(
            sorted(
                [path.resolve() for path in data_dir.glob("*.zarr") if path.is_dir()],
                key=_numeric_key,
            )
        )
    return sequences


def _preflight_inpaint(
    zarr_path: Path,
    start_frame: int,
    max_frames: int | None,
    frame_stride: int,
    target_fps: float | None,
) -> dict[str, Any]:
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root:
        return {"ok": False, "reason": "missing data group", "missing_frames": []}
    frame_keys = sorted(root["data"].keys())
    timestamps = (
        frame_timestamps_from_data_group(root["data"], frame_keys)
        if target_fps is not None
        else None
    )
    indices = selected_indices(
        len(frame_keys),
        start_frame,
        max_frames,
        frame_stride,
        target_fps=target_fps,
        timestamps=timestamps,
    )
    if not indices:
        return {"ok": False, "reason": "frame selection is empty", "missing_frames": []}
    missing: list[int] = []
    for index in indices:
        frame = root["data"][frame_keys[index]]
        if "inpaint_rgb" not in frame or "inpaint_mask" not in frame:
            missing.append(index)
    return {
        "ok": not missing,
        "reason": None if not missing else "selected frames missing inpaint_rgb or inpaint_mask",
        "missing_frames": missing,
        "selected_frame_count": len(indices),
        "total_frame_count": len(frame_keys),
    }


def _sequence_paths(
    source: Path,
    output_root: Path,
    diagnostics_root: Path,
) -> tuple[Path, Path]:
    parent = source.parent.name if re.fullmatch(r"data_\d+", source.parent.name) else "single"
    name = source.name.removesuffix(".zarr")
    return output_root / parent / f"{name}.zarr", diagnostics_root / parent / name


def _complete_sequence(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        import zarr

        return bool(zarr.open(str(path), mode="r").attrs.get("export_complete", False))
    except Exception:
        return False


def _has_retarget_work(work_dir: Path) -> bool:
    required = (
        "metadata.json",
        "legwaist.npy",
        "left_arm.npy",
        "right_arm.npy",
        "left_hand.npy",
        "right_hand.npy",
        "whole_body_joints.npy",
        "camera_trajectory_cam0.npy",
        "base_pose_cam0.npy",
        "whole_body_joint_names.json",
        "wrist_free/wrist_left.npy",
        "wrist_free/wrist_right.npy",
    )
    return work_dir.is_dir() and all((work_dir / name).exists() for name in required)


def _clear_outputs_for_overwrite(
    final_path: Path,
    sequence_diag: Path,
) -> None:
    if final_path.exists():
        shutil.rmtree(final_path)
    if sequence_diag.exists():
        shutil.rmtree(sequence_diag)


def _command(
    args: argparse.Namespace,
    source: Path,
    sequence_output: Path | None,
    work_dir: Path,
) -> list[str]:
    command = [
        sys.executable, "-m", "whole_body_retarget.cli", "run",
        "--zarr_path", str(source),
        "--output_dir", str(work_dir),
        "--urdf", str(args.urdf),
        "--start_frame", str(args.start_frame),
        "--frame_stride", str(args.frame_stride),
        "--calibration", str(args.calibration),
        "--calibration_key", args.calibration_key,
        "--config", str(args.config),
        "--layout", str(args.layout),
        "--base_in_tag", str(args.base_in_tag),
        "--apriltag_max_reprojection_error_px", str(args.apriltag_max_reprojection_error_px),
        "--apriltag_max_gap_sec", str(args.apriltag_max_gap_sec),
        "--apriltag_gate_half_window", str(args.apriltag_gate_half_window),
        "--apriltag_gate_mad_scale", str(args.apriltag_gate_mad_scale),
        "--apriltag_gate_translation_floor_m", str(args.apriltag_gate_translation_floor_m),
        "--apriltag_gate_translation_ceiling_m", str(args.apriltag_gate_translation_ceiling_m),
        "--apriltag_gate_rotation_floor_deg", str(args.apriltag_gate_rotation_floor_deg),
        "--apriltag_gate_rotation_ceiling_deg", str(args.apriltag_gate_rotation_ceiling_deg),
        "--apriltag_gate_dilation_frames", str(args.apriltag_gate_dilation_frames),
        "--apriltag_lowpass_cutoff_hz", str(args.apriltag_lowpass_cutoff_hz),
        "--apriltag_pixel_sigma", str(args.apriltag_pixel_sigma),
        "--apriltag_translation_accel_sigma", str(args.apriltag_translation_accel_sigma),
        "--apriltag_rotation_accel_sigma", str(args.apriltag_rotation_accel_sigma),
        "--apriltag_max_nfev", str(args.apriltag_max_nfev),
        "--wrist_free_chunk_size", str(args.wrist_free_chunk_size),
        "--wrist_free_chunk_timeout_sec", str(args.wrist_free_chunk_timeout_sec),
        "--wrist_free_workers", "1",
        "--tactile_max_gap_sec", str(args.tactile_max_gap_sec),
        "--ik_max_iters", str(args.ik_max_iters),
        "--camera_pos_weight", str(args.camera_pos_weight),
        "--camera_rot_weight", str(args.camera_rot_weight),
        "--arm_pos_weight", str(args.arm_pos_weight),
        "--arm_rot_weight", str(args.arm_rot_weight),
        "--joint_rest_weight", str(args.joint_rest_weight),
        "--joint_smooth_weight", str(args.joint_smooth_weight),
        "--rgb_o3d_occlusion_epsilon_m", str(args.rgb_o3d_occlusion_epsilon_m),
        "--rgb_o3d_near_clip_m", str(args.rgb_o3d_near_clip_m),
        "--rgb_o3d_backend", args.rgb_o3d_backend,
        "--rgb_o3d_mp4_fps", str(args.rgb_o3d_mp4_fps),
        "--initial_legwaist", *(str(value) for value in args.initial_legwaist),
        "--fixed_head", *(str(value) for value in args.fixed_head),
    ]
    if sequence_output is not None:
        command.extend(["--sequence_output", str(sequence_output)])
    if args.target_fps is not None:
        command.extend(["--target_fps", str(args.target_fps)])
    if args.expected_dof is not None:
        command.extend(["--expected_dof", str(args.expected_dof)])
    if args.post_filter_motion:
        command.extend(
            [
                "--post_filter_motion",
                "--post_filter_window", str(args.post_filter_window),
                "--post_filter_polyorder", str(args.post_filter_polyorder),
            ]
        )
    if args.post_filter_hands:
        command.extend(
            [
                "--post_filter_hands",
                "--post_filter_window", str(args.post_filter_window),
                "--post_filter_polyorder", str(args.post_filter_polyorder),
            ]
        )
    if args.max_frames is not None:
        command.extend(["--max_frames", str(args.max_frames)])
    for enabled, flag in (
        (args.force_wrist_free, "--force_wrist_free"),
        (args.no_wrist_free_fallback, "--no_wrist_free_fallback"),
        (args.with_no_compliance_branch, "--run_no_compliance"),
        (args.force_rgb_o3d, "--force_rgb_o3d"),
        (args.rgb_o3d_only, "--rgb_o3d_only"),
        (args.camera_only, "--camera_only"),
        (args.skip_rgb_o3d, "--skip_rgb_o3d"),
        (args.overwrite and sequence_output is not None, "--overwrite_sequence_output"),
    ):
        if enabled:
            command.append(flag)
    return command


def _visualizer_command(args: argparse.Namespace, source: Path, work_dir: Path) -> list[str]:
    command = [
        sys.executable, "-m", "whole_body_retarget.open3d_visualizer",
        "--output_dir", str(work_dir),
        "--zarr_path", str(source),
        "--urdf", str(args.urdf),
        "--frame", str(args.visualizer_frame),
        "--fps", str(args.visualizer_fps),
        "--point_sample_step", str(args.visualizer_point_sample_step),
        "--max_depth", str(args.visualizer_max_depth),
        "--width", str(args.visualizer_width),
        "--height", str(args.visualizer_height),
        "--point_size", str(args.visualizer_point_size),
    ]
    if args.expected_dof is not None:
        command.extend(["--expected_dof", str(args.expected_dof)])
    if args.visualizer_static:
        command.append("--static")
    if args.visualizer_play:
        command.append("--play")
    if args.visualizer_loop:
        command.append("--loop")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--zarr-path", type=Path, help="One aligned input Zarr.")
    inputs.add_argument("--data-dir", type=Path, help="All direct *.zarr children of one data_N.")
    inputs.add_argument("--scripts-root", type=Path, help="All direct *.zarr children of numeric data_* dirs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diagnostics-root", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Downsample visual/data frames to this FPS before retarget, "
            "rgb_o3d, and final export. Raw tactile_stream is preserved."
        ),
    )

    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--expected-dof",
        type=int,
        default=None,
        help="Optional URDF DoF validation; omitted means count from the selected full URDF.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Optical-view calibration only; Base is anchored by Tag10/Tag11 transforms.",
    )
    parser.add_argument("--calibration-key", default=DEFAULT_CALIBRATION_KEY)
    parser.add_argument(
        "--config",
        "--retarget-config",
        dest="config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Retarget YAML config. --retarget-config is an alias kept for tuning clarity.",
    )
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument(
        "--base-in-tag", type=Path, default=DEFAULT_BASE_IN_TAG,
        help="Robot base pose in the stationary scene-tag frame (JSON or .npy); default identity.",
    )
    parser.add_argument("--apriltag-max-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--apriltag-max-gap-sec", type=float, default=0.4)
    parser.add_argument("--apriltag-gate-half-window", type=int, default=5)
    parser.add_argument("--apriltag-gate-mad-scale", type=float, default=6.0)
    parser.add_argument("--apriltag-gate-translation-floor-m", type=float, default=0.02)
    parser.add_argument("--apriltag-gate-translation-ceiling-m", type=float, default=0.06)
    parser.add_argument("--apriltag-gate-rotation-floor-deg", type=float, default=1.0)
    parser.add_argument("--apriltag-gate-rotation-ceiling-deg", type=float, default=5.0)
    parser.add_argument("--apriltag-gate-dilation-frames", type=int, default=1)
    parser.add_argument("--apriltag-lowpass-cutoff-hz", type=float, default=3.0)
    parser.add_argument("--apriltag-pixel-sigma", type=float, default=0.25)
    parser.add_argument("--apriltag-translation-accel-sigma", type=float, default=2.0)
    parser.add_argument("--apriltag-rotation-accel-sigma", type=float, default=8.0)
    parser.add_argument("--apriltag-max-nfev", type=int, default=120)
    parser.add_argument("--force-wrist-free", action="store_true")
    parser.add_argument("--wrist-free-chunk-size", type=int, default=16)
    parser.add_argument("--wrist-free-chunk-timeout-sec", type=float, default=240.0)
    parser.add_argument("--no-wrist-free-fallback", action="store_true")
    parser.add_argument("--tactile-max-gap-sec", type=float, default=0.03)
    parser.add_argument(
        "--with-no-compliance-branch",
        action="store_true",
        help="Run the optional full counterfactual branch used only for inward vectors.",
    )

    parser.add_argument("--ik-max-iters", type=int, default=120)
    parser.add_argument("--camera-pos-weight", type=float, default=40.0)
    parser.add_argument("--camera-rot-weight", type=float, default=8.0)
    parser.add_argument("--arm-pos-weight", type=float, default=55.0)
    parser.add_argument("--arm-rot-weight", type=float, default=8.0)
    parser.add_argument("--joint-rest-weight", type=float, default=0.02)
    parser.add_argument("--joint-smooth-weight", type=float, default=0.20)
    parser.add_argument(
        "--post-filter-motion",
        action="store_true",
        help=(
            "After retargeting, smooth only legwaist/left_arm/right_arm and matching "
            "whole_body_joints columns before rgb_o3d/final export."
        ),
    )
    parser.add_argument(
        "--post-filter-hands",
        action="store_true",
        help=(
            "After retargeting or before --rgb-o3d-only export, smooth left_hand/right_hand "
            "and matching whole_body_joints finger columns."
        ),
    )
    parser.add_argument(
        "--post-filter-window",
        type=int,
        default=7,
        help="Odd Savitzky-Golay window length used by --post-filter-motion.",
    )
    parser.add_argument(
        "--post-filter-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order used by --post-filter-motion.",
    )
    parser.add_argument("--initial-legwaist", type=float, nargs=6, default=DEFAULT_INITIAL_LEGWAIST)
    parser.add_argument("--fixed-head", type=float, nargs=2, default=DEFAULT_FIXED_HEAD)
    parser.add_argument(
        "--rgb-o3d-occlusion-epsilon-m",
        type=float,
        default=0.005,
        help="Depth tolerance for robot-versus-scene occlusion in the rgb_o3d composite.",
    )
    parser.add_argument(
        "--rgb-o3d-near-clip-m",
        type=float,
        default=0.0,
        help="Optional debug near clipping; default 0 renders from the camera origin.",
    )
    parser.add_argument(
        "--rgb-o3d-backend",
        choices=("cpu", "egl", "cuda"),
        default="cpu",
        help="rgb_o3d renderer: CPU, headless EGL, or NVIDIA CUDA via nvdiffrast.",
    )
    parser.add_argument(
        "--rgb-o3d-mp4-fps",
        type=float,
        default=15.0,
        help="Frame rate for the MP4 exported from rgb_o3d.zarr.",
    )
    parser.add_argument(
        "--force-rgb-o3d",
        action="store_true",
        help="Re-render rgb_o3d even if a matching complete render cache exists.",
    )
    parser.add_argument(
        "--rgb-o3d-only",
        action="store_true",
        help="Only regenerate the existing work/rgb_o3d.zarr cache; skip camera estimation and retarget.",
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Only generate work/apriltag_camera, then skip wrist-free, whole-body, rgb_o3d, and final export.",
    )
    parser.add_argument(
        "--skip-rgb-o3d",
        action="store_true",
        help="Debug/tuning only: skip rgb_o3d generation and final strict Zarr export.",
    )

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualizer-frame", type=int, default=0)
    parser.add_argument("--visualizer-static", action="store_true")
    parser.add_argument("--visualizer-play", action="store_true")
    parser.add_argument("--visualizer-loop", action="store_true")
    parser.add_argument("--visualizer-fps", type=float, default=15.0)
    parser.add_argument("--visualizer-point-sample-step", type=int, default=1)
    parser.add_argument("--visualizer-max-depth", type=float, default=3.0)
    parser.add_argument("--visualizer-width", type=int, default=1400)
    parser.add_argument("--visualizer-height", type=int, default=950)
    parser.add_argument("--visualizer-point-size", type=float, default=3.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.start_frame < 0 or args.frame_stride <= 0:
        parser.error("start-frame must be >=0 and frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    if args.target_fps is not None and args.target_fps <= 0.0:
        parser.error("--target-fps must be positive")
    if args.target_fps is not None and args.frame_stride != 1:
        parser.error("--target-fps cannot be combined with --frame-stride other than 1")
    if args.visualize and args.zarr_path is None:
        parser.error("--visualize is supported only with --zarr-path")
    if args.rgb_o3d_only and args.skip_rgb_o3d:
        parser.error("--rgb-o3d-only cannot be combined with --skip-rgb-o3d")
    if args.camera_only and args.rgb_o3d_only:
        parser.error("--camera-only cannot be combined with --rgb-o3d-only")
    if args.rgb_o3d_mp4_fps <= 0.0:
        parser.error("--rgb-o3d-mp4-fps must be positive")
    if args.post_filter_motion or args.post_filter_hands:
        if args.post_filter_window < 3:
            parser.error("--post-filter-window must be at least 3")
        if args.post_filter_polyorder < 0:
            parser.error("--post-filter-polyorder must be non-negative")
    if args.apriltag_max_gap_sec <= 0.0:
        parser.error("--apriltag-max-gap-sec must be positive")
    if args.apriltag_gate_half_window < 1:
        parser.error("--apriltag-gate-half-window must be at least 1")
    if args.apriltag_translation_accel_sigma <= 0.0 or args.apriltag_rotation_accel_sigma <= 0.0:
        parser.error("AprilTag acceleration sigma values must be positive")
    if args.apriltag_pixel_sigma <= 0.0 or args.apriltag_max_nfev <= 0:
        parser.error("AprilTag pixel sigma and max-nfev must be positive")

    for name in ("urdf", "calibration", "config", "layout"):
        setattr(args, name, _require(getattr(args, name), name, directory=False))
    args.base_in_tag = _require(args.base_in_tag, "base-in-tag", directory=False)
    output_root = args.output_root.expanduser().resolve()
    diagnostics_root = (
        args.diagnostics_root.expanduser().resolve()
        if args.diagnostics_root is not None
        else output_root / "_diagnostics"
    )
    sequences = _discover(args)
    if not sequences:
        parser.error("No direct *.zarr input sequences were discovered")

    print(f"Discovered {len(sequences)} sequence(s)", flush=True)
    report: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_mode": (
            "zarr_path" if args.zarr_path is not None
            else "data_dir" if args.data_dir is not None
            else "scripts_root"
        ),
        "output_root": str(output_root),
        "diagnostics_root": str(diagnostics_root),
        "target_fps": args.target_fps,
        "with_no_compliance_branch": bool(args.with_no_compliance_branch),
        "camera_only": bool(args.camera_only),
        "entries": [],
    }
    failures = 0
    for ordinal, source in enumerate(sequences, start=1):
        final_path, sequence_diag = _sequence_paths(source, output_root, diagnostics_root)
        work_dir = sequence_diag / "work"
        entry: dict[str, Any] = {
            "ordinal": ordinal,
            "source": str(source),
            "output": str(final_path),
            "diagnostics": str(sequence_diag),
            "camera_method": "fixed_apriltag_10_se3_trajectory",
        }
        preflight = (
            {
                "ok": True,
                "reason": None,
                "mode": (
                    "rgb_o3d_only_uses_existing_retarget_metadata"
                    if args.rgb_o3d_only
                    else "camera_only_does_not_require_inpaint"
                ),
            }
            if args.rgb_o3d_only or args.camera_only
            else _preflight_inpaint(
                source,
                args.start_frame,
                args.max_frames,
                args.frame_stride,
                args.target_fps,
            )
        )
        entry["inpaint_preflight"] = preflight
        if not preflight["ok"]:
            entry["status"] = "skipped_missing_inpaint"
            entry["reason"] = preflight["reason"]
            report["entries"].append(entry)
            print(
                f"[{ordinal}/{len(sequences)}] SKIP {source}: {preflight['reason']} "
                f"(missing frames={len(preflight['missing_frames'])})",
                flush=True,
            )
            continue
        if args.resume and args.rgb_o3d_only and not _has_retarget_work(work_dir):
            entry["status"] = "skipped_missing_retarget_work"
            entry["reason"] = "rgb_o3d_only requires existing retarget work outputs"
            report["entries"].append(entry)
            print(
                f"[{ordinal}/{len(sequences)}] RGB_O3D SKIP missing retarget work {work_dir}",
                flush=True,
            )
            continue
        if args.resume and not (args.rgb_o3d_only or args.camera_only) and _complete_sequence(final_path):
            entry["status"] = "skipped_complete"
            report["entries"].append(entry)
            print(f"[{ordinal}/{len(sequences)}] COMPLETE {final_path}", flush=True)
            continue
        if (
            (final_path.exists() or work_dir.exists())
            and not (args.overwrite or args.resume or args.rgb_o3d_only)
        ):
            entry["status"] = "failed"
            entry["reason"] = "existing output; use --overwrite or --resume"
            report["entries"].append(entry)
            failures += 1
            print(f"[{ordinal}/{len(sequences)}] FAIL existing output for {source}", flush=True)
            continue
        if args.overwrite and not args.dry_run and not (args.rgb_o3d_only or args.camera_only):
            _clear_outputs_for_overwrite(final_path, sequence_diag)
        sequence_output = None if (args.skip_rgb_o3d or args.camera_only) else final_path
        command = _command(args, source, sequence_output, work_dir)
        entry["command"] = command
        entry["command_shell"] = shlex.join(command)
        print(f"[{ordinal}/{len(sequences)}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            entry["status"] = "dry_run_ready"
            report["entries"].append(entry)
            continue
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            entry["status"] = "failed"
            entry["returncode"] = exc.returncode
            failures += 1
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
        else:
            entry["status"] = "success"
        report["entries"].append(entry)

        if entry["status"] == "success" and args.visualize:
            viewer = _visualizer_command(args, source, work_dir)
            entry["visualizer_command"] = viewer
            print(f"Open3D command: {shlex.join(viewer)}", flush=True)
            subprocess.run(viewer, cwd=PROJECT_ROOT, check=True)

    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["summary"] = {
        status: sum(item["status"] == status for item in report["entries"])
        for status in sorted({item["status"] for item in report["entries"]})
    }
    report["failure_count"] = failures
    if not args.dry_run:
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = diagnostics_root / f"batch_report_{stamp}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (diagnostics_root / "batch_report_latest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Batch report: {report_path}", flush=True)
    print(f"Summary: {json.dumps(report['summary'], ensure_ascii=False)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
