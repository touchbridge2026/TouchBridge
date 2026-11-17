#!/usr/bin/env python3
"""CLI for AprilTag-driven whole-body MOZ/Wuji retargeting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .io_utils import (
    DEFAULT_CALIBRATION,
    DEFAULT_CALIBRATION_KEY,
    DEFAULT_CONFIG,
    DEFAULT_LAYOUT,
    DEFAULT_BASE_IN_TAG,
    DEFAULT_URDF,
    assert_same_length,
    complete_wrist_free_output,
    ensure_dir,
    ensure_project_imports,
    frame_selection_metadata,
    frame_timestamps_from_dataset,
    load_transform_from_json,
    load_base_in_tag,
    load_wrist_free_output,
    selected_indices,
    write_json,
)

ensure_project_imports()

from retarget.config import load_retarget_config  # noqa: E402
from retarget.data.zarr_loader import ZarrHumanDataset  # noqa: E402
from retarget.robot.urdf_model import parse_urdf_model  # noqa: E402

from .diagnostics_export import write_visualization_diagnostics  # noqa: E402
from .apriltag_camera import run_apriltag_camera_estimation  # noqa: E402
from .full_body_kinematics import FullBodyKinematics  # noqa: E402
from .hand_models import dual_hand_submodels_from_full  # noqa: E402
from .lighting import DEFAULT_LIGHTING_CONFIG  # noqa: E402
from .rgb_o3d_renderer import export_rgb_o3d_mp4, render_rgb_o3d_sequence  # noqa: E402
from .safe_wrist_free import WristFreeRunConfig, run_safe_wrist_free  # noqa: E402
from .whole_body_solver import WholeBodyWeights, solve_whole_body  # noqa: E402


DEFAULT_INITIAL_LEGWAIST = [0.001396, 0.357096, -0.576134, 0.383624, 0.0, 0.0]
DEFAULT_FIXED_HEAD = [0.007854, 0.739495]
def _active_dof_from_urdf(path: str | Path) -> int:
    root = ET.parse(path).getroot()
    return sum(
        joint.get("type", "fixed") != "fixed"
        for joint in root.findall("joint")
    )


def _complete_wrist_diagnostics(
    wrist_dir: Path,
    frame_indices: list[int],
) -> bool:
    if not complete_wrist_free_output(wrist_dir):
        return False
    try:
        optimization = json.loads(
            (wrist_dir / "optimization_info.json").read_text(encoding="utf-8")
        )
        with np.load(wrist_dir / "contact_features.npz", allow_pickle=True) as archive:
            contacts = archive["records"].tolist()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    for records in (optimization, contacts):
        if not isinstance(records, list) or len(records) != len(frame_indices):
            return False
        if [int(record.get("frame_index", -1)) for record in records] != frame_indices:
            return False
    return True


def _run_wrist_free(
    args: argparse.Namespace,
    dataset: ZarrHumanDataset,
    output_dir: Path,
    *,
    frame_indices: list[int],
    config_path: Path | None = None,
    force: bool | None = None,
) -> dict:
    wrist_dir = output_dir / "wrist_free"
    config_path = Path(args.config if config_path is None else config_path)
    force = bool(args.force_wrist_free if force is None else force)
    if complete_wrist_free_output(wrist_dir) and not force:
        if _complete_wrist_diagnostics(wrist_dir, frame_indices):
            print(f"Using cached wrist-free output: {wrist_dir}")
            return load_wrist_free_output(wrist_dir)
        print(
            f"Ignoring incomplete/legacy wrist-free cache without per-frame diagnostics: "
            f"{wrist_dir}",
            flush=True,
        )

    full_model = parse_urdf_model(args.urdf, expected_dof=args.expected_dof)
    left_model, right_model = dual_hand_submodels_from_full(full_model)
    cfg = WristFreeRunConfig(
        zarr_path=str(Path(args.zarr_path).resolve()),
        config=str(config_path.resolve()),
        layout=str(Path(args.layout).resolve()),
        urdf=str(Path(args.urdf).resolve()),
        expected_dof=int(args.expected_dof),
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        frame_indices=tuple(frame_indices),
        chunk_size=args.wrist_free_chunk_size,
        timeout_sec=args.wrist_free_chunk_timeout_sec,
        retry_single_frames=True,
        allow_fallback=not args.no_wrist_free_fallback,
        max_workers=args.wrist_free_workers,
    )
    print(
        "Running safe wrist-free retarget -> "
        f"{wrist_dir} (chunk_size={cfg.chunk_size}, timeout={cfg.timeout_sec:.1f}s, "
        f"workers={cfg.max_workers})"
    )
    return run_safe_wrist_free(
        cfg=cfg,
        output_dir=wrist_dir,
        total_frames=len(dataset),
        left_dof=left_model.dof,
        right_dof=right_model.dof,
        joint_names_left=left_model.joint_names,
        joint_names_right=right_model.joint_names,
    )


def inspect_command(args: argparse.Namespace) -> int:
    if args.expected_dof is None:
        args.expected_dof = _active_dof_from_urdf(args.urdf)
    dataset = ZarrHumanDataset(args.zarr_path)
    timestamps = frame_timestamps_from_dataset(dataset) if args.target_fps is not None else None
    indices = selected_indices(
        len(dataset),
        args.start_frame,
        args.max_frames,
        args.frame_stride,
        target_fps=args.target_fps,
        timestamps=timestamps,
    )
    model = parse_urdf_model(args.urdf, expected_dof=args.expected_dof)
    kin = FullBodyKinematics(model)
    print(f"zarr_path={Path(args.zarr_path).resolve()}")
    print(f"total_frames={len(dataset)}")
    print(f"selected_frames={len(indices)}")
    print(f"first_selected_frame={indices[0]}")
    print(f"last_selected_frame={indices[-1]}")
    print(f"urdf_root_link={model.root_link}")
    print(f"parsed_dof={model.dof}")
    print(f"has_torso_flange={'torso_flange' in model.link_names}")
    print(f"has_wuji_left_base_link={'wuji_left_base_link' in model.link_names}")
    print(f"has_wuji_right_base_link={'wuji_right_base_link' in model.link_names}")
    print(f"legwaist_indices={kin.group_indices([f'LegWaist-{i}' for i in range(6)])}")
    print(f"left_arm_indices={kin.group_indices([f'LeftArm-{i}' for i in range(7)])}")
    print(f"right_arm_indices={kin.group_indices([f'RightArm-{i}' for i in range(7)])}")
    return 0


def _whole_body_weights(args: argparse.Namespace) -> WholeBodyWeights:
    return WholeBodyWeights(
        camera_pos=args.camera_pos_weight,
        camera_rot=args.camera_rot_weight,
        arm_pos=args.arm_pos_weight,
        arm_rot=args.arm_rot_weight,
        joint_rest=args.joint_rest_weight,
        joint_smooth=args.joint_smooth_weight,
    )


def _solve_whole_body_branch(
    args: argparse.Namespace,
    *,
    kin: FullBodyKinematics,
    camera_trajectory_cam0: np.ndarray,
    torso_from_cam_high: np.ndarray,
    cam0_from_base_fixed: np.ndarray,
    wrist: dict,
    indices: list[int],
) -> dict:
    return solve_whole_body(
        kin=kin,
        camera_trajectory_cam0=camera_trajectory_cam0,
        torso_from_cam_high=torso_from_cam_high,
        cam0_from_base_fixed=cam0_from_base_fixed,
        left_wrist_cam=wrist["wrist_left"],
        right_wrist_cam=wrist["wrist_right"],
        left_hand_joints=wrist["joints_left"],
        right_hand_joints=wrist["joints_right"],
        left_hand_joint_names=wrist["joint_names_left"],
        right_hand_joint_names=wrist["joint_names_right"],
        frame_indices=indices,
        initial_legwaist=np.asarray(args.initial_legwaist, dtype=np.float64),
        fixed_head=np.asarray(args.fixed_head, dtype=np.float64),
        weights=_whole_body_weights(args),
        max_iters=args.ik_max_iters,
    )


def _write_no_compliance_config(source: Path, destination: Path) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    optimization = payload.setdefault("optimization", {})
    optimization["lambda_compliance_target"] = 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def _save_branch_arrays(output_dir: Path, result: dict, wrist: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "whole_body_joints.npy", result["whole_q"])
    np.save(output_dir / "legwaist.npy", result["legwaist"])
    np.save(output_dir / "left_arm.npy", result["left_arm"])
    np.save(output_dir / "right_arm.npy", result["right_arm"])
    np.save(output_dir / "left_hand.npy", wrist["joints_left"])
    np.save(output_dir / "right_hand.npy", wrist["joints_right"])
    write_json(output_dir / "whole_body_joint_names.json", result["joint_names"])
    write_json(output_dir / "whole_body_diagnostics.json", result["diagnostics"])


def _post_filter_motion_groups(output_dir: Path, result: dict, args: argparse.Namespace) -> dict | None:
    filter_body = bool(getattr(args, "post_filter_motion", False))
    filter_hands = bool(getattr(args, "post_filter_hands", False))
    if not (filter_body or filter_hands):
        return None
    try:
        from scipy.signal import savgol_filter
    except Exception as exc:
        raise RuntimeError(
            "post-filtering requires scipy.signal.savgol_filter"
        ) from exc

    frame_count = int(result["whole_q"].shape[0])
    if frame_count < 3:
        return {
            "enabled": True,
            "skipped": True,
            "reason": f"need at least 3 frames, got {frame_count}",
        }
    window = int(args.post_filter_window)
    if window < 3:
        raise ValueError("--post_filter_window must be at least 3")
    if window % 2 == 0:
        window -= 1
    if window > frame_count:
        window = frame_count if frame_count % 2 == 1 else frame_count - 1
    polyorder = int(args.post_filter_polyorder)
    if polyorder < 0 or polyorder >= window:
        raise ValueError("--post_filter_polyorder must satisfy 0 <= polyorder < effective window")

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            previous = metadata.get("post_filter")
            if isinstance(previous, dict):
                already_body = (not filter_body) or bool(previous.get("body_motion_filtered"))
                if filter_body and "body_motion_filtered" not in previous:
                    already_body = all(
                        name in previous.get("filtered_groups", [])
                        for name in ("legwaist", "left_arm", "right_arm")
                    )
                already_hands = (not filter_hands) or bool(previous.get("hand_joints_filtered"))
                same_parameters = (
                    previous.get("method") == "scipy.signal.savgol_filter"
                    and int(previous.get("window_length", -1)) == window
                    and int(previous.get("polyorder", -1)) == polyorder
                    and previous.get("mode") == "interp"
                )
                if same_parameters and already_body and already_hands:
                    print(
                        "Using existing post-filtered motion groups "
                        f"(window={window}, polyorder={polyorder})",
                        flush=True,
                    )
                    return previous
        except Exception as exc:
            print(
                f"Warning: failed to read existing post_filter from {metadata_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    joint_names = list(result["joint_names"])
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    groups: dict[str, list[int]] = {}
    if filter_body:
        body_specs = {
            "legwaist": ("LegWaist", int(result["legwaist"].shape[1])),
            "left_arm": ("LeftArm", int(result["left_arm"].shape[1])),
            "right_arm": ("RightArm", int(result["right_arm"].shape[1])),
        }
        for array_name, (prefix, width) in body_specs.items():
            groups[array_name] = [name_to_index[f"{prefix}-{i}"] for i in range(width)]
    if filter_hands:
        hand_specs = {
            "left_hand": "left_finger",
            "right_hand": "right_finger",
        }
        for array_name, prefix in hand_specs.items():
            if array_name not in result:
                array_path = output_dir / f"{array_name}.npy"
                if not array_path.exists():
                    raise KeyError(
                        f"{array_name!r} is missing from result and {array_path} does not exist"
                    )
                result[array_name] = np.load(array_path)
            columns = [
                index
                for index, name in enumerate(joint_names)
                if name.startswith(prefix)
            ]
            expected_width = int(result[array_name].shape[1])
            if len(columns) != expected_width:
                raise ValueError(
                    f"{array_name} has width {expected_width}, but found "
                    f"{len(columns)} whole_body_joints columns with prefix {prefix!r}"
                )
            groups[array_name] = columns
    original = {
        "whole_q": np.asarray(result["whole_q"], dtype=np.float64).copy(),
    }
    for array_name in groups:
        original[array_name] = np.asarray(result[array_name], dtype=np.float64).copy()
    whole_q = original["whole_q"].copy()
    stats: dict[str, dict] = {}
    for array_name, columns in groups.items():
        source = original[array_name]
        filtered = savgol_filter(
            source,
            window_length=window,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )
        whole_q[:, columns] = filtered
        result[array_name] = filtered
        np.save(output_dir / f"{array_name}.npy", filtered)
        delta = filtered - source
        stats[array_name] = {
            "columns": columns,
            "max_abs_delta": float(np.max(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        }

    result["whole_q"] = whole_q
    np.save(output_dir / "whole_body_joints.npy", whole_q)
    whole_delta = whole_q - original["whole_q"]
    summary = {
        "enabled": True,
        "method": "scipy.signal.savgol_filter",
        "window_length": window,
        "polyorder": polyorder,
        "mode": "interp",
        "filtered_groups": list(groups),
        "body_motion_filtered": filter_body,
        "hand_joints_filtered": filter_hands,
        "stats": stats,
        "whole_body_joints_max_abs_delta": float(np.max(np.abs(whole_delta))),
        "whole_body_joints_rms_delta": float(np.sqrt(np.mean(whole_delta * whole_delta))),
    }
    write_json(output_dir / "post_filter.json", summary)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["post_filter"] = summary
            write_json(metadata_path, metadata)
        except Exception as exc:
            print(
                f"Warning: failed to update post_filter in {metadata_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    print(
        "Post-filtered motion groups "
        f"(window={window}, polyorder={polyorder}, groups={','.join(groups)})",
        flush=True,
    )
    return summary


def _load_existing_retarget_result(output_dir: Path) -> dict:
    return {
        "whole_q": np.load(output_dir / "whole_body_joints.npy"),
        "legwaist": np.load(output_dir / "legwaist.npy"),
        "left_arm": np.load(output_dir / "left_arm.npy"),
        "right_arm": np.load(output_dir / "right_arm.npy"),
        "left_hand": np.load(output_dir / "left_hand.npy"),
        "right_hand": np.load(output_dir / "right_hand.npy"),
        "joint_names": json.loads(
            (output_dir / "whole_body_joint_names.json").read_text(encoding="utf-8")
        ),
    }


def _write_loss_jsonl(
    output_dir: Path,
    *,
    actual_result: dict,
    baseline_result: dict | None,
) -> None:
    rows: list[dict] = []
    branches = [("actual", actual_result)]
    if baseline_result is not None:
        branches.append(("no_compliance", baseline_result))
    for branch, result in branches:
        for record in result["diagnostics"]:
            row = {"branch": branch, "stage": "whole_body", **record}
            rows.append(row)
    branch_dirs = [("actual", output_dir)]
    if baseline_result is not None:
        branch_dirs.append(("no_compliance", output_dir / "no_compliance"))
    for branch, branch_dir in branch_dirs:
        path = branch_dir / "wrist_free" / "optimization_info.json"
        if not path.exists():
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            rows.append({"branch": branch, "stage": "wrist_free", **record})
    with (output_dir / "losses.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rgb_o3d_mp4_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.sequence_output is not None:
        sequence_output = Path(args.sequence_output).resolve()
        name = sequence_output.name.removesuffix(".zarr")
        return sequence_output.parent / f"{name}_rgb_o3d.mp4"
    return output_dir / "rgb_o3d.mp4"


def _write_rgb_o3d_mp4(args: argparse.Namespace, output_dir: Path, rgb_o3d_path: Path) -> Path:
    mp4_path = export_rgb_o3d_mp4(
        rgb_o3d_path,
        _rgb_o3d_mp4_path(args, output_dir),
        fps=args.rgb_o3d_mp4_fps,
    )
    print(f"Saved rgb_o3d MP4: {mp4_path}", flush=True)
    return mp4_path


def _run_rgb_o3d_only(args: argparse.Namespace, output_dir: Path) -> int:
    required = [
        "metadata.json",
        "whole_body_joints.npy",
        "legwaist.npy",
        "left_arm.npy",
        "right_arm.npy",
        "left_hand.npy",
        "right_hand.npy",
        "camera_trajectory_cam0.npy",
        "base_pose_cam0.npy",
        "whole_body_joint_names.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot regenerate rgb_o3d without existing retarget outputs in "
            f"{output_dir}: {missing}"
        )
    if args.post_filter_motion or args.post_filter_hands:
        result = _load_existing_retarget_result(output_dir)
        _post_filter_motion_groups(output_dir, result, args)
    rgb_o3d_path = render_rgb_o3d_sequence(
        source_zarr_path=args.zarr_path,
        work_dir=output_dir,
        urdf_path=args.urdf,
        output_path=output_dir / "rgb_o3d.zarr",
        occlusion_epsilon_m=args.rgb_o3d_occlusion_epsilon_m,
        near_clip_m=args.rgb_o3d_near_clip_m,
        backend=args.rgb_o3d_backend,
        lighting_config_path=args.rgb_o3d_light_config,
        force=args.force_rgb_o3d,
    )
    print(f"Prepared rgb_o3d render cache: {rgb_o3d_path}", flush=True)
    _write_rgb_o3d_mp4(args, output_dir, rgb_o3d_path)
    if args.sequence_output is not None:
        from .sequence_export import export_sequence

        export_sequence(
            source_zarr_path=args.zarr_path,
            work_dir=output_dir,
            destination_zarr_path=args.sequence_output,
            layout_path=args.layout,
            urdf_path=args.urdf,
            overwrite=True,
        )
        print(
            "Updated strict training sequence rgb_o3d: "
            f"{Path(args.sequence_output).resolve()}",
            flush=True,
        )
    return 0


def run_command(args: argparse.Namespace) -> int:
    if args.expected_dof is None:
        args.expected_dof = _active_dof_from_urdf(args.urdf)
    output_dir = ensure_dir(args.output_dir)
    if args.rgb_o3d_only:
        if args.skip_rgb_o3d:
            raise ValueError("--rgb_o3d_only cannot be combined with --skip_rgb_o3d")
        return _run_rgb_o3d_only(args, output_dir)

    dataset = ZarrHumanDataset(args.zarr_path, tactile_max_gap_sec=args.tactile_max_gap_sec)
    timestamps = frame_timestamps_from_dataset(dataset) if args.target_fps is not None else None
    indices = selected_indices(
        len(dataset),
        args.start_frame,
        args.max_frames,
        args.frame_stride,
        target_fps=args.target_fps,
        timestamps=timestamps,
    )
    selection_metadata = frame_selection_metadata(
        total_frames=len(dataset),
        indices=indices,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        target_fps=args.target_fps,
        timestamps=timestamps,
    )
    if args.target_fps is not None:
        print(
            "Selected "
            f"{len(indices)}/{len(dataset)} visual frames at target_fps={args.target_fps:g} "
            f"(source_frames={indices[0]}..{indices[-1]}, "
            f"effective_fps={selection_metadata.get('effective_fps')})",
            flush=True,
        )

    camera_dir = output_dir / "apriltag_camera"
    print(
        f"Building cleaned AprilTag camera trajectory for {len(indices)} selected frames "
        f"-> {camera_dir}",
        flush=True,
    )
    camera = run_apriltag_camera_estimation(
        dataset=dataset,
        indices=indices,
        output_dir=camera_dir,
        max_reprojection_error_px=args.apriltag_max_reprojection_error_px,
        max_gap_sec=args.apriltag_max_gap_sec,
        gate_half_window=args.apriltag_gate_half_window,
        gate_mad_scale=args.apriltag_gate_mad_scale,
        gate_translation_floor_m=args.apriltag_gate_translation_floor_m,
        gate_translation_ceiling_m=args.apriltag_gate_translation_ceiling_m,
        gate_rotation_floor_deg=args.apriltag_gate_rotation_floor_deg,
        gate_rotation_ceiling_deg=args.apriltag_gate_rotation_ceiling_deg,
        gate_dilation_frames=args.apriltag_gate_dilation_frames,
        lowpass_cutoff_hz=args.apriltag_lowpass_cutoff_hz,
        pixel_sigma=args.apriltag_pixel_sigma,
        translation_accel_sigma=args.apriltag_translation_accel_sigma,
        rotation_accel_sigma=args.apriltag_rotation_accel_sigma,
        max_nfev=args.apriltag_max_nfev,
    )
    if camera.camera_trajectory_cam0.shape[0] != len(indices):
        raise ValueError(
            f"AprilTag trajectory length {camera.camera_trajectory_cam0.shape[0]} "
            f"does not match selected frames {len(indices)}"
        )
    if args.camera_only:
        print(
            "Saved cleaned AprilTag camera trajectory only: "
            f"{camera_dir} (frames={len(indices)})",
            flush=True,
        )
        return 0

    tag_from_base = load_base_in_tag(args.base_in_tag)
    cam0_from_tag = camera.cleaned_tag_to_camera[0]
    cam0_from_base_fixed = cam0_from_tag @ tag_from_base

    wrist = _run_wrist_free(args, dataset, output_dir, frame_indices=indices)
    assert_same_length(
        {
            "joints_left": wrist["joints_left"],
            "joints_right": wrist["joints_right"],
            "wrist_left": wrist["wrist_left"],
            "wrist_right": wrist["wrist_right"],
        },
        len(indices),
    )

    full_model = parse_urdf_model(args.urdf, expected_dof=args.expected_dof)
    kin = FullBodyKinematics(full_model)
    torso_from_cam_high = load_transform_from_json(args.calibration, args.calibration_key)
    result = _solve_whole_body_branch(
        args,
        kin=kin,
        camera_trajectory_cam0=camera.camera_trajectory_cam0,
        torso_from_cam_high=torso_from_cam_high,
        cam0_from_base_fixed=cam0_from_base_fixed,
        wrist=wrist,
        indices=indices,
    )

    _save_branch_arrays(output_dir, result, wrist)
    np.save(output_dir / "camera_trajectory_cam0.npy", camera.camera_trajectory_cam0)
    np.save(output_dir / "base_pose_cam0.npy", result["base_pose_cam0"])
    np.save(
        output_dir / "frame_timestamps.npy",
        np.asarray(
            [
                float(
                    dataset.frame_group(index)["realsense_timestamp"][...]
                    if "realsense_timestamp" in dataset.frame_group(index)
                    else dataset.frame_group(index)["timestamp"][...]
                )
                for index in indices
            ],
            dtype=np.float64,
        ),
    )
    post_filter_summary = _post_filter_motion_groups(output_dir, result, args)
    baseline_result = None
    if args.run_no_compliance:
        baseline_dir = output_dir / "no_compliance"
        baseline_config = _write_no_compliance_config(
            Path(args.config),
            baseline_dir / "retarget_no_compliance.yaml",
        )
        print(
            "Running counterfactual no-compliance branch "
            "(lambda_compliance_target=0) for inward-vector visualization"
        )
        baseline_wrist = _run_wrist_free(
            args,
            dataset,
            baseline_dir,
            frame_indices=indices,
            config_path=baseline_config,
        )
        assert_same_length(
            {
                "joints_left": baseline_wrist["joints_left"],
                "joints_right": baseline_wrist["joints_right"],
                "wrist_left": baseline_wrist["wrist_left"],
                "wrist_right": baseline_wrist["wrist_right"],
            },
            len(indices),
        )
        baseline_result = _solve_whole_body_branch(
            args,
            kin=kin,
            camera_trajectory_cam0=camera.camera_trajectory_cam0,
            torso_from_cam_high=torso_from_cam_high,
            cam0_from_base_fixed=cam0_from_base_fixed,
            wrist=baseline_wrist,
            indices=indices,
        )
        _save_branch_arrays(baseline_dir, baseline_result, baseline_wrist)
        np.save(baseline_dir / "base_pose_cam0.npy", baseline_result["base_pose_cam0"])
        write_json(
            baseline_dir / "metadata.json",
            {
                "branch": "counterfactual_no_compliance",
                "lambda_compliance_target": 0.0,
                "purpose": (
                    "The FK fingertip position from this branch is the inward-vector start; "
                    "the actual compliant branch fingertip position is the vector end."
                ),
            },
        )

    _write_loss_jsonl(
        output_dir,
        actual_result=result,
        baseline_result=baseline_result,
    )
    write_visualization_diagnostics(
        destination=output_dir / "visualization_diagnostics.npz",
        dataset=dataset,
        frame_indices=indices,
        camera_trajectory_cam0=camera.camera_trajectory_cam0,
        kin=kin,
        actual_q=result["whole_q"],
        base_pose_cam0=result["base_pose_cam0"],
        layout_path=args.layout,
        tactile_config=load_retarget_config(args.config).tactile,
        no_compliance_q=None if baseline_result is None else baseline_result["whole_q"],
        source_config=args.config,
    )
    write_json(
        output_dir / "diagnostics.json",
        {
            "apriltag_camera": camera.diagnostics,
            "whole_body": result["diagnostics"],
            "missing_left_hand_joints": result["missing_left_hand_joints"],
            "missing_right_hand_joints": result["missing_right_hand_joints"],
            "groups": result["groups"],
            "arm_end_links": result["arm_end_links"],
            "hand_root_links": result["hand_root_links"],
            "initial_legwaist": result["initial_legwaist"].tolist(),
            "fixed_head": result["fixed_head"].tolist(),
            "base_anchor": {
                "mode": result["base_anchor_mode"],
                "T_cam0_base": result["base_pose_cam0"].tolist(),
            },
            "initial_joint_validation": {
                "legwaist_first_frame_was_optimized": True,
                "head_all_frames_exact": bool(
                    np.all(result["whole_q"][:, kin.group_indices(["Neck-0", "Neck-1"])] == result["fixed_head"][None, :])
                ),
            },
        },
    )
    write_json(
        output_dir / "metadata.json",
        {
            "zarr_path": str(Path(args.zarr_path).resolve()),
            "urdf": str(Path(args.urdf).resolve()),
            "hand_model_source": (
                "Both palm-rooted wrist-free hand models are extracted from the full-body URDF; "
                "no standalone hand URDF is loaded."
            ),
            "calibration": str(Path(args.calibration).resolve()),
            "calibration_key": args.calibration_key,
            "frames": indices,
            "world_frame": "cam0, the first selected human camera frame",
            "fixed_base_link": "base_link",
            "program_base_frame": "torso_flange",
            "camera_frame": "cam_high",
            "arm_end_links": result["arm_end_links"],
            "hand_root_links": result["hand_root_links"],
            "fingertip_target_source": {
                "left": "T_left_sensor_to_cam",
                "right": "T_right_sensor_to_cam",
            },
            "fingertip_tracking_semantics": (
                "The full-URDF left/right_finger1..5_tip_link frames track the dataset sensor "
                "poses. T_left/right_finger_to_cam are not used as fingertip IK targets."
            ),
            "camera_trajectory_semantics": "T_cam0_cam_t",
            "camera_trajectory_source": "fixed AprilTag 10 with validity checks, outlier rejection, short-gap interpolation, and SE(3) temporal smoothing",
            "base_pose_semantics": "T_cam0_base_link, fixed for the whole sequence",
            "base_pose_source": {
                "formula": "T_cam0_tag_clean @ T_tag_base",
                "base_in_tag": str(Path(args.base_in_tag).resolve()),
                "T_tag_base": tag_from_base.tolist(),
                "tag10_pose_source": "cleaned AprilTag 10 pose at the first selected source frame",
            },
            "robot_view_calibration": {
                "source": str(Path(args.calibration).resolve()),
                "key": args.calibration_key,
                "usage": "Defines fixed head22-to-optical-view only; it does not place base_link.",
                "T_head22_camera": result["head22_from_camera"].tolist(),
            },
            "wrist_free_pose_semantics": "wrist_left.npy / wrist_right.npy are hand root palm poses; whole-body arm IK converts them to flange targets.",
            "joint_output_semantics": "whole_body_joints.npy follows whole_body_joint_names.json",
            "initial_legwaist": result["initial_legwaist"].tolist(),
            "fixed_head": result["fixed_head"].tolist(),
            "tactile_alignment": dataset.tactile_alignment_summary(),
            "frame_selection": selection_metadata,
            "post_filter": post_filter_summary,
            "no_compliance_branch": (
                "no_compliance" if baseline_result is not None else None
            ),
        },
    )

    if not args.skip_rgb_o3d:
        rgb_o3d_path = render_rgb_o3d_sequence(
            source_zarr_path=args.zarr_path,
            work_dir=output_dir,
            urdf_path=args.urdf,
            output_path=output_dir / "rgb_o3d.zarr",
            occlusion_epsilon_m=args.rgb_o3d_occlusion_epsilon_m,
            near_clip_m=args.rgb_o3d_near_clip_m,
            backend=args.rgb_o3d_backend,
            lighting_config_path=args.rgb_o3d_light_config,
            force=args.force_rgb_o3d,
        )
        print(f"Saved rgb_o3d render cache: {rgb_o3d_path}", flush=True)
        _write_rgb_o3d_mp4(args, output_dir, rgb_o3d_path)

    if args.sequence_output is not None:
        from .sequence_export import export_sequence

        export_sequence(
            source_zarr_path=args.zarr_path,
            work_dir=output_dir,
            destination_zarr_path=args.sequence_output,
            layout_path=args.layout,
            urdf_path=args.urdf,
            overwrite=args.overwrite_sequence_output,
        )
        print(f"Saved strict training sequence: {Path(args.sequence_output).resolve()}")

    print(f"Saved whole-body retarget outputs to: {output_dir}")
    print(f"whole_body_joints_shape={result['whole_q'].shape}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--zarr_path", required=True)
    common.add_argument("--urdf", default=str(DEFAULT_URDF))
    common.add_argument("--start_frame", type=int, default=0)
    common.add_argument("--max_frames", type=int, default=None)
    common.add_argument("--frame_stride", type=int, default=1)
    common.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help=(
            "Downsample visual/data frames to this FPS before retarget, "
            "rgb_o3d, and final export. Raw tactile_stream is preserved."
        ),
    )
    common.add_argument(
        "--expected_dof",
        type=int,
        default=None,
        help="Optional validation value; when omitted it is counted from the selected full URDF.",
    )

    inspect_parser = subparsers.add_parser("inspect", parents=[common], help="Inspect input zarr and URDF.")
    inspect_parser.set_defaults(func=inspect_command)

    run_parser = subparsers.add_parser("run", parents=[common], help="Run whole-body retargeting.")
    run_parser.add_argument("--output_dir", required=True)
    run_parser.add_argument(
        "--sequence_output",
        default=None,
        help="Optional strict final training-sequence Zarr destination.",
    )
    run_parser.add_argument(
        "--overwrite_sequence_output",
        action="store_true",
        help="Replace only the requested final sequence Zarr after a successful solve.",
    )
    run_parser.add_argument(
        "--run_no_compliance",
        action="store_true",
        help=(
            "Also run lambda_compliance_target=0 wrist and whole-body solves under "
            "output_dir/no_compliance for real counterfactual inward vectors."
        ),
    )
    run_parser.add_argument(
        "--skip_rgb_o3d",
        action="store_true",
        help="Skip the post-retarget full-URDF mesh render (debug use only).",
    )
    run_parser.add_argument(
        "--force_rgb_o3d",
        action="store_true",
        help="Re-render rgb_o3d even when a matching complete render cache exists.",
    )
    run_parser.add_argument(
        "--rgb_o3d_only",
        action="store_true",
        help="Only regenerate output_dir/rgb_o3d.zarr from existing retarget outputs.",
    )
    run_parser.add_argument(
        "--camera_only",
        action="store_true",
        help="Only create output_dir/apriltag_camera, then exit before retargeting.",
    )
    run_parser.add_argument(
        "--rgb_o3d_occlusion_epsilon_m",
        type=float,
        default=0.005,
        help="Depth tolerance used when compositing robot mesh against valid scene depth.",
    )
    run_parser.add_argument(
        "--rgb_o3d_near_clip_m",
        type=float,
        default=0.0,
        help="Optional debug near clipping; default 0 renders from the camera origin.",
    )
    run_parser.add_argument(
        "--rgb_o3d_backend",
        choices=("cpu", "egl", "cuda"),
        default="cpu",
        help=(
            "Renderer backend: CPU Open3D ray casting or GPU Open3D "
            "Filament through headless EGL, or NVIDIA CUDA via nvdiffrast."
        ),
    )
    run_parser.add_argument(
        "--rgb_o3d_light_config",
        type=Path,
        default=DEFAULT_LIGHTING_CONFIG,
        help=(
            "YAML file defining key/fill point lights in robot-base coordinates "
            "and the Lambert/Blinn-Phong material parameters."
        ),
    )
    run_parser.add_argument(
        "--rgb_o3d_mp4_fps",
        type=float,
        default=15.0,
        help="Frame rate for the MP4 exported from rgb_o3d.zarr.",
    )
    run_parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION),
        help=(
            "Legacy optical-view calibration. It is converted to a fixed head22-to-camera "
            "transform and is not used to place base_link."
        ),
    )
    run_parser.add_argument("--calibration_key", default=DEFAULT_CALIBRATION_KEY)
    run_parser.add_argument(
        "--base_in_tag", default=str(DEFAULT_BASE_IN_TAG),
        help="JSON {T_tag_base: 4x4 matrix} or .npy, metres; default identity.",
    )
    run_parser.add_argument(
        "--config",
        "--retarget_config",
        dest="config",
        default=str(DEFAULT_CONFIG),
        help="Retarget YAML config. --retarget_config is an alias.",
    )
    run_parser.add_argument("--layout", default=str(DEFAULT_LAYOUT))
    run_parser.add_argument("--apriltag_max_reprojection_error_px", type=float, default=1.0)
    run_parser.add_argument("--apriltag_max_gap_sec", type=float, default=0.4)
    run_parser.add_argument("--apriltag_gate_half_window", type=int, default=5)
    run_parser.add_argument("--apriltag_gate_mad_scale", type=float, default=6.0)
    run_parser.add_argument("--apriltag_gate_translation_floor_m", type=float, default=0.02)
    run_parser.add_argument("--apriltag_gate_translation_ceiling_m", type=float, default=0.06)
    run_parser.add_argument("--apriltag_gate_rotation_floor_deg", type=float, default=1.0)
    run_parser.add_argument("--apriltag_gate_rotation_ceiling_deg", type=float, default=5.0)
    run_parser.add_argument("--apriltag_gate_dilation_frames", type=int, default=1)
    run_parser.add_argument("--apriltag_lowpass_cutoff_hz", type=float, default=3.0)
    run_parser.add_argument("--apriltag_pixel_sigma", type=float, default=0.25)
    run_parser.add_argument("--apriltag_translation_accel_sigma", type=float, default=2.0)
    run_parser.add_argument("--apriltag_rotation_accel_sigma", type=float, default=8.0)
    run_parser.add_argument("--apriltag_max_nfev", type=int, default=120)
    run_parser.add_argument("--force_wrist_free", action="store_true")
    run_parser.add_argument(
        "--wrist_free_chunk_size",
        type=int,
        default=16,
        help="Run wrist-free retarget in subprocess chunks of this many selected frames.",
    )
    run_parser.add_argument(
        "--wrist_free_chunk_timeout_sec",
        type=float,
        default=240.0,
        help="Kill and split a wrist-free subprocess chunk after this many seconds.",
    )
    run_parser.add_argument(
        "--wrist_free_workers",
        type=int,
        default=1,
        help="Must remain 1 so temporal velocity/acceleration state is preserved across chunks.",
    )
    run_parser.add_argument(
        "--no_wrist_free_fallback",
        action="store_true",
        help="Fail instead of copying the previous frame if a single-frame wrist-free subprocess crashes or times out.",
    )
    run_parser.add_argument("--ik_max_iters", type=int, default=120)
    run_parser.add_argument("--camera_pos_weight", type=float, default=40.0)
    run_parser.add_argument("--camera_rot_weight", type=float, default=8.0)
    run_parser.add_argument("--arm_pos_weight", type=float, default=55.0)
    run_parser.add_argument("--arm_rot_weight", type=float, default=8.0)
    run_parser.add_argument("--joint_rest_weight", type=float, default=0.02)
    run_parser.add_argument("--joint_smooth_weight", type=float, default=0.20)
    run_parser.add_argument(
        "--post_filter_motion",
        action="store_true",
        help=(
            "After retargeting, smooth only legwaist/left_arm/right_arm and the "
            "matching whole_body_joints columns before rgb_o3d/final export."
        ),
    )
    run_parser.add_argument(
        "--post_filter_hands",
        action="store_true",
        help=(
            "After retargeting or before --rgb_o3d_only export, smooth left_hand/right_hand "
            "and matching whole_body_joints finger columns."
        ),
    )
    run_parser.add_argument(
        "--post_filter_window",
        type=int,
        default=7,
        help="Odd Savitzky-Golay window length used by --post_filter_motion.",
    )
    run_parser.add_argument(
        "--post_filter_polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order used by --post_filter_motion.",
    )
    run_parser.add_argument(
        "--initial_legwaist",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_LEGWAIST,
        metavar=("LW0", "LW1", "LW2", "LW3", "LW4", "LW5"),
    )
    run_parser.add_argument(
        "--fixed_head",
        type=float,
        nargs=2,
        default=DEFAULT_FIXED_HEAD,
        metavar=("NECK0", "NECK1"),
    )
    run_parser.add_argument(
        "--tactile_max_gap_sec",
        type=float,
        default=0.03,
        help="Maximum tactile bracket/endpoint distance accepted for timestamp alignment.",
    )
    run_parser.set_defaults(func=run_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.start_frame < 0 or args.frame_stride <= 0:
        parser.error("--start_frame must be >= 0 and --frame_stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max_frames must be positive")
    if args.target_fps is not None and args.target_fps <= 0.0:
        parser.error("--target_fps must be positive")
    if args.target_fps is not None and args.frame_stride != 1:
        parser.error("--target_fps cannot be combined with --frame_stride other than 1")
    if getattr(args, "command", None) == "run":
        if args.apriltag_max_gap_sec <= 0.0:
            parser.error("--apriltag_max_gap_sec must be positive")
        if args.apriltag_gate_half_window < 1 or args.apriltag_gate_dilation_frames < 0:
            parser.error("AprilTag gate window must be positive and dilation non-negative")
        if args.apriltag_gate_translation_floor_m > args.apriltag_gate_translation_ceiling_m:
            parser.error("AprilTag translation gate floor cannot exceed its ceiling")
        if args.apriltag_gate_rotation_floor_deg > args.apriltag_gate_rotation_ceiling_deg:
            parser.error("AprilTag rotation gate floor cannot exceed its ceiling")
        if min(
            args.apriltag_pixel_sigma,
            args.apriltag_translation_accel_sigma,
            args.apriltag_rotation_accel_sigma,
        ) <= 0.0:
            parser.error("AprilTag smoothing sigma values must be positive")
        if args.apriltag_max_nfev <= 0:
            parser.error("--apriltag_max_nfev must be positive")
    if getattr(args, "post_filter_motion", False) or getattr(args, "post_filter_hands", False):
        if args.post_filter_window < 3:
            parser.error("--post_filter_window must be at least 3")
        if args.post_filter_polyorder < 0:
            parser.error("--post_filter_polyorder must be non-negative")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
