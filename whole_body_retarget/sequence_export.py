"""Export the strict training-sequence Zarr from source data and retarget work files."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .full_body_kinematics import FullBodyKinematics
from .io_utils import ensure_project_imports

ensure_project_imports()

from retarget.robot.urdf_model import parse_urdf_model  # noqa: E402


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _sensor_names(root: Any) -> list[str]:
    value = root.attrs.get("left_tactile_sensor_names_json")
    if value is None and "left_tactile" in root:
        value = root["left_tactile"].attrs.get("sensor_names_json")
    if value is None:
        raise KeyError("Missing left tactile sensor_names_json")
    names = _json_value(value)
    if not isinstance(names, list) or len(names) != 5:
        raise ValueError(f"Expected five tactile finger names, got {names!r}")
    right_value = (
        root["right_tactile"].attrs.get("sensor_names_json")
        if "right_tactile" in root
        else None
    )
    if right_value is not None and list(_json_value(right_value)) != list(names):
        raise ValueError("Left and right tactile finger names differ")
    return [str(name) for name in names]


def _timestamp_array(group: Any) -> np.ndarray:
    key = (
        "timestamps_realsense_aligned"
        if "timestamps_realsense_aligned" in group
        else "timestamps"
    )
    if key not in group:
        raise KeyError("Tactile group has neither timestamps_realsense_aligned nor timestamps")
    return np.asarray(group[key][...], dtype=np.float64).reshape(-1)


def _tactile_array(group: Any, names: list[str], count: int) -> np.ndarray:
    if "distribution_force_json" not in group:
        raise KeyError("Tactile group is missing distribution_force_json")
    source = group["distribution_force_json"]
    if len(source) != count:
        raise ValueError(f"Tactile timestamp/JSON length mismatch: {count} vs {len(source)}")
    # These object arrays are commonly stored as one large Zarr chunk.  Reading
    # one scalar at a time would decompress that same chunk thousands of times.
    raw_samples = source[...]
    result = np.zeros((count, 5, 52, 3), dtype=np.float32)
    for sample_index in range(count):
        payload = _json_value(raw_samples[sample_index])
        if not isinstance(payload, dict):
            raise ValueError(f"Tactile sample {sample_index} is not a JSON object")
        for finger_index, name in enumerate(names):
            cells = payload.get(name, [])
            if not isinstance(cells, list):
                raise ValueError(f"Tactile sample {sample_index}, finger {name} is not a list")
            for taxel_index, cell in enumerate(cells[:52]):
                if not isinstance(cell, dict):
                    continue
                for axis, key in enumerate(("X", "Y", "Z")):
                    try:
                        result[sample_index, finger_index, taxel_index, axis] = float(
                            cell.get(key, 0.0)
                        )
                    except (TypeError, ValueError):
                        result[sample_index, finger_index, taxel_index, axis] = 0.0
        if sample_index % 500 == 0:
            print(f"[export tactile] sample={sample_index}/{count}", flush=True)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _taxel_to_fingertip(root: Any) -> np.ndarray:
    owners = [root]
    owners.extend(root[name] for name in ("left_tactile", "right_tactile") if name in root)
    for owner in owners:
        if owner is None:
            continue
        for key in ("T_sensor_to_finger_json", "taxel_to_fingertip"):
            if key in owner.attrs:
                value = np.asarray(_json_value(owner.attrs[key]), dtype=np.float64)
                if value.shape == (4, 4):
                    return value
    raise KeyError("Missing 4x4 T_sensor_to_finger_json/taxel_to_fingertip calibration")


def _strings(values: list[str]) -> np.ndarray:
    width = max(1, max(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def _create_like(group: Any, name: str, sample: np.ndarray, count: int, compressor: Any):
    return group.create_dataset(
        name,
        shape=(count, *sample.shape),
        chunks=(1, *sample.shape),
        dtype=sample.dtype,
        compressor=compressor,
    )


def _frame_timestamp(frame: Any) -> float:
    key = "realsense_timestamp" if "realsense_timestamp" in frame else "timestamp"
    return float(np.asarray(frame[key][...]).item())


def export_sequence(
    *,
    source_zarr_path: str | Path,
    work_dir: str | Path,
    destination_zarr_path: str | Path,
    layout_path: str | Path,
    urdf_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    import zarr
    from numcodecs import Blosc

    source_path = Path(source_zarr_path).resolve()
    work_dir = Path(work_dir).resolve()
    destination = Path(destination_zarr_path).resolve()
    layout_path = Path(layout_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Final sequence already exists: {destination}")
    metadata = json.loads((work_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata_source = Path(metadata["zarr_path"]).resolve()
    if metadata_source != source_path:
        raise ValueError(
            "Retarget work/source mismatch: "
            f"metadata was solved from {metadata_source}, export requested {source_path}"
        )
    frame_indices = [int(value) for value in metadata["frames"]]
    frame_count = len(frame_indices)
    if not frame_indices:
        raise ValueError("Cannot export an empty frame selection")

    source = zarr.open(str(source_path), mode="r")
    if "T_apriltag_10_to_cam" not in source:
        raise KeyError("Source Zarr is missing T_apriltag_10_to_cam")
    rgb_o3d_path = work_dir / "rgb_o3d.zarr"
    if not rgb_o3d_path.is_dir():
        raise FileNotFoundError(
            f"Missing post-retarget rgb_o3d render cache: {rgb_o3d_path}"
        )
    rgb_o3d_root = zarr.open(str(rgb_o3d_path), mode="r")
    if not bool(rgb_o3d_root.attrs.get("render_complete", False)):
        raise ValueError(f"Incomplete rgb_o3d render cache: {rgb_o3d_path}")
    if not np.array_equal(
        np.asarray(rgb_o3d_root["source_frame_index"][...], dtype=np.int64),
        np.asarray(frame_indices, dtype=np.int64),
    ):
        raise ValueError("rgb_o3d source_frame_index does not match retarget metadata")
    rgb_o3d_source = rgb_o3d_root["rgb_o3d"]
    if rgb_o3d_source.shape[0] != frame_count:
        raise ValueError(
            f"rgb_o3d frame count {rgb_o3d_source.shape[0]} != {frame_count}"
        )
    frame_keys = sorted(source["data"].keys())
    raw_tag_to_cam = np.asarray(source["T_apriltag_10_to_cam"][...], dtype=np.float64)
    if raw_tag_to_cam.shape != (len(frame_keys), 4, 4):
        raise ValueError(
            "T_apriltag_10_to_cam shape "
            f"{raw_tag_to_cam.shape} != ({len(frame_keys)}, 4, 4)"
        )
    tag_to_cam = raw_tag_to_cam[np.asarray(frame_indices, dtype=np.int64)]
    required_visual_arrays = ("inpaint_rgb", "inpaint_mask", "depth")
    optional_visual_arrays = ("mvs_left_rgb", "mvs_right_rgb")
    visual_availability = {
        name: np.zeros(frame_count, dtype=bool) for name in optional_visual_arrays
    }
    for output_index, index in enumerate(frame_indices):
        frame = source["data"][frame_keys[index]]
        missing = [
            name
            for name in required_visual_arrays
            if name not in frame
        ]
        if missing:
            raise KeyError(f"Source frame {index} missing required arrays: {missing}")
        for name in optional_visual_arrays:
            visual_availability[name][output_index] = name in frame

    arrays = {
        "legwaist": np.load(work_dir / "legwaist.npy"),
        "left_arm": np.load(work_dir / "left_arm.npy"),
        "right_arm": np.load(work_dir / "right_arm.npy"),
        "left_hand": np.load(work_dir / "left_hand.npy"),
        "right_hand": np.load(work_dir / "right_hand.npy"),
        "whole_body_joints": np.load(work_dir / "whole_body_joints.npy"),
        "camera": np.load(work_dir / "camera_trajectory_cam0.npy"),
        "base": np.load(work_dir / "base_pose_cam0.npy"),
        "wrist_left": np.load(work_dir / "wrist_free" / "wrist_left.npy"),
        "wrist_right": np.load(work_dir / "wrist_free" / "wrist_right.npy"),
    }
    for name, array in arrays.items():
        if name != "base" and array.shape[0] != frame_count:
            raise ValueError(f"{name} frame count {array.shape[0]} != {frame_count}")
    if arrays["base"].shape != (4, 4):
        raise ValueError(f"base_pose_cam0 must be 4x4, got {arrays['base'].shape}")
    joint_names = json.loads(
        (work_dir / "whole_body_joint_names.json").read_text(encoding="utf-8")
    )
    if len(joint_names) != arrays["whole_body_joints"].shape[1]:
        raise ValueError("whole-body joint name count does not match joint width")

    if urdf_path is None:
        urdf_path = metadata["urdf"]
    model = parse_urdf_model(
        urdf_path,
        expected_dof=int(arrays["whole_body_joints"].shape[1]),
    )
    kin = FullBodyKinematics(model)
    if kin.joint_names != list(joint_names):
        raise ValueError("Full URDF joint order differs from the retarget output")
    flange = {
        side: np.zeros((frame_count, 4, 4), dtype=np.float64)
        for side in ("left", "right")
    }
    for output_index in range(frame_count):
        camera_from_cam0 = np.linalg.inv(arrays["camera"][output_index])
        for side in ("left", "right"):
            flange[side][output_index] = (
                camera_from_cam0
                @ arrays["base"]
                @ kin.link_transform(
                    arrays["whole_body_joints"][output_index],
                    f"{side}_flange",
                )
            )

    names = _sensor_names(source)
    tactile_timestamps: dict[str, np.ndarray] = {}
    tactile_values: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        group = source[f"{side}_tactile"]
        tactile_timestamps[side] = _timestamp_array(group)
        tactile_values[side] = _tactile_array(
            group, names, len(tactile_timestamps[side])
        )

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    positions = np.asarray(layout["points_mm"], dtype=np.float64) / 1000.0
    rotations = np.asarray(layout["rotation_matrices"], dtype=np.float64)
    if positions.shape != (52, 3) or rotations.shape != (52, 3, 3):
        raise ValueError(
            f"Expected 52-taxel calibration, got {positions.shape}, {rotations.shape}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    if temporary.exists():
        shutil.rmtree(temporary)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    try:
        root = zarr.open(str(temporary), mode="w")
        data = root.create_group("data")
        first_frame = source["data"][frame_keys[frame_indices[0]]]
        output_samples = {
            "inpaint_rgb": np.asarray(first_frame["inpaint_rgb"][...]),
            "depth": np.asarray(first_frame["depth"][...]),
        }
        for name in optional_visual_arrays:
            sample = None
            for source_index in frame_indices:
                frame = source["data"][frame_keys[source_index]]
                if name in frame:
                    sample = np.asarray(frame[name][...])
                    break
            if sample is None:
                # Preserve a uniform training schema without fabricating a camera
                # view.  The per-frame availability array and root provenance make
                # the zero sentinel unambiguous to downstream consumers.
                sample = np.zeros_like(output_samples["inpaint_rgb"])
            output_samples[name] = sample
        output_data = {
            name: _create_like(data, name, sample, frame_count, compressor)
            for name, sample in output_samples.items()
        }
        for name in optional_visual_arrays:
            data.create_dataset(
                f"{name}_available",
                data=visual_availability[name],
                compressor=compressor,
            )
        rgb_o3d_out = _create_like(
            data,
            "rgb_o3d",
            np.asarray(rgb_o3d_source[0], dtype=np.uint8),
            frame_count,
            compressor,
        )
        intrinsic_sample = (
            np.asarray(first_frame["intrinsic"][...], dtype=np.float64)
            if "intrinsic" in first_frame
            else np.asarray(source["intrinsic"][...], dtype=np.float64)
        )
        intrinsic_out = data.create_dataset(
            "intrinsic",
            shape=(frame_count, 3, 3),
            chunks=(1, 3, 3),
            dtype=np.float64,
            compressor=compressor,
        )
        timestamps_out = data.create_dataset(
            "realsense_timestamp",
            shape=(frame_count,),
            chunks=(min(4096, frame_count),),
            dtype=np.float64,
            compressor=compressor,
        )
        data.create_dataset(
            "source_frame_index",
            data=np.asarray(frame_indices, dtype=np.int64),
            compressor=compressor,
        )
        for output_index, source_index in enumerate(frame_indices):
            frame = source["data"][frame_keys[source_index]]
            for name, target in output_data.items():
                target[output_index] = (
                    frame[name][...]
                    if name in frame
                    else np.zeros(output_samples[name].shape, dtype=output_samples[name].dtype)
                )
            rgb_o3d_out[output_index] = rgb_o3d_source[output_index]
            intrinsic_out[output_index] = (
                frame["intrinsic"][...]
                if "intrinsic" in frame
                else intrinsic_sample
            )
            timestamps_out[output_index] = _frame_timestamp(frame)
            if output_index % 100 == 0:
                print(f"[export frames] frame={output_index}/{frame_count}", flush=True)

        tactile = root.create_group("tactile_stream")
        tactile.create_dataset(
            "left_timestamp", data=tactile_timestamps["left"], compressor=compressor
        )
        tactile.create_dataset(
            "left_tactile", data=tactile_values["left"], compressor=compressor
        )
        tactile.create_dataset(
            "right_timestamp", data=tactile_timestamps["right"], compressor=compressor
        )
        tactile.create_dataset(
            "right_tactile", data=tactile_values["right"], compressor=compressor
        )

        calibration = root.create_group("calibration")
        calibration.create_dataset(
            "taxel_positions_sensor", data=positions, compressor=compressor
        )
        calibration.create_dataset(
            "taxel_rotations_sensor", data=rotations, compressor=compressor
        )
        calibration.create_dataset(
            "taxel_to_fingertip",
            data=_taxel_to_fingertip(source),
            compressor=compressor,
        )
        calibration.create_dataset("finger_names", data=_strings(names))
        calibration.create_dataset(
            "taxel_indices", data=np.arange(52, dtype=np.int64), compressor=compressor
        )

        retarget = root.create_group("retarget")
        for name in (
            "legwaist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "wrist_left",
            "wrist_right",
            "whole_body_joints",
        ):
            retarget.create_dataset(name, data=arrays[name], compressor=compressor)
        retarget.create_dataset("left_flange", data=flange["left"], compressor=compressor)
        retarget.create_dataset("right_flange", data=flange["right"], compressor=compressor)
        retarget.create_dataset("whole_body_joint_names", data=_strings(joint_names))
        retarget.create_dataset("base_pose_cam0", data=arrays["base"], compressor=compressor)
        camera_group = root.create_group("camera")
        camera_group.create_dataset(
            "camera_trajectory_cam0", data=arrays["camera"], compressor=compressor
        )
        camera_group.create_dataset(
            "T_apriltag_10_to_cam", data=tag_to_cam, compressor=compressor
        )

        attrs = {
            "schema_version": "1.1",
            "source_zarr_path": str(source_path),
            "num_frames": frame_count,
            "num_tactile_samples_left": len(tactile_timestamps["left"]),
            "num_tactile_samples_right": len(tactile_timestamps["right"]),
            "world_frame": "cam0",
            "apriltag_pose_semantics": (
                "T_apriltag_10_to_cam = T_cam_t_tag10; raw source pose copied "
                "per selected frame, without temporal filtering"
            ),
            "depth_unit": source.attrs.get("depth_unit", "meters"),
            "timestamp_unit": "seconds",
            "visual_stream_missing_policy": "zero_fill_with_per_frame_availability",
            "visual_stream_availability": {
                name: {
                    "available_frames": int(np.count_nonzero(visual_availability[name])),
                    "missing_frames": int(frame_count - np.count_nonzero(visual_availability[name])),
                }
                for name in optional_visual_arrays
            },
            "fingertip_target_source": "T_sensor_to_cam",
            "wrist_pose_frame": "camera_t",
            "wrist_left_link": "left_palm_link",
            "wrist_right_link": "right_palm_link",
            "flange_pose_frame": "camera_t",
            "left_flange_link": "left_flange",
            "right_flange_link": "right_flange",
            "export_complete": True,
        }
        frame_selection = metadata.get("frame_selection")
        if isinstance(frame_selection, dict):
            attrs.update(
                {
                    "target_fps": frame_selection.get("target_fps"),
                    "original_num_frames": frame_selection.get("original_num_frames"),
                    "downsample_source": frame_selection.get("downsample_source"),
                    "downsample_method": frame_selection.get("downsample_method"),
                    "source_frame_index_preserved": frame_selection.get(
                        "source_frame_index_preserved", True
                    ),
                    "frame_selection": frame_selection,
                }
            )
        post_filter = metadata.get("post_filter")
        if isinstance(post_filter, dict):
            attrs["post_filter"] = post_filter
        attrs["rgb_o3d_render"] = {
            "schema_version": rgb_o3d_root.attrs.get("schema_version"),
            "backend": rgb_o3d_root.attrs.get("render_backend"),
            "device": rgb_o3d_root.attrs.get("render_device"),
            "urdf": rgb_o3d_root.attrs.get("full_urdf"),
            "urdf_sha256": rgb_o3d_root.attrs.get("urdf_sha256"),
            "visual_color_semantics": rgb_o3d_root.attrs.get(
                "visual_color_semantics"
            ),
            "hand_color_override": rgb_o3d_root.attrs.get("hand_color_override"),
            "lighting_config_sha256": rgb_o3d_root.attrs.get(
                "lighting_config_sha256"
            ),
            "lighting_config": rgb_o3d_root.attrs.get("lighting_config"),
        }
        root.attrs.update(attrs)
        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
