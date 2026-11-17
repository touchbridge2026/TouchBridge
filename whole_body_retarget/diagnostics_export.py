"""Write visualization-only diagnostics outside the training sequence Zarr."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from retarget.config import TactileConfig
from retarget.tactile.features import TaxelLayout, extract_tactile_features
from retarget.utils import FINGER_ORDER

from .full_body_kinematics import FullBodyKinematics


TIP_LINKS = {
    "left": [f"left_finger{i}_tip_link" for i in range(1, 6)],
    "right": [f"right_finger{i}_tip_link" for i in range(1, 6)],
}


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _finger_tip_positions(
    kin: FullBodyKinematics,
    q: np.ndarray,
    base_pose_cam0: np.ndarray,
) -> np.ndarray:
    result = np.zeros((q.shape[0], 2, 5, 3), dtype=np.float64)
    for output_index, q_frame in enumerate(q):
        links = kin.fk_all_links(q_frame)
        for side_index, side in enumerate(("left", "right")):
            for finger_index, link_name in enumerate(TIP_LINKS[side]):
                result[output_index, side_index, finger_index] = (
                    base_pose_cam0 @ links[link_name]
                )[:3, 3]
    return result


def write_visualization_diagnostics(
    *,
    destination: str | Path,
    dataset: Any,
    frame_indices: list[int],
    camera_trajectory_cam0: np.ndarray,
    kin: FullBodyKinematics,
    actual_q: np.ndarray,
    base_pose_cam0: np.ndarray,
    layout_path: str | Path,
    tactile_config: TactileConfig,
    no_compliance_q: np.ndarray | None,
    source_config: str | Path,
) -> Path:
    """Save aligned taxel/contact data and optional counterfactual fingertip positions."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    layout = TaxelLayout.from_json(layout_path)
    frame_count = len(frame_indices)
    target_taxels = np.zeros((frame_count, 2, 5, layout.point_count, 3), dtype=np.float32)
    robot_taxels = np.zeros_like(target_taxels)
    force_norms = np.zeros((frame_count, 2, 5, layout.point_count), dtype=np.float32)
    active_taxels = np.zeros((frame_count, 2, 5, layout.point_count), dtype=bool)
    contact_fingers = np.zeros((frame_count, 2, 5), dtype=bool)
    patch_centers = np.full((frame_count, 2, 5, 3), np.nan, dtype=np.float32)
    human_sensor_origins = np.full((frame_count, 2, 5, 3), np.nan, dtype=np.float32)
    tactile_valid = np.zeros((frame_count, 2), dtype=bool)
    tactile_gap_sec = np.full((frame_count, 2), np.nan, dtype=np.float64)
    actual_tip_positions = _finger_tip_positions(kin, actual_q, base_pose_cam0)

    for output_index, source_index in enumerate(frame_indices):
        frame = dataset.get_frame(source_index)
        features = extract_tactile_features(
            frame=frame,
            sensor_names=dataset.sensor_names,
            layout=layout,
            tactile_cfg=tactile_config,
        )
        links = kin.fk_all_links(actual_q[output_index])
        for side_index, side in enumerate(("left", "right")):
            tactile_valid[output_index, side_index] = bool(
                frame.left_tactile_valid if side == "left" else frame.right_tactile_valid
            )
            tactile_gap_sec[output_index, side_index] = float(
                frame.left_tactile_bracket_gap_sec
                if side == "left"
                else frame.right_tactile_bracket_gap_sec
            )
            for finger_index, finger in enumerate(FINGER_ORDER):
                feature = features[side][finger]
                source_sensor_transforms = (
                    frame.T_left_sensor_to_cam if side == "left" else frame.T_right_sensor_to_cam
                )
                if finger_index < source_sensor_transforms.shape[0]:
                    human_sensor_origins[output_index, side_index, finger_index] = (
                        _transform_points(
                            camera_trajectory_cam0[output_index],
                            np.asarray(source_sensor_transforms[finger_index, :3, 3])[None],
                        )[0]
                    )
                target_taxels[output_index, side_index, finger_index] = _transform_points(
                    camera_trajectory_cam0[output_index],
                    feature.taxel_positions_cam,
                )
                robot_transform = base_pose_cam0 @ links[TIP_LINKS[side][finger_index]]
                robot_taxels[output_index, side_index, finger_index] = _transform_points(
                    robot_transform,
                    layout.corrected_points_m,
                )
                force_norms[output_index, side_index, finger_index] = feature.force_norms
                active_taxels[output_index, side_index, finger_index] = feature.active_mask
                contact_fingers[output_index, side_index, finger_index] = feature.contact_label
                if feature.patch_center_cam is not None:
                    patch_centers[output_index, side_index, finger_index] = _transform_points(
                        camera_trajectory_cam0[output_index],
                        np.asarray(feature.patch_center_cam, dtype=np.float64)[None],
                    )[0]

    no_compliance_tip_positions = (
        _finger_tip_positions(kin, no_compliance_q, base_pose_cam0)
        if no_compliance_q is not None
        else np.zeros((0, 2, 5, 3), dtype=np.float64)
    )
    np.savez_compressed(
        destination,
        source_frame_index=np.asarray(frame_indices, dtype=np.int64),
        target_taxels_cam0=target_taxels,
        robot_taxels_cam0=robot_taxels,
        force_norms=force_norms,
        active_taxel_mask=active_taxels,
        contact_finger_mask=contact_fingers,
        patch_centers_cam0=patch_centers,
        human_sensor_origins_cam0=human_sensor_origins,
        tactile_valid=tactile_valid,
        tactile_bracket_gap_sec=tactile_gap_sec,
        actual_tip_positions_cam0=actual_tip_positions.astype(np.float32),
        no_compliance_tip_positions_cam0=no_compliance_tip_positions.astype(np.float32),
        has_no_compliance_branch=np.asarray(no_compliance_q is not None, dtype=bool),
    )
    metadata = {
        "source_zarr_path": str(dataset.zarr_path),
        "source_config": str(Path(source_config).resolve()),
        "layout": str(Path(layout_path).resolve()),
        "world_frame": "cam0",
        "inward_vector_semantics": (
            "For contact fingers only: no-compliance full-URDF fingertip position -> "
            "normal compliance-enabled full-URDF fingertip position."
        ),
        "has_no_compliance_branch": no_compliance_q is not None,
        "finger_order": list(FINGER_ORDER),
        "side_order": ["left", "right"],
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def write_loss_jsonl(
    destination: str | Path,
    branches: dict[str, dict[str, list[dict[str, Any]]]],
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for branch_name, stages in branches.items():
            for stage_name, records in stages.items():
                for output_index, record in enumerate(records):
                    payload = {
                        "branch": branch_name,
                        "stage": stage_name,
                        "output_index": output_index,
                        **record,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return destination
