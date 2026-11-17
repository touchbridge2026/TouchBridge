"""Tactile feature extraction in local and camera frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retarget.config import TactileConfig, TactileVisualizationConfig
from retarget.data.zarr_loader import HumanFrameData, ZarrHumanDataset
from retarget.utils import (
    FINGER_ORDER,
    FINGER_SLUGS,
    compose_homogeneous,
    css_rgb,
    rotation_x,
    rotation_y,
    rotation_z,
    safe_float,
    transform_points,
    yellow_to_red_css,
)


@dataclass(frozen=True)
class TaxelLayout:
    points_mm: np.ndarray
    points_m: np.ndarray
    rotation_matrices: np.ndarray
    corrected_rotation_matrices: np.ndarray
    corrected_points_m: np.ndarray
    corrected_transform: np.ndarray
    point_count: int

    @classmethod
    def from_json(cls, layout_path: str | Path) -> "TaxelLayout":
        payload = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        points_mm = np.asarray(payload["points_mm"], dtype=np.float64)
        points_m = points_mm / 1000.0
        if "taxel_to_fingertip_matrix" in payload:
            correction = np.asarray(payload["taxel_to_fingertip_matrix"], dtype=np.float64)
            if correction.shape != (3, 3):
                raise ValueError(f"taxel_to_fingertip_matrix must be 3x3, got {correction.shape}")
        elif "taxel_to_fingertip_rpy_deg" in payload:
            rpy_deg = payload["taxel_to_fingertip_rpy_deg"]
            if len(rpy_deg) != 3:
                raise ValueError("taxel_to_fingertip_rpy_deg must contain [roll, pitch, yaw] degrees")
            roll, pitch, yaw = [float(value) for value in rpy_deg]
            correction = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
        else:
            correction = np.eye(3, dtype=np.float64)
        rotation_matrices = np.asarray(payload["rotation_matrices"], dtype=np.float64)
        corrected_rotation_matrices = correction @ rotation_matrices
        corrected_transform = compose_homogeneous(rotation=correction)
        corrected_points_m = transform_points(corrected_transform, points_m)
        return cls(
            points_mm=points_mm,
            points_m=points_m,
            rotation_matrices=rotation_matrices,
            corrected_rotation_matrices=corrected_rotation_matrices,
            corrected_points_m=corrected_points_m,
            corrected_transform=corrected_transform,
            point_count=int(payload["point_count"]),
        )


@dataclass(frozen=True)
class FingerTactileFeature:
    hand: str
    finger: str
    finger_label: str
    taxel_count: int
    forces: np.ndarray
    forces_fingertip: np.ndarray
    force_norms: np.ndarray
    active_mask: np.ndarray
    active_taxel_count: int
    total_force: float
    contact_label: bool
    taxel_positions_local: np.ndarray
    taxel_positions_cam: np.ndarray
    patch_center_local: np.ndarray | None
    patch_center_cam: np.ndarray | None
    max_force_norm: float
    mean_force_norm: float
    raw_contact_label: bool = False
    contact_weight: float = 0.0
    contact_hold_age: int = 0


def force_vector_from_cell(cell: Any) -> np.ndarray:
    if not isinstance(cell, dict):
        return np.zeros(3, dtype=np.float64)
    vector = np.array(
        [safe_float(cell.get("X")), safe_float(cell.get("Y")), safe_float(cell.get("Z"))],
        dtype=np.float64,
    )
    vector[~np.isfinite(vector)] = 0.0
    return vector


def _taxel_size(cfg: TactileVisualizationConfig, force_norm: float, active: bool) -> float:
    if not active:
        return cfg.inactive_taxel_size
    low = cfg.active_taxel_force_threshold
    high = max(low + cfg.eps, low * 4.0)
    alpha = max(0.0, min(1.0, (force_norm - low) / (high - low)))
    return cfg.active_taxel_min_size + alpha * (cfg.active_taxel_max_size - cfg.active_taxel_min_size)


def _patch_center_size(cfg: TactileVisualizationConfig, total_force: float) -> float:
    low = cfg.patch_center_min_total_force
    high = max(low + cfg.eps, low * 6.0)
    alpha = max(0.0, min(1.0, (total_force - low) / (high - low)))
    return cfg.patch_center_min_size + alpha * (cfg.patch_center_max_size - cfg.patch_center_min_size)


def compute_finger_feature(
    hand: str,
    finger: str,
    finger_label: str,
    finger_transform_to_cam: np.ndarray,
    tactile_payload: dict[str, Any],
    layout: TaxelLayout,
    tactile_cfg: TactileConfig,
) -> FingerTactileFeature:
    cells = tactile_payload.get(finger_label, [])
    if not isinstance(cells, list):
        cells = []
    taxel_count = layout.point_count
    if taxel_count == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        empty_norm = np.zeros((0,), dtype=np.float64)
        return FingerTactileFeature(
            hand=hand,
            finger=finger,
            finger_label=finger_label,
            taxel_count=0,
            forces=empty,
            forces_fingertip=empty,
            force_norms=empty_norm,
            active_mask=np.zeros((0,), dtype=bool),
            active_taxel_count=0,
            total_force=0.0,
            contact_label=False,
            taxel_positions_local=empty,
            taxel_positions_cam=empty,
            patch_center_local=None,
            patch_center_cam=None,
            max_force_norm=0.0,
            mean_force_norm=0.0,
            raw_contact_label=False,
            contact_weight=0.0,
            contact_hold_age=0,
        )

    forces = np.zeros((taxel_count, 3), dtype=np.float64)
    force_norms = np.zeros((taxel_count,), dtype=np.float64)
    for taxel_index in range(taxel_count):
        vector = force_vector_from_cell(cells[taxel_index] if taxel_index < len(cells) else {})
        forces[taxel_index] = vector
        force_norms[taxel_index] = float(np.linalg.norm(vector))
    taxel_rotations = layout.corrected_rotation_matrices[:taxel_count]
    if taxel_rotations.shape != (taxel_count, 3, 3):
        raise ValueError(
            f"Expected {taxel_count} corrected taxel rotations, got {taxel_rotations.shape}"
        )
    forces_fingertip = np.einsum("nij,nj->ni", taxel_rotations, forces)
    local_positions = layout.corrected_points_m.copy()
    cam_positions = transform_points(finger_transform_to_cam, local_positions)
    total_force = float(np.sum(force_norms))
    active_mask = force_norms > tactile_cfg.active_taxel_force_threshold
    active_taxel_count = int(np.count_nonzero(active_mask))
    contact_label = bool(
        total_force > tactile_cfg.contact_threshold
        and active_taxel_count >= tactile_cfg.min_active_taxels
    )
    patch_center_local = None
    patch_center_cam = None
    if contact_label:
        weights = force_norms.reshape(-1, 1)
        patch_center_local = np.sum(local_positions * weights, axis=0) / (total_force + tactile_cfg.eps)
        patch_center_cam = np.sum(cam_positions * weights, axis=0) / (total_force + tactile_cfg.eps)
    return FingerTactileFeature(
        hand=hand,
        finger=finger,
        finger_label=finger_label,
        taxel_count=taxel_count,
        forces=forces,
        forces_fingertip=forces_fingertip,
        force_norms=force_norms,
        active_mask=active_mask,
        active_taxel_count=active_taxel_count,
        total_force=total_force,
        contact_label=contact_label,
        taxel_positions_local=local_positions,
        taxel_positions_cam=cam_positions,
        patch_center_local=patch_center_local,
        patch_center_cam=patch_center_cam,
        max_force_norm=float(np.max(force_norms)) if taxel_count else 0.0,
        mean_force_norm=float(np.mean(force_norms)) if taxel_count else 0.0,
        raw_contact_label=contact_label,
        contact_weight=1.0 if contact_label else 0.0,
        contact_hold_age=0,
    )


def extract_tactile_features(
    frame: HumanFrameData,
    sensor_names: list[str],
    layout: TaxelLayout,
    tactile_cfg: TactileConfig,
) -> dict[str, dict[str, FingerTactileFeature]]:
    result: dict[str, dict[str, FingerTactileFeature]] = {"left": {}, "right": {}}
    side_payloads = {
        "left": frame.left_tactile_distribution_force_json,
        "right": frame.right_tactile_distribution_force_json,
    }
    side_transforms = {
        "left": frame.T_left_sensor_to_cam,
        "right": frame.T_right_sensor_to_cam,
    }
    for side in ("left", "right"):
        for finger_index, finger_label in enumerate(sensor_names):
            finger_slug = FINGER_SLUGS.get(finger_label, finger_label)
            result[side][finger_slug] = compute_finger_feature(
                hand=side,
                finger=finger_slug,
                finger_label=finger_label,
                finger_transform_to_cam=np.asarray(side_transforms[side][finger_index], dtype=np.float64),
                tactile_payload=side_payloads[side],
                layout=layout,
                tactile_cfg=tactile_cfg,
            )
    return result


def tactile_payload_for_viewer(
    frame: HumanFrameData,
    sensor_names: list[str],
    layout: TaxelLayout,
    tactile_cfg: TactileConfig,
    visual_cfg: TactileVisualizationConfig,
    show_left: bool = True,
    show_right: bool = True,
) -> dict[str, Any]:
    features = extract_tactile_features(frame, sensor_names, layout, tactile_cfg)
    taxels: list[dict[str, Any]] = []
    patch_centers: list[dict[str, Any]] = []
    debug_stats: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    active_force_norms: list[float] = []
    for side, enabled in (("left", show_left), ("right", show_right)):
        if not enabled:
            continue
        for finger in FINGER_ORDER:
            feature = features[side][finger]
            if feature.taxel_count == 0:
                debug_stats[side][finger] = {
                    "finger_label": feature.finger_label,
                    "taxel_count": 0,
                    "max_force_norm": 0.0,
                    "mean_force_norm": 0.0,
                    "active_taxel_count": 0,
                    "total_force": 0.0,
                    "patch_center_valid": False,
                    "patch_center": None,
                }
                continue
            for index in range(feature.taxel_count):
                force_norm = float(feature.force_norms[index])
                active = bool(feature.active_mask[index])
                if active:
                    active_force_norms.append(force_norm)
                taxels.append(
                    {
                        "hand": side,
                        "finger": finger,
                        "finger_label": feature.finger_label,
                        "index": index,
                        "position": feature.taxel_positions_cam[index].tolist(),
                        "position_local": feature.taxel_positions_local[index].tolist(),
                        "force": feature.forces[index].tolist(),
                        "force_norm": force_norm,
                        "active": active,
                        "inactive_color_css": css_rgb(visual_cfg.inactive_taxel_color),
                        "display_size": _taxel_size(visual_cfg, force_norm, active),
                    }
                )
            if feature.contact_label and feature.patch_center_cam is not None and feature.patch_center_local is not None:
                patch_centers.append(
                    {
                        "hand": side,
                        "finger": finger,
                        "finger_label": feature.finger_label,
                        "position": feature.patch_center_cam.tolist(),
                        "position_local": feature.patch_center_local.tolist(),
                        "total_force": feature.total_force,
                        "active_taxel_count": feature.active_taxel_count,
                        "color_css": css_rgb(visual_cfg.patch_center_color),
                        "display_size": _patch_center_size(visual_cfg, feature.total_force),
                    }
                )
            debug_stats[side][finger] = {
                "finger_label": feature.finger_label,
                "taxel_count": feature.taxel_count,
                "max_force_norm": feature.max_force_norm,
                "mean_force_norm": feature.mean_force_norm,
                "active_taxel_count": feature.active_taxel_count,
                "total_force": feature.total_force,
                "patch_center_valid": feature.contact_label,
                "patch_center": None
                if feature.patch_center_cam is None
                else feature.patch_center_cam.tolist(),
            }

    force_scale_min = visual_cfg.active_taxel_force_threshold
    force_scale_max = max(active_force_norms) if active_force_norms else force_scale_min + 1e-6
    for item in taxels:
        item["active_color_css"] = (
            yellow_to_red_css(item["force_norm"], force_scale_min, force_scale_max)
            if item["active"]
            else item["inactive_color_css"]
        )
    return {
        "taxels": taxels,
        "patch_centers": patch_centers,
        "force_scale": {"min": force_scale_min, "max": force_scale_max},
        "debug_stats": debug_stats,
        "features": features,
    }


def print_tactile_debug(
    dataset: ZarrHumanDataset,
    layout: TaxelLayout,
    tactile_cfg: TactileConfig,
    visual_cfg: TactileVisualizationConfig,
    frame_index: int,
) -> int:
    frame = dataset.get_frame(frame_index)
    payload = tactile_payload_for_viewer(
        frame=frame,
        sensor_names=dataset.sensor_names,
        layout=layout,
        tactile_cfg=tactile_cfg,
        visual_cfg=visual_cfg,
    )
    print(f"frame_index={frame.frame_index} frame_key={frame.frame_key}")
    print(f"timestamp={frame.timestamp:.6f}")
    for side in ("left", "right"):
        print(f"[{side}]")
        for finger in FINGER_ORDER:
            stats = payload["debug_stats"][side][finger]
            print(
                f"  {finger} ({stats['finger_label']}): "
                f"taxel_count={stats['taxel_count']} "
                f"max_force_norm={stats['max_force_norm']:.6f} "
                f"mean_force_norm={stats['mean_force_norm']:.6f} "
                f"active_taxel_count={stats['active_taxel_count']} "
                f"total_force={stats['total_force']:.6f} "
                f"patch_center_valid={stats['patch_center_valid']} "
                f"patch_center={stats['patch_center']}"
            )
    return 0
