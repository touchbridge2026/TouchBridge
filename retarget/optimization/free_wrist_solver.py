"""Base-soft dual-hand retargeting solver driven by human fingertips."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from retarget.config import RetargetConfig
from retarget.data.zarr_loader import HumanFrameData, ZarrHumanDataset
from retarget.robot.fk import CompiledHandKinematics, HandKinematics
from retarget.robot.urdf_model import URDFModel
from retarget.tactile.features import FingerTactileFeature, TaxelLayout, extract_tactile_features
from retarget.utils import (
    FINGER_ORDER,
    finite_or_zeros,
    homogeneous_from_rotvec_translation,
    invert_transform,
    is_valid_transform,
    pairwise,
    relative_rotation_log,
    relative_rotation_log_so3,
    rotation_log_so3,
    transform_point,
)


@dataclass
class FrameRetargetTargets:
    left_base_anchor: np.ndarray
    right_base_anchor: np.ndarray
    left_tip_targets: dict[str, np.ndarray]
    right_tip_targets: dict[str, np.ndarray]
    left_tip_targets_anchor_local: dict[str, np.ndarray]
    right_tip_targets_anchor_local: dict[str, np.ndarray]
    left_tip_target_rotations: dict[str, np.ndarray]
    right_tip_target_rotations: dict[str, np.ndarray]
    tactile_features: dict[str, dict[str, FingerTactileFeature]]


def _hand_scale(config: RetargetConfig, hand: str) -> float:
    return config.mapping.hand_scale_left if hand == "left" else config.mapping.hand_scale_right


def _tip_name_map(model: URDFModel) -> dict[str, str]:
    finger_keys = [f"finger{index + 1}" for index in range(5)]
    mapping: dict[str, str] = {}
    lower_links = {link.lower(): link for link in model.tip_links}
    for index, finger in enumerate(FINGER_ORDER):
        key = finger_keys[index]
        matched = next((name for low, name in lower_links.items() if key in low), None)
        if matched is None and index < len(model.tip_links):
            matched = model.tip_links[index]
        if matched is not None:
            mapping[finger] = matched
    return mapping


def _q_midpoint(model: URDFModel) -> np.ndarray:
    return 0.5 * (model.joint_limits_lower + model.joint_limits_upper)


def _rigid_kabsch(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch alignment expects Nx3 source/target point arrays.")
    if source.shape[0] == 0:
        return np.eye(4, dtype=np.float64)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u, _, vh = np.linalg.svd(covariance, full_matrices=False)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _graph_edges(active_fingers_left: list[str], active_fingers_right: list[str]) -> list[tuple[str, str]]:
    # Only fingertip/contact nodes belong to the tactile graph.
    edges: list[tuple[str, str]] = []
    for finger in active_fingers_left:
        edges.append((f"left_tip_{finger}", f"left_patch_{finger}"))
    for finger in active_fingers_right:
        edges.append((f"right_tip_{finger}", f"right_patch_{finger}"))
    for first, second in pairwise([f"left_tip_{finger}" for finger in active_fingers_left]):
        edges.append((first, second))
    for first, second in pairwise([f"right_tip_{finger}" for finger in active_fingers_right]):
        edges.append((first, second))
    for left in active_fingers_left:
        for right in active_fingers_right:
            edges.append((f"left_tip_{left}", f"right_tip_{right}"))
            edges.append((f"left_patch_{left}", f"right_patch_{right}"))
    return edges


def _laplacian_residual(
    source_nodes: dict[str, np.ndarray],
    robot_nodes: dict[str, np.ndarray],
    node_force: dict[str, float],
    config: RetargetConfig,
) -> np.ndarray:
    node_names = [name for name in source_nodes if name in robot_nodes]
    if not node_names:
        return np.zeros((0,), dtype=np.float64)
    active_left = [name.split("_")[-1] for name in node_names if name.startswith("left_patch_")]
    active_right = [name.split("_")[-1] for name in node_names if name.startswith("right_patch_")]
    edges = _graph_edges(active_left, active_right)
    neighbor_map: dict[str, list[tuple[str, float]]] = {name: [] for name in node_names}
    for first, second in edges:
        if first not in source_nodes or second not in source_nodes:
            continue
        distance = float(np.linalg.norm(source_nodes[first] - source_nodes[second]))
        weight_raw = np.exp(-config.optimization.kappa * distance) * (
            1.0
            + config.optimization.tactile_gamma
            * (node_force.get(first, 0.0) + node_force.get(second, 0.0))
        )
        neighbor_map.setdefault(first, []).append((second, weight_raw))
        neighbor_map.setdefault(second, []).append((first, weight_raw))
    residuals: list[np.ndarray] = []
    for node in node_names:
        source_neighbors = neighbor_map.get(node, [])
        if not source_neighbors:
            continue
        normalizer = sum(weight for _, weight in source_neighbors)
        if normalizer <= 1e-12:
            continue
        source_delta = np.zeros(3, dtype=np.float64)
        robot_delta = np.zeros(3, dtype=np.float64)
        for neighbor, raw_weight in source_neighbors:
            weight = raw_weight / normalizer
            source_delta += weight * (source_nodes[node] - source_nodes[neighbor])
            robot_delta += weight * (robot_nodes[node] - robot_nodes[neighbor])
        residuals.append(
            np.sqrt(config.optimization.lambda_tactile_lap) * (robot_delta - source_delta)
        )
    return np.concatenate(residuals) if residuals else np.zeros((0,), dtype=np.float64)


def _collision_proxy(points: dict[str, np.ndarray], config: RetargetConfig) -> np.ndarray:
    values = np.asarray(list(points.values()), dtype=np.float64)
    if values.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    first, second = np.triu_indices(values.shape[0], k=1)
    distances = np.linalg.norm(values[first] - values[second], axis=1)
    return (
        np.sqrt(config.optimization.lambda_finger_collision)
        * np.maximum(0.0, config.optimization.d_min - distances)
    )


class FreeWristRetargetSolver:
    """TopoRetarget-inspired retargeting without trusted human wrist supervision.

    This implementation is fingertip-driven with soft base constraints. Since object
    mesh and object pose are unavailable, it preserves bimanual wrist-fingertip-tactile
    interaction structure rather than exact hand-object surface contact. It cannot
    compute true hand-object penetration.
    """

    def __init__(
        self,
        dataset: ZarrHumanDataset,
        layout: TaxelLayout,
        left_model: URDFModel,
        right_model: URDFModel,
        config: RetargetConfig,
    ) -> None:
        self.dataset = dataset
        self.layout = layout
        self.left_model = left_model
        self.right_model = right_model
        kinematics = (
            CompiledHandKinematics
            if config.optimization.fast_cpu_math
            else HandKinematics
        )
        self.left_fk = kinematics(left_model)
        self.right_fk = kinematics(right_model)
        self._rotation_log = (
            rotation_log_so3
            if config.optimization.fast_cpu_math
            else lambda matrix: Rotation.from_matrix(matrix).as_rotvec()
        )
        self._relative_rotation_log = (
            relative_rotation_log_so3
            if config.optimization.fast_cpu_math
            else relative_rotation_log
        )
        self.config = config
        self._finger_weights = {
            hand: {
                finger: float((config.optimization.finger_weights or {}).get(finger, 1.0))
                * float(
                    (
                        config.optimization.left_finger_weights
                        if hand == "left"
                        else config.optimization.right_finger_weights
                    ).get(finger, 1.0)
                )
                for finger in FINGER_ORDER
            }
            for hand in ("left", "right")
        }
        invalid_weights = {
            f"{hand}.{finger}": weight
            for hand, values in self._finger_weights.items()
            for finger, weight in values.items()
            if not np.isfinite(weight) or weight < 0.0
        }
        if invalid_weights:
            raise ValueError(f"Finger weights must be finite and non-negative: {invalid_weights}")
        self.left_tip_map = _tip_name_map(left_model)
        self.right_tip_map = _tip_name_map(right_model)
        self.left_neutral_q = _q_midpoint(self.left_model)
        self.right_neutral_q = _q_midpoint(self.right_model)
        self.left_neutral_tip_positions = self._neutral_tip_positions(
            self.left_fk, self.left_neutral_q, self.left_tip_map
        )
        self.right_neutral_tip_positions = self._neutral_tip_positions(
            self.right_fk, self.right_neutral_q, self.right_tip_map
        )
        self._frame_targets_cache: dict[int, FrameRetargetTargets] = {}
        self._laplacian_cache: dict[
            int,
            list[tuple[str, tuple[tuple[str, float], ...], np.ndarray]],
        ] = {}

    def _neutral_tip_positions(
        self,
        fk: HandKinematics,
        q: np.ndarray,
        tip_map: dict[str, str],
    ) -> dict[str, np.ndarray]:
        local_tips = fk.fk_tips(q)
        result: dict[str, np.ndarray] = {}
        for finger, tip_name in tip_map.items():
            if tip_name in local_tips:
                result[finger] = local_tips[tip_name][:3, 3].copy()
        return result

    def _human_tip_targets(
        self,
        sensor_to_cam: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        tip_targets: dict[str, np.ndarray] = {}
        tip_rotations: dict[str, np.ndarray] = {}
        for finger_index, finger in enumerate(FINGER_ORDER):
            if finger_index >= sensor_to_cam.shape[0]:
                continue
            transform = np.asarray(sensor_to_cam[finger_index], dtype=np.float64)
            if not is_valid_transform(transform):
                continue
            tip_targets[finger] = transform
            # Sensor poses can arrive as float32-derived matrices with ~1e-7
            # orthogonality error. Project once here instead of making every
            # residual evaluation project the same target rotation again.
            tip_rotations[finger] = Rotation.from_matrix(
                transform[:3, :3]
            ).as_matrix()
        return tip_targets, tip_rotations

    def _compute_base_anchor(
        self,
        hand: str,
        tip_targets: dict[str, np.ndarray],
    ) -> np.ndarray:
        neutral_positions = (
            self.left_neutral_tip_positions if hand == "left" else self.right_neutral_tip_positions
        )
        source_points: list[np.ndarray] = []
        target_points: list[np.ndarray] = []
        for finger in FINGER_ORDER:
            if finger not in neutral_positions or finger not in tip_targets:
                continue
            source_points.append(neutral_positions[finger])
            target_points.append(tip_targets[finger][:3, 3])
        if len(source_points) < 3:
            anchor = np.eye(4, dtype=np.float64)
            if target_points:
                anchor[:3, 3] = np.mean(np.asarray(target_points, dtype=np.float64), axis=0)
            return anchor
        return _rigid_kabsch(np.asarray(source_points, dtype=np.float64), np.asarray(target_points, dtype=np.float64))

    def _anchor_local_tip_targets(
        self,
        hand: str,
        anchor: np.ndarray,
        tip_targets: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        _ = hand
        anchor_inv = invert_transform(anchor)
        result: dict[str, np.ndarray] = {}
        for finger, target in tip_targets.items():
            result[finger] = transform_point(anchor_inv, target[:3, 3])
        return result

    def _frame_targets(self, frame: HumanFrameData) -> FrameRetargetTargets:
        tactile_features = extract_tactile_features(
            frame=frame,
            sensor_names=self.dataset.sensor_names,
            layout=self.layout,
            tactile_cfg=self.config.tactile,
        )
        # The full-URDF finger_tip_link frames are calibrated to the tactile
        # sensor origins, so their IK targets must use the sensor poses rather
        # than the legacy human finger frames.
        left_tip_targets, left_tip_rotations = self._human_tip_targets(frame.T_left_sensor_to_cam)
        right_tip_targets, right_tip_rotations = self._human_tip_targets(frame.T_right_sensor_to_cam)
        left_base_anchor = self._compute_base_anchor("left", left_tip_targets)
        right_base_anchor = self._compute_base_anchor("right", right_tip_targets)
        return FrameRetargetTargets(
            left_base_anchor=left_base_anchor,
            right_base_anchor=right_base_anchor,
            left_tip_targets=left_tip_targets,
            right_tip_targets=right_tip_targets,
            left_tip_targets_anchor_local=self._anchor_local_tip_targets(
                "left", left_base_anchor, left_tip_targets
            ),
            right_tip_targets_anchor_local=self._anchor_local_tip_targets(
                "right", right_base_anchor, right_tip_targets
            ),
            left_tip_target_rotations=left_tip_rotations,
            right_tip_target_rotations=right_tip_rotations,
            tactile_features=tactile_features,
        )

    def _cached_frame_targets(
        self,
        frame_index: int,
        frame: HumanFrameData | None = None,
    ) -> FrameRetargetTargets:
        cached = self._frame_targets_cache.get(int(frame_index))
        if cached is not None:
            return cached
        if frame is None:
            frame = self.dataset.get_frame(int(frame_index))
        targets = self._frame_targets(frame)
        self._frame_targets_cache[int(frame_index)] = targets
        return targets

    def _pack(
        self,
        left_base: np.ndarray,
        left_q: np.ndarray,
        right_base: np.ndarray,
        right_q: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                left_base[:3, 3],
                Rotation.from_matrix(left_base[:3, :3]).as_rotvec(),
                left_q,
                right_base[:3, 3],
                Rotation.from_matrix(right_base[:3, :3]).as_rotvec(),
                right_q,
            ]
        )

    def _unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        left_pos = z[0:3]
        left_rot = z[3:6]
        left_q = z[6 : 6 + self.left_model.dof]
        offset = 6 + self.left_model.dof
        right_pos = z[offset : offset + 3]
        right_rot = z[offset + 3 : offset + 6]
        right_q = z[offset + 6 : offset + 6 + self.right_model.dof]
        return (
            homogeneous_from_rotvec_translation(left_rot, left_pos),
            left_q,
            homogeneous_from_rotvec_translation(right_rot, right_pos),
            right_q,
        )

    def _state_dict(
        self,
        left_base: np.ndarray,
        left_q: np.ndarray,
        right_base: np.ndarray,
        right_q: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "left_wrist": left_base.copy(),
            "right_wrist": right_base.copy(),
            "left_q": np.asarray(left_q, dtype=np.float64).copy(),
            "right_q": np.asarray(right_q, dtype=np.float64).copy(),
            "z": self._pack(left_base, left_q, right_base, right_q),
        }

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        rot_bound = self.config.optimization.rotvec_bound
        lower = np.concatenate(
            [
                np.full(3, -np.inf),
                np.full(3, -rot_bound),
                self.left_model.joint_limits_lower,
                np.full(3, -np.inf),
                np.full(3, -rot_bound),
                self.right_model.joint_limits_lower,
            ]
        )
        upper = np.concatenate(
            [
                np.full(3, np.inf),
                np.full(3, rot_bound),
                self.left_model.joint_limits_upper,
                np.full(3, np.inf),
                np.full(3, rot_bound),
                self.right_model.joint_limits_upper,
            ]
        )
        return lower, upper

    def _finger_beta(self, feature: FingerTactileFeature) -> float:
        return self._finger_weight(feature.hand, feature.finger) * (
            self.config.optimization.beta_base
            + self.config.optimization.beta_contact * float(feature.raw_contact_label)
        )

    def _tip_tracking_beta(self, feature: FingerTactileFeature) -> float:
        scale = self.config.optimization.contact_tip_tracking_scale if feature.raw_contact_label else 1.0
        return self._finger_beta(feature) * max(0.0, float(scale))

    def _tactile_beta(self, feature: FingerTactileFeature) -> float:
        contact_weight = max(0.0, float(feature.contact_weight))
        return (
            self._finger_weight(feature.hand, feature.finger)
            * (self.config.optimization.beta_base + self.config.optimization.beta_contact)
            * contact_weight
        )

    def _finger_weight(self, hand: str, finger: str) -> float:
        return self._finger_weights[hand][finger]

    def _fast_laplacian_residual(
        self,
        targets: FrameRetargetTargets,
        source_nodes: dict[str, np.ndarray],
        robot_nodes: dict[str, np.ndarray],
        node_force: dict[str, float],
    ) -> np.ndarray:
        cache_key = id(targets)
        prepared = self._laplacian_cache.get(cache_key)
        if prepared is None:
            node_names = [name for name in source_nodes if name in robot_nodes]
            active_left = [
                name.split("_")[-1]
                for name in node_names
                if name.startswith("left_patch_")
            ]
            active_right = [
                name.split("_")[-1]
                for name in node_names
                if name.startswith("right_patch_")
            ]
            neighbors: dict[str, list[tuple[str, float]]] = {
                name: [] for name in node_names
            }
            for first, second in _graph_edges(active_left, active_right):
                if first not in source_nodes or second not in source_nodes:
                    continue
                distance = float(np.linalg.norm(source_nodes[first] - source_nodes[second]))
                raw_weight = np.exp(-self.config.optimization.kappa * distance) * (
                    1.0
                    + self.config.optimization.tactile_gamma
                    * (node_force.get(first, 0.0) + node_force.get(second, 0.0))
                )
                neighbors.setdefault(first, []).append((second, raw_weight))
                neighbors.setdefault(second, []).append((first, raw_weight))
            prepared = []
            for node in node_names:
                entries = neighbors.get(node, [])
                normalizer = sum(weight for _, weight in entries)
                if not entries or normalizer <= 1.0e-12:
                    continue
                normalized = tuple(
                    (neighbor, raw_weight / normalizer)
                    for neighbor, raw_weight in entries
                )
                source_delta = np.zeros(3, dtype=np.float64)
                for neighbor, weight in normalized:
                    source_delta += weight * (source_nodes[node] - source_nodes[neighbor])
                prepared.append((node, normalized, source_delta))
            self._laplacian_cache[cache_key] = prepared
        residuals = []
        scale = np.sqrt(self.config.optimization.lambda_tactile_lap)
        for node, neighbors, source_delta in prepared:
            robot_delta = np.zeros(3, dtype=np.float64)
            for neighbor, weight in neighbors:
                robot_delta += weight * (robot_nodes[node] - robot_nodes[neighbor])
            residuals.append(scale * (robot_delta - source_delta))
        return np.concatenate(residuals) if residuals else np.zeros((0,), dtype=np.float64)

    def _frame_finger_weights(self) -> dict[str, dict[str, float]]:
        return {
            hand: {
                finger: self._finger_weight(hand, finger)
                for finger in FINGER_ORDER
            }
            for hand in ("left", "right")
        }

    def _hand_beta(self, targets: FrameRetargetTargets, hand: str) -> float:
        has_contact = any(
            targets.tactile_features[hand][finger].contact_label for finger in FINGER_ORDER
        )
        return self.config.optimization.beta_base + self.config.optimization.beta_contact * float(has_contact)

    def _active_taxel_count(self, targets: FrameRetargetTargets) -> int:
        return sum(
            int(targets.tactile_features[hand][finger].active_taxel_count)
            for hand in ("left", "right")
            for finger in FINGER_ORDER
        )

    def _effective_contact_targets(
        self,
        targets: FrameRetargetTargets,
        frame_index: int,
        frame_stride: int,
        previous_frame_indices: Sequence[int] | None = None,
    ) -> FrameRetargetTargets:
        hold_frames = max(0, int(self.config.optimization.contact_hold_frames))
        if hold_frames <= 0:
            return targets

        features = {
            hand: dict(targets.tactile_features[hand])
            for hand in ("left", "right")
        }
        for hand in ("left", "right"):
            for finger in FINGER_ORDER:
                current = features[hand][finger]
                if current.raw_contact_label:
                    features[hand][finger] = replace(
                        current,
                        contact_label=True,
                        contact_weight=1.0,
                        contact_hold_age=0,
                    )
                    continue

                held_source: FingerTactileFeature | None = None
                hold_age = 0
                if previous_frame_indices is None:
                    history = [
                        frame_index - age * frame_stride
                        for age in range(1, hold_frames + 1)
                        if frame_index - age * frame_stride >= 0
                    ]
                else:
                    history = list(reversed(list(previous_frame_indices)[-hold_frames:]))
                for age, previous_index in enumerate(history, start=1):
                    previous_targets = self._cached_frame_targets(previous_index)
                    previous_feature = previous_targets.tactile_features[hand][finger]
                    if previous_feature.raw_contact_label and previous_feature.patch_center_local is not None:
                        held_source = previous_feature
                        hold_age = age
                        break
                if held_source is None:
                    features[hand][finger] = replace(
                        current,
                        contact_label=False,
                        raw_contact_label=False,
                        contact_weight=0.0,
                        contact_hold_age=0,
                    )
                    continue

                decay = (hold_frames - hold_age + 1) / (hold_frames + 1)
                current_sensor_from_local = _rigid_kabsch(
                    current.taxel_positions_local,
                    current.taxel_positions_cam,
                )
                held_patch_cam = transform_point(
                    current_sensor_from_local,
                    held_source.patch_center_local,
                )
                features[hand][finger] = replace(
                    current,
                    forces=held_source.forces * decay,
                    forces_fingertip=held_source.forces_fingertip * decay,
                    force_norms=held_source.force_norms * decay,
                    active_mask=held_source.active_mask.copy(),
                    active_taxel_count=int(held_source.active_taxel_count),
                    total_force=float(held_source.total_force) * decay,
                    contact_label=True,
                    patch_center_local=held_source.patch_center_local.copy(),
                    patch_center_cam=held_patch_cam,
                    max_force_norm=float(held_source.max_force_norm) * decay,
                    mean_force_norm=float(held_source.mean_force_norm) * decay,
                    raw_contact_label=False,
                    contact_weight=float(decay),
                    contact_hold_age=int(hold_age),
                )

        return replace(targets, tactile_features=features)

    def _compliance_target(
        self,
        feature: FingerTactileFeature,
        human_tip: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        cfg = self.config.optimization
        if (
            cfg.lambda_compliance_target <= 0.0
            or cfg.compliance_stiffness <= self.config.tactile.eps
            or cfg.compliance_max_depth <= 0.0
            or not feature.contact_label
            or feature.patch_center_cam is None
        ):
            return None
        force = max(0.0, float(feature.total_force) - cfg.compliance_min_force)
        depth = min(cfg.compliance_max_depth, force / cfg.compliance_stiffness)
        if depth <= self.config.tactile.eps:
            return None

        direction_cam = feature.patch_center_cam - human_tip[:3, 3]
        direction_norm = float(np.linalg.norm(direction_cam))
        if direction_norm <= self.config.tactile.eps:
            return None
        direction_cam = direction_cam / direction_norm
        # +1 extends from the sensor origin through the measured patch into
        # the object (paper Eq. 15). This is a geometric direction, not a force vector.
        direction_sign = float(cfg.compliance_force_direction_sign)
        if not np.isfinite(direction_sign):
            raise ValueError("compliance_force_direction_sign must be finite")
        return (
            feature.patch_center_cam + direction_sign * depth * direction_cam,
            float(depth),
        )

    def _realign_hand_state(
        self,
        previous_base: np.ndarray,
        previous_q: np.ndarray,
        tip_targets: dict[str, np.ndarray],
        fk: HandKinematics,
        tip_map: dict[str, str],
    ) -> np.ndarray:
        local_tips = fk.fk_tips(previous_q)
        source_points: list[np.ndarray] = []
        target_points: list[np.ndarray] = []
        for finger in FINGER_ORDER:
            tip_name = tip_map.get(finger)
            if tip_name is None or tip_name not in local_tips or finger not in tip_targets:
                continue
            source_points.append((previous_base @ local_tips[tip_name])[:3, 3])
            target_points.append(tip_targets[finger][:3, 3])
        if len(source_points) < 3:
            return previous_base.copy()
        align = _rigid_kabsch(np.asarray(source_points, dtype=np.float64), np.asarray(target_points, dtype=np.float64))
        return align @ previous_base

    def _make_initial_state_from_neutral(self, targets: FrameRetargetTargets) -> dict[str, Any]:
        return self._state_dict(
            left_base=targets.left_base_anchor,
            left_q=self.left_neutral_q,
            right_base=targets.right_base_anchor,
            right_q=self.right_neutral_q,
        )

    def _candidate_score(self, state: dict[str, Any], targets: FrameRetargetTargets) -> float:
        diagnostics = self._diagnostics(
            state["left_wrist"],
            state["left_q"],
            state["right_wrist"],
            state["right_q"],
            targets,
        )
        return float(
            diagnostics["tip_global_rmse"]
            + 0.35 * diagnostics["tip_local_anchor_rmse"]
            + 0.10 * diagnostics["patch_rmse"]
        )

    def _choose_initial_state(
        self,
        targets: FrameRetargetTargets,
        previous: dict[str, Any] | None,
        previous_previous: dict[str, Any] | None,
    ) -> dict[str, Any]:
        neutral_candidate = self._make_initial_state_from_neutral(targets)
        if previous is None:
            return neutral_candidate
        raw_candidate = self._state_dict(
            left_base=previous["left_wrist"],
            left_q=previous["left_q"],
            right_base=previous["right_wrist"],
            right_q=previous["right_q"],
        )
        aligned_candidate = self._state_dict(
            left_base=self._realign_hand_state(
                previous["left_wrist"],
                previous["left_q"],
                targets.left_tip_targets,
                self.left_fk,
                self.left_tip_map,
            ),
            left_q=previous["left_q"],
            right_base=self._realign_hand_state(
                previous["right_wrist"],
                previous["right_q"],
                targets.right_tip_targets,
                self.right_fk,
                self.right_tip_map,
            ),
            right_q=previous["right_q"],
        )
        candidates = [raw_candidate, aligned_candidate, neutral_candidate]
        return min(candidates, key=lambda state: self._candidate_score(state, targets))

    def _solve_from_init(
        self,
        targets: FrameRetargetTargets,
        init_state: dict[str, Any],
        previous: dict[str, Any] | None,
        previous_previous: dict[str, Any] | None,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, bool, str | None, dict[str, Any]]:
        z0 = init_state["z"].copy()
        failure_reason = None
        solve_info: dict[str, Any] = {}
        try:
            result = least_squares(
                lambda z: self._residual(z, targets, init_state, previous, previous_previous),
                x0=z0,
                bounds=bounds,
                max_nfev=self.config.optimization.max_iters,
                verbose=self.config.optimization.verbose,
            )
            z_best = result.x
            success = bool(result.success)
            if not success:
                failure_reason = result.message
            solve_info = {
                "scipy_cost": float(result.cost),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "optimality": float(result.optimality),
                "status": int(result.status),
                "message": str(result.message),
            }
        except Exception as exc:  # pragma: no cover - fallback path
            success = False
            failure_reason = str(exc)
            z_best = previous["z"] if previous is not None else z0
            solve_info = {
                "scipy_cost": None,
                "nfev": 0,
                "njev": None,
                "optimality": None,
                "status": None,
                "message": str(exc),
            }
        return z_best, success, failure_reason, solve_info

    def _nodes_from_state(
        self,
        left_base: np.ndarray,
        left_tip_transforms: dict[str, np.ndarray],
        right_base: np.ndarray,
        right_tip_transforms: dict[str, np.ndarray],
        targets: FrameRetargetTargets,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
        source_nodes: dict[str, np.ndarray] = {}
        robot_nodes: dict[str, np.ndarray] = {}
        node_force: dict[str, float] = {}
        for hand, tip_targets, tip_transforms, tip_map in (
            ("left", targets.left_tip_targets, left_tip_transforms, self.left_tip_map),
            ("right", targets.right_tip_targets, right_tip_transforms, self.right_tip_map),
        ):
            for finger in FINGER_ORDER:
                tip_name = tip_map.get(finger)
                if tip_name is None or tip_name not in tip_transforms or finger not in tip_targets:
                    continue
                source_nodes[f"{hand}_tip_{finger}"] = tip_targets[finger][:3, 3]
                robot_nodes[f"{hand}_tip_{finger}"] = tip_transforms[tip_name][:3, 3]
                feature = targets.tactile_features[hand][finger]
                node_force[f"{hand}_tip_{finger}"] = feature.total_force
                if (
                    feature.contact_label
                    and feature.patch_center_local is not None
                    and feature.patch_center_cam is not None
                ):
                    robot_patch = (
                        tip_transforms[tip_name][:3, :3] @ feature.patch_center_local
                        + tip_transforms[tip_name][:3, 3]
                    )
                    source_nodes[f"{hand}_patch_{finger}"] = feature.patch_center_cam
                    robot_nodes[f"{hand}_patch_{finger}"] = robot_patch
                    node_force[f"{hand}_patch_{finger}"] = feature.total_force
        return source_nodes, robot_nodes, node_force

    def _tip_sets(
        self,
        left_base: np.ndarray,
        left_q: np.ndarray,
        right_base: np.ndarray,
        right_q: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        left_tip_local = self.left_fk.fk_tips(left_q)
        right_tip_local = self.right_fk.fk_tips(right_q)
        left_tip_world = {
            name: left_base @ transform for name, transform in left_tip_local.items()
        }
        right_tip_world = {
            name: right_base @ transform for name, transform in right_tip_local.items()
        }
        return left_tip_local, left_tip_world, right_tip_local, right_tip_world

    def _diagnostics(
        self,
        left_base: np.ndarray,
        left_q: np.ndarray,
        right_base: np.ndarray,
        right_q: np.ndarray,
        targets: FrameRetargetTargets,
    ) -> dict[str, float]:
        left_tip_local, left_tip_world, right_tip_local, right_tip_world = self._tip_sets(
            left_base, left_q, right_base, right_q
        )
        global_errors: list[float] = []
        local_anchor_errors: list[float] = []
        patch_errors: list[float] = []
        taxel_errors: list[float] = []
        compliance_errors: list[float] = []
        compliance_depths: list[float] = []
        for hand, tip_map, tip_targets, anchor_targets, tip_local, tip_world in (
            (
                "left",
                self.left_tip_map,
                targets.left_tip_targets,
                targets.left_tip_targets_anchor_local,
                left_tip_local,
                left_tip_world,
            ),
            (
                "right",
                self.right_tip_map,
                targets.right_tip_targets,
                targets.right_tip_targets_anchor_local,
                right_tip_local,
                right_tip_world,
            ),
        ):
            scale = _hand_scale(self.config, hand)
            for finger in FINGER_ORDER:
                tip_name = tip_map.get(finger)
                if tip_name is None or tip_name not in tip_local or tip_name not in tip_world:
                    continue
                if finger in tip_targets:
                    global_errors.append(
                        float(np.linalg.norm(tip_world[tip_name][:3, 3] - tip_targets[finger][:3, 3]))
                    )
                if finger in anchor_targets:
                    local_anchor_errors.append(
                        float(
                            np.linalg.norm(
                                tip_local[tip_name][:3, 3] - scale * anchor_targets[finger]
                            )
                        )
                    )
                feature = targets.tactile_features[hand][finger]
                if (
                    feature.contact_label
                    and feature.patch_center_local is not None
                    and feature.patch_center_cam is not None
                ):
                    robot_patch = (
                        tip_world[tip_name][:3, :3] @ feature.patch_center_local
                        + tip_world[tip_name][:3, 3]
                    )
                    patch_errors.append(
                        float(np.linalg.norm(robot_patch - feature.patch_center_cam))
                    )
                    if finger in tip_targets:
                        compliance = self._compliance_target(feature, tip_targets[finger])
                        if compliance is not None:
                            compliance_target, compliance_depth = compliance
                            compliance_errors.append(float(np.linalg.norm(robot_patch - compliance_target)))
                            compliance_depths.append(compliance_depth)
                    for taxel_index in np.flatnonzero(feature.active_mask):
                        robot_taxel = (
                            tip_world[tip_name][:3, :3] @ feature.taxel_positions_local[taxel_index]
                            + tip_world[tip_name][:3, 3]
                        )
                        taxel_errors.append(
                            float(np.linalg.norm(robot_taxel - feature.taxel_positions_cam[taxel_index]))
                        )
        return {
            "tip_global_rmse": _rmse(global_errors),
            "tip_local_anchor_rmse": _rmse(local_anchor_errors),
            "patch_rmse": _rmse(patch_errors),
            "taxel_cloud_rmse": _rmse(taxel_errors),
            "compliance_rmse": _rmse(compliance_errors),
            "compliance_depth_mean": float(np.mean(compliance_depths)) if compliance_depths else 0.0,
            "compliance_depth_max": float(np.max(compliance_depths)) if compliance_depths else 0.0,
            "compliance_target_count": float(len(compliance_depths)),
        }

    def _residual(
        self,
        z: np.ndarray,
        targets: FrameRetargetTargets,
        init_state: dict[str, Any],
        previous: dict[str, Any] | None,
        previous_previous: dict[str, Any] | None,
    ) -> np.ndarray:
        left_base, left_q, right_base, right_q = self._unpack(z)
        left_tip_local, left_tip_world, right_tip_local, right_tip_world = self._tip_sets(
            left_base, left_q, right_base, right_q
        )
        residuals: list[np.ndarray] = []
        left_base_beta = self._hand_beta(targets, "left")
        right_base_beta = self._hand_beta(targets, "right")

        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_anchor_pos * left_base_beta)
            * (left_base[:3, 3] - targets.left_base_anchor[:3, 3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_anchor_pos * right_base_beta)
            * (right_base[:3, 3] - targets.right_base_anchor[:3, 3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_anchor_rot * left_base_beta)
            * self._relative_rotation_log(targets.left_base_anchor[:3, :3], left_base[:3, :3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_anchor_rot * right_base_beta)
            * self._relative_rotation_log(targets.right_base_anchor[:3, :3], right_base[:3, :3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_step_pos)
            * (left_base[:3, 3] - init_state["left_wrist"][:3, 3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_step_pos)
            * (right_base[:3, 3] - init_state["right_wrist"][:3, 3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_step_rot)
            * self._relative_rotation_log(init_state["left_wrist"][:3, :3], left_base[:3, :3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_base_step_rot)
            * self._relative_rotation_log(init_state["right_wrist"][:3, :3], right_base[:3, :3])
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_joint_rest)
            * (left_q - self.left_neutral_q)
        )
        residuals.append(
            np.sqrt(self.config.optimization.lambda_joint_rest)
            * (right_q - self.right_neutral_q)
        )

        robot_tip_points: dict[str, np.ndarray] = {}
        target_tip_points: dict[str, np.ndarray] = {}
        for hand, tip_map, tip_targets, anchor_targets, tip_target_rotations, tip_local, tip_world in (
            (
                "left",
                self.left_tip_map,
                targets.left_tip_targets,
                targets.left_tip_targets_anchor_local,
                targets.left_tip_target_rotations,
                left_tip_local,
                left_tip_world,
            ),
            (
                "right",
                self.right_tip_map,
                targets.right_tip_targets,
                targets.right_tip_targets_anchor_local,
                targets.right_tip_target_rotations,
                right_tip_local,
                right_tip_world,
            ),
        ):
            scale = _hand_scale(self.config, hand)
            for finger in FINGER_ORDER:
                tip_name = tip_map.get(finger)
                if tip_name is None or tip_name not in tip_local or tip_name not in tip_world:
                    continue
                robot_tip = tip_world[tip_name]
                robot_tip_points[f"{hand}_{finger}"] = robot_tip[:3, 3]
                feature = targets.tactile_features[hand][finger]
                tactile_beta = self._tactile_beta(feature)
                tip_beta = self._tip_tracking_beta(feature)
                if finger in tip_targets:
                    target_tip = tip_targets[finger]
                    target_tip_points[f"{hand}_{finger}"] = target_tip[:3, 3]
                    residuals.append(
                        np.sqrt(self.config.optimization.lambda_tip_global * tip_beta)
                        * (robot_tip[:3, 3] - target_tip[:3, 3])
                    )
                if finger in anchor_targets:
                    residuals.append(
                        np.sqrt(self.config.optimization.lambda_tip_local * tip_beta)
                        * (tip_local[tip_name][:3, 3] - scale * anchor_targets[finger])
                    )
                if (
                    feature.contact_label
                    and feature.patch_center_local is not None
                    and feature.patch_center_cam is not None
                ):
                    robot_patch = (
                        robot_tip[:3, :3] @ feature.patch_center_local + robot_tip[:3, 3]
                    )
                    residuals.append(
                        np.sqrt(self.config.optimization.lambda_patch * tactile_beta)
                        * (robot_patch - feature.patch_center_cam)
                    )
                    if finger in tip_targets:
                        compliance = self._compliance_target(feature, tip_targets[finger])
                        if compliance is not None:
                            compliance_target, _ = compliance
                            residuals.append(
                                np.sqrt(self.config.optimization.lambda_compliance_target * tactile_beta)
                                * (robot_patch - compliance_target)
                            )
                    active_indices = np.flatnonzero(feature.active_mask)
                    if active_indices.size and self.config.optimization.lambda_taxel_cloud > 0.0:
                        active_forces = feature.force_norms[active_indices]
                        force_scale = max(float(np.max(active_forces)), self.config.tactile.eps)
                        normalizer = max(1, int(active_indices.size))
                        for taxel_index in active_indices:
                            robot_taxel = (
                                robot_tip[:3, :3] @ feature.taxel_positions_local[taxel_index]
                                + robot_tip[:3, 3]
                            )
                            force_weight = max(
                                0.0,
                                min(1.0, float(feature.force_norms[taxel_index]) / force_scale),
                            )
                            residuals.append(
                                np.sqrt(
                                    self.config.optimization.lambda_taxel_cloud
                                    * tactile_beta
                                    * force_weight
                                    / normalizer
                                )
                                * (robot_taxel - feature.taxel_positions_cam[taxel_index])
                            )
                if self.config.optimization.enable_tip_rot and finger in tip_target_rotations:
                    residuals.append(
                        np.sqrt(self.config.optimization.lambda_tip_rot * tip_beta)
                        * self._relative_rotation_log(tip_target_rotations[finger], robot_tip[:3, :3])
                    )

        source_nodes, robot_nodes, node_force = self._nodes_from_state(
            left_base=left_base,
            left_tip_transforms=left_tip_world,
            right_base=right_base,
            right_tip_transforms=right_tip_world,
            targets=targets,
        )
        residuals.append(
            self._fast_laplacian_residual(
                targets, source_nodes, robot_nodes, node_force
            )
            if self.config.optimization.fast_cpu_math
            else _laplacian_residual(source_nodes, robot_nodes, node_force, self.config)
        )

        active_left = [
            finger for finger in FINGER_ORDER if targets.tactile_features["left"][finger].raw_contact_label
        ]
        active_right = [
            finger for finger in FINGER_ORDER if targets.tactile_features["right"][finger].raw_contact_label
        ]
        for left_finger in active_left:
            for right_finger in active_right:
                left_name = f"left_{left_finger}"
                right_name = f"right_{right_finger}"
                if left_name not in robot_tip_points or right_name not in robot_tip_points:
                    continue
                if left_name not in target_tip_points or right_name not in target_tip_points:
                    continue
                residuals.append(
                    np.asarray(
                        [
                            np.sqrt(self.config.optimization.lambda_cross_hand)
                            * (
                                np.linalg.norm(
                                    robot_tip_points[left_name] - robot_tip_points[right_name]
                                )
                                - np.linalg.norm(
                                    target_tip_points[left_name] - target_tip_points[right_name]
                                )
                            )
                        ],
                        dtype=np.float64,
                    )
                )
        active_names = [
            f"{hand}_{finger}"
            for hand in ("left", "right")
            for finger in FINGER_ORDER
            if targets.tactile_features[hand][finger].raw_contact_label
        ]
        for first, second in pairwise(active_names):
            if first not in robot_tip_points or second not in robot_tip_points:
                continue
            if first not in target_tip_points or second not in target_tip_points:
                continue
            residuals.append(
                np.asarray(
                    [
                        np.sqrt(self.config.optimization.lambda_pair_dist)
                        * (
                            np.linalg.norm(robot_tip_points[first] - robot_tip_points[second])
                            - np.linalg.norm(target_tip_points[first] - target_tip_points[second])
                        )
                    ],
                    dtype=np.float64,
                )
            )
        residuals.append(_collision_proxy(robot_tip_points, self.config))

        if previous is not None:
            residuals.append(
                np.sqrt(self.config.optimization.lambda_joint_smooth)
                * (left_q - previous["left_q"])
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_joint_smooth)
                * (right_q - previous["right_q"])
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_base_pos_smooth)
                * (left_base[:3, 3] - previous["left_wrist"][:3, 3])
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_base_pos_smooth)
                * (right_base[:3, 3] - previous["right_wrist"][:3, 3])
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_base_rot_smooth)
                * self._relative_rotation_log(previous["left_wrist"][:3, :3], left_base[:3, :3])
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_base_rot_smooth)
                * self._relative_rotation_log(previous["right_wrist"][:3, :3], right_base[:3, :3])
            )
        if previous is not None and previous_previous is not None:
            left_base_acc = np.concatenate(
                [
                    left_base[:3, 3] - 2.0 * previous["left_wrist"][:3, 3] + previous_previous["left_wrist"][:3, 3],
                    self._rotation_log(left_base[:3, :3])
                    - 2.0 * self._rotation_log(previous["left_wrist"][:3, :3])
                    + self._rotation_log(previous_previous["left_wrist"][:3, :3]),
                ]
            )
            right_base_acc = np.concatenate(
                [
                    right_base[:3, 3] - 2.0 * previous["right_wrist"][:3, 3] + previous_previous["right_wrist"][:3, 3],
                    self._rotation_log(right_base[:3, :3])
                    - 2.0 * self._rotation_log(previous["right_wrist"][:3, :3])
                    + self._rotation_log(previous_previous["right_wrist"][:3, :3]),
                ]
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_base_acc) * left_base_acc
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_base_acc) * right_base_acc
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_joint_acc)
                * (left_q - 2.0 * previous["left_q"] + previous_previous["left_q"])
            )
            residuals.append(
                np.sqrt(self.config.optimization.lambda_joint_acc)
                * (right_q - 2.0 * previous["right_q"] + previous_previous["right_q"])
            )
        if not residuals:
            return np.zeros((0,), dtype=np.float64)
        return finite_or_zeros(np.concatenate([np.atleast_1d(item) for item in residuals]))

    def solve_sequence(
        self,
        output_dir: str | Path,
        start_frame: int = 0,
        max_frames: int | None = None,
        frame_stride: int = 1,
        frame_indices: Sequence[int] | None = None,
        initial_previous: dict[str, Any] | None = None,
        initial_previous_previous: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        if frame_indices is None:
            indices = list(range(start_frame, len(self.dataset), frame_stride))
            if max_frames is not None:
                indices = indices[:max_frames]
        else:
            indices = [int(value) for value in frame_indices]
            if max_frames is not None and len(indices) > max_frames:
                indices = indices[:max_frames]
        if not indices:
            raise ValueError("No frames selected")
        left_joints = np.zeros((len(indices), self.left_model.dof), dtype=np.float64)
        right_joints = np.zeros((len(indices), self.right_model.dof), dtype=np.float64)
        left_wrists = np.zeros((len(indices), 4, 4), dtype=np.float64)
        right_wrists = np.zeros((len(indices), 4, 4), dtype=np.float64)
        contact_records: list[dict[str, Any]] = []
        optimization_info: list[dict[str, Any]] = []

        # Chunked execution must preserve both history states because the objective
        # contains velocity and acceleration regularizers.  Starting each chunk
        # from neutral creates artificial jumps at every chunk boundary.
        previous: dict[str, Any] | None = initial_previous
        previous_previous: dict[str, Any] | None = initial_previous_previous
        bounds = self._bounds()
        for output_index, frame_index in enumerate(indices):
            frame = self.dataset.get_frame(frame_index)
            targets = self._effective_contact_targets(
                self._cached_frame_targets(frame_index, frame),
                frame_index,
                frame_stride,
                indices[:output_index],
            )
            init_state = (
                self._state_dict(
                    previous["left_wrist"],
                    previous["left_q"],
                    previous["right_wrist"],
                    previous["right_q"],
                )
                if previous is not None
                else self._make_initial_state_from_neutral(targets)
            )
            initial_residual = self._residual(
                init_state["z"], targets, init_state, previous, previous_previous
            )
            initial_residual_norm = float(np.linalg.norm(initial_residual))
            initial_cost = float(0.5 * np.dot(initial_residual, initial_residual))
            z_best, success, failure_reason, solve_info = self._solve_from_init(
                targets, init_state, previous, previous_previous, bounds
            )
            retry_from_neutral = False
            left_base, left_q, right_base, right_q = self._unpack(z_best)
            diagnostics = self._diagnostics(left_base, left_q, right_base, right_q, targets)

            if not success and previous is not None:
                left_base = previous["left_wrist"].copy()
                right_base = previous["right_wrist"].copy()
                left_q = previous["left_q"].copy()
                right_q = previous["right_q"].copy()
                z_best = previous["z"].copy()
                diagnostics = self._diagnostics(left_base, left_q, right_base, right_q, targets)
            final_residual = self._residual(z_best, targets, init_state, previous, previous_previous)
            final_residual_norm = float(np.linalg.norm(final_residual))
            final_cost = float(0.5 * np.dot(final_residual, final_residual))
            print(
                "[wrist-free loss] "
                f"frame={frame_index} output={output_index} "
                f"cost={initial_cost:.6e}->{final_cost:.6e} "
                f"residual_norm={initial_residual_norm:.6e}->{final_residual_norm:.6e} "
                f"tip_global_rmse={diagnostics['tip_global_rmse']:.6e} "
                f"tip_local_rmse={diagnostics['tip_local_anchor_rmse']:.6e} "
                f"patch_rmse={diagnostics['patch_rmse']:.6e} "
                f"taxel_rmse={diagnostics['taxel_cloud_rmse']:.6e} "
                f"active_taxels={self._active_taxel_count(targets)} "
                f"nfev={solve_info.get('nfev')} "
                f"success={success}"
                f"{' retry=neutral' if retry_from_neutral else ''}",
                flush=True,
            )

            left_wrists[output_index] = left_base
            right_wrists[output_index] = right_base
            left_joints[output_index] = left_q
            right_joints[output_index] = right_q
            contact_records.append(
                {
                    "frame_index": frame_index,
                    "frame_key": frame.frame_key,
                    "left": {
                        finger: {
                            "contact_label": bool(targets.tactile_features["left"][finger].contact_label),
                            "raw_contact_label": bool(targets.tactile_features["left"][finger].raw_contact_label),
                            "contact_weight": float(targets.tactile_features["left"][finger].contact_weight),
                            "contact_hold_age": int(targets.tactile_features["left"][finger].contact_hold_age),
                            "total_force": float(targets.tactile_features["left"][finger].total_force),
                            "active_taxel_count": int(
                                targets.tactile_features["left"][finger].active_taxel_count
                            ),
                            "patch_center_local": None
                            if targets.tactile_features["left"][finger].patch_center_local is None
                            else targets.tactile_features["left"][finger].patch_center_local.tolist(),
                            "patch_center_cam": None
                            if targets.tactile_features["left"][finger].patch_center_cam is None
                            else targets.tactile_features["left"][finger].patch_center_cam.tolist(),
                        }
                        for finger in FINGER_ORDER
                    },
                    "right": {
                        finger: {
                            "contact_label": bool(targets.tactile_features["right"][finger].contact_label),
                            "raw_contact_label": bool(targets.tactile_features["right"][finger].raw_contact_label),
                            "contact_weight": float(targets.tactile_features["right"][finger].contact_weight),
                            "contact_hold_age": int(targets.tactile_features["right"][finger].contact_hold_age),
                            "total_force": float(targets.tactile_features["right"][finger].total_force),
                            "active_taxel_count": int(
                                targets.tactile_features["right"][finger].active_taxel_count
                            ),
                            "patch_center_local": None
                            if targets.tactile_features["right"][finger].patch_center_local is None
                            else targets.tactile_features["right"][finger].patch_center_local.tolist(),
                            "patch_center_cam": None
                            if targets.tactile_features["right"][finger].patch_center_cam is None
                            else targets.tactile_features["right"][finger].patch_center_cam.tolist(),
                        }
                        for finger in FINGER_ORDER
                    },
                }
            )
            optimization_info.append(
                {
                    "frame_index": frame_index,
                    "frame_key": frame.frame_key,
                    "success": success,
                    "failure_reason": failure_reason,
                    "left_base_anchor": targets.left_base_anchor.tolist(),
                    "right_base_anchor": targets.right_base_anchor.tolist(),
                    "left_base_init": init_state["left_wrist"].tolist(),
                    "right_base_init": init_state["right_wrist"].tolist(),
                    "left_base_result": left_base.tolist(),
                    "right_base_result": right_base.tolist(),
                    "contact_frame": self._active_taxel_count(targets) > 0,
                    "active_taxel_count": self._active_taxel_count(targets),
                    "finger_weights": self._frame_finger_weights(),
                    "initial_residual_norm": initial_residual_norm,
                    "final_residual_norm": final_residual_norm,
                    "initial_cost": initial_cost,
                    "final_cost": final_cost,
                    "cost_reduction": initial_cost - final_cost,
                    "retry_from_neutral": retry_from_neutral,
                    "scipy_cost": solve_info.get("scipy_cost"),
                    "scipy_nfev": solve_info.get("nfev"),
                    "scipy_njev": solve_info.get("njev"),
                    "scipy_optimality": solve_info.get("optimality"),
                    "scipy_status": solve_info.get("status"),
                    "scipy_message": solve_info.get("message"),
                    "tip_global_rmse": diagnostics["tip_global_rmse"],
                    "tip_local_anchor_rmse": diagnostics["tip_local_anchor_rmse"],
                    "patch_rmse": diagnostics["patch_rmse"],
                    "taxel_cloud_rmse": diagnostics["taxel_cloud_rmse"],
                    "compliance_rmse": diagnostics["compliance_rmse"],
                    "compliance_depth_mean": diagnostics["compliance_depth_mean"],
                    "compliance_depth_max": diagnostics["compliance_depth_max"],
                    "compliance_target_count": diagnostics["compliance_target_count"],
                }
            )
            previous_previous = previous
            previous = self._state_dict(left_base, left_q, right_base, right_q)

        np.save(output_path / "joints_left.npy", left_joints)
        np.save(output_path / "joints_right.npy", right_joints)
        np.save(output_path / "wrist_left.npy", left_wrists)
        np.save(output_path / "wrist_right.npy", right_wrists)
        np.savez(output_path / "contact_features.npz", records=np.asarray(contact_records, dtype=object))
        (output_path / "joint_names_left.json").write_text(
            json.dumps(self.left_model.joint_names, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "joint_names_right.json").write_text(
            json.dumps(self.right_model.joint_names, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "optimization_info.json").write_text(
            json.dumps(optimization_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "zarr_path": str(self.dataset.zarr_path),
            # Save all resolved values, including defaults, to make results
            # reproducible when the configuration changes.
            "resolved_config": self.config.to_dict(),
            "expected_dof_per_hand": self.config.expected_dof_per_hand,
            "left_dof": self.left_model.dof,
            "right_dof": self.right_model.dof,
            "left_tip_links": self.left_model.tip_links,
            "right_tip_links": self.right_model.tip_links,
            "left_root_link": self.left_model.root_link,
            "right_root_link": self.right_model.root_link,
            "frames": indices,
            "mapping_mode": "fingertip_driven_base_soft_single_stage_tactile",
            "initialization_strategy": "first frame from neutral, every subsequent frame strictly from the previous optimized state",
            "neutral_retry": False,
            "contact_hold_frames": int(self.config.optimization.contact_hold_frames),
            "contact_hold_decay": "after raw contact turns off, held tactile constraints use (hold_frames-age+1)/(hold_frames+1)",
            "human_wrist_pose_used": False,
            "wrist_output_semantics": "wrist_left.npy / wrist_right.npy store robot base poses for compatibility.",
            "toporetarget_notice": (
                "This implementation is TopoRetarget-inspired. Since object mesh and object pose "
                "are unavailable, it preserves bimanual wrist-fingertip-tactile interaction structure "
                "rather than exact hand-object surface contact. It cannot compute true hand-object penetration."
            ),
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "left_joints": left_joints,
            "right_joints": right_joints,
            "left_wrists": left_wrists,
            "right_wrists": right_wrists,
            "frames": indices,
            "optimization_info": optimization_info,
            "metadata": metadata,
        }
