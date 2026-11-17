"""Centralized configuration for tactile retargeting and visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


FINGER_WEIGHT_KEYS = ("thumb", "index", "middle", "ring", "pinky")


def _default_finger_weights() -> dict[str, float]:
    return {finger: 1.0 for finger in FINGER_WEIGHT_KEYS}


@dataclass(frozen=True)
class TactileVisualizationConfig:
    active_taxel_force_threshold: float = 0.02
    patch_center_min_total_force: float = 0.05
    patch_center_min_active_taxels: int = 2
    active_taxel_colormap: str = "yellow_to_red"
    inactive_taxel_color: tuple[float, float, float] = (0.2, 0.4, 1.0)
    patch_center_color: tuple[float, float, float] = (0.0, 1.0, 0.2)
    patch_center_radius: float = 0.008
    inactive_taxel_size: float = 2.0
    active_taxel_min_size: float = 3.8
    active_taxel_max_size: float = 7.5
    patch_center_min_size: float = 6.0
    patch_center_max_size: float = 14.0
    eps: float = 1e-12


@dataclass(frozen=True)
class TactileConfig:
    contact_threshold: float = 0.02
    active_taxel_force_threshold: float = 0.01
    min_active_taxels: int = 2
    smooth_contact_window: int = 3
    eps: float = 1e-8


@dataclass(frozen=True)
class MappingConfig:
    mode: str = "relative_initial"
    hand_scale_left: float = 1.0
    hand_scale_right: float = 1.0
    left_robot_wrist_init_to_cam: list[list[float]] | None = None
    right_robot_wrist_init_to_cam: list[list[float]] | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    max_iters: int = 160
    verbose: int = 1
    lambda_base_anchor_pos: float = 0.1
    lambda_base_anchor_rot: float = 0.03
    lambda_base_step_pos: float = 0.06
    lambda_base_step_rot: float = 0.05
    lambda_tip_global: float = 100.0
    lambda_tip_local: float = 0.4
    lambda_tip_rot: float = 1.0
    contact_tip_tracking_scale: float = 0.05
    contact_hold_frames: int = 3
    enable_tip_rot: bool = True
    finger_weights: dict[str, float] = field(default_factory=_default_finger_weights)
    left_finger_weights: dict[str, float] = field(default_factory=dict)
    right_finger_weights: dict[str, float] = field(default_factory=dict)
    lambda_patch: float = 20.0
    lambda_taxel_cloud: float = 4.0
    lambda_compliance_target: float = 16.0
    compliance_stiffness: float = 20.0
    compliance_max_depth: float = 0.01
    compliance_min_force: float = 0.0
    compliance_force_direction_sign: float = 1.0
    lambda_tactile_lap: float = 1.0
    lambda_cross_hand: float = 0.4
    lambda_pair_dist: float = 0.25
    lambda_finger_collision: float = 0.05
    lambda_joint_rest: float = 0.005
    lambda_joint_smooth: float = 0.45
    lambda_base_pos_smooth: float = 10.0
    lambda_base_rot_smooth: float = 5.0
    lambda_base_acc: float = 0.08
    lambda_joint_acc: float = 0.08
    beta_base: float = 1.0
    beta_contact: float = 2.0
    d_min: float = 0.01
    kappa: float = 20.0
    tactile_gamma: float = 1.0
    rotvec_bound: float = 12.566370614359172
    # Algebraically equivalent SO(3), precompiled FK, and static residual caches.
    fast_cpu_math: bool = True


@dataclass(frozen=True)
class VisualizationConfig:
    pointcloud_stride: int = 4
    max_depth: float = 3.0
    show_mesh_default: bool = True
    show_skeleton_default: bool = True
    wrist_axis_scale: float = 0.03
    fingertip_size: float = 5.0
    target_fingertip_size: float = 4.0
    human_fingertip_size: float = 4.5
    skeleton_joint_size: float = 3.5


@dataclass(frozen=True)
class RetargetConfig:
    expected_dof_per_hand: int = 20
    mapping: MappingConfig = field(default_factory=MappingConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    tactile_visualization: TactileVisualizationConfig = field(default_factory=TactileVisualizationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass_dict(defaults: Any, overrides: dict[str, Any] | None) -> Any:
    if overrides is None:
        return defaults
    values = asdict(defaults)
    for key, value in overrides.items():
        if key in values:
            values[key] = value
    return type(defaults)(**values)


def load_retarget_config(path: str | Path | None) -> RetargetConfig:
    config = RetargetConfig()
    if path is None:
        return config
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RetargetConfig(
        expected_dof_per_hand=int(payload.get("expected_dof_per_hand", config.expected_dof_per_hand)),
        mapping=_merge_dataclass_dict(config.mapping, payload.get("mapping")),
        tactile=_merge_dataclass_dict(config.tactile, payload.get("tactile")),
        tactile_visualization=_merge_dataclass_dict(
            config.tactile_visualization, payload.get("tactile_visualization")
        ),
        optimization=_merge_dataclass_dict(config.optimization, payload.get("optimization")),
        visualization=_merge_dataclass_dict(config.visualization, payload.get("visualization")),
    )
