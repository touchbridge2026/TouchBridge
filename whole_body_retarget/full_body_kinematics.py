"""Full-body FK utilities over the lightweight URDF model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from retarget.robot.urdf_model import URDFJoint, URDFModel
from retarget.utils import axis_angle_transform, compose_homogeneous


LEGWAIST_JOINTS = [f"LegWaist-{index}" for index in range(6)]
LEFT_ARM_JOINTS = [f"LeftArm-{index}" for index in range(7)]
RIGHT_ARM_JOINTS = [f"RightArm-{index}" for index in range(7)]


@dataclass
class FullBodyKinematics:
    model: URDFModel

    @property
    def joint_names(self) -> list[str]:
        return self.model.joint_names

    @property
    def joint_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.model.joint_names)}

    @property
    def lower(self) -> np.ndarray:
        return self.model.joint_limits_lower

    @property
    def upper(self) -> np.ndarray:
        return self.model.joint_limits_upper

    def zero_q(self) -> np.ndarray:
        return np.clip(np.zeros(self.model.dof, dtype=np.float64), self.lower, self.upper)

    def group_indices(self, names: Iterable[str]) -> list[int]:
        index = self.joint_index
        missing = [name for name in names if name not in index]
        if missing:
            raise KeyError(f"Missing joints in URDF: {missing}")
        return [index[name] for name in names]

    def left_hand_joint_names(self) -> list[str]:
        return [name for name in self.model.joint_names if name.startswith("left_finger")]

    def right_hand_joint_names(self) -> list[str]:
        return [name for name in self.model.joint_names if name.startswith("right_finger")]

    def _joint_motion_transform(self, joint: URDFJoint, value: float) -> np.ndarray:
        if joint.joint_type in {"revolute", "continuous"}:
            return axis_angle_transform(joint.axis, value)
        if joint.joint_type == "prismatic":
            axis = np.asarray(joint.axis, dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                return np.eye(4, dtype=np.float64)
            return compose_homogeneous(translation=(axis / norm) * float(value))
        return np.eye(4, dtype=np.float64)

    def fk_all_links(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.dof:
            raise ValueError(f"Expected q shape ({self.model.dof},), got {q.shape}")
        transforms: dict[str, np.ndarray] = {self.model.root_link: np.eye(4, dtype=np.float64)}
        joint_values = {joint.name: q[index] for index, joint in enumerate(self.model.joints)}
        unresolved = set(self.model.child_to_all_joint.keys())
        while unresolved:
            progress = False
            for child_link in list(unresolved):
                joint = self.model.child_to_all_joint[child_link]
                if joint.parent_link not in transforms:
                    continue
                value = joint_values.get(joint.name, 0.0)
                transforms[child_link] = (
                    transforms[joint.parent_link]
                    @ joint.origin
                    @ self._joint_motion_transform(joint, value)
                )
                unresolved.remove(child_link)
                progress = True
            if not progress:
                raise RuntimeError("Failed to resolve FK chain; URDF may contain a disconnected graph.")
        return transforms

    def link_transform(self, q: np.ndarray, link_name: str) -> np.ndarray:
        transforms = self.fk_all_links(q)
        if link_name not in transforms:
            raise KeyError(f"Link {link_name!r} is not reachable from root {self.model.root_link!r}")
        return transforms[link_name]


def map_named_joints(
    target_q: np.ndarray,
    target_names: list[str],
    source_q: np.ndarray,
    source_names: list[str],
) -> list[str]:
    """Copy source joint values into target_q by exact joint name and return missing names."""
    target_index = {name: index for index, name in enumerate(target_names)}
    missing: list[str] = []
    for src_index, name in enumerate(source_names):
        if name not in target_index:
            missing.append(name)
            continue
        target_q[target_index[name]] = float(source_q[src_index])
    return missing
