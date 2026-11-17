"""Forward kinematics for lightweight URDF hand models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retarget.robot.urdf_model import URDFJoint, URDFModel
from retarget.utils import axis_angle_transform, compose_homogeneous


@dataclass
class HandKinematics:
    model: URDFModel

    def _joint_motion_transform(self, joint: URDFJoint, value: float) -> np.ndarray:
        if joint.joint_type in {"revolute", "continuous"}:
            return axis_angle_transform(joint.axis, value)
        if joint.joint_type == "prismatic":
            axis = np.asarray(joint.axis, dtype=np.float64)
            axis = axis / max(1e-12, np.linalg.norm(axis))
            return compose_homogeneous(translation=axis * float(value))
        return np.eye(4, dtype=np.float64)

    def fk_all_links(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.dof:
            raise ValueError(f"Expected q with shape ({self.model.dof},), got {q.shape}")
        transforms: dict[str, np.ndarray] = {self.model.root_link: np.eye(4, dtype=np.float64)}
        joint_values = {joint.name: q[index] for index, joint in enumerate(self.model.joints)}
        unresolved = set(self.model.child_to_all_joint.keys())
        while unresolved:
            progress = False
            for child_link in list(unresolved):
                joint = self.model.child_to_all_joint[child_link]
                if joint.parent_link not in transforms:
                    continue
                parent_transform = transforms[joint.parent_link]
                value = joint_values.get(joint.name, 0.0)
                transforms[child_link] = parent_transform @ joint.origin @ self._joint_motion_transform(joint, value)
                unresolved.remove(child_link)
                progress = True
            if not progress:
                raise RuntimeError("Failed to resolve FK chain; URDF may contain a disconnected graph.")
        return transforms

    def fk_tips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        link_transforms = self.fk_all_links(q)
        return {name: link_transforms[name] for name in self.model.tip_links if name in link_transforms}

    def fk_tip_list(self, q: np.ndarray) -> list[np.ndarray]:
        tip_dict = self.fk_tips(q)
        return [tip_dict[name] for name in self.model.tip_links if name in tip_dict]

    def fk_joint_positions(self, q: np.ndarray) -> dict[str, np.ndarray]:
        transforms = self.fk_all_links(q)
        return {name: transform[:3, 3].copy() for name, transform in transforms.items()}


@dataclass(frozen=True)
class _CompiledJoint:
    parent_link: str
    child_link: str
    origin: np.ndarray
    q_index: int | None
    joint_type: str
    axis: np.ndarray
    axis_skew: np.ndarray
    axis_skew_sq: np.ndarray


class CompiledHandKinematics(HandKinematics):
    """Hand FK with a precomputed traversal and Rodrigues constants."""

    def __init__(self, model: URDFModel) -> None:
        super().__init__(model)
        q_indices = {joint.name: index for index, joint in enumerate(model.joints)}
        ordered: list[Any] = []
        pending = [model.root_link]
        while pending:
            parent = pending.pop()
            children = list(model.link_children.get(parent, []))
            for child in reversed(children):
                pending.append(child)
            for child in children:
                joint = model.child_to_all_joint.get(child)
                if joint is not None:
                    ordered.append(joint)
        # A depth-first stack can place grandchildren after their parents but
        # interleave finger branches. Resolve once here, never in hot FK calls.
        unresolved = list(ordered)
        topological: list[Any] = []
        resolved = {model.root_link}
        while unresolved:
            next_unresolved = []
            for joint in unresolved:
                if joint.parent_link in resolved:
                    topological.append(joint)
                    resolved.add(joint.child_link)
                else:
                    next_unresolved.append(joint)
            if len(next_unresolved) == len(unresolved):
                raise RuntimeError("Failed to precompile hand FK traversal")
            unresolved = next_unresolved
        compiled = []
        for joint in topological:
            axis = np.asarray(joint.axis, dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm > 1.0e-12:
                axis = axis / norm
            x, y, z = axis
            skew = np.asarray(
                [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
                dtype=np.float64,
            )
            compiled.append(
                _CompiledJoint(
                    parent_link=joint.parent_link,
                    child_link=joint.child_link,
                    origin=np.asarray(joint.origin, dtype=np.float64),
                    q_index=q_indices.get(joint.name),
                    joint_type=joint.joint_type,
                    axis=axis,
                    axis_skew=skew,
                    axis_skew_sq=skew @ skew,
                )
            )
        self._compiled_joints = tuple(compiled)

    def fk_all_links(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.dof:
            raise ValueError(f"Expected q with shape ({self.model.dof},), got {q.shape}")
        transforms: dict[str, np.ndarray] = {
            self.model.root_link: np.eye(4, dtype=np.float64)
        }
        identity3 = np.eye(3, dtype=np.float64)
        for joint in self._compiled_joints:
            transform = transforms[joint.parent_link] @ joint.origin
            if joint.q_index is None:
                pass
            elif joint.joint_type in {"revolute", "continuous"}:
                value = float(q[joint.q_index])
                motion_rotation = (
                    identity3
                    + np.sin(value) * joint.axis_skew
                    + (1.0 - np.cos(value)) * joint.axis_skew_sq
                )
                transform[:3, :3] = transform[:3, :3] @ motion_rotation
            elif joint.joint_type == "prismatic":
                transform[:3, 3] += (
                    transform[:3, :3]
                    @ (joint.axis * float(q[joint.q_index]))
                )
            transforms[joint.child_link] = transform
        return transforms
