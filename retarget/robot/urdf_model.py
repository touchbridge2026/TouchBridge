"""Lightweight URDF parsing without external URDF dependencies."""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retarget.utils import compose_homogeneous, rpy_to_matrix


TIP_LINK_PATTERN = re.compile(r"(finger\d+.*tip.*link|tip_link)$", re.IGNORECASE)


@dataclass(frozen=True)
class LinkVisual:
    link_name: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray
    format: str
    exists: bool


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis: np.ndarray
    origin: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class URDFModel:
    name: str
    urdf_path: Path
    root_link: str
    joints: list[URDFJoint]
    all_joints: list[URDFJoint]
    joint_order: list[str]
    link_names: list[str]
    link_visuals: dict[str, list[LinkVisual]]
    tip_links: list[str]
    link_children: dict[str, list[str]]
    child_to_joint: dict[str, URDFJoint]
    child_to_all_joint: dict[str, URDFJoint]

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def joint_limits_lower(self) -> np.ndarray:
        return np.asarray([joint.lower for joint in self.joints], dtype=np.float64)

    @property
    def joint_limits_upper(self) -> np.ndarray:
        return np.asarray([joint.upper for joint in self.joints], dtype=np.float64)

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    def mesh_status(self) -> dict[str, Any]:
        visuals = [visual for items in self.link_visuals.values() for visual in items]
        supported = sum(1 for visual in visuals if visual.exists and visual.format in {"stl", "obj"})
        unsupported = sum(1 for visual in visuals if visual.exists and visual.format not in {"stl", "obj"})
        missing = sum(1 for visual in visuals if not visual.exists)
        return {
            "visual_count": len(visuals),
            "supported_visual_count": supported,
            "unsupported_visual_count": unsupported,
            "missing_visual_count": missing,
            "formats": sorted({visual.format for visual in visuals}),
        }


def _parse_xyz_rpy(node: ET.Element | None) -> np.ndarray:
    if node is None:
        return np.eye(4, dtype=np.float64)
    xyz = [float(value) for value in (node.get("xyz", "0 0 0").split())]
    rpy = [float(value) for value in (node.get("rpy", "0 0 0").split())]
    return compose_homogeneous(rotation=rpy_to_matrix(rpy), translation=xyz)


def _resolve_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    if mesh_filename.startswith("package://"):
        mesh_filename = mesh_filename[len("package://") :]
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _parse_scale(value: str | None) -> np.ndarray:
    if not value:
        return np.ones(3, dtype=np.float64)
    fields = [float(item) for item in value.split()]
    if len(fields) == 1:
        fields = fields * 3
    return np.asarray(fields, dtype=np.float64)


def parse_urdf_model(
    urdf_path: str | Path,
    expected_dof: int = 22,
    tip_link_regex: str | None = None,
    manual_tip_links: list[str] | None = None,
) -> URDFModel:
    path = Path(urdf_path).resolve()
    root = ET.parse(path).getroot()
    robot_name = root.get("name", path.stem)
    link_visuals: dict[str, list[LinkVisual]] = {}
    link_names: list[str] = []
    for link_node in root.findall("link"):
        link_name = link_node.get("name")
        if link_name is None:
            continue
        link_names.append(link_name)
        visuals: list[LinkVisual] = []
        for visual_node in link_node.findall("visual"):
            mesh_node = visual_node.find("./geometry/mesh")
            if mesh_node is None or mesh_node.get("filename") is None:
                continue
            mesh_path = _resolve_mesh_path(path, mesh_node.get("filename", ""))
            visuals.append(
                LinkVisual(
                    link_name=link_name,
                    mesh_path=mesh_path,
                    origin=_parse_xyz_rpy(visual_node.find("origin")),
                    scale=_parse_scale(mesh_node.get("scale")),
                    format=mesh_path.suffix.lower().lstrip("."),
                    exists=mesh_path.exists(),
                )
            )
        link_visuals[link_name] = visuals

    joints: list[URDFJoint] = []
    all_joints: list[URDFJoint] = []
    joint_order: list[str] = []
    parent_links: set[str] = set()
    child_links: set[str] = set()
    link_children: dict[str, list[str]] = {name: [] for name in link_names}
    child_to_joint: dict[str, URDFJoint] = {}
    child_to_all_joint: dict[str, URDFJoint] = {}
    for joint_node in root.findall("joint"):
        joint_name = joint_node.get("name")
        joint_type = joint_node.get("type", "fixed")
        parent_node = joint_node.find("parent")
        child_node = joint_node.find("child")
        if parent_node is None or child_node is None:
            continue
        parent_link = parent_node.get("link", "")
        child_link = child_node.get("link", "")
        parent_links.add(parent_link)
        child_links.add(child_link)
        link_children.setdefault(parent_link, []).append(child_link)
        axis_node = joint_node.find("axis")
        axis_values = axis_node.get("xyz", "0 0 1").split() if axis_node is not None else ["0", "0", "1"]
        limit_node = joint_node.find("limit")
        lower = float(limit_node.get("lower", "-3.141592653589793")) if limit_node is not None else -np.pi
        upper = float(limit_node.get("upper", "3.141592653589793")) if limit_node is not None else np.pi
        if joint_type == "continuous":
            lower, upper = -np.pi, np.pi
        joint = URDFJoint(
            name=joint_name or f"joint_{len(joints)}",
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            axis=np.asarray([float(value) for value in axis_values], dtype=np.float64),
            origin=_parse_xyz_rpy(joint_node.find("origin")),
            lower=lower,
            upper=upper,
        )
        all_joints.append(joint)
        child_to_all_joint[child_link] = joint
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        joints.append(joint)
        joint_order.append(joint.name)
        child_to_joint[child_link] = joint

    root_candidates = [name for name in link_names if name not in child_links]
    root_link = root_candidates[0] if root_candidates else link_names[0]
    pattern = re.compile(tip_link_regex, re.IGNORECASE) if tip_link_regex else TIP_LINK_PATTERN
    tip_links = [name for name in link_names if pattern.search(name)]
    if manual_tip_links:
        tip_links = manual_tip_links
    if len(joints) != expected_dof:
        warnings.warn(
            f"Expected DoF = {expected_dof}, parsed DoF = {len(joints)}, using parsed URDF joints.",
            stacklevel=2,
        )
    return URDFModel(
        name=robot_name,
        urdf_path=path,
        root_link=root_link,
        joints=joints,
        all_joints=all_joints,
        joint_order=joint_order,
        link_names=link_names,
        link_visuals=link_visuals,
        tip_links=tip_links,
        link_children=link_children,
        child_to_joint=child_to_joint,
        child_to_all_joint=child_to_all_joint,
    )
