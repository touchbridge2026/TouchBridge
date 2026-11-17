"""Render the retargeted full-URDF mesh into every inpaint RGB frame."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .full_body_kinematics import FullBodyKinematics
from .io_utils import ensure_project_imports
from .lighting import DEFAULT_LIGHTING_CONFIG, LightingConfig, load_lighting_config

ensure_project_imports()

from retarget.robot.urdf_model import parse_urdf_model  # noqa: E402


RENDER_SCHEMA_VERSION = "6.0"
EXCLUDED_VISUAL_LINKS = frozenset(
    {
        "base_link",
        "wheel01",
        "wheel02",
        "wheel03",
        "wheel04",
        "head21",
        "head22",
    }
)
HAND_LINK_PREFIXES = (
    "left_palm",
    "right_palm",
    "left_finger",
    "right_finger",
    "wuji_left",
    "wuji_right",
)
DEFAULT_VISUAL_COLOR = np.asarray((0.7, 0.7, 0.7), dtype=np.float32)


@dataclass(frozen=True)
class VisualTemplate:
    link_name: str
    visual_origin: np.ndarray
    vertices: np.ndarray
    vertex_normals: np.ndarray
    triangles: np.ndarray
    triangle_color: np.ndarray
    lighting_strength: float


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Open3D is required to render rgb_o3d") from exc
    return o3d


def _parse_rgb(rgba: str | None) -> np.ndarray | None:
    if not rgba:
        return None
    values = np.fromstring(rgba, sep=" ", dtype=np.float32)
    if values.size < 3 or not np.all(np.isfinite(values[:3])):
        return None
    return np.clip(values[:3], 0.0, 1.0)


def _urdf_visual_colors(urdf_path: Path) -> dict[str, list[np.ndarray]]:
    """Read mesh visual colors in the same link/visual order as the URDF parser."""
    root = ET.parse(urdf_path).getroot()
    named_materials: dict[str, np.ndarray] = {}
    for material in root.findall("material"):
        color = material.find("color")
        rgb = _parse_rgb(None if color is None else color.get("rgba"))
        name = material.get("name")
        if name and rgb is not None:
            named_materials[name] = rgb

    colors_by_link: dict[str, list[np.ndarray]] = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        visual_colors: list[np.ndarray] = []
        for visual in link.findall("visual"):
            if visual.find("./geometry/mesh") is None:
                continue
            rgb = None
            material = visual.find("material")
            if material is not None:
                inline_color = material.find("color")
                rgb = _parse_rgb(
                    None if inline_color is None else inline_color.get("rgba")
                )
                if rgb is None:
                    rgb = named_materials.get(material.get("name", ""))
            visual_colors.append(
                DEFAULT_VISUAL_COLOR.copy()
                if rgb is None
                else np.asarray(rgb, dtype=np.float32)
            )
        colors_by_link[link_name] = visual_colors
    return colors_by_link


def _is_hand_link(link_name: str) -> bool:
    return link_name.lower().startswith(HAND_LINK_PREFIXES)


def _is_arm_shell_link(link_name: str) -> bool:
    lowered = link_name.lower()
    return lowered.startswith("left0") or lowered.startswith("right0")


def _lighting_strength_for_link(
    link_name: str,
    lighting: LightingConfig,
) -> float:
    if _is_hand_link(link_name):
        return lighting.hand_lighting_strength
    if _is_arm_shell_link(link_name):
        return lighting.arm_lighting_strength
    return lighting.other_lighting_strength


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _srgb_to_linear_numpy(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32)
    return np.where(
        color <= 0.04045,
        color / 12.92,
        ((color + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb_numpy(color: np.ndarray) -> np.ndarray:
    color = np.maximum(np.asarray(color, dtype=np.float32), 0.0)
    return np.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * np.power(color, 1.0 / 2.4) - 0.055,
    )


def _shade_points_numpy(
    *,
    positions_camera: np.ndarray,
    normals_camera: np.ndarray,
    base_colors: np.ndarray,
    lighting_strengths: np.ndarray,
    camera_from_base: np.ndarray,
    lighting: LightingConfig,
) -> np.ndarray:
    """Apply the production Lambert + Blinn-Phong model on CPU hit points."""
    positions = np.asarray(positions_camera, dtype=np.float32)
    normals = np.asarray(normals_camera, dtype=np.float32)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, 1.0e-12)
    view = -positions
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1.0e-12)
    # Keep imported STL normals usable for double-sided rasterization.
    flip = np.sum(normals * view, axis=1, keepdims=True) < 0.0
    normals = np.where(flip, -normals, normals)

    illumination = np.full(
        (positions.shape[0], 3), lighting.ambient_intensity, dtype=np.float32
    )
    specular = np.zeros_like(illumination)
    for light in lighting.lights:
        light_position = _transform_points(
            camera_from_base,
            np.asarray(light.position_base_m, dtype=np.float64)[None],
        )[0].astype(np.float32)
        to_light = light_position[None] - positions
        distance_squared = np.sum(to_light * to_light, axis=1, keepdims=True)
        to_light /= np.maximum(np.sqrt(distance_squared), 1.0e-12)
        radius_squared = float(light.falloff_radius_m) ** 2
        attenuation = radius_squared / (radius_squared + distance_squared)
        ndotl = np.clip(np.sum(normals * to_light, axis=1, keepdims=True), 0.0, 1.0)
        light_color = np.asarray(light.color, dtype=np.float32)[None]
        weighted = float(light.intensity) * attenuation * light_color
        illumination += ndotl * weighted
        half_vector = to_light + view
        half_vector /= np.maximum(
            np.linalg.norm(half_vector, axis=1, keepdims=True), 1.0e-12
        )
        ndoth = np.clip(
            np.sum(normals * half_vector, axis=1, keepdims=True), 0.0, 1.0
        )
        specular += (
            float(lighting.specular_intensity)
            * np.power(ndoth, float(lighting.shininess))
            * weighted
        )

    base = np.asarray(base_colors, dtype=np.float32)
    base_work = _srgb_to_linear_numpy(base) if lighting.gamma_correct else base
    lit = base_work * illumination + specular
    strength = np.asarray(lighting_strengths, dtype=np.float32).reshape(-1, 1)
    result = base_work + strength * (lit - base_work)
    if lighting.gamma_correct:
        result = _linear_to_srgb_numpy(result)
    return np.clip(result, 0.0, 1.0)


def _load_visual_templates(
    o3d: Any,
    kin: FullBodyKinematics,
    urdf_path: Path,
    lighting: LightingConfig | None = None,
) -> list[VisualTemplate]:
    templates: list[VisualTemplate] = []
    colors_by_link = _urdf_visual_colors(urdf_path)
    for link_name, visuals in kin.model.link_visuals.items():
        if link_name in EXCLUDED_VISUAL_LINKS:
            continue
        urdf_colors = colors_by_link.get(link_name, [])
        for visual_index, visual in enumerate(visuals):
            if not visual.exists or visual.format not in {"stl", "obj", "ply"}:
                continue
            mesh = o3d.io.read_triangle_mesh(str(visual.mesh_path))
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            triangles = np.asarray(mesh.triangles, dtype=np.int32)
            if vertices.size == 0 or triangles.size == 0:
                continue
            vertices = vertices * np.asarray(visual.scale, dtype=np.float64)[None]
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if vertex_normals.shape != vertices.shape:
                raise RuntimeError(f"Could not compute vertex normals for {visual.mesh_path}")
            templates.append(
                VisualTemplate(
                    link_name=link_name,
                    visual_origin=np.asarray(visual.origin, dtype=np.float64),
                    vertices=vertices,
                    vertex_normals=vertex_normals,
                    triangles=triangles,
                    triangle_color=(
                        urdf_colors[visual_index]
                        if visual_index < len(urdf_colors)
                        else DEFAULT_VISUAL_COLOR.copy()
                    ),
                    lighting_strength=(
                        1.0
                        if lighting is None
                        else _lighting_strength_for_link(link_name, lighting)
                    ),
                )
            )
    if not templates:
        raise RuntimeError("No triangle visual meshes could be loaded from the full URDF")
    return templates


def _retarget_fingerprint(
    q: np.ndarray,
    camera: np.ndarray,
    base: np.ndarray,
    frame_indices: list[int],
    occlusion_epsilon_m: float,
    near_clip_m: float,
    urdf_path: Path,
    backend: str,
    lighting: LightingConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"render_schema_version={RENDER_SCHEMA_VERSION}".encode("ascii"))
    digest.update(b"urdf_sha256=")
    digest.update(hashlib.sha256(urdf_path.read_bytes()).hexdigest().encode("ascii"))
    for array in (q, camera, base, np.asarray(frame_indices, dtype=np.int64)):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    digest.update(f"occlusion_epsilon_m={occlusion_epsilon_m:.12g}".encode("ascii"))
    digest.update(f"near_clip_m={near_clip_m:.12g}".encode("ascii"))
    digest.update(f"backend={backend}".encode("ascii"))
    digest.update(b"lighting=")
    digest.update(lighting.canonical_json().encode("utf-8"))
    return digest.hexdigest()


def _complete_cache(
    path: Path,
    *,
    source_path: Path,
    frame_indices: list[int],
    fingerprint: str,
) -> bool:
    if not path.is_dir():
        return False
    try:
        import zarr

        root = zarr.open(str(path), mode="r")
        return bool(root.attrs.get("render_complete", False)) and (
            Path(root.attrs["source_zarr_path"]).resolve() == source_path
        ) and root.attrs.get("retarget_fingerprint") == fingerprint and np.array_equal(
            np.asarray(root["source_frame_index"][...], dtype=np.int64),
            np.asarray(frame_indices, dtype=np.int64),
        )
    except Exception:
        return False


def _posed_mesh(
    templates: list[VisualTemplate],
    link_transforms: dict[str, np.ndarray],
    base_pose_cam0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    lighting_strengths: list[np.ndarray] = []
    vertex_offset = 0
    for template in templates:
        link_transform = link_transforms.get(template.link_name)
        if link_transform is None:
            continue
        transform = base_pose_cam0 @ link_transform @ template.visual_origin
        posed = _transform_points(transform, template.vertices)
        vertices.append(posed.astype(np.float32))
        triangles.append(template.triangles + vertex_offset)
        colors.append(
            np.repeat(
                template.triangle_color[None],
                template.triangles.shape[0],
                axis=0,
            )
        )
        lighting_strengths.append(
            np.full(
                template.triangles.shape[0],
                template.lighting_strength,
                dtype=np.float32,
            )
        )
        vertex_offset += posed.shape[0]
    if not vertices:
        raise RuntimeError("No full-URDF mesh triangles were posed for the frame")
    return (
        np.ascontiguousarray(np.concatenate(vertices, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(triangles, axis=0), dtype=np.int32),
        np.ascontiguousarray(np.concatenate(colors, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(lighting_strengths), dtype=np.float32),
    )


def _render_frame(
    *,
    o3d: Any,
    templates: list[VisualTemplate],
    kin: FullBodyKinematics,
    q: np.ndarray,
    base_pose_cam0: np.ndarray,
    cam0_from_cam_t: np.ndarray,
    intrinsic: np.ndarray,
    background: np.ndarray,
    scene_depth: np.ndarray,
    inpaint_mask: np.ndarray,
    occlusion_epsilon_m: float,
    near_clip_m: float,
    lighting: LightingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = background.shape[:2]
    links = kin.fk_all_links(q)
    vertices, triangles, triangle_colors, triangle_lighting_strengths = _posed_mesh(
        templates, links, base_pose_cam0
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    camera_from_cam0 = np.linalg.inv(cam0_from_cam_t)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(
            np.asarray(intrinsic, dtype=np.float64),
            dtype=o3d.core.Dtype.Float64,
        ),
        extrinsic_matrix=o3d.core.Tensor(
            camera_from_cam0,
            dtype=o3d.core.Dtype.Float64,
        ),
        width_px=int(width),
        height_px=int(height),
    )
    ray_values = rays.numpy()
    ray_values[..., :3] += float(near_clip_m) * ray_values[..., 3:]
    rays = o3d.core.Tensor(
        np.ascontiguousarray(ray_values),
        dtype=o3d.core.Dtype.Float32,
    )
    hits = scene.cast_rays(rays)
    t_hit = hits["t_hit"].numpy()
    primitive_ids = hits["primitive_ids"].numpy().astype(np.int64)
    silhouette = np.isfinite(t_hit)
    robot_depth = np.full((height, width), np.inf, dtype=np.float32)
    robot_rgb = np.zeros((height, width, 3), dtype=np.float32)
    if np.any(silhouette):
        hit_cam0 = (
            ray_values[..., :3]
            + t_hit[..., None] * ray_values[..., 3:]
        )
        hit_camera = _transform_points(
            camera_from_cam0,
            hit_cam0[silhouette],
        )
        robot_depth[silhouette] = hit_camera[:, 2].astype(np.float32)
        valid_ids = primitive_ids[silhouette]
        colors = triangle_colors[valid_ids]
        if "primitive_normals" in hits:
            normals_cam0 = hits["primitive_normals"].numpy()[silhouette]
            normals_camera = normals_cam0 @ camera_from_cam0[:3, :3].T
            colors = _shade_points_numpy(
                positions_camera=hit_camera,
                normals_camera=normals_camera,
                base_colors=colors,
                lighting_strengths=triangle_lighting_strengths[valid_ids],
                camera_from_base=camera_from_cam0 @ base_pose_cam0,
                lighting=lighting,
            )
        robot_rgb[silhouette] = np.clip(colors, 0.0, 1.0)

    depth = np.asarray(scene_depth, dtype=np.float32)
    mask = np.asarray(inpaint_mask)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if mask.ndim == 3:
        mask = mask[..., 0]
    scene_valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & (mask == 0)
    )
    robot_visible = (
        silhouette
        & (robot_depth > 0.0)
        & (
            ~scene_valid
            | (robot_depth <= depth + float(occlusion_epsilon_m))
        )
    )
    composite = np.asarray(background, dtype=np.uint8).copy()
    composite[robot_visible] = np.rint(
        robot_rgb[robot_visible] * 255.0
    ).astype(np.uint8)
    return composite, robot_visible, silhouette, robot_depth


class _EglOffscreenRenderer:
    """GPU rasterizer backed by Open3D Filament through headless EGL.

    Visual meshes are uploaded once.  Per frame we update only their rigid link
    transforms and the camera, avoiding the CPU path's repeated 2.1M-vertex
    posing and ray-scene rebuild.
    """

    def __init__(
        self,
        *,
        o3d: Any,
        templates: list[VisualTemplate],
        width: int,
        height: int,
        lighting: LightingConfig,
    ) -> None:
        self.o3d = o3d
        self.templates = templates
        self.width = int(width)
        self.height = int(height)
        self.lighting = lighting
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.width, self.height
        )
        scene = self.renderer.scene
        scene.set_background(np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32))
        scene.show_skybox(False)
        scene.scene.enable_indirect_light(True)
        scene.scene.set_indirect_light_intensity(
            30000.0 * float(lighting.ambient_intensity)
        )
        scene.scene.enable_sun_light(False)
        scene.view.set_post_processing(False)
        scene.view.set_antialiasing(False)
        scene.view.set_sample_count(1)
        scene.view.set_ambient_occlusion(False, False)
        scene.view.set_shadowing(False, scene.view.ShadowType.PCF)

        self.geometry_names: list[str] = []
        for light in lighting.lights:
            scene.scene.add_point_light(
                f"rgb_o3d_{light.name}",
                np.asarray(light.color, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                100000.0 * float(light.intensity),
                float(light.falloff_radius_m),
                False,
            )
        for visual_index, template in enumerate(templates):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(template.vertices, dtype=np.float64)
            )
            mesh.triangles = o3d.utility.Vector3iVector(
                np.asarray(template.triangles, dtype=np.int32)
            )
            mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.asarray(template.vertex_normals, dtype=np.float64)
            )
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = (
                "defaultUnlit" if _is_hand_link(template.link_name) else "defaultLit"
            )
            color = np.asarray(template.triangle_color, dtype=np.float32)
            material.base_color = [
                float(color[0]),
                float(color[1]),
                float(color[2]),
                1.0,
            ]
            material.base_roughness = float(
                np.clip(1.0 - lighting.specular_intensity, 0.05, 1.0)
            )
            material.base_reflectance = float(
                np.clip(lighting.specular_intensity, 0.0, 1.0)
            )
            name = f"visual_{visual_index:03d}"
            scene.add_geometry(name, mesh, material)
            self.geometry_names.append(name)

    def render(
        self,
        *,
        kin: FullBodyKinematics,
        q: np.ndarray,
        base_pose_cam0: np.ndarray,
        cam0_from_cam_t: np.ndarray,
        intrinsic: np.ndarray,
        background: np.ndarray,
        scene_depth: np.ndarray,
        inpaint_mask: np.ndarray,
        occlusion_epsilon_m: float,
        near_clip_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        links = kin.fk_all_links(q)
        scene = self.renderer.scene
        for light in self.lighting.lights:
            position_cam0 = _transform_points(
                base_pose_cam0,
                np.asarray(light.position_base_m, dtype=np.float64)[None],
            )[0]
            scene.scene.update_light_position(
                f"rgb_o3d_{light.name}",
                np.asarray(position_cam0, dtype=np.float32),
            )
        for name, template in zip(
            self.geometry_names, self.templates, strict=True
        ):
            link_transform = links.get(template.link_name)
            if link_transform is None:
                raise KeyError(
                    f"Missing FK transform for visual link {template.link_name}"
                )
            transform = base_pose_cam0 @ link_transform @ template.visual_origin
            scene.set_geometry_transform(
                name, np.asarray(transform, dtype=np.float64)
            )

        camera_from_cam0 = np.linalg.inv(cam0_from_cam_t)
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.renderer.setup_camera(
            intrinsic,
            np.asarray(camera_from_cam0, dtype=np.float64),
            self.width,
            self.height,
        )
        scene.camera.set_projection(
            intrinsic,
            max(float(near_clip_m), 1.0e-4),
            100.0,
            float(self.width),
            float(self.height),
        )
        robot_rgb_u8 = np.asarray(
            self.renderer.render_to_image(), dtype=np.uint8
        ).copy()
        robot_depth = np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True),
            dtype=np.float32,
        ).copy()
        silhouette = np.isfinite(robot_depth) & (robot_depth > 0.0)

        depth = np.asarray(scene_depth, dtype=np.float32)
        mask = np.asarray(inpaint_mask)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if mask.ndim == 3:
            mask = mask[..., 0]
        scene_valid = np.isfinite(depth) & (depth > 0.0) & (mask == 0)
        robot_visible = silhouette & (
            ~scene_valid
            | (robot_depth <= depth + float(occlusion_epsilon_m))
        )
        composite = np.asarray(background, dtype=np.uint8).copy()
        composite[robot_visible] = robot_rgb_u8[robot_visible]
        return composite, robot_visible, silhouette, robot_depth


class _CudaRasterizer:
    """Single-pass CUDA triangle rasterizer using NVIDIA nvdiffrast."""

    def __init__(
        self,
        *,
        templates: list[VisualTemplate],
        width: int,
        height: int,
        lighting: LightingConfig,
        device: str = "cuda:0",
    ) -> None:
        try:
            import torch
            import nvdiffrast.torch as dr
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "CUDA rgb_o3d rendering requires PyTorch and nvdiffrast"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA rgb_o3d rendering requested but CUDA is unavailable")
        self.torch = torch
        self.dr = dr
        self.device = torch.device(device)
        self.width = int(width)
        self.height = int(height)
        self.templates = templates
        self.lighting = lighting
        self.context = dr.RasterizeCudaContext(device=self.device)

        local_vertices: list[np.ndarray] = []
        local_normals: list[np.ndarray] = []
        vertex_template_ids: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        triangle_colors: list[np.ndarray] = []
        triangle_lighting_strengths: list[np.ndarray] = []
        vertex_offset = 0
        for template_index, template in enumerate(templates):
            vertices = np.asarray(template.vertices, dtype=np.float32)
            faces = np.asarray(template.triangles, dtype=np.int32)
            local_vertices.append(vertices)
            local_normals.append(
                np.asarray(template.vertex_normals, dtype=np.float32)
            )
            vertex_template_ids.append(
                np.full(vertices.shape[0], template_index, dtype=np.int64)
            )
            triangles.append(faces + vertex_offset)
            triangle_colors.append(
                np.repeat(
                    template.triangle_color[None], faces.shape[0], axis=0
                )
            )
            triangle_lighting_strengths.append(
                np.full(
                    faces.shape[0], template.lighting_strength, dtype=np.float32
                )
            )
            vertex_offset += vertices.shape[0]
        vertices_np = np.ascontiguousarray(
            np.concatenate(local_vertices, axis=0), dtype=np.float32
        )
        local_h = np.concatenate(
            [vertices_np, np.ones((vertices_np.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        self.local_h = torch.as_tensor(
            np.ascontiguousarray(local_h), device=self.device
        )
        self.local_normals = torch.as_tensor(
            np.ascontiguousarray(np.concatenate(local_normals), dtype=np.float32),
            device=self.device,
        )
        self.vertex_template_id = torch.as_tensor(
            np.ascontiguousarray(np.concatenate(vertex_template_ids)),
            dtype=torch.long,
            device=self.device,
        )
        self.triangles = torch.as_tensor(
            np.ascontiguousarray(np.concatenate(triangles, axis=0)),
            dtype=torch.int32,
            device=self.device,
        )
        self.triangle_colors = torch.as_tensor(
            np.ascontiguousarray(np.concatenate(triangle_colors, axis=0)),
            device=self.device,
        )
        self.triangle_lighting_strengths = torch.as_tensor(
            np.ascontiguousarray(np.concatenate(triangle_lighting_strengths)),
            device=self.device,
        )

    def _camera_vertices(
        self,
        *,
        kin: FullBodyKinematics,
        q: np.ndarray,
        base_pose_cam0: np.ndarray,
        cam0_from_cam_t: np.ndarray,
    ):
        torch = self.torch
        links = kin.fk_all_links(q)
        camera_from_cam0 = np.linalg.inv(cam0_from_cam_t)
        transforms = []
        for template in self.templates:
            link_transform = links.get(template.link_name)
            if link_transform is None:
                raise KeyError(
                    f"Missing FK transform for visual link {template.link_name}"
                )
            transforms.append(
                camera_from_cam0
                @ base_pose_cam0
                @ link_transform
                @ template.visual_origin
            )
        transform_tensor = torch.as_tensor(
            np.ascontiguousarray(np.stack(transforms), dtype=np.float32),
            device=self.device,
        )
        selected = transform_tensor[self.vertex_template_id]
        camera_vertices = torch.bmm(
            selected, self.local_h.unsqueeze(-1)
        ).squeeze(-1)[:, :3]
        camera_normals = torch.bmm(
            selected[:, :3, :3], self.local_normals.unsqueeze(-1)
        ).squeeze(-1)
        camera_normals = torch.nn.functional.normalize(
            camera_normals, dim=1, eps=1.0e-12
        )
        return camera_vertices, camera_normals

    def render(
        self,
        *,
        kin: FullBodyKinematics,
        q: np.ndarray,
        base_pose_cam0: np.ndarray,
        cam0_from_cam_t: np.ndarray,
        intrinsic: np.ndarray,
        background: np.ndarray,
        scene_depth: np.ndarray,
        inpaint_mask: np.ndarray,
        occlusion_epsilon_m: float,
        near_clip_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        dr = self.dr
        with torch.inference_mode():
            camera_vertices, camera_vertex_normals = self._camera_vertices(
                kin=kin,
                q=q,
                base_pose_cam0=base_pose_cam0,
                cam0_from_cam_t=cam0_from_cam_t,
            )
            k = np.asarray(intrinsic, dtype=np.float64)
            near = max(float(near_clip_m), 1.0e-4)
            far = 100.0
            x, y, z = camera_vertices.unbind(dim=1)
            # OpenCV (+x right, +y down, +z forward) to OpenGL clip space.
            x_clip = (
                (2.0 * float(k[0, 0]) / self.width) * x
                + (2.0 * float(k[0, 2]) / self.width - 1.0) * z
            )
            y_clip = (
                (-2.0 * float(k[1, 1]) / self.height) * y
                + (1.0 - 2.0 * float(k[1, 2]) / self.height) * z
            )
            z_clip = ((far + near) / (far - near)) * z - (
                2.0 * far * near / (far - near)
            )
            clip = torch.stack((x_clip, y_clip, z_clip, z), dim=1).contiguous()
            rast, _ = dr.rasterize(
                self.context,
                clip.unsqueeze(0),
                self.triangles,
                resolution=(self.height, self.width),
                grad_db=False,
            )
            # nvdiffrast follows OpenGL's bottom-left image origin.
            rast = torch.flip(rast, dims=(1,)).contiguous()
            triangle_id = rast[0, :, :, 3].to(torch.long) - 1
            silhouette = triangle_id >= 0
            safe_id = triangle_id.clamp(min=0)

            depth_values, _ = dr.interpolate(
                camera_vertices[:, 2:3].unsqueeze(0).contiguous(),
                rast,
                self.triangles,
            )
            robot_depth = depth_values[0, :, :, 0]
            robot_depth = torch.where(
                silhouette,
                robot_depth,
                torch.full_like(robot_depth, float("inf")),
            )

            surface_position, _ = dr.interpolate(
                camera_vertices.unsqueeze(0).contiguous(), rast, self.triangles
            )
            surface_normal, _ = dr.interpolate(
                camera_vertex_normals.unsqueeze(0).contiguous(),
                rast,
                self.triangles,
            )
            surface_position = surface_position[0]
            surface_normal = torch.nn.functional.normalize(
                surface_normal[0], dim=2, eps=1.0e-12
            )
            view_direction = torch.nn.functional.normalize(
                -surface_position, dim=2, eps=1.0e-12
            )
            flip_normal = (
                torch.sum(surface_normal * view_direction, dim=2, keepdim=True) < 0.0
            )
            surface_normal = torch.where(
                flip_normal, -surface_normal, surface_normal
            )

            base_color = self.triangle_colors[safe_id]
            if self.lighting.gamma_correct:
                base_work = torch.where(
                    base_color <= 0.04045,
                    base_color / 12.92,
                    torch.pow((base_color + 0.055) / 1.055, 2.4),
                )
            else:
                base_work = base_color
            illumination = torch.full_like(
                base_work, float(self.lighting.ambient_intensity)
            )
            specular = torch.zeros_like(base_work)
            camera_from_base = np.linalg.inv(cam0_from_cam_t) @ base_pose_cam0
            for light in self.lighting.lights:
                light_camera = _transform_points(
                    camera_from_base,
                    np.asarray(light.position_base_m, dtype=np.float64)[None],
                )[0]
                light_position = torch.as_tensor(
                    np.ascontiguousarray(light_camera, dtype=np.float32),
                    device=self.device,
                )
                to_light_raw = light_position[None, None] - surface_position
                distance_squared = torch.sum(
                    to_light_raw * to_light_raw, dim=2, keepdim=True
                )
                to_light = torch.nn.functional.normalize(
                    to_light_raw, dim=2, eps=1.0e-12
                )
                radius_squared = float(light.falloff_radius_m) ** 2
                attenuation = radius_squared / (radius_squared + distance_squared)
                ndotl = torch.clamp(
                    torch.sum(surface_normal * to_light, dim=2, keepdim=True),
                    0.0,
                    1.0,
                )
                light_color = torch.as_tensor(
                    light.color, dtype=torch.float32, device=self.device
                )[None, None]
                weighted_light = (
                    float(light.intensity) * attenuation * light_color
                )
                illumination = illumination + ndotl * weighted_light
                half_vector = torch.nn.functional.normalize(
                    to_light + view_direction, dim=2, eps=1.0e-12
                )
                ndoth = torch.clamp(
                    torch.sum(surface_normal * half_vector, dim=2, keepdim=True),
                    0.0,
                    1.0,
                )
                specular = specular + (
                    float(self.lighting.specular_intensity)
                    * torch.pow(ndoth, float(self.lighting.shininess))
                    * weighted_light
                )
            lit_color = base_work * illumination + specular
            strength = self.triangle_lighting_strengths[safe_id][..., None]
            robot_rgb = base_work + strength * (lit_color - base_work)
            if self.lighting.gamma_correct:
                robot_rgb = torch.where(
                    robot_rgb <= 0.0031308,
                    12.92 * robot_rgb,
                    1.055 * torch.pow(torch.clamp(robot_rgb, min=0.0), 1.0 / 2.4)
                    - 0.055,
                )

            depth_np = np.asarray(scene_depth, dtype=np.float32)
            mask_np = np.asarray(inpaint_mask)
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 0]
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            scene_depth_gpu = torch.as_tensor(
                np.ascontiguousarray(depth_np), device=self.device
            )
            mask_gpu = torch.as_tensor(
                np.ascontiguousarray(mask_np), device=self.device
            )
            scene_valid = (
                torch.isfinite(scene_depth_gpu)
                & (scene_depth_gpu > 0.0)
                & (mask_gpu == 0)
            )
            robot_visible = silhouette & (
                ~scene_valid
                | (
                    robot_depth
                    <= scene_depth_gpu + float(occlusion_epsilon_m)
                )
            )
            composite = torch.as_tensor(
                np.ascontiguousarray(background),
                dtype=torch.uint8,
                device=self.device,
            ).clone()
            robot_rgb_u8 = torch.clamp(
                torch.round(robot_rgb * 255.0), 0.0, 255.0
            ).to(torch.uint8)
            composite[robot_visible] = robot_rgb_u8[robot_visible]
            return (
                composite.cpu().numpy(),
                robot_visible.cpu().numpy(),
                silhouette.cpu().numpy(),
                robot_depth.cpu().numpy().astype(np.float32, copy=False),
            )


def render_rgb_o3d_sequence(
    *,
    source_zarr_path: str | Path,
    work_dir: str | Path,
    urdf_path: str | Path,
    output_path: str | Path | None = None,
    occlusion_epsilon_m: float = 0.005,
    near_clip_m: float = 0.0,
    backend: str = "cpu",
    lighting_config_path: str | Path = DEFAULT_LIGHTING_CONFIG,
    force: bool = False,
) -> Path:
    """Render every selected frame and cache RGB plus diagnostic mask/depth."""
    import zarr
    from numcodecs import Blosc

    source_path = Path(source_zarr_path).resolve()
    work_dir = Path(work_dir).resolve()
    output_path = (
        Path(output_path).resolve()
        if output_path is not None
        else work_dir / "rgb_o3d.zarr"
    )
    metadata = json.loads((work_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata_source = Path(metadata["zarr_path"]).resolve()
    if metadata_source != source_path:
        raise ValueError(
            f"Retarget/source mismatch: {metadata_source} != {source_path}"
        )
    frame_indices = [int(value) for value in metadata["frames"]]
    q = np.load(work_dir / "whole_body_joints.npy")
    camera = np.load(work_dir / "camera_trajectory_cam0.npy")
    base = np.load(work_dir / "base_pose_cam0.npy")
    joint_names = json.loads(
        (work_dir / "whole_body_joint_names.json").read_text(encoding="utf-8")
    )
    if q.shape[0] != len(frame_indices) or camera.shape != (len(frame_indices), 4, 4):
        raise ValueError("Retarget frame count does not match metadata/camera trajectory")
    if near_clip_m < 0.0:
        raise ValueError("near_clip_m must be non-negative")
    backend = str(backend).lower()
    if backend not in {"cpu", "egl", "cuda"}:
        raise ValueError(f"Unsupported rgb_o3d backend: {backend!r}")
    urdf_path = Path(urdf_path).resolve()
    urdf_sha256 = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    lighting_path = Path(lighting_config_path).expanduser().resolve()
    lighting = load_lighting_config(lighting_path)
    fingerprint = _retarget_fingerprint(
        q,
        camera,
        base,
        frame_indices,
        occlusion_epsilon_m,
        near_clip_m,
        urdf_path,
        backend,
        lighting,
    )
    if not force and _complete_cache(
        output_path,
        source_path=source_path,
        frame_indices=frame_indices,
        fingerprint=fingerprint,
    ):
        print(f"Using cached rgb_o3d render: {output_path}", flush=True)
        return output_path

    model = parse_urdf_model(urdf_path, expected_dof=int(q.shape[1]))
    kin = FullBodyKinematics(model)
    if kin.joint_names != list(joint_names):
        raise ValueError("Saved whole-body joint order does not match the full URDF")
    if backend == "egl":
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    o3d = _import_open3d()
    templates = _load_visual_templates(o3d, kin, urdf_path, lighting)
    source = zarr.open(str(source_path), mode="r")
    source_frame_keys = sorted(source["data"].keys())
    first = source["data"][source_frame_keys[frame_indices[0]]]
    background_first = np.asarray(first["inpaint_rgb"][...], dtype=np.uint8)
    height, width = background_first.shape[:2]
    egl_renderer = (
        _EglOffscreenRenderer(
            o3d=o3d,
            templates=templates,
            width=width,
            height=height,
            lighting=lighting,
        )
        if backend == "egl"
        else None
    )
    cuda_renderer = (
        _CudaRasterizer(
            templates=templates,
            width=width,
            height=height,
            lighting=lighting,
        )
        if backend == "cuda"
        else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    try:
        root = zarr.open(str(temporary), mode="w")
        rgb_out = root.create_dataset(
            "rgb_o3d",
            shape=(len(frame_indices), height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.uint8,
            compressor=compressor,
        )
        visible_out = root.create_dataset(
            "robot_visible_mask",
            shape=(len(frame_indices), height, width),
            chunks=(1, height, width),
            dtype=bool,
            compressor=compressor,
        )
        silhouette_out = root.create_dataset(
            "robot_silhouette_mask",
            shape=(len(frame_indices), height, width),
            chunks=(1, height, width),
            dtype=bool,
            compressor=compressor,
        )
        depth_out = root.create_dataset(
            "robot_depth",
            shape=(len(frame_indices), height, width),
            chunks=(1, height, width),
            dtype=np.float32,
            compressor=compressor,
        )
        root.create_dataset(
            "source_frame_index",
            data=np.asarray(frame_indices, dtype=np.int64),
            compressor=compressor,
        )
        for output_index, source_index in enumerate(frame_indices):
            frame = source["data"][source_frame_keys[source_index]]
            intrinsic = (
                np.asarray(frame["intrinsic"][...], dtype=np.float64)
                if "intrinsic" in frame
                else np.asarray(source["intrinsic"][...], dtype=np.float64)
            )
            render_args = {
                "kin": kin,
                "q": q[output_index],
                "base_pose_cam0": base,
                "cam0_from_cam_t": camera[output_index],
                "intrinsic": intrinsic,
                "background": np.asarray(frame["inpaint_rgb"][...], dtype=np.uint8),
                "scene_depth": np.asarray(frame["depth"][...], dtype=np.float32),
                "inpaint_mask": np.asarray(frame["inpaint_mask"][...]),
                "occlusion_epsilon_m": occlusion_epsilon_m,
                "near_clip_m": near_clip_m,
            }
            if cuda_renderer is not None:
                composite, visible, silhouette, robot_depth = cuda_renderer.render(
                    **render_args
                )
            elif egl_renderer is not None:
                composite, visible, silhouette, robot_depth = egl_renderer.render(
                    **render_args
                )
            else:
                composite, visible, silhouette, robot_depth = _render_frame(
                    o3d=o3d,
                    templates=templates,
                    lighting=lighting,
                    **render_args,
                )
            rgb_out[output_index] = composite
            visible_out[output_index] = visible
            silhouette_out[output_index] = silhouette
            depth_out[output_index] = robot_depth
            print(
                f"[rgb_o3d:{backend}] "
                f"frame={source_index} output={output_index} "
                f"silhouette_pixels={int(np.count_nonzero(silhouette))} "
                f"visible_pixels={int(np.count_nonzero(visible))}",
                flush=True,
            )
        root.attrs.update(
            {
                "schema_version": RENDER_SCHEMA_VERSION,
                "source_zarr_path": str(source_path),
                "retarget_fingerprint": fingerprint,
                "world_frame": "cam0",
                "camera_semantics": "T_cam0_cam_t; inverse used as world-to-camera",
                "background": "inpaint_rgb",
                "scene_occlusion": "depth is used only where inpaint_mask == 0",
                "occlusion_epsilon_m": float(occlusion_epsilon_m),
                "near_clip_m": float(near_clip_m),
                "ray_origin": "current camera optical center",
                "excluded_visual_links": sorted(EXCLUDED_VISUAL_LINKS),
                "visual_color_semantics": (
                    "Exact URDF material RGB for every visual, including all hand "
                    "links; per-link lighting strength from the lighting config"
                ),
                "hand_color_override": False,
                "full_urdf": str(urdf_path),
                "urdf_sha256": urdf_sha256,
                "visual_mesh_count": len(templates),
                "render_backend": backend,
                "lighting_config_path": str(lighting_path),
                "lighting_config_sha256": lighting.fingerprint(),
                "lighting_config": lighting.to_dict(),
                "render_device": (
                    "Open3D Filament EGL/OpenGL GPU"
                    if backend == "egl"
                    else (
                        "NVIDIA CUDA via nvdiffrast"
                        if backend == "cuda"
                        else "Open3D tensor RaycastingScene CPU"
                    )
                ),
                "render_complete": True,
            }
        )
        if output_path.exists():
            shutil.rmtree(output_path)
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_path


def export_rgb_o3d_mp4(
    rgb_o3d_zarr_path: str | Path,
    output_path: str | Path,
    *,
    fps: float = 15.0,
) -> Path:
    """Encode the cached rgb_o3d frames into an MP4 video."""
    if fps <= 0.0:
        raise ValueError("rgb_o3d MP4 fps must be positive")

    import imageio.v2 as imageio
    import zarr

    rgb_o3d_zarr_path = Path(rgb_o3d_zarr_path).resolve()
    output_path = Path(output_path).resolve()
    root = zarr.open(str(rgb_o3d_zarr_path), mode="r")
    if not bool(root.attrs.get("render_complete", False)):
        raise ValueError(f"Incomplete rgb_o3d render cache: {rgb_o3d_zarr_path}")
    frames = root["rgb_o3d"]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected rgb_o3d frames with shape (N,H,W,3), got {frames.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}.mp4"
    try:
        with imageio.get_writer(
            str(temporary),
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            for frame_index in range(int(frames.shape[0])):
                writer.append_data(np.asarray(frames[frame_index], dtype=np.uint8))
                if frame_index % 100 == 0:
                    print(
                        f"[rgb_o3d mp4] frame={frame_index}/{int(frames.shape[0])}",
                        flush=True,
                    )
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return output_path
