#!/usr/bin/env python3
"""Open3D viewer for retarget diagnostics.

The point cloud can use raw ``rgb + depth`` directly, or ``inpaint_rgb`` while
excluding every pixel where ``inpaint_mask == 1``.  It shows the complete
full-URDF robot mesh, human
target taxels, achieved robot taxels, contact/active taxels, human sensor
origins, and coordinate frames for both palm links and both flange links.
Optional robot and measured-sensor fingertip axes make orientation agreement
directly inspectable.

If ``visualization_diagnostics.npz`` contains a complete no-compliance branch,
contact-finger inward vectors are drawn from that branch's full-URDF fingertip
position to the normal compliance-enabled full-URDF fingertip position.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import DEFAULT_URDF, ensure_project_imports

ensure_project_imports()

from retarget.robot.urdf_model import parse_urdf_model  # noqa: E402

from .full_body_kinematics import FullBodyKinematics  # noqa: E402


AXIS_LINKS = ("left_palm_link", "right_palm_link", "left_flange", "right_flange")
FINGERTIP_LINKS = tuple(
    f"{side}_finger{finger}_tip_link"
    for side in ("left", "right")
    for finger in range(1, 6)
)


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Open3D is not installed in the active Python environment.") from exc
    return o3d


@dataclass
class GeometryHandle:
    geometry: object
    template_vertices: np.ndarray
    template_normals: np.ndarray | None
    link_name: str | None = None
    visual_origin: np.ndarray | None = None


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _normals(mesh) -> np.ndarray | None:
    return (
        np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
        if mesh.has_vertex_normals()
        else None
    )


def _copy_mesh(o3d, mesh):
    copied = o3d.geometry.TriangleMesh()
    copied.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices, dtype=np.float64).copy()
    )
    copied.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles, dtype=np.int32).copy()
    )
    if mesh.has_vertex_normals():
        copied.vertex_normals = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
        )
    if mesh.has_vertex_colors():
        copied.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors, dtype=np.float64).copy()
        )
    return copied


def _paint_link(mesh, link_name: str) -> None:
    name = link_name.lower()
    if "finger" in name or "palm" in name:
        color = (0.72, 0.72, 0.72)
    elif "left" in name:
        color = (0.25, 0.48, 0.92)
    elif "right" in name:
        color = (0.94, 0.42, 0.26)
    elif "wheel" in name:
        color = (0.05, 0.05, 0.05)
    else:
        color = (0.66, 0.68, 0.66)
    mesh.paint_uniform_color(color)


def _update_handle(o3d, handle: GeometryHandle, transform: np.ndarray) -> None:
    handle.geometry.vertices = o3d.utility.Vector3dVector(
        np.ascontiguousarray(_transform_points(transform, handle.template_vertices))
    )
    if handle.template_normals is not None:
        handle.geometry.vertex_normals = o3d.utility.Vector3dVector(
            np.ascontiguousarray(handle.template_normals @ transform[:3, :3].T)
        )


def _load_meshes(o3d, kin: FullBodyKinematics) -> tuple[list[GeometryHandle], int]:
    handles: list[GeometryHandle] = []
    skipped = 0
    for link_name, visuals in kin.model.link_visuals.items():
        for visual in visuals:
            if not visual.exists or visual.format not in {"stl", "obj", "ply"}:
                skipped += 1
                continue
            mesh = o3d.io.read_triangle_mesh(str(visual.mesh_path))
            if mesh.is_empty():
                skipped += 1
                continue
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            vertices *= np.asarray(visual.scale, dtype=np.float64)[None]
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            _paint_link(mesh, link_name)
            mesh.compute_vertex_normals()
            mesh = _copy_mesh(o3d, mesh)
            handles.append(
                GeometryHandle(
                    geometry=mesh,
                    template_vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
                    template_normals=_normals(mesh),
                    link_name=link_name,
                    visual_origin=np.asarray(visual.origin, dtype=np.float64),
                )
            )
    return handles, skipped


def _axis(o3d, size: float) -> GeometryHandle:
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    mesh.compute_vertex_normals()
    return GeometryHandle(
        geometry=mesh,
        template_vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
        template_normals=_normals(mesh),
    )


def _empty_cloud(o3d):
    return o3d.geometry.PointCloud()


def _set_cloud(o3d, cloud, points: np.ndarray, colors: np.ndarray) -> None:
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    )


def _set_lines(o3d, line_set, starts: np.ndarray, ends: np.ndarray, color) -> None:
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
    count = min(len(starts), len(ends))
    if count == 0:
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        return
    points = np.vstack((starts[:count], ends[:count]))
    lines = np.column_stack((np.arange(count), np.arange(count) + count)).astype(np.int32)
    colors = np.tile(np.asarray(color, dtype=np.float64)[None], (count, 1))
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)


def _load_outputs(output_dir: Path) -> dict:
    required = {
        "q": output_dir / "whole_body_joints.npy",
        "camera": output_dir / "camera_trajectory_cam0.npy",
        "base": output_dir / "base_pose_cam0.npy",
        "joint_names": output_dir / "whole_body_joint_names.json",
        "metadata": output_dir / "metadata.json",
        "visualization": output_dir / "visualization_diagnostics.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing visualization inputs:\n" + "\n".join(missing))
    with np.load(required["visualization"]) as archive:
        diagnostics = {name: archive[name] for name in archive.files}
    return {
        "q": np.load(required["q"]),
        "camera": np.load(required["camera"]),
        "base": np.load(required["base"]),
        "joint_names": json.loads(required["joint_names"].read_text(encoding="utf-8")),
        "metadata": json.loads(required["metadata"].read_text(encoding="utf-8")),
        "diagnostics": diagnostics,
    }


def _open_source(zarr_path: Path):
    import zarr

    return zarr.open(str(zarr_path), mode="r")


def _sensor_origins_cam0(source_root, frame_key: str, cam0_from_cam: np.ndarray) -> np.ndarray:
    group = source_root["data"][frame_key]
    origins = np.full((2, 5, 3), np.nan, dtype=np.float64)
    for side_index, key in enumerate(("T_left_sensor_to_cam", "T_right_sensor_to_cam")):
        if key not in group:
            continue
        transforms = np.asarray(group[key][...], dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
            continue
        count = min(5, transforms.shape[0])
        points_cam = transforms[:count, :3, 3]
        origins[side_index, :count] = _transform_points(cam0_from_cam, points_cam)
    return origins


def _scene_point_cloud(
    o3d,
    source_root,
    frame_key: str,
    cam0_from_cam: np.ndarray,
    *,
    sample_step: int,
    max_depth: float,
    point_cloud_source: str,
):
    group = source_root["data"][frame_key]
    depth = np.asarray(group["depth"][...], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    use_inpaint = point_cloud_source == "inpaint" or (
        point_cloud_source == "auto"
        and "inpaint_rgb" in group
        and "inpaint_mask" in group
    )
    if use_inpaint:
        if "inpaint_rgb" not in group or "inpaint_mask" not in group:
            raise KeyError(
                f"Frame {frame_key} has no inpaint_rgb/inpaint_mask; use "
                "--point_cloud_source raw"
            )
        rgb = np.asarray(group["inpaint_rgb"][...], dtype=np.uint8)
        mask = np.asarray(group["inpaint_mask"][...])
        if mask.ndim == 3:
            mask = mask[..., 0]
    else:
        if "rgb" not in group:
            raise KeyError(f"Frame {frame_key} has no raw rgb")
        rgb = np.asarray(group["rgb"][...], dtype=np.uint8)
        mask = np.zeros(depth.shape, dtype=bool)
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError(
            f"rgb/depth/mask shape mismatch: rgb={rgb.shape}, mask={mask.shape}, "
            f"depth={depth.shape}"
        )
    intrinsic = (
        np.asarray(group["intrinsic"][...], dtype=np.float64)
        if "intrinsic" in group
        else np.asarray(source_root["intrinsic"][...], dtype=np.float64)
    )
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    valid = (depth > 0.0) & (mask == 0)
    if max_depth > 0:
        valid &= depth <= max_depth
    valid &= (yy % sample_step == 0) & (xx % sample_step == 0)
    cloud = _empty_cloud(o3d)
    if not np.any(valid):
        return cloud
    z = depth[valid]
    x = (xx[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = _transform_points(cam0_from_cam, np.column_stack((x, y, z)))
    _set_cloud(o3d, cloud, points, rgb[valid].astype(np.float64) / 255.0)
    return cloud


def build_scene(args: argparse.Namespace):
    o3d = _import_open3d()
    outputs = _load_outputs(args.output_dir)
    q = np.asarray(outputs["q"], dtype=np.float64)
    camera = np.asarray(outputs["camera"], dtype=np.float64)
    base = np.asarray(outputs["base"], dtype=np.float64)
    metadata = outputs["metadata"]
    diagnostics = outputs["diagnostics"]
    source_path = Path(args.zarr_path or metadata["zarr_path"]).resolve()
    source = _open_source(source_path)
    frame_keys = sorted(source["data"].keys())
    source_indices = np.asarray(diagnostics["source_frame_index"], dtype=np.int64)

    model = parse_urdf_model(
        args.urdf,
        expected_dof=int(q.shape[1] if args.expected_dof is None else args.expected_dof),
    )
    kin = FullBodyKinematics(model)
    if list(outputs["joint_names"]) != kin.joint_names:
        raise ValueError("Saved joint order does not match the selected full URDF.")
    if q.shape[0] != len(source_indices) or camera.shape != (q.shape[0], 4, 4):
        raise ValueError("Retarget, camera, and source-frame counts do not match.")

    mesh_handles, skipped_meshes = _load_meshes(o3d, kin)
    if getattr(args, "hide_camera_housing", False):
        mesh_handles = [h for h in mesh_handles if h.link_name not in {"head21", "head22"}]
    axis_handles = {name: _axis(o3d, args.link_axis_size) for name in AXIS_LINKS}
    show_fingertip_axes = bool(getattr(args, "show_fingertip_axes", False))
    fingertip_axis_size = float(getattr(args, "fingertip_axis_size", 0.025))
    robot_fingertip_axes = (
        {name: _axis(o3d, fingertip_axis_size) for name in FINGERTIP_LINKS}
        if show_fingertip_axes
        else {}
    )
    # Shorter target axes remain visible inside the longer robot axes when the
    # two measured/achieved coordinate frames agree.
    target_fingertip_axes = (
        {
            (side, finger): _axis(o3d, 0.65 * fingertip_axis_size)
            for side in ("left", "right")
            for finger in range(5)
        }
        if show_fingertip_axes
        else {}
    )
    point_cloud = _empty_cloud(o3d)
    target_taxels = _empty_cloud(o3d)
    robot_taxels = _empty_cloud(o3d)
    patch_centers = _empty_cloud(o3d)
    human_sensor_origins = _empty_cloud(o3d)
    inward_endpoints = _empty_cloud(o3d)
    taxel_correspondence = o3d.geometry.LineSet()
    inward_vectors = o3d.geometry.LineSet()

    target_all = np.asarray(diagnostics["target_taxels_cam0"], dtype=np.float64)
    robot_all = np.asarray(diagnostics["robot_taxels_cam0"], dtype=np.float64)
    force_norms_all = np.asarray(diagnostics["force_norms"], dtype=np.float64)
    active_all = np.asarray(diagnostics["active_taxel_mask"], dtype=bool)
    contact_all = np.asarray(diagnostics["contact_finger_mask"], dtype=bool)
    patch_all = np.asarray(diagnostics["patch_centers_cam0"], dtype=np.float64)
    actual_tip_all = np.asarray(diagnostics["actual_tip_positions_cam0"], dtype=np.float64)
    baseline_tip_all = np.asarray(
        diagnostics["no_compliance_tip_positions_cam0"], dtype=np.float64
    )
    has_baseline = bool(np.asarray(diagnostics["has_no_compliance_branch"]).item())
    has_baseline = has_baseline and baseline_tip_all.shape == actual_tip_all.shape

    target_inactive = np.asarray([0.0, 0.75, 0.95])
    target_active = np.asarray([1.0, 0.82, 0.0])
    robot_inactive = np.asarray([0.85, 0.18, 0.90])
    robot_active = np.asarray([1.0, 0.05, 0.02])
    human_sensor_color = np.asarray([[1.0, 1.0, 1.0]])

    def update_frame(output_index: int) -> None:
        output_index = int(np.clip(output_index, 0, q.shape[0] - 1))
        source_index = int(source_indices[output_index])
        cloud = _scene_point_cloud(
            o3d,
            source,
            frame_keys[source_index],
            camera[output_index],
            sample_step=args.point_sample_step,
            max_depth=args.max_depth,
            point_cloud_source=args.point_cloud_source,
        )
        _set_cloud(
            o3d,
            point_cloud,
            np.asarray(cloud.points),
            np.asarray(cloud.colors),
        )

        target = target_all[output_index].reshape(-1, 3)
        robot = robot_all[output_index].reshape(-1, 3)
        active = active_all[output_index].reshape(-1)
        force = force_norms_all[output_index].reshape(-1)
        target_colors = np.tile(target_inactive[None], (len(target), 1))
        target_colors[active] = target_active
        robot_colors = np.tile(robot_inactive[None], (len(robot), 1))
        robot_colors[active] = robot_active
        if np.any(active):
            scale = np.clip(force[active] / max(args.force_color_scale, 1e-12), 0.0, 1.0)
            robot_colors[active, 1] = 0.25 * (1.0 - scale)
        _set_cloud(o3d, target_taxels, target, target_colors)
        _set_cloud(o3d, robot_taxels, robot, robot_colors)
        if args.show_taxel_correspondence:
            _set_lines(o3d, taxel_correspondence, target, robot, (1.0, 0.45, 0.0))
        else:
            _set_lines(o3d, taxel_correspondence, [], [], (1.0, 0.45, 0.0))

        contact = contact_all[output_index]
        patches = patch_all[output_index][contact]
        patches = patches[np.all(np.isfinite(patches), axis=1)]
        _set_cloud(
            o3d,
            patch_centers,
            patches,
            np.tile(np.asarray([[0.0, 1.0, 0.2]]), (len(patches), 1)),
        )

        if args.show_human_sensor_origins:
            sensor_origins = _sensor_origins_cam0(
                source,
                frame_keys[source_index],
                camera[output_index],
            ).reshape(-1, 3)
            sensor_origins = sensor_origins[np.all(np.isfinite(sensor_origins), axis=1)]
            _set_cloud(
                o3d,
                human_sensor_origins,
                sensor_origins,
                np.tile(human_sensor_color, (len(sensor_origins), 1)),
            )
        else:
            _set_cloud(
                o3d,
                human_sensor_origins,
                np.zeros((0, 3)),
                np.zeros((0, 3)),
            )

        if has_baseline:
            starts = baseline_tip_all[output_index][contact]
            ends = actual_tip_all[output_index][contact]
            finite = np.all(np.isfinite(starts), axis=1) & np.all(np.isfinite(ends), axis=1)
            starts, ends = starts[finite], ends[finite]
            _set_lines(o3d, inward_vectors, starts, ends, (0.15, 1.0, 0.15))
            _set_cloud(
                o3d,
                inward_endpoints,
                ends,
                np.tile(np.asarray([[1.0, 0.0, 0.0]]), (len(ends), 1)),
            )
        else:
            _set_lines(o3d, inward_vectors, [], [], (0.15, 1.0, 0.15))
            _set_cloud(o3d, inward_endpoints, np.zeros((0, 3)), np.zeros((0, 3)))

        links = kin.fk_all_links(q[output_index])
        for handle in mesh_handles:
            if handle.link_name in links:
                _update_handle(
                    o3d,
                    handle,
                    base @ links[handle.link_name] @ handle.visual_origin,
                )
        for link_name, handle in axis_handles.items():
            if link_name not in links:
                raise KeyError(f"Required visualization link missing from URDF: {link_name}")
            _update_handle(o3d, handle, base @ links[link_name])
        for link_name, handle in robot_fingertip_axes.items():
            if link_name not in links:
                raise KeyError(f"Required fingertip link missing from URDF: {link_name}")
            _update_handle(o3d, handle, base @ links[link_name])
        if target_fingertip_axes:
            source_group = source["data"][frame_keys[source_index]]
            for side in ("left", "right"):
                transforms = np.asarray(
                    source_group[f"T_{side}_sensor_to_cam"][...], dtype=np.float64
                )
                for finger in range(5):
                    _update_handle(
                        o3d,
                        target_fingertip_axes[(side, finger)],
                        camera[output_index] @ transforms[finger],
                    )

    update_frame(args.frame)
    geometries = [
        point_cloud,
        target_taxels,
        robot_taxels,
        patch_centers,
        human_sensor_origins,
        taxel_correspondence,
        inward_vectors,
        inward_endpoints,
        *(handle.geometry for handle in mesh_handles),
        *(handle.geometry for handle in axis_handles.values()),
        *(handle.geometry for handle in robot_fingertip_axes.values()),
        *(handle.geometry for handle in target_fingertip_axes.values()),
    ]
    print(f"Source: {source_path}")
    print(f"Frames: {q.shape[0]}; complete robot mesh visuals: {len(mesh_handles)}")
    print(f"Point-cloud RGB source: {args.point_cloud_source}")
    print(f"Skipped missing/unsupported mesh visuals: {skipped_meshes}")
    print("Link axes: left_palm_link, right_palm_link, left_flange, right_flange")
    if show_fingertip_axes:
        print(
            "Fingertip axes: long=robot URDF tip frame, short=measured sensor target frame"
        )
    print(
        "Taxels/targets: cyan=human taxels, yellow=active human taxels, "
        "lime=contact patch centers, white=human sensor origins, "
        "magenta=robot taxels, red=active robot taxels"
    )
    print(
        "Inward vectors: enabled (no-compliance -> actual, green with red endpoint)"
        if has_baseline
        else "Inward vectors: hidden (no complete no-compliance branch detected)"
    )
    return o3d, geometries, update_frame, q.shape[0]


def run_visualizer(args: argparse.Namespace) -> None:
    o3d, geometries, update_frame, frame_count = build_scene(args)
    if args.static:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Retarget Open3D",
            width=args.width,
            height=args.height,
        )
        return

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    if not visualizer.create_window("Retarget Open3D", width=args.width, height=args.height):
        raise RuntimeError("Open3D needs a desktop display. Use the demo's --no-view option on a headless machine.")
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.reset_view_point(True)
    if getattr(args, "scene_camera", False):
        outputs = _load_outputs(args.output_dir)
        source = _open_source(Path(args.zarr_path or outputs["metadata"]["zarr_path"]))
        frame = int(np.clip(args.frame, 0, frame_count - 1))
        index = int(outputs["diagnostics"]["source_frame_index"][frame])
        key = sorted(source["data"].keys())[index]
        height, width = source["data"][key]["depth"].shape[:2]
        K = np.asarray(source["intrinsic"], dtype=np.float64)
        sx, sy = args.width / width, args.height / height
        parameters = o3d.camera.PinholeCameraParameters()
        parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            args.width, args.height, K[0, 0] * sx, K[1, 1] * sy,
            K[0, 2] * sx, K[1, 2] * sy,
        )
        parameters.extrinsic = np.linalg.inv(outputs["camera"][frame])
        visualizer.get_view_control().convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
    render = visualizer.get_render_option()
    render.point_size = args.point_size
    state = {
        "frame": int(np.clip(args.frame, 0, frame_count - 1)),
        "playing": bool(args.play),
        "last": time.monotonic(),
    }

    def redraw(frame: int) -> None:
        state["frame"] = int(np.clip(frame, 0, frame_count - 1))
        update_frame(state["frame"])
        for geometry in geometries:
            visualizer.update_geometry(geometry)
        print(f"\rframe {state['frame'] + 1}/{frame_count}", end="", flush=True)

    def previous(_):
        state["playing"] = False
        redraw(state["frame"] - 1)
        return False

    def following(_):
        state["playing"] = False
        redraw(state["frame"] + 1)
        return False

    def toggle(_):
        state["playing"] = not state["playing"]
        state["last"] = time.monotonic()
        return False

    visualizer.register_key_callback(263, previous)
    visualizer.register_key_callback(262, following)
    visualizer.register_key_callback(32, toggle)

    def animate(_):
        now = time.monotonic()
        if state["playing"] and now - state["last"] >= 1.0 / args.fps:
            next_frame = state["frame"] + 1
            if next_frame >= frame_count:
                if not args.loop:
                    state["playing"] = False
                    return False
                next_frame = 0
            redraw(next_frame)
            state["last"] = now
        return False

    visualizer.register_animation_callback(animate)
    print("Controls: Left/Right step, Space play/pause, Q/Esc close")
    visualizer.run()
    visualizer.destroy_window()
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--zarr_path", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--expected_dof",
        type=int,
        default=None,
        help="Optional URDF DoF validation; defaults to saved whole_body_joints width.",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max_depth", type=float, default=3.0)
    parser.add_argument("--point_sample_step", type=int, default=1)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=950)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--scene_camera", action="store_true", help="Start at the selected RGB-D camera viewpoint.")
    parser.add_argument("--hide_camera_housing", action="store_true", help="Hide head meshes enclosing the first-person camera.")
    parser.add_argument(
        "--point_cloud_source",
        choices=("auto", "raw", "inpaint"),
        default="auto",
        help="Use raw rgb, inpaint_rgb, or automatically prefer inpaint when present.",
    )
    parser.add_argument("--link_axis_size", type=float, default=0.04)
    parser.add_argument("--show_fingertip_axes", action="store_true")
    parser.add_argument("--fingertip_axis_size", type=float, default=0.025)
    parser.add_argument("--force_color_scale", type=float, default=0.2)
    parser.add_argument("--show_taxel_correspondence", action="store_true")
    parser.add_argument(
        "--hide_human_sensor_origins",
        dest="show_human_sensor_origins",
        action="store_false",
        help="Hide white human fingertip/sensor-origin target points.",
    )
    parser.set_defaults(show_human_sensor_origins=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.point_sample_step <= 0 or args.fps <= 0 or args.point_size <= 0:
        raise SystemExit("point_sample_step, fps, and point_size must be positive")
    run_visualizer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
