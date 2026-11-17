#!/usr/bin/env python3
"""Retarget the bundled light-bulb demonstration and inspect it in Open3D."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(__file__).resolve().parent / "data"


def unpack_demo(work_dir: Path) -> Path:
    manifest = json.loads((DATA / "manifest.json").read_text())
    archive = DATA / manifest["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Demo archive checksum mismatch; download it again from the repository.")
    destination = work_dir / "input"
    marker = destination / ".archive-sha256"
    source = destination / "screw_bulb.zarr"
    if marker.exists() and marker.read_text() == manifest["sha256"] and source.is_dir():
        return source
    if destination.exists():
        raise FileExistsError(f"Incomplete or different demo data in {destination}; choose a fresh --work-dir.")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Invalid path in demo archive")
        z.extractall(destination)
    marker.write_text(manifest["sha256"])
    return source


def export_scene(args, path: Path) -> None:
    """Save a colored point-cloud snapshot without opening a window."""
    import numpy as np
    from whole_body_retarget.open3d_visualizer import build_scene

    o3d, geometries, _, _ = build_scene(args)
    o3d.utility.random.seed(0)
    cloud = o3d.geometry.PointCloud()
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.PointCloud):
            cloud += geometry
        elif isinstance(geometry, o3d.geometry.TriangleMesh) and len(geometry.triangles):
            count = max(50, min(15000, int(geometry.get_surface_area() * 20000)))
            cloud += geometry.sample_points_uniformly(number_of_points=count)
    if not np.isfinite(np.asarray(cloud.points)).all() or not len(cloud.points):
        raise ValueError("Scene contains invalid or empty geometry")
    cloud.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(cloud.colors), 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f"Could not write {path}")
    print(f"Saved {path}: {len(cloud.points)} colored points")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, default=ROOT / "work/bulb_demo")
    p.add_argument("--no-view", action="store_true", help="Run retargeting without a desktop window")
    p.add_argument("--view-only", action="store_true", help="Reuse previously computed results")
    p.add_argument("--max-frames", type=int, help="Limit computation for a quick check; use a fresh work directory")
    p.add_argument("--frame", type=int, default=0, help="Initial viewer/export frame")
    p.add_argument("--raw-cloud", action="store_true", help="Include the human RGB-D points instead of masking them")
    p.add_argument("--export-frame", type=Path, help="Export selected scene and robot geometry to a colored PLY")
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")
    if args.frame < 0:
        p.error("--frame must be nonnegative")
    work = args.work_dir.expanduser().resolve()
    source = unpack_demo(work)
    result = work / "retarget"
    if not args.view_only:
        command = [sys.executable, "-m", "whole_body_retarget.cli", "run",
                   "--zarr_path", str(source), "--output_dir", str(result),
                   "--base_in_tag", str(DATA / "base_in_scene_tag.json"),
                   "--skip_rgb_o3d", "--no_wrist_free_fallback"]
        if args.max_frames is not None:
            command += ["--max_frames", str(args.max_frames)]
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Retarget results: {result}")
    if args.no_view and args.export_frame is None:
        return
    from whole_body_retarget.open3d_visualizer import build_parser, run_visualizer
    view = build_parser().parse_args([
        "--output_dir", str(result), "--zarr_path", str(source),
        "--fps", "5", "--play", "--loop", "--frame", str(args.frame),
        "--point_cloud_source", "raw" if args.raw_cloud else "inpaint",
        "--scene_camera", "--hide_camera_housing", "--width", "1280", "--height", "960",
    ])
    if args.export_frame:
        export_scene(view, args.export_frame.expanduser().resolve())
    if not args.no_view:
        run_visualizer(view)


if __name__ == "__main__":
    main()
