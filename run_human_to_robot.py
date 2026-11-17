#!/usr/bin/env python3
"""Copy human demonstrations to a workspace, then synthesize robot-domain data."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "preprocess"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Source raw_data_N/ or data_N/ directories; read-only")
    p.add_argument("--work-root", type=Path, required=True, help="Separate working copies and masks")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--data-index", type=int, help="Process only this numeric session")
    p.add_argument("--mask-mode", choices=["text", "manual"], default="text")
    p.add_argument("--prompt", default="glove and its attachments")
    p.add_argument("--sam3-checkpoint", type=Path)
    p.add_argument("--sam2-checkpoint", type=Path)
    p.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam-python", default=sys.executable, help="Segmentation environment Python")
    p.add_argument("--inpaint-python", default=sys.executable, help="ProPainter environment Python")
    p.add_argument("--sam-no-interactive", action="store_true", help="Reuse desktop SAM2 annotation JSON")
    p.add_argument("--sam-overwrite", action="store_true")
    p.add_argument("--sam-n-samples", type=int, default=3)
    p.add_argument("--human-calibration", type=Path, default=ROOT / "configs/human_calibration.json")
    p.add_argument("--base-in-tag", type=Path, default=ROOT / "configs/base_in_scene_tag.json")
    p.add_argument("--scene-tag-id", type=int, default=10)
    p.add_argument("--scene-tag-size", type=float, default=0.080, help="Printed black tag edge, metres")
    p.add_argument("--retarget-config", type=Path, default=ROOT / "configs/retarget_free_wrist.yaml")
    p.add_argument("--target-fps", type=float, default=10.)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--rgb-o3d-backend", choices=["cpu", "egl", "cuda"], default="cpu")
    p.add_argument("--propainter-root", type=Path, default=ROOT / ".third_party/ProPainter")
    p.add_argument("--inpaint-weights-dir", type=Path)
    for stage in ["interp", "calib", "sam", "inpaint"]:
        p.add_argument(f"--skip-{stage}", action="store_true")
    stages = p.add_mutually_exclusive_group()
    stages.add_argument("--preprocess-only", action="store_true")
    stages.add_argument("--retarget-only", action="store_true")
    p.add_argument("--overwrite-retarget", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print planned commands without copying or running")
    return p


def run(command, dry_run=False, cwd=ROOT):
    command = [str(x) for x in command]
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=cwd)


def stage_copy(source, destination, dry_run=False):
    if destination.exists():
        return  # Reuse the workspace; never merge source data into processed Zarrs.
    print(f"Copy {source} -> {destination}", flush=True)
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, symlinks=False)


def complete_inpaint(path):
    import zarr
    root = zarr.open(str(path), mode="r")
    frames = root["data"]
    keys = list(frames.group_keys())
    return bool(keys) and all(
        "inpaint_rgb" in frames[k] and "inpaint_mask" in frames[k]
        and frames[k]["inpaint_rgb"].shape == frames[k]["rgb"].shape
        and frames[k]["inpaint_mask"].shape == frames[k]["rgb"].shape[:2]
        for k in keys
    )


def segmentation_command(args, data_dir, mask_dir, annotation_dir):
    if args.mask_mode == "text":
        cmd = [args.sam_python, PRE / "segment_sam3.py", "--data-dir", data_dir,
               "--mask-dir", mask_dir, "--prompt", args.prompt]
        if args.sam3_checkpoint:
            cmd += ["--checkpoint", args.sam3_checkpoint]
    else:
        cmd = [args.sam_python, PRE / "glove_mask_sam2_video.py", "--data-dir", data_dir,
               "--mask-dir", mask_dir, "--annotation-dir", annotation_dir,
               "--n-samples", args.sam_n_samples, "--model-cfg", args.sam2_model_cfg]
        if args.sam2_checkpoint:
            cmd += ["--checkpoint", args.sam2_checkpoint]
        if args.sam_no_interactive:
            cmd += ["--no-interactive"]
    if args.sam_overwrite:
        cmd += ["--overwrite"]
    return cmd


def retarget_command(args, data_dir):
    cmd = [sys.executable, ROOT / "run_sensor_tip_retarget.py", "--data-dir", data_dir,
           "--output-root", args.output_root, "--base-in-tag", args.base_in_tag,
           "--retarget-config", args.retarget_config, "--target-fps", args.target_fps,
           "--rgb-o3d-backend", args.rgb_o3d_backend, "--rgb-o3d-mp4-fps", args.target_fps,
           "--overwrite" if args.overwrite_retarget else "--resume"]
    if args.max_frames is not None:
        cmd += ["--max-frames", args.max_frames]
    return cmd


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Resolve user paths before subprocesses change their working directory.
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if not args.data_root.is_dir():
        parser.error("--data-root does not exist")
    # Neither output tree may overlap the source tree or each other.
    roots = [args.data_root, args.work_root, args.output_root]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i+1:]):
        parser.error("data-root, work-root and output-root must be separate, non-nested directories")
    indices = sorted({int(m.group(1)) for d in args.data_root.iterdir()
                      if d.is_dir() and (m := re.fullmatch(r"(?:raw_)?data_(\d+)", d.name))})
    if args.data_index is not None:
        indices = [i for i in indices if i == args.data_index]
    if not indices:
        parser.error("No matching raw_data_N/ or data_N/ sessions")
    for index in indices:
        data = args.work_root / f"data_{index}"
        source_data = args.data_root / data.name
        if source_data.is_dir():
            stage_copy(source_data, data, args.dry_run)
        if not args.retarget_only:
            source_raw = args.data_root / f"raw_data_{index}"
            if not args.skip_interp and source_raw.is_dir():
                raw = args.work_root / source_raw.name
                stage_copy(source_raw, raw, args.dry_run)
                for original in sorted(source_raw.glob("*.zarr")):
                    sequence = raw / original.name
                    tag_cmd = [sys.executable, PRE / "offline_detect_table_apriltag_raw.py",
                               "--raw-zarr", sequence, "--tag-id", args.scene_tag_id,
                               "--tag-size", args.scene_tag_size]
                    run(tag_cmd, args.dry_run)
                    if not list(data.glob(f"{original.stem}_aligned*.zarr")):
                        run([sys.executable, PRE / "offline_detect_cube_pose_raw.py", "--raw-zarr", sequence], args.dry_run)
                        run([sys.executable, PRE / "interpolate_process_double.py", "--input", sequence,
                             "--output", data / f"{original.stem}_aligned.zarr"], args.dry_run)
                    run(tag_cmd + ["--data-dir", data, "--sync-only"], args.dry_run)
            if not args.dry_run and not list(data.glob("*.zarr")):
                raise FileNotFoundError(f"No aligned Zarr in {data}")
            if not args.skip_calib:
                run([sys.executable, PRE / "write_final_t_finger_to_cam.py", "--input", data,
                     "--pattern", "*.zarr", "--calibration", args.human_calibration], args.dry_run)
            masks = args.work_root / f"mask_{index}"
            if not args.skip_sam:
                run(segmentation_command(args, data, masks, args.work_root / f"sam2_annotations_{index}"), args.dry_run)
            if not args.skip_inpaint:
                sources = sorted(data.glob("*.zarr"))
                if args.dry_run and not sources:
                    print(f"Inpaint each aligned Zarr in {data} using masks from {masks}")
                for sequence in sources:
                    if not args.dry_run and not args.sam_overwrite and complete_inpaint(sequence):
                        continue
                    mask = masks / f"{sequence.stem}.npz"
                    if not args.dry_run and not mask.is_file():
                        raise FileNotFoundError(f"Missing mask: {mask}")
                    cmd = [args.inpaint_python, args.propainter_root / "inference_propainter.py",
                           "--zarr-rgb", sequence, "--npz-mask", mask, "--output", data,
                           "--mode", "video_inpainting", "--rewrite-zarr"]
                    if args.inpaint_weights_dir:
                        cmd += ["--weights-dir", args.inpaint_weights_dir]
                    run(cmd, args.dry_run, cwd=args.propainter_root)
        if not args.preprocess_only:
            run(retarget_command(args, data), args.dry_run)


if __name__ == "__main__":
    main()
