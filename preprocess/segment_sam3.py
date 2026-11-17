"""Text-prompt video masks using SAM3; output is compatible with ProPainter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "glove and its attachments"


def merge_frame_masks(outputs, height, width):
    """SAM3 returns N x H x W binary masks, including N=0 for absent objects."""
    masks = outputs["out_binary_masks"]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks, dtype=bool)
    if masks.size == 0:
        return np.zeros((height, width), dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(f"Unexpected SAM3 mask shape: {masks.shape}")
    return masks.any(axis=0)


def segment_sequence(predictor, zarr_path, mask_path, prompt=DEFAULT_PROMPT):
    import cv2
    import zarr

    source = zarr.open(str(zarr_path), mode="r")
    frames = source["data"]
    keys = sorted(frames.group_keys())
    if not keys:
        raise ValueError(f"No frames in {zarr_path}")
    height, width = frames[keys[0]]["rgb"].shape[:2]
    masks = np.zeros((len(keys), height, width), dtype=bool)
    seen = set()
    with tempfile.TemporaryDirectory(prefix="touchbridge-sam3-") as tmp:
        for i, key in enumerate(keys):
            rgb = np.asarray(frames[key]["rgb"][:], dtype=np.uint8)
            if rgb.shape != (height, width, 3):
                raise ValueError(f"Inconsistent RGB shape at {key}")
            if not cv2.imwrite(str(Path(tmp) / f"{i:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not export frame {i}")
        session = predictor.handle_request(dict(type="start_session", resource_path=tmp))
        sid = session["session_id"]
        try:
            response = predictor.handle_request(dict(
                type="add_prompt", session_id=sid, frame_index=0, text=prompt,
            ))
            masks[0] = merge_frame_masks(response["outputs"], height, width)
            seen.add(0)
            for response in predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=sid,
                start_frame_index=0, propagation_direction="forward",
            )):
                i = int(response["frame_index"])
                if not 0 <= i < len(keys):
                    raise ValueError(f"SAM3 returned invalid frame index {i}")
                masks[i] = merge_frame_masks(response["outputs"], height, width)
                seen.add(i)
        finally:
            predictor.handle_request(dict(type="close_session", session_id=sid))
    if len(seen) != len(keys):
        raise RuntimeError(f"SAM3 returned {len(seen)}/{len(keys)} frames; refusing incomplete masks")
    if not masks.any():
        raise RuntimeError("SAM3 found no matching objects; revise --prompt or use desktop SAM2")
    mask_path = Path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=mask_path.parent, suffix=".npz", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        np.savez_compressed(tmp_path, masks=masks, prompt=prompt, backend="sam3")
        os.replace(tmp_path, mask_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"Saved {mask_path}: {len(keys)} frames")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--checkpoint", type=Path, default=ROOT / ".third_party/sam3/checkpoints/sam3.pt")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not args.prompt.strip():
        p.error("--prompt cannot be empty")
    sequences = sorted(args.data_dir.glob("*.zarr"))
    if not sequences:
        p.error("--data-dir contains no Zarr sequences")
    pending = [(s, args.mask_dir / f"{s.stem}.npz") for s in sequences
               if args.overwrite or not (args.mask_dir / f"{s.stem}.npz").is_file()]
    if not pending:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint missing: {args.checkpoint}; see README.md")
    sys.path.insert(0, str(ROOT / ".third_party/sam3"))
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.checkpoint.resolve()), gpus_to_use=[args.gpu],
        apply_temporal_disambiguation=False,
    )
    try:
        for source, destination in pending:
            segment_sequence(predictor, source, destination, args.prompt)
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()


if __name__ == "__main__":
    main()
