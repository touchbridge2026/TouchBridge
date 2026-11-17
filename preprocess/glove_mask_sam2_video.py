#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop SAM2 video segmentation for gloves and attached hardware.

Draw polygons or positive/negative points on selected video frames. Polygons
are rasterized into mask prompts. Tracking starts from frame 0 by default;
corrections on later keyframes also trigger backward propagation.

Output NPZ fields:
- masks: (n_frames, H, W) bool, union of all removal regions
- object_masks: (n_objects, n_frames, H, W) bool
- object_ids, object_names: identifiers for annotated regions
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_ROOT = os.path.join(BASE_DIR, ".third_party", "sam2")

DATA_DIR = os.path.join(SCRIPT_DIR, "data_0")
MASK_DIR = os.path.join(SCRIPT_DIR, "mask_sam2")
ANNOTATION_DIR = os.path.join(SCRIPT_DIR, "sam2_annotations")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_CHECKPOINT = os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_large.pt")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, SAM2_ROOT)

DEFAULT_PROMPT_FRAME = 0
OBJECT_SPECS = [
    {"name": "left", "label": "左手", "obj_id": 1},
    {"name": "right", "label": "右手", "obj_id": 2},
    {"name": "env", "label": "环境", "obj_id": 3},
]
OBJECT_ID_TO_NAME = {spec["obj_id"]: spec["name"] for spec in OBJECT_SPECS}
OBJECT_NAME_TO_ID = {spec["name"]: spec["obj_id"] for spec in OBJECT_SPECS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="输入 data_N/ 目录")
    p.add_argument("--mask-dir", default=MASK_DIR, help="输出 mask 目录")
    p.add_argument("--annotation-dir", default=ANNOTATION_DIR, help="标注 JSON 保存目录")
    p.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG, help="SAM2 config")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="SAM2 checkpoint")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="cuda / cpu")
    p.add_argument("--vos-optimized", action="store_true", help="启用 SAM2 vos_optimized predictor")
    p.add_argument("--offload-video-to-cpu", action="store_true", help="init_state 时把视频帧放 CPU")
    p.add_argument("--offload-state-to-cpu", action="store_true", help="init_state 时把部分 state 放 CPU")
    p.add_argument("--n-samples", type=int, default=3,
                   help="每个对象标注关键帧数；默认 3（第 0 帧 + 其余均匀）")
    p.add_argument("--reverse-propagation", action="store_true",
                   help="启用反向传播；默认关闭以提升速度")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有 mask 文件")
    p.add_argument("--no-interactive", action="store_true", help="Reuse saved desktop annotations")
    return p.parse_args()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def export_zarr_to_jpegs(zarr_path, output_dir):
    import zarr as _z

    root = _z.open(zarr_path, mode="r")
    data = root["data"]
    n_frames = int(root.attrs.get("n_frames", len(data)))
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx in tqdm(range(n_frames), desc="  Exporting frames", leave=False):
        rgb = data[f"frame_{frame_idx:05d}/rgb"][:].astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(output_dir, f"{frame_idx:05d}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    return n_frames


def polygon_to_mask_bool(polygon_points, img_h, img_w):
    if polygon_points is None or len(polygon_points) < 3:
        return None
    poly = np.asarray(polygon_points, dtype=np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, img_w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, img_h - 1)
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(poly).astype(np.int32)], 1)
    return canvas > 0


def mask_to_box_xyxy(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def sample_points_from_mask(mask_bool, n_pos=24, n_neg=12, rng=None):
    rng = rng or np.random.default_rng(0)
    if mask_bool is None or not mask_bool.any():
        return [], []

    mask_u8 = mask_bool.astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.erode(mask_u8, k3, iterations=2) > 0
    boundary = (mask_bool & (~eroded)) if eroded.any() else mask_bool
    outer = cv2.dilate(mask_u8, k9, iterations=2) > 0
    outer &= ~mask_bool

    def _sample(region, n):
        ys, xs = np.where(region)
        if len(xs) == 0 or n <= 0:
            return []
        n = min(n, len(xs))
        pick = rng.choice(len(xs), size=n, replace=False)
        return [[float(xs[i]), float(ys[i])] for i in pick]

    pts_pos = _sample(eroded if eroded.any() else mask_bool, n_pos)
    pts_pos += _sample(boundary, max(1, n_pos // 2))
    pts_neg = _sample(outer, n_neg)
    pts = pts_pos + pts_neg
    lbs = [1] * len(pts_pos) + [0] * len(pts_neg)
    return pts, lbs


def merge_points_and_labels(base_pts, base_lbs, extra_pts, extra_lbs):
    pts = list(base_pts) if base_pts else []
    lbs = list(base_lbs) if base_lbs else []
    if extra_pts:
        pts.extend(extra_pts)
        lbs.extend(extra_lbs)
    return pts, lbs


def filter_annotations_for_object(annotations, obj_id):
    filtered = {}
    for frame_idx, frame_ann in annotations.items():
        if obj_id in frame_ann:
            filtered[frame_idx] = {obj_id: frame_ann[obj_id]}
    return filtered


def annotation_json_path(annotation_dir, zarr_path):
    stem = os.path.splitext(os.path.basename(zarr_path.rstrip("/")))[0]
    return os.path.join(annotation_dir, f"{stem}.json")


def save_annotation_info(annotation_dir, zarr_path, ann_info):
    os.makedirs(annotation_dir, exist_ok=True)
    payload = {
        "zarr_name": os.path.basename(zarr_path.rstrip("/")),
        "n_frames": int(ann_info["n_frames"]),
        "img_h": int(ann_info["img_h"]),
        "img_w": int(ann_info["img_w"]),
        "annotations": ann_info["annotations"],
        "object_prompt_frames": {k: int(v) for k, v in ann_info["object_prompt_frames"].items()},
        "object_sample_indices": {
            k: [int(x) for x in v] for k, v in ann_info["object_sample_indices"].items()
        },
    }
    annotation_path = annotation_json_path(annotation_dir, zarr_path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".sam2_annotation_", suffix=".tmp", dir=annotation_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, annotation_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_annotation_info(annotation_dir, zarr_path):
    path = annotation_json_path(annotation_dir, zarr_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        annotations = {}
        for frame_idx_str, frame_ann in payload.get("annotations", {}).items():
            frame_idx = int(frame_idx_str)
            annotations[frame_idx] = {}
            for obj_id_str, ann in frame_ann.items():
                obj_id = int(obj_id_str)
                annotations[frame_idx][obj_id] = {
                    "points": ann.get("points", []),
                    "labels": ann.get("labels", []),
                }
                if "polygon" in ann:
                    annotations[frame_idx][obj_id]["polygon"] = ann["polygon"]
        return {
            "annotations": annotations,
            "n_frames": int(payload["n_frames"]),
            "img_h": int(payload["img_h"]),
            "img_w": int(payload["img_w"]),
            "object_prompt_frames": {
                k: int(v) for k, v in payload.get("object_prompt_frames", {}).items()
            },
            "object_sample_indices": {
                k: [int(x) for x in vals]
                for k, vals in payload.get("object_sample_indices", {}).items()
            },
        }
    except Exception as e:
        print(f"[annotation] 读取失败，忽略旧标注 {path}: {e}")
        return None


def prompt_existing_annotation_action(zarr_name, ann_path, loaded_ann):
    desc = ", ".join(
        f"{spec['label']}={loaded_ann['object_prompt_frames'].get(spec['name'], DEFAULT_PROMPT_FRAME)}"
        for spec in OBJECT_SPECS
    )
    existing_frames = sorted(loaded_ann.get("annotations", {}).keys())
    print(f"  [annotation] 已找到历史标注: {ann_path}")
    print(f"  [annotation] 已标注 {len(existing_frames)} 帧 {existing_frames} | prompt 帧: {desc}")
    print("  请选择：")
    print("    [Enter]/r = 直接复用")
    print("    e = 在已有标注基础上继续修改")
    print("    a = 在已有标注基础上选额外帧补标注")
    print("    n = 丢弃旧标注，重新标")
    while True:
        choice = input(f"  [{zarr_name}] 你的选择 [r/e/a/n]: ").strip().lower()
        if choice in ("", "r", "reuse"):
            return "reuse"
        if choice in ("e", "edit"):
            return "edit"
        if choice in ("a", "append", "add"):
            return "append"
        if choice in ("n", "new", "reset"):
            return "new"


def build_sample_indices_with_prompt_frame(n_frames, n_samples, prompt_frame, max_frame=None):
    """构建采样帧索引列表。

    Args:
        n_frames: zarr 总帧数
        n_samples: 希望采样的帧数
        prompt_frame: 必须包含的起始 prompt 帧（一般为 0）
        max_frame: 采样范围上限（含）。None 表示使用 n_frames-1。
                   传入 n_frames//2 可将采样限制在前半段，
                   配合 SAM2 仅前向传播时效果最好。
    """
    if n_frames <= 0:
        return [DEFAULT_PROMPT_FRAME]
    prompt_frame = int(np.clip(prompt_frame, 0, n_frames - 1))
    n_samples = max(1, int(n_samples))
    end = int(np.clip(max_frame if max_frame is not None else n_frames - 1, 0, n_frames - 1))
    sample_indices = np.linspace(0, end, n_samples, dtype=int).tolist()
    sample_indices.append(prompt_frame)
    return sorted(set(int(fi) for fi in sample_indices))


def collect_pre_annotations_for_zarr(
    zarr_data,
    n_frames,
    img_w,
    img_h,
    zarr_name,
    n_samples=5,
    initial_annotations=None,
    object_prompt_frames_override=None,
    object_sample_indices_override=None,
    fresh_annotation=False,
):
    """
    fresh_annotation=True 时，若无 override，将采样范围限定在前 n_frames//2 帧，
    以匹配 SAM2 仅前向传播的特性（帧 0, n_frames//4, n_frames//2）。
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    del object_prompt_frames_override  # SAM2 版固定第 0 帧为传播起点
    object_prompt_frames = {
        spec["name"]: DEFAULT_PROMPT_FRAME for spec in OBJECT_SPECS
    }
    object_sample_indices = {}
    for spec in OBJECT_SPECS:
        name = spec["name"]
        sample_indices = []
        if object_sample_indices_override:
            sample_indices = sorted(set(int(x) for x in object_sample_indices_override.get(name, [])))
        if not sample_indices:
            max_frame = n_frames // 2 if fresh_annotation else None
            sample_indices = build_sample_indices_with_prompt_frame(
                n_frames, n_samples, DEFAULT_PROMPT_FRAME, max_frame=max_frame
            )
        if DEFAULT_PROMPT_FRAME not in sample_indices:
            sample_indices = sorted(set(sample_indices + [DEFAULT_PROMPT_FRAME]))
        object_sample_indices[name] = sample_indices

    sample_indices = sorted({
        fi for vals in object_sample_indices.values() for fi in vals
    })
    ordered_indices = []
    for spec in OBJECT_SPECS:
        pf = object_prompt_frames[spec["name"]]
        if pf not in ordered_indices:
            ordered_indices.append(pf)
    ordered_indices.extend(fi for fi in sample_indices if fi not in ordered_indices)

    annotations = {}
    total_frames = len(sample_indices)

    print(f"\n  [{zarr_name}] 预标注：共 {total_frames} 帧")
    for spec in OBJECT_SPECS:
        name = spec["name"]
        pf = object_prompt_frames[name]
        rest = [fi for fi in object_sample_indices[name] if fi != pf]
        print(f"  {spec['label']} prompt 帧固定为 {pf}；其余关键帧: {rest}")
    print("  建议：第 0 帧优先画 polygon。SAM2 会把 polygon 作为 mask prompt。")
    print("  操作：1=左手 2=右手 3=环境")
    print("       左键=正点/顶点  右键=负点/撤销顶点")
    print("       m=多边形模式  f=完成多边形  u=撤销顶点  x=删除当前对象多边形")
    print("       回车/n=下一帧  q=结束  c=清空本帧")

    style_map = {
        OBJECT_NAME_TO_ID["left"]: {"pos": ("c+", 18, 3), "neg": ("bx", 18, 3)},
        OBJECT_NAME_TO_ID["right"]: {"pos": ("g+", 18, 3), "neg": ("rx", 18, 3)},
        OBJECT_NAME_TO_ID["env"]: {"pos": ("y+", 18, 3), "neg": ("mx", 18, 3)},
    }
    object_colors = {
        OBJECT_NAME_TO_ID["left"]: (0.0, 1.0, 1.0),
        OBJECT_NAME_TO_ID["right"]: (0.0, 1.0, 0.0),
        OBJECT_NAME_TO_ID["env"]: (1.0, 0.6, 0.0),
    }
    object_color_names = {
        OBJECT_NAME_TO_ID["left"]: "cyan",
        OBJECT_NAME_TO_ID["right"]: "lime",
        OBJECT_NAME_TO_ID["env"]: "orange",
    }

    for order_idx, frame_idx in enumerate(ordered_indices, start=1):
        frame_rgb = zarr_data[f"frame_{frame_idx:05d}/rgb"][:].astype(np.uint8)
        existing_frame_ann = (initial_annotations or {}).get(frame_idx, {})
        obj_points = {
            spec["obj_id"]: list(existing_frame_ann.get(spec["obj_id"], {}).get("points", []))
            for spec in OBJECT_SPECS
        }
        obj_labels = {
            spec["obj_id"]: list(existing_frame_ann.get(spec["obj_id"], {}).get("labels", []))
            for spec in OBJECT_SPECS
        }
        obj_polygons = {
            spec["obj_id"]: list(existing_frame_ann.get(spec["obj_id"], {}).get("polygon", []))
            if existing_frame_ann.get(spec["obj_id"], {}).get("polygon") else None
            for spec in OBJECT_SPECS
        }
        active_obj = [OBJECT_SPECS[0]["obj_id"]]
        draw_mode = ["points"]
        temp_polygon = []
        done_flag = [False]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.tight_layout()

        def _redraw():
            ax.cla()
            ax.imshow(frame_rgb)
            for spec in OBJECT_SPECS:
                oid = spec["obj_id"]
                poly = obj_polygons[oid]
                if poly and len(poly) >= 3:
                    poly_np = np.asarray(poly, dtype=np.float32)
                    closed = np.vstack([poly_np, poly_np[:1]])
                    color = object_colors[oid]
                    ax.fill(poly_np[:, 0], poly_np[:, 1], color=color, alpha=0.18)
                    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
                for (px, py), lb in zip(obj_points[oid], obj_labels[oid]):
                    kind = "pos" if lb == 1 else "neg"
                    marker, ms, mw = style_map[oid][kind]
                    ax.plot(px, py, marker, markersize=ms, markeredgewidth=mw)
            if draw_mode[0] == "polygon" and temp_polygon:
                temp_np = np.asarray(temp_polygon, dtype=np.float32)
                color = object_color_names[active_obj[0]]
                ax.plot(temp_np[:, 0], temp_np[:, 1], color=color, linewidth=2, linestyle="--")
                ax.scatter(temp_np[:, 0], temp_np[:, 1], c=color, s=20)

            prompt_tags = []
            for spec in OBJECT_SPECS:
                if frame_idx == object_prompt_frames[spec["name"]]:
                    prompt_tags.append(f"{spec['label']} prompt 帧")
            prompt_tag_text = " / ".join(prompt_tags) if prompt_tags else "普通关键帧"
            ax.set_title(
                f"{zarr_name} | 关键帧 {order_idx}/{total_frames} | 帧 {frame_idx}/{n_frames-1}\n"
                f"{prompt_tag_text}\n"
                f"当前对象={OBJECT_ID_TO_NAME[active_obj[0]]} | 模式={draw_mode[0]} | "
                f"1=左手 2=右手 3=环境 | m=polygon f=完成 回车=下一帧",
                fontsize=9,
            )
            ax.axis("off")
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            x, y = event.xdata, event.ydata
            if draw_mode[0] == "polygon":
                if event.button == 1:
                    temp_polygon.append([x, y])
                elif event.button == 3 and temp_polygon:
                    temp_polygon.pop()
            else:
                if event.button == 1:
                    obj_points[active_obj[0]].append([x, y])
                    obj_labels[active_obj[0]].append(1)
                elif event.button == 3:
                    obj_points[active_obj[0]].append([x, y])
                    obj_labels[active_obj[0]].append(0)
            _redraw()

        def on_key(event):
            if event.key in ("n", "enter"):
                plt.close(fig)
            elif event.key == "q":
                done_flag[0] = True
                plt.close(fig)
            elif event.key == "c":
                for oid in obj_points:
                    obj_points[oid].clear()
                    obj_labels[oid].clear()
                    obj_polygons[oid] = None
                temp_polygon.clear()
                draw_mode[0] = "points"
                _redraw()
            elif event.key == "1":
                active_obj[0] = OBJECT_NAME_TO_ID["left"]
                _redraw()
            elif event.key == "2":
                active_obj[0] = OBJECT_NAME_TO_ID["right"]
                _redraw()
            elif event.key == "3":
                active_obj[0] = OBJECT_NAME_TO_ID["env"]
                _redraw()
            elif event.key == "m":
                draw_mode[0] = "polygon"
                temp_polygon.clear()
                _redraw()
            elif event.key == "f":
                if len(temp_polygon) >= 3:
                    obj_polygons[active_obj[0]] = list(temp_polygon)
                    temp_polygon.clear()
                    draw_mode[0] = "points"
                _redraw()
            elif event.key == "u":
                if draw_mode[0] == "polygon" and temp_polygon:
                    temp_polygon.pop()
                _redraw()
            elif event.key == "x":
                obj_polygons[active_obj[0]] = None
                if draw_mode[0] == "polygon":
                    temp_polygon.clear()
                _redraw()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        _redraw()
        plt.show(block=True)

        frame_ann = {}
        for spec in OBJECT_SPECS:
            oid = spec["obj_id"]
            has_points = bool(obj_points[oid])
            has_poly = obj_polygons[oid] is not None and len(obj_polygons[oid]) >= 3
            if has_points or has_poly:
                frame_ann[oid] = {
                    "points": list(obj_points[oid]),
                    "labels": list(obj_labels[oid]),
                }
                if has_poly:
                    frame_ann[oid]["polygon"] = list(obj_polygons[oid])
        if frame_ann:
            annotations[frame_idx] = frame_ann
            stats = []
            for spec in OBJECT_SPECS:
                oid = spec["obj_id"]
                if oid not in frame_ann:
                    continue
                lbs = frame_ann[oid]["labels"]
                n_pos = sum(lbs)
                poly_info = ""
                if "polygon" in frame_ann[oid]:
                    poly_info = f" + {len(frame_ann[oid]['polygon'])} 边界顶点"
                stats.append(f"{spec['label']} {n_pos} 正点 + {len(lbs)-n_pos} 负点{poly_info}")
            print(f"    帧 {frame_idx}: " + " | ".join(stats))
        else:
            print(f"    帧 {frame_idx}: 未标注，跳过")

        if done_flag[0]:
            print("  用户按 q，结束当前条数据标注")
            break

    return annotations, object_prompt_frames, object_sample_indices


def collect_all_pre_annotations(zarr_files, annotation_dir, n_samples):
    import zarr as _zarr

    os.makedirs(annotation_dir, exist_ok=True)
    all_ann = {}
    total = len(zarr_files)
    for zi, zarr_path in enumerate(zarr_files, start=1):
        zarr_name = os.path.splitext(os.path.basename(zarr_path.rstrip("/")))[0]
        print(f"\n{'─'*60}")
        print(f"  预标注进度 [{zi}/{total}]: {zarr_name}")
        print(f"{'─'*60}")

        loaded_ann = load_annotation_info(annotation_dir, zarr_path)
        action = None
        if loaded_ann is not None:
            ann_path = annotation_json_path(annotation_dir, zarr_path)
            action = prompt_existing_annotation_action(zarr_name, ann_path, loaded_ann)
            if action == "reuse":
                all_ann[zarr_path] = loaded_ann
                print("  [annotation] 复用已有标注")
                continue

        root = _zarr.open(zarr_path, mode="r")
        data = root["data"]
        n_frames = int(root.attrs.get("n_frames", len(data)))
        sample_rgb = data["frame_00000/rgb"][:]
        img_h, img_w = sample_rgb.shape[:2]

        # ── 补标注模式（append）：在已有标注基础上选额外帧 ──────────────────
        if action == "append" and loaded_ann is not None:
            existing_frames = sorted(loaded_ann.get("annotations", {}).keys())
            print(f"  [append] 现有标注帧: {existing_frames}")
            while True:
                raw = input(
                    f"  [append] 输入要补标注的帧索引（0~{n_frames-1}，逗号分隔）: "
                ).strip()
                try:
                    extra_frames = sorted(set(
                        int(x.strip()) for x in raw.split(",") if x.strip()
                    ))
                    if not extra_frames:
                        print("  请至少输入一个帧索引")
                        continue
                    invalid = [f for f in extra_frames if not (0 <= f < n_frames)]
                    if invalid:
                        print(f"  以下帧索引超出范围 [0, {n_frames-1}]：{invalid}")
                        continue
                    break
                except ValueError:
                    print("  输入格式错误，请用逗号分隔的整数")

            extra_frame_override = {spec["name"]: extra_frames for spec in OBJECT_SPECS}
            extra_annotations, _, _ = collect_pre_annotations_for_zarr(
                data,
                n_frames,
                img_w,
                img_h,
                zarr_name,
                n_samples=len(extra_frames),
                initial_annotations=loaded_ann.get("annotations"),
                object_prompt_frames_override=None,
                object_sample_indices_override=extra_frame_override,
                fresh_annotation=False,
            )

            merged_annotations = dict(loaded_ann.get("annotations", {}))
            merged_annotations.update(extra_annotations)
            merged_sample_indices = {
                spec["name"]: sorted(set(
                    list(loaded_ann.get("object_sample_indices", {}).get(spec["name"], []))
                    + extra_frames
                ))
                for spec in OBJECT_SPECS
            }
            ann_info = {
                **loaded_ann,
                "annotations": merged_annotations,
                "object_sample_indices": merged_sample_indices,
            }
            save_annotation_info(annotation_dir, zarr_path, ann_info)
            all_ann[zarr_path] = ann_info
            print(f"  [append] 补标注完成，合并后共 {len(merged_annotations)} 帧标注")
            continue

        # ── 常规标注（new / edit / 首次） ────────────────────────────────────
        fresh_annotation = (loaded_ann is None or action == "new")
        annotations, object_prompt_frames, object_sample_indices = collect_pre_annotations_for_zarr(
            data,
            n_frames,
            img_w,
            img_h,
            zarr_name,
            n_samples=n_samples,
            initial_annotations=(loaded_ann["annotations"] if loaded_ann is not None and action == "edit" else None),
            object_prompt_frames_override=(loaded_ann.get("object_prompt_frames") if loaded_ann is not None and action == "edit" else None),
            object_sample_indices_override=(loaded_ann.get("object_sample_indices") if loaded_ann is not None and action == "edit" else None),
            fresh_annotation=fresh_annotation,
        )

        ann_info = {
            "annotations": annotations,
            "n_frames": n_frames,
            "img_h": img_h,
            "img_w": img_w,
            "object_prompt_frames": dict(object_prompt_frames),
            "object_sample_indices": {k: list(v) for k, v in object_sample_indices.items()},
        }
        save_annotation_info(annotation_dir, zarr_path, ann_info)
        all_ann[zarr_path] = ann_info
    return all_ann


def update_object_masks_from_logits(object_masks_per_frame, frame_idx, object_ids, mask_logits):
    if isinstance(mask_logits, torch.Tensor):
        masks_np = mask_logits.detach().cpu().numpy()
    else:
        masks_np = np.asarray(mask_logits)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    for i, obj_id in enumerate(object_ids):
        mask = masks_np[i] > 0.0
        if not mask.any():
            continue
        frame_masks = object_masks_per_frame.setdefault(int(frame_idx), {})
        obj_id = int(obj_id)
        if obj_id in frame_masks:
            frame_masks[obj_id] |= mask
        else:
            frame_masks[obj_id] = mask


def apply_annotation_to_state(predictor, inference_state, frame_idx, obj_id, ann, img_h, img_w):
    poly = ann.get("polygon")
    if poly:
        poly_mask = polygon_to_mask_bool(poly, img_h, img_w)
        if poly_mask is not None and poly_mask.any():
            predictor.add_new_mask(
                inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask=poly_mask,
            )
            return "mask"

    pts = list(ann.get("points", []))
    lbs = list(ann.get("labels", []))
    box = None
    if pts:
        points = np.asarray(pts, dtype=np.float32)
        labels = np.asarray(lbs, dtype=np.int32)
    else:
        points = None
        labels = None
    if points is None and box is None:
        return None

    predictor.add_new_points_or_box(
        inference_state,
        frame_idx=int(frame_idx),
        obj_id=int(obj_id),
        points=points,
        labels=labels,
        clear_old_points=True,
        normalize_coords=True,  # Desktop clicks are image-pixel coordinates.
        box=box,
    )
    return "points"


def propagate_bidirectionally(predictor, inference_state, seed_frames, enable_reverse=False):
    object_masks_per_frame = {}
    start_forward = DEFAULT_PROMPT_FRAME
    start_reverse = max(seed_frames)

    print(f"  Forward propagate from frame {start_forward} ...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=int(start_forward), reverse=False
    ):
        update_object_masks_from_logits(object_masks_per_frame, frame_idx, obj_ids, mask_logits)

    if enable_reverse and start_reverse > 0:
        print(f"  Reverse propagate from frame {start_reverse} ...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=int(start_reverse), reverse=True
        ):
            update_object_masks_from_logits(object_masks_per_frame, frame_idx, obj_ids, mask_logits)

    return object_masks_per_frame


def merge_object_masks(object_masks_per_frame, n_frames, img_h, img_w):
    merged = {}
    for fi in range(n_frames):
        frame_masks = object_masks_per_frame.get(fi)
        if not frame_masks:
            continue
        combined = np.zeros((img_h, img_w), dtype=bool)
        for mask in frame_masks.values():
            combined |= mask
        merged[fi] = combined
    return merged


def stack_object_masks(object_masks_per_frame, n_frames, img_h, img_w, object_ids):
    stacked = np.zeros((len(object_ids), n_frames, img_h, img_w), dtype=bool)
    for oi, obj_id in enumerate(object_ids):
        for fi, frame_masks in object_masks_per_frame.items():
            if obj_id in frame_masks:
                stacked[oi, fi] = frame_masks[obj_id]
    return stacked


def process_one_zarr(zarr_path, predictor, ann_info, mask_dir, args):
    import zarr

    if ann_info is None:
        raise RuntimeError("SAM2 版本必须先做视觉标注；当前 zarr 没有可用 ann_info。")

    zarr_name = os.path.splitext(os.path.basename(zarr_path.rstrip("/")))[0]
    mask_path = os.path.join(mask_dir, f"{zarr_name}.npz")
    if os.path.exists(mask_path) and not args.overwrite:
        print(f"  [跳过] 已存在: {mask_path}")
        return mask_path

    root = zarr.open(zarr_path, mode="r")
    data = root["data"]
    n_frames = int(root.attrs.get("n_frames", len(data)))
    sample = data["frame_00000/rgb"][:]
    img_h, img_w = sample.shape[:2]

    jpeg_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    print(f"  Exporting frames → {jpeg_dir}")
    export_zarr_to_jpegs(zarr_path, jpeg_dir)

    try:
        inference_state = predictor.init_state(
            video_path=jpeg_dir,
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )
        predictor.reset_state(inference_state)

        annotations = ann_info.get("annotations", {})
        object_prompt_frames = {
            spec["name"]: DEFAULT_PROMPT_FRAME for spec in OBJECT_SPECS
        }
        object_sample_indices = ann_info.get("object_sample_indices", {})
        seed_frames = []

        for spec in OBJECT_SPECS:
            obj_id = spec["obj_id"]
            obj_ann = filter_annotations_for_object(annotations, obj_id)
            prompt_frame = int(object_prompt_frames.get(spec["name"], DEFAULT_PROMPT_FRAME))
            ordered_frames = [prompt_frame] + [
                fi for fi in object_sample_indices.get(spec["name"], sorted(obj_ann.keys()))
                if fi != prompt_frame
            ]
            ordered_frames = [fi for fi in ordered_frames if fi in obj_ann]
            if not ordered_frames:
                print(f"  [{spec['label']}] 无任何标注，跳过该对象")
                continue

            print(f"  [{spec['label']}] prompt 帧: {prompt_frame} | 标注帧: {ordered_frames}")
            for frame_idx in ordered_frames:
                ann = obj_ann[frame_idx][obj_id]
                prompt_mode = apply_annotation_to_state(
                    predictor, inference_state, frame_idx, obj_id, ann, img_h, img_w
                )
                if prompt_mode is not None:
                    print(f"    帧 {frame_idx}: 以 {prompt_mode} prompt 注入 {spec['label']}")
                    seed_frames.append(int(frame_idx))

        if not seed_frames:
            raise RuntimeError("没有可用于 SAM2 的视觉提示。请至少为一个对象提供 polygon 或点标注。")

        object_masks_per_frame = propagate_bidirectionally(
            predictor,
            inference_state,
            sorted(set(seed_frames)),
            enable_reverse=bool(getattr(args, "reverse_propagation", False)),
        )
        merged_masks = merge_object_masks(object_masks_per_frame, n_frames, img_h, img_w)

        all_masks = np.zeros((n_frames, img_h, img_w), dtype=bool)
        for fi, mask in merged_masks.items():
            all_masks[fi] = mask

        object_ids = sorted({
            int(obj_id)
            for frame_masks in object_masks_per_frame.values()
            for obj_id in frame_masks.keys()
        })
        object_masks = stack_object_masks(
            object_masks_per_frame, n_frames, img_h, img_w, object_ids
        ) if object_ids else np.zeros((0, n_frames, img_h, img_w), dtype=bool)

        # Re-create the destination immediately before writing so a resumed batch
        # cannot lose an otherwise completed inference to a missing mask_N folder.
        # Publish atomically to prevent an interrupted compression from leaving a
        # truncated NPZ that a later resume would mistake for a completed result.
        os.makedirs(mask_dir, exist_ok=True)
        fd, temporary_mask_path = tempfile.mkstemp(
            prefix=f".{zarr_name}.", suffix=".npz", dir=mask_dir
        )
        os.close(fd)
        try:
            np.savez_compressed(
                temporary_mask_path,
                masks=all_masks,
                object_masks=object_masks,
                object_ids=np.asarray(object_ids, dtype=np.int32),
                object_names=np.asarray(
                    [OBJECT_ID_TO_NAME.get(obj_id, f"obj_{obj_id}") for obj_id in object_ids],
                    dtype="<U16",
                ),
            )
            os.replace(temporary_mask_path, mask_path)
        finally:
            if os.path.exists(temporary_mask_path):
                os.unlink(temporary_mask_path)
        print(f"  Saved: {mask_path}")
        return mask_path
    finally:
        shutil.rmtree(jpeg_dir, ignore_errors=True)
        free_gpu()


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"SAM2 checkpoint 不存在: {args.checkpoint}\n"
            f"请先在 `{SAM2_ROOT}` 下下载 checkpoint。"
        )

    zarr_files = sorted(
        p for p in glob.glob(os.path.join(args.data_dir, "*.zarr"))
        if os.path.isdir(p)
    )
    if not zarr_files:
        print(f"未找到任何 .zarr: {args.data_dir}")
        return

    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.annotation_dir, exist_ok=True)

    print(f"Found {len(zarr_files)} zarr file(s) in {args.data_dir}/:")
    for zf in zarr_files:
        print(f"  - {os.path.basename(zf)}")
    print(f"  model_cfg   = {args.model_cfg}")
    print(f"  checkpoint  = {args.checkpoint}")
    print(f"  device      = {args.device}")
    print("  visual prompt 策略：第 0 帧 polygon 优先 -> mask prompt；其余关键帧点/边界补充")
    print(f"  reverse propagation = {'ON' if args.reverse_propagation else 'OFF'}")

    print(f"\n{'='*70}")
    print("  Phase 1: 预标注")
    print(f"{'='*70}")
    if args.no_interactive:
        all_annotations = {p: load_annotation_info(args.annotation_dir, p) for p in zarr_files}
        missing = [p for p, info in all_annotations.items() if info is None]
        if missing:
            raise FileNotFoundError(f"Missing desktop annotations: {missing}")
    else:
        all_annotations = collect_all_pre_annotations(
            zarr_files, annotation_dir=args.annotation_dir, n_samples=args.n_samples,
        )

    print(f"\n{'='*70}")
    print("  Phase 2: 构建 SAM2 predictor")
    print(f"{'='*70}")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        args.model_cfg,
        args.checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )

    try:
        with torch.inference_mode(), amp_ctx:
            for idx, zarr_path in enumerate(zarr_files, start=1):
                print(f"\n{'='*70}")
                print(f"  [{idx}/{len(zarr_files)}] {os.path.basename(zarr_path)}")
                print(f"{'='*70}")
                ann_info = all_annotations.get(zarr_path)
                process_one_zarr(zarr_path, predictor, ann_info, args.mask_dir, args)
    finally:
        del predictor
        free_gpu()


if __name__ == "__main__":
    main()
