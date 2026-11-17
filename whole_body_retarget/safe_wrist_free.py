"""Crash-tolerant wrist-free retarget runner."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import complete_wrist_free_output, ensure_dir, load_wrist_free_output, selected_indices, write_json


@dataclass(frozen=True)
class WristFreeRunConfig:
    zarr_path: str
    config: str
    layout: str
    urdf: str
    expected_dof: int
    start_frame: int
    max_frames: int | None
    frame_stride: int
    frame_indices: tuple[int, ...] | None
    chunk_size: int
    timeout_sec: float
    retry_single_frames: bool
    allow_fallback: bool
    max_workers: int = 1


def _selected_indices(total_frames: int, cfg: WristFreeRunConfig) -> list[int]:
    if cfg.frame_indices is not None:
        return [int(value) for value in cfg.frame_indices]
    return selected_indices(total_frames, cfg.start_frame, cfg.max_frames, cfg.frame_stride)


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def _write_seed(path: Path, previous: dict[str, np.ndarray] | None) -> Path | None:
    if previous is None:
        return None
    count = int(previous["joints_left"].shape[0])
    if count < 1:
        return None
    payload = {
        "previous_joints_left": previous["joints_left"][-1],
        "previous_joints_right": previous["joints_right"][-1],
        "previous_wrist_left": previous["wrist_left"][-1],
        "previous_wrist_right": previous["wrist_right"][-1],
    }
    if count >= 2:
        payload.update(
            {
                "previous_previous_joints_left": previous["joints_left"][-2],
                "previous_previous_joints_right": previous["joints_right"][-2],
                "previous_previous_wrist_left": previous["wrist_left"][-2],
                "previous_previous_wrist_right": previous["wrist_right"][-2],
            }
        )
    np.savez(path, **payload)
    return path


def _run_worker(
    cfg: WristFreeRunConfig,
    output_dir: Path,
    frame_indices: list[int],
    previous: dict[str, np.ndarray] | None,
) -> tuple[bool, str | None]:
    frame_indices_path = output_dir / "input_frame_indices.npy"
    cmd = [
        sys.executable,
        "-m",
        "whole_body_retarget.wrist_free_worker",
        "--zarr_path",
        cfg.zarr_path,
        "--output_dir",
        str(output_dir),
        "--config",
        cfg.config,
        "--layout",
        cfg.layout,
        "--urdf",
        cfg.urdf,
        "--expected_dof",
        str(cfg.expected_dof),
        "--frame_indices_path",
        str(frame_indices_path),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(frame_indices_path, np.asarray(frame_indices, dtype=np.int64))
    seed_path = _write_seed(output_dir / "input_seed.npz", previous)
    if seed_path is not None:
        cmd.extend(["--seed_path", str(seed_path)])
    stdout_path = output_dir / "worker.stdout.log"
    stderr_path = output_dir / "worker.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=_worker_env(),
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        if proc.stdout is None:  # pragma: no cover - subprocess contract
            raise RuntimeError("wrist-free worker stdout pipe was not created")

        def tee_stdout() -> None:
            try:
                for line in proc.stdout:
                    stdout_file.write(line)
                    stdout_file.flush()
                    if line.startswith("[wrist-free loss]"):
                        print(line.rstrip("\r\n"), flush=True)
            except (OSError, ValueError):
                # The parent closes the pipe only as a last resort after the
                # worker process has exited or its process group was killed.
                pass

        stdout_thread = threading.Thread(
            target=tee_stdout,
            name=f"wrist-free-stdout-{frame_indices[0]}",
            daemon=True,
        )
        stdout_thread.start()
        timed_out = False
        try:
            proc.wait(timeout=cfg.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        finally:
            stdout_thread.join(timeout=5.0)
            if stdout_thread.is_alive():
                proc.stdout.close()
                stdout_thread.join(timeout=1.0)

        if timed_out:
            stderr_tail = _tail_text(stderr_path)
            return False, (
                f"timeout after {cfg.timeout_sec:.1f}s; killed process group {proc.pid}\n"
                + stderr_tail
            )
    if proc.returncode != 0:
        stderr_tail = _tail_text(stderr_path)
        return False, f"exit_code={proc.returncode}\n{stderr_tail}"
    if not complete_wrist_free_output(output_dir):
        return False, "worker exited successfully but output is incomplete"
    return True, None


def _tail_text(path: Path, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])


def _load_optimization_records(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "optimization_info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing wrist-free optimization records: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a list of optimization records in {path}")
    return payload


def _load_contact_records(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "contact_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing wrist-free contact records: {path}")
    with np.load(path, allow_pickle=True) as payload:
        if "records" not in payload:
            raise KeyError(f"Missing 'records' array in {path}")
        records = payload["records"].tolist()
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Expected a list of contact records in {path}")
    return records


def _load_result_with_records(
    output_dir: Path,
    frame_indices: list[int],
) -> dict[str, Any]:
    result = load_wrist_free_output(output_dir)
    optimization_records = _load_optimization_records(output_dir)
    contact_records = _load_contact_records(output_dir)
    expected = len(frame_indices)
    for label, records in (
        ("optimization", optimization_records),
        ("contact", contact_records),
    ):
        if len(records) != expected:
            raise ValueError(
                f"{label} record count mismatch in {output_dir}: "
                f"expected {expected}, got {len(records)}"
            )
        actual_indices = [int(record.get("frame_index", -1)) for record in records]
        if actual_indices != frame_indices:
            raise ValueError(
                f"{label} frame indices mismatch in {output_dir}: "
                f"expected {frame_indices}, got {actual_indices}"
            )
    result["_optimization_records"] = optimization_records
    result["_contact_records"] = contact_records
    return result


def _load_chunk_loss_summary(output_dir: Path) -> dict[str, Any]:
    info_path = output_dir / "optimization_info.json"
    if not info_path.exists():
        return {}
    try:
        entries = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"could not parse {info_path}"}
    if not entries:
        return {"frame_count": 0}

    def values(key: str) -> list[float]:
        result: list[float] = []
        for entry in entries:
            value = entry.get(key)
            if value is None:
                continue
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                continue
        return result

    final_costs = values("final_cost")
    initial_costs = values("initial_cost")
    final_norms = values("final_residual_norm")
    tip_rmses = values("tip_global_rmse")
    cost_reductions = values("cost_reduction")
    nfev_values = values("scipy_nfev")

    def aggregate(items: list[float]) -> dict[str, float] | None:
        if not items:
            return None
        array = np.asarray(items, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    summary = {
        "frame_count": len(entries),
        "first_frame": entries[0].get("frame_index"),
        "last_frame": entries[-1].get("frame_index"),
        "initial_cost": aggregate(initial_costs),
        "final_cost": aggregate(final_costs),
        "cost_reduction": aggregate(cost_reductions),
        "final_residual_norm": aggregate(final_norms),
        "tip_global_rmse": aggregate(tip_rmses),
        "mean_nfev": float(np.mean(nfev_values)) if nfev_values else None,
        "success_count": int(sum(bool(entry.get("success")) for entry in entries)),
        "failure_count": int(sum(not bool(entry.get("success")) for entry in entries)),
        "retry_from_neutral_count": int(sum(bool(entry.get("retry_from_neutral")) for entry in entries)),
        "log_path": str(output_dir / "worker.stdout.log"),
    }
    return summary


def _print_chunk_loss_summary(chunk_id: str, summary: dict[str, Any]) -> None:
    if not summary:
        print(f"[wrist-free chunk {chunk_id}] no loss summary available", flush=True)
        return
    if "error" in summary:
        print(f"[wrist-free chunk {chunk_id}] {summary['error']}", flush=True)
        return
    final_cost = summary.get("final_cost") or {}
    initial_cost = summary.get("initial_cost") or {}
    final_norm = summary.get("final_residual_norm") or {}
    tip_rmse = summary.get("tip_global_rmse") or {}
    print(
        "[wrist-free chunk "
        f"{chunk_id}] frames={summary.get('first_frame')}..{summary.get('last_frame')} "
        f"count={summary.get('frame_count')} "
        f"cost_mean={initial_cost.get('mean', float('nan')):.6e}->{final_cost.get('mean', float('nan')):.6e} "
        f"cost_max_final={final_cost.get('max', float('nan')):.6e} "
        f"residual_norm_mean_final={final_norm.get('mean', float('nan')):.6e} "
        f"tip_global_rmse_mean={tip_rmse.get('mean', float('nan')):.6e} "
        f"nfev_mean={summary.get('mean_nfev', float('nan')):.1f} "
        f"success={summary.get('success_count')} failure={summary.get('failure_count')} "
        f"retry_neutral={summary.get('retry_from_neutral_count')} "
        f"log={summary.get('log_path')}",
        flush=True,
    )


def _fallback_chunk(
    output_dir: Path,
    frame_indices: list[int],
    left_dof: int,
    right_dof: int,
    joint_names_left: list[str],
    joint_names_right: list[str],
    previous: dict[str, np.ndarray] | None,
    reason: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(frame_indices)
    left_joints = np.zeros((count, left_dof), dtype=np.float64)
    right_joints = np.zeros((count, right_dof), dtype=np.float64)
    left_wrist = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    right_wrist = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    if previous is not None:
        left_joints[:] = previous["joints_left"][-1]
        right_joints[:] = previous["joints_right"][-1]
        left_wrist[:] = previous["wrist_left"][-1]
        right_wrist[:] = previous["wrist_right"][-1]
    optimization_records = [
        {
            "frame_index": int(frame_index),
            "frame_key": None,
            "success": False,
            "failure_reason": reason,
            "fallback": True,
            "initial_residual_norm": None,
            "final_residual_norm": None,
            "initial_cost": None,
            "final_cost": None,
            "cost_reduction": None,
            "scipy_cost": None,
            "scipy_nfev": 0,
        }
        for frame_index in frame_indices
    ]
    contact_records = [
        {
            "frame_index": int(frame_index),
            "frame_key": None,
            "fallback": True,
            "fallback_reason": reason,
            "left": {},
            "right": {},
        }
        for frame_index in frame_indices
    ]
    np.save(output_dir / "joints_left.npy", left_joints)
    np.save(output_dir / "joints_right.npy", right_joints)
    np.save(output_dir / "wrist_left.npy", left_wrist)
    np.save(output_dir / "wrist_right.npy", right_wrist)
    np.savez(
        output_dir / "contact_features.npz",
        records=np.asarray(contact_records, dtype=object),
    )
    write_json(output_dir / "joint_names_left.json", joint_names_left)
    write_json(output_dir / "joint_names_right.json", joint_names_right)
    write_json(output_dir / "optimization_info.json", optimization_records)
    write_json(
        output_dir / "metadata.json",
        {
            "frames": frame_indices,
            "fallback": True,
            "fallback_reason": reason,
            "wrist_output_semantics": "Fallback copied previous frame if available, otherwise identity wrists and zero joints.",
        },
    )
    return _load_result_with_records(output_dir, frame_indices)


def _copy_complete(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "joints_left.npy",
        "joints_right.npy",
        "wrist_left.npy",
        "wrist_right.npy",
        "contact_features.npz",
        "joint_names_left.json",
        "joint_names_right.json",
        "optimization_info.json",
        "safe_runner_events.json",
        "metadata.json",
    ):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def _solve_chunk_recursive(
    cfg: WristFreeRunConfig,
    chunk_root: Path,
    frame_indices: list[int],
    chunk_id: str,
    previous: dict[str, np.ndarray] | None,
    left_dof: int,
    right_dof: int,
    joint_names_left: list[str],
    joint_names_right: list[str],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_dir = chunk_root / f"chunk_{chunk_id}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    success, reason = _run_worker(cfg, output_dir, frame_indices, previous)
    if success:
        try:
            result = _load_result_with_records(output_dir, frame_indices)
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            success = False
            reason = f"worker diagnostics are incomplete or invalid: {exc}"
        else:
            loss_summary = _load_chunk_loss_summary(output_dir)
            _print_chunk_loss_summary(chunk_id, loss_summary)
            events.append(
                {
                    "chunk": chunk_id,
                    "frames": frame_indices,
                    "status": "ok",
                    "loss_summary": loss_summary,
                }
            )
            return [result]

    print(
        f"[wrist-free chunk {chunk_id}] failed frames={frame_indices[0]}..{frame_indices[-1]} "
        f"reason={reason}",
        flush=True,
    )
    events.append({"chunk": chunk_id, "frames": frame_indices, "status": "failed", "reason": reason})
    if len(frame_indices) > 1:
        mid = len(frame_indices) // 2
        left_results = _solve_chunk_recursive(
            cfg,
            chunk_root,
            frame_indices[:mid],
            f"{chunk_id}a",
            previous,
            left_dof,
            right_dof,
            joint_names_left,
            joint_names_right,
            events,
        )
        next_previous = left_results[-1] if left_results else previous
        right_results = _solve_chunk_recursive(
            cfg,
            chunk_root,
            frame_indices[mid:],
            f"{chunk_id}b",
            next_previous,
            left_dof,
            right_dof,
            joint_names_left,
            joint_names_right,
            events,
        )
        return left_results + right_results

    if not cfg.allow_fallback:
        raise RuntimeError(f"Wrist-free failed for frame {frame_indices[0]} and fallback is disabled:\n{reason}")
    fallback_dir = chunk_root / f"fallback_{chunk_id}"
    if fallback_dir.exists():
        shutil.rmtree(fallback_dir)
    print(
        f"[wrist-free chunk {chunk_id}] fallback frame={frame_indices[0]} reason={reason}",
        flush=True,
    )
    events.append({"chunk": chunk_id, "frames": frame_indices, "status": "fallback", "reason": reason})
    return [
        _fallback_chunk(
            fallback_dir,
            frame_indices,
            left_dof,
            right_dof,
            joint_names_left,
            joint_names_right,
            previous,
            reason or "unknown failure",
        )
    ]


def _combine_results(
    cfg: WristFreeRunConfig,
    results: list[dict[str, Any]],
    output_dir: Path,
    frame_indices: list[int],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    left_joints = np.concatenate([item["joints_left"] for item in results], axis=0)
    right_joints = np.concatenate([item["joints_right"] for item in results], axis=0)
    left_wrist = np.concatenate([item["wrist_left"] for item in results], axis=0)
    right_wrist = np.concatenate([item["wrist_right"] for item in results], axis=0)
    optimization_records = [
        record
        for item in results
        for record in item.get("_optimization_records", [])
    ]
    contact_records = [
        record
        for item in results
        for record in item.get("_contact_records", [])
    ]
    joint_names_left = results[0]["joint_names_left"]
    joint_names_right = results[0]["joint_names_right"]
    if left_joints.shape[0] != len(frame_indices):
        raise ValueError(f"Combined wrist-free length mismatch: {left_joints.shape[0]} vs {len(frame_indices)}")
    for label, records in (
        ("optimization", optimization_records),
        ("contact", contact_records),
    ):
        if len(records) != len(frame_indices):
            raise ValueError(
                f"Combined {label} record count mismatch: "
                f"{len(records)} vs {len(frame_indices)}"
            )
        actual_indices = [int(record.get("frame_index", -1)) for record in records]
        if actual_indices != frame_indices:
            raise ValueError(
                f"Combined {label} frame indices mismatch: "
                f"expected {frame_indices}, got {actual_indices}"
            )
    np.save(output_dir / "joints_left.npy", left_joints)
    np.save(output_dir / "joints_right.npy", right_joints)
    np.save(output_dir / "wrist_left.npy", left_wrist)
    np.save(output_dir / "wrist_right.npy", right_wrist)
    np.savez(
        output_dir / "contact_features.npz",
        records=np.asarray(contact_records, dtype=object),
    )
    write_json(output_dir / "joint_names_left.json", joint_names_left)
    write_json(output_dir / "joint_names_right.json", joint_names_right)
    write_json(output_dir / "optimization_info.json", optimization_records)
    write_json(output_dir / "safe_runner_events.json", events)
    write_json(
        output_dir / "metadata.json",
        {
            "frames": frame_indices,
            "mapping_mode": "safe_subprocess_chunked_wrist_free",
            "initialization_strategy": (
                "first selected frame from neutral, every subsequent frame strictly "
                "from the previous optimized state; previous states are passed across chunks"
            ),
            "neutral_retry": False,
            "contact_hold_config_source": str(Path(cfg.config).resolve()),
            "human_wrist_pose_used": False,
            "wrist_output_semantics": "wrist_left.npy / wrist_right.npy store robot base poses for compatibility.",
            "safe_runner_events_path": "safe_runner_events.json",
        },
    )
    return load_wrist_free_output(output_dir)


def run_safe_wrist_free(
    cfg: WristFreeRunConfig,
    output_dir: str | Path,
    total_frames: int,
    left_dof: int,
    right_dof: int,
    joint_names_left: list[str],
    joint_names_right: list[str],
) -> dict[str, Any]:
    if cfg.chunk_size <= 0:
        raise ValueError("wrist_free_chunk_size must be positive")
    if cfg.timeout_sec <= 0:
        raise ValueError("wrist_free_chunk_timeout_sec must be positive")
    if cfg.max_workers <= 0:
        raise ValueError("wrist_free_workers must be positive")
    if cfg.max_workers != 1:
        raise ValueError(
            "wrist_free_workers must be 1: the temporal velocity/acceleration objective "
            "requires each chunk to receive the preceding two optimized states"
        )
    final_dir = ensure_dir(output_dir)
    frame_indices = _selected_indices(total_frames, cfg)
    chunk_root = final_dir / "_safe_chunks"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    chunks = [
        (
            frame_indices[start : start + cfg.chunk_size],
            str(start // cfg.chunk_size).zfill(4),
        )
        for start in range(0, len(frame_indices), cfg.chunk_size)
    ]
    if cfg.max_workers > 1 and not cfg.allow_fallback:
        print(
            f"Running {len(chunks)} independent wrist-free chunks with "
            f"{cfg.max_workers} parallel workers"
        )

        def solve_independent(
            chunk_frames: list[int], chunk_id: str
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            local_events: list[dict[str, Any]] = []
            chunk_results = _solve_chunk_recursive(
                cfg,
                chunk_root,
                chunk_frames,
                chunk_id,
                None,
                left_dof,
                right_dof,
                joint_names_left,
                joint_names_right,
                local_events,
            )
            return chunk_results, local_events

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = [
                executor.submit(solve_independent, chunk_frames, chunk_id)
                for chunk_frames, chunk_id in chunks
            ]
            for future in futures:
                chunk_results, chunk_events = future.result()
                results.extend(chunk_results)
                events.extend(chunk_events)
    else:
        previous: dict[str, np.ndarray] | None = None
        for chunk_frames, chunk_id in chunks:
            chunk_results = _solve_chunk_recursive(
                cfg,
                chunk_root,
                chunk_frames,
                chunk_id,
                previous,
                left_dof,
                right_dof,
                joint_names_left,
                joint_names_right,
                events,
            )
            results.extend(chunk_results)
            previous = results[-1]
    events.sort(key=lambda item: min(item.get("frames", [10**18])))
    combined = _combine_results(cfg, results, final_dir, frame_indices, events)
    return combined
