#!/usr/bin/env python3
"""Subprocess worker for wrist-free retarget chunks.

This file intentionally sets BLAS/OpenMP thread limits before importing numpy/scipy.
It is launched by ``safe_wrist_free.py`` so native crashes or hangs do not kill the
whole-body retarget parent process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")


from .io_utils import ensure_project_imports  # noqa: E402

ensure_project_imports()

import numpy as np  # noqa: E402

from retarget.config import load_retarget_config  # noqa: E402
from retarget.data.zarr_loader import ZarrHumanDataset  # noqa: E402
from retarget.optimization.free_wrist_solver import FreeWristRetargetSolver  # noqa: E402
from retarget.robot.urdf_model import parse_urdf_model  # noqa: E402
from retarget.tactile.features import TaxelLayout  # noqa: E402

from .hand_models import dual_hand_submodels_from_full  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--urdf", required=True, help="The only URDF used by this worker.")
    parser.add_argument("--expected_dof", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--frame_indices_path",
        default=None,
        help="Optional NPY array of exact source frame indices for this chunk.",
    )
    parser.add_argument(
        "--seed_path",
        default=None,
        help="Optional NPZ containing the last one or two states of the preceding chunk.",
    )
    return parser


def _load_seed(path: str | None, solver: FreeWristRetargetSolver):
    if path is None:
        return None, None
    with np.load(path, allow_pickle=False) as seed:
        previous = solver._state_dict(
            seed["previous_wrist_left"],
            seed["previous_joints_left"],
            seed["previous_wrist_right"],
            seed["previous_joints_right"],
        )
        previous_previous = None
        if "previous_previous_wrist_left" in seed.files:
            previous_previous = solver._state_dict(
                seed["previous_previous_wrist_left"],
                seed["previous_previous_joints_left"],
                seed["previous_previous_wrist_right"],
                seed["previous_previous_joints_right"],
            )
    return previous, previous_previous


def main() -> int:
    args = build_parser().parse_args()
    dataset = ZarrHumanDataset(args.zarr_path)
    config = load_retarget_config(args.config)
    layout = TaxelLayout.from_json(args.layout)
    full_model = parse_urdf_model(args.urdf, expected_dof=args.expected_dof)
    left_model, right_model = dual_hand_submodels_from_full(
        full_model,
        expected_dof_per_hand=config.expected_dof_per_hand,
    )
    solver = FreeWristRetargetSolver(
        dataset=dataset,
        layout=layout,
        left_model=left_model,
        right_model=right_model,
        config=config,
    )
    previous, previous_previous = _load_seed(args.seed_path, solver)
    frame_indices = None
    if args.frame_indices_path is not None:
        frame_indices = [
            int(value)
            for value in np.load(args.frame_indices_path, allow_pickle=False).reshape(-1)
        ]
    solver.solve_sequence(
        output_dir=Path(args.output_dir),
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        frame_indices=frame_indices,
        initial_previous=previous,
        initial_previous_previous=previous_previous,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
