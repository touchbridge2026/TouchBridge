"""Whole-body IK solver driven by camera and wrist-free targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from retarget.utils import relative_rotation_log

from .full_body_kinematics import (
    LEGWAIST_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    FullBodyKinematics,
    map_named_joints,
)


LEFT_ARM_EE_LINK = "left_flange"
RIGHT_ARM_EE_LINK = "right_flange"
LEFT_HAND_ROOT_LINK = "left_palm_link"
RIGHT_HAND_ROOT_LINK = "right_palm_link"
NECK_JOINTS = ["Neck-0", "Neck-1"]


@dataclass(frozen=True)
class WholeBodyWeights:
    camera_pos: float = 40.0
    camera_rot: float = 8.0
    arm_pos: float = 55.0
    arm_rot: float = 8.0
    joint_rest: float = 0.02
    joint_smooth: float = 0.20


def _pose_residual(current: np.ndarray, target: np.ndarray, pos_weight: float, rot_weight: float) -> np.ndarray:
    return np.concatenate(
        [
            np.sqrt(pos_weight) * (current[:3, 3] - target[:3, 3]),
            np.sqrt(rot_weight) * relative_rotation_log(target[:3, :3], current[:3, :3]),
        ]
    )


def _relative_link_transform(kin: FullBodyKinematics, q: np.ndarray, parent_link: str, child_link: str) -> np.ndarray:
    parent = kin.link_transform(q, parent_link)
    child = kin.link_transform(q, child_link)
    return np.linalg.inv(parent) @ child


def _solve_group_pose(
    kin: FullBodyKinematics,
    q_seed: np.ndarray,
    variable_indices: list[int],
    target_link: str,
    target_pose: np.ndarray,
    previous_values: np.ndarray | None,
    rest_values: np.ndarray | None,
    weights: WholeBodyWeights,
    pos_weight: float,
    rot_weight: float,
    max_iters: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    lower = kin.lower[variable_indices]
    upper = kin.upper[variable_indices]
    x0 = np.clip(q_seed[variable_indices], lower, upper)

    def unpack(x: np.ndarray) -> np.ndarray:
        q = q_seed.copy()
        q[variable_indices] = x
        return q

    def residual(x: np.ndarray) -> np.ndarray:
        q = unpack(x)
        current = kin.link_transform(q, target_link)
        items = [
            _pose_residual(current, target_pose, pos_weight, rot_weight),
            np.sqrt(weights.joint_rest)
            * (x - (np.zeros_like(x) if rest_values is None else rest_values)),
        ]
        if previous_values is not None:
            items.append(np.sqrt(weights.joint_smooth) * (x - previous_values))
        return np.concatenate(items)

    initial_residual = residual(x0)
    initial_residual_norm = float(np.linalg.norm(initial_residual))
    initial_cost = float(0.5 * np.dot(initial_residual, initial_residual))
    nfev = 0
    njev: int | None = None
    optimality: float | None = None
    status: int | None = None
    scipy_cost: float | None = None
    try:
        result = least_squares(residual, x0=x0, bounds=(lower, upper), max_nfev=max_iters)
        x_best = result.x
        success = bool(result.success)
        failure_reason = None if success else str(result.message)
        nfev = int(result.nfev)
        njev = None if result.njev is None else int(result.njev)
        optimality = float(result.optimality)
        status = int(result.status)
        scipy_cost = float(result.cost)
    except Exception as exc:  # pragma: no cover - defensive fallback
        x_best = x0
        success = False
        failure_reason = str(exc)

    final_residual = residual(x_best)
    final_residual_norm = float(np.linalg.norm(final_residual))
    final_cost = float(0.5 * np.dot(final_residual, final_residual))
    q_out = unpack(x_best)
    final_pose = kin.link_transform(q_out, target_link)
    pose_error = _pose_residual(final_pose, target_pose, 1.0, 1.0)
    clipped = np.logical_or(np.isclose(x_best, lower), np.isclose(x_best, upper))
    return q_out, {
        "success": success,
        "failure_reason": failure_reason,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_reduction": initial_cost - final_cost,
        "initial_residual_norm": initial_residual_norm,
        "final_residual_norm": final_residual_norm,
        "scipy_cost": scipy_cost,
        "scipy_nfev": nfev,
        "scipy_njev": njev,
        "scipy_optimality": optimality,
        "scipy_status": status,
        "pos_error_m": float(np.linalg.norm(final_pose[:3, 3] - target_pose[:3, 3])),
        "rot_error_rad": float(np.linalg.norm(pose_error[3:])),
        "limit_active_count": int(np.count_nonzero(clipped)),
    }


def solve_whole_body(
    kin: FullBodyKinematics,
    camera_trajectory_cam0: np.ndarray,
    torso_from_cam_high: np.ndarray,
    left_wrist_cam: np.ndarray,
    right_wrist_cam: np.ndarray,
    left_hand_joints: np.ndarray,
    right_hand_joints: np.ndarray,
    left_hand_joint_names: list[str],
    right_hand_joint_names: list[str],
    frame_indices: list[int],
    initial_legwaist: np.ndarray,
    fixed_head: np.ndarray,
    weights: WholeBodyWeights | None = None,
    max_iters: int = 120,
    cam0_from_base_fixed: np.ndarray | None = None,
) -> dict[str, Any]:
    weights = weights or WholeBodyWeights()
    frame_count = len(frame_indices)
    if camera_trajectory_cam0.shape != (frame_count, 4, 4):
        raise ValueError(f"camera_trajectory_cam0 must have shape ({frame_count}, 4, 4)")
    for name, array in {
        "left_wrist_cam": left_wrist_cam,
        "right_wrist_cam": right_wrist_cam,
    }.items():
        if array.shape != (frame_count, 4, 4):
            raise ValueError(f"{name} must have shape ({frame_count}, 4, 4), got {array.shape}")

    leg_indices = kin.group_indices(LEGWAIST_JOINTS)
    left_arm_indices = kin.group_indices(LEFT_ARM_JOINTS)
    right_arm_indices = kin.group_indices(RIGHT_ARM_JOINTS)
    neck_indices = kin.group_indices(NECK_JOINTS)

    initial_legwaist = np.asarray(initial_legwaist, dtype=np.float64).reshape(-1)
    fixed_head = np.asarray(fixed_head, dtype=np.float64).reshape(-1)
    if initial_legwaist.shape != (len(leg_indices),):
        raise ValueError(
            f"initial_legwaist must contain {len(leg_indices)} values, got {initial_legwaist.shape}"
        )
    if fixed_head.shape != (len(neck_indices),):
        raise ValueError(f"fixed_head must contain {len(neck_indices)} values, got {fixed_head.shape}")
    for values, indices, name in (
        (initial_legwaist, leg_indices, "initial_legwaist"),
        (fixed_head, neck_indices, "fixed_head"),
    ):
        lower = kin.lower[indices]
        upper = kin.upper[indices]
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"{name} is outside URDF limits: values={values}, lower={lower}, upper={upper}")

    q_reference = kin.zero_q()
    q_reference[leg_indices] = initial_legwaist
    q_reference[neck_indices] = fixed_head
    if not np.array_equal(q_reference[leg_indices], initial_legwaist):
        raise AssertionError("LegWaist initialization does not match the requested values")
    if not np.array_equal(q_reference[neck_indices], fixed_head):
        raise AssertionError("Neck initialization does not match the requested fixed values")

    base_from_torso0 = kin.link_transform(q_reference, "torso_flange")
    legacy_base_from_cam0 = base_from_torso0 @ torso_from_cam_high
    externally_anchored_base = cam0_from_base_fixed is not None
    if externally_anchored_base:
        cam0_from_base_fixed = np.asarray(cam0_from_base_fixed, dtype=np.float64)
        if cam0_from_base_fixed.shape != (4, 4) or not np.all(
            np.isfinite(cam0_from_base_fixed)
        ):
            raise ValueError("cam0_from_base_fixed must be one finite 4x4 transform")
        base_from_cam0 = np.linalg.inv(cam0_from_base_fixed)
        base_from_head_reference = kin.link_transform(q_reference, "head22")
        # The legacy calibration is no longer used to place Base.  It only defines
        # the fixed optical frame attached to head22 at the requested fixed Neck pose.
        head22_from_camera = (
            np.linalg.inv(base_from_head_reference)
            @ base_from_torso0
            @ torso_from_cam_high
        )
    else:
        base_from_cam0 = legacy_base_from_cam0
        cam0_from_base_fixed = np.linalg.inv(base_from_cam0)
        head22_from_camera = None
    left_ee_from_hand_root = _relative_link_transform(
        kin, q_reference, LEFT_ARM_EE_LINK, LEFT_HAND_ROOT_LINK
    )
    right_ee_from_hand_root = _relative_link_transform(
        kin, q_reference, RIGHT_ARM_EE_LINK, RIGHT_HAND_ROOT_LINK
    )

    joint_names = kin.joint_names
    whole_q = np.zeros((frame_count, kin.model.dof), dtype=np.float64)
    leg_q = np.zeros((frame_count, len(leg_indices)), dtype=np.float64)
    left_arm_q = np.zeros((frame_count, len(left_arm_indices)), dtype=np.float64)
    right_arm_q = np.zeros((frame_count, len(right_arm_indices)), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    missing_left_hand: set[str] = set()
    missing_right_hand: set[str] = set()

    previous_leg: np.ndarray | None = None
    previous_left_arm: np.ndarray | None = None
    previous_right_arm: np.ndarray | None = None

    for output_index, frame_index in enumerate(frame_indices):
        cam0_from_cam_t = camera_trajectory_cam0[output_index]
        base_from_cam_target = base_from_cam0 @ cam0_from_cam_t

        q_seed = q_reference.copy()
        if previous_leg is not None:
            q_seed[leg_indices] = previous_leg
        if output_index == 0 and not externally_anchored_base:
            q_after_leg = q_seed.copy()
            leg_diag = {
                "success": True,
                "failure_reason": None,
                "pos_error_m": 0.0,
                "rot_error_rad": 0.0,
                "limit_active_count": int(
                    np.count_nonzero(
                        np.logical_or(
                            np.isclose(initial_legwaist, kin.lower[leg_indices]),
                            np.isclose(initial_legwaist, kin.upper[leg_indices]),
                        )
                    )
                ),
                "fixed_to_requested_initial_state": True,
                "initial_cost": 0.0,
                "final_cost": 0.0,
                "cost_reduction": 0.0,
                "initial_residual_norm": 0.0,
                "final_residual_norm": 0.0,
                "scipy_cost": None,
                "scipy_nfev": 0,
                "scipy_njev": None,
                "scipy_optimality": None,
                "scipy_status": None,
            }
        else:
            if externally_anchored_base:
                leg_target_link = "head22"
                leg_target_pose = base_from_cam_target @ np.linalg.inv(
                    head22_from_camera
                )
            else:
                leg_target_link = "torso_flange"
                leg_target_pose = base_from_cam_target @ np.linalg.inv(
                    torso_from_cam_high
                )
            q_after_leg, leg_diag = _solve_group_pose(
                kin=kin,
                q_seed=q_seed,
                variable_indices=leg_indices,
                target_link=leg_target_link,
                target_pose=leg_target_pose,
                previous_values=previous_leg,
                rest_values=initial_legwaist,
                weights=weights,
                pos_weight=weights.camera_pos,
                rot_weight=weights.camera_rot,
                max_iters=max_iters,
            )
        previous_leg = q_after_leg[leg_indices].copy()

        base_from_left_palm = base_from_cam0 @ cam0_from_cam_t @ left_wrist_cam[output_index]
        base_from_right_palm = base_from_cam0 @ cam0_from_cam_t @ right_wrist_cam[output_index]
        base_from_left_ee = base_from_left_palm @ np.linalg.inv(left_ee_from_hand_root)
        base_from_right_ee = base_from_right_palm @ np.linalg.inv(right_ee_from_hand_root)

        q_left_seed = q_after_leg.copy()
        if previous_left_arm is not None:
            q_left_seed[left_arm_indices] = previous_left_arm
        q_after_left, left_diag = _solve_group_pose(
            kin=kin,
            q_seed=q_left_seed,
            variable_indices=left_arm_indices,
            target_link=LEFT_ARM_EE_LINK,
            target_pose=base_from_left_ee,
            previous_values=previous_left_arm,
            rest_values=None,
            weights=weights,
            pos_weight=weights.arm_pos,
            rot_weight=weights.arm_rot,
            max_iters=max_iters,
        )
        previous_left_arm = q_after_left[left_arm_indices].copy()

        q_right_seed = q_after_left.copy()
        if previous_right_arm is not None:
            q_right_seed[right_arm_indices] = previous_right_arm
        q_after_right, right_diag = _solve_group_pose(
            kin=kin,
            q_seed=q_right_seed,
            variable_indices=right_arm_indices,
            target_link=RIGHT_ARM_EE_LINK,
            target_pose=base_from_right_ee,
            previous_values=previous_right_arm,
            rest_values=None,
            weights=weights,
            pos_weight=weights.arm_pos,
            rot_weight=weights.arm_rot,
            max_iters=max_iters,
        )
        previous_right_arm = q_after_right[right_arm_indices].copy()

        q_frame = q_after_right.copy()
        q_frame[neck_indices] = fixed_head
        missing_left_hand.update(
            map_named_joints(q_frame, joint_names, left_hand_joints[output_index], left_hand_joint_names)
        )
        missing_right_hand.update(
            map_named_joints(q_frame, joint_names, right_hand_joints[output_index], right_hand_joint_names)
        )
        q_frame = np.clip(q_frame, kin.lower, kin.upper)

        whole_q[output_index] = q_frame
        leg_q[output_index] = q_frame[leg_indices]
        left_arm_q[output_index] = q_frame[left_arm_indices]
        right_arm_q[output_index] = q_frame[right_arm_indices]

        final_base_from_cam = (
            kin.link_transform(q_frame, "head22") @ head22_from_camera
            if externally_anchored_base
            else kin.link_transform(q_frame, "torso_flange") @ torso_from_cam_high
        )
        diagnostics.append(
            {
                "frame_index": int(frame_index),
                "output_index": int(output_index),
                "legwaist": leg_diag,
                "left_arm": left_diag,
                "right_arm": right_diag,
                "camera_pos_error_m": float(
                    np.linalg.norm(final_base_from_cam[:3, 3] - base_from_cam_target[:3, 3])
                ),
                "camera_rot_error_rad": float(
                    np.linalg.norm(
                        relative_rotation_log(base_from_cam_target[:3, :3], final_base_from_cam[:3, :3])
                    )
                ),
            }
        )
        print(
            "[whole-body loss] "
            f"frame={frame_index} output={output_index} "
            f"leg_cost={leg_diag['initial_cost']:.6e}->{leg_diag['final_cost']:.6e} "
            f"left_arm_cost={left_diag['initial_cost']:.6e}->{left_diag['final_cost']:.6e} "
            f"right_arm_cost={right_diag['initial_cost']:.6e}->{right_diag['final_cost']:.6e} "
            f"camera_pos_err={diagnostics[-1]['camera_pos_error_m']:.6e} "
            f"camera_rot_err={diagnostics[-1]['camera_rot_error_rad']:.6e} "
            f"left_pos_err={left_diag['pos_error_m']:.6e} "
            f"right_pos_err={right_diag['pos_error_m']:.6e}",
            flush=True,
        )

    if not np.allclose(whole_q[:, neck_indices], fixed_head[None, :], atol=0.0, rtol=0.0):
        raise AssertionError("Neck-0/1 changed even though they must remain fixed")
    if (
        not externally_anchored_base
        and not np.allclose(whole_q[0, leg_indices], initial_legwaist, atol=0.0, rtol=0.0)
    ):
        raise AssertionError("First-frame LegWaist does not match the requested initial values")

    return {
        "whole_q": whole_q,
        "joint_names": joint_names,
        "legwaist": leg_q,
        "left_arm": left_arm_q,
        "right_arm": right_arm_q,
        "base_pose_cam0": cam0_from_base_fixed,
        "base_from_cam0": base_from_cam0,
        "base_anchor_mode": (
            "fixed_scene_tag_base" if externally_anchored_base else "legacy_reference_pose"
        ),
        "head22_from_camera": head22_from_camera,
        "initial_legwaist": initial_legwaist,
        "fixed_head": fixed_head,
        "diagnostics": diagnostics,
        "missing_left_hand_joints": sorted(missing_left_hand),
        "missing_right_hand_joints": sorted(missing_right_hand),
        "groups": {
            "legwaist": LEGWAIST_JOINTS,
            "left_arm": LEFT_ARM_JOINTS,
            "right_arm": RIGHT_ARM_JOINTS,
            "left_hand": kin.left_hand_joint_names(),
            "right_hand": kin.right_hand_joint_names(),
            "head": NECK_JOINTS,
        },
        "arm_end_links": {
            "left": LEFT_ARM_EE_LINK,
            "right": RIGHT_ARM_EE_LINK,
        },
        "hand_root_links": {
            "left": LEFT_HAND_ROOT_LINK,
            "right": RIGHT_HAND_ROOT_LINK,
        },
    }
