"""Camera-anchored action construction (paper Sec. III-C, Eq. 7)."""
import numpy as np


def camera_anchored_chunk(end_effector_in_camera, tag_in_camera, hand_joints, anchor=0):
    """Return chunk-relative arm SE(3) poses and absolute hand joint targets.

    Inputs have shapes (T,4,4), (T,4,4), and (T,D). T_camera_tag
    maps stationary-tag coordinates into that frame's camera. Arm targets
    are expressed relative to the anchor end-effector frame after camera
    motion compensation. Call independently for each arm and each chunk.
    """
    poses = np.asarray(end_effector_in_camera, dtype=np.float64)
    tags = np.asarray(tag_in_camera, dtype=np.float64)
    joints = np.asarray(hand_joints, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or tags.shape != poses.shape:
        raise ValueError("Pose arrays must both have shape (T,4,4)")
    if joints.ndim != 2 or len(joints) != len(poses) or not 0 <= anchor < len(poses):
        raise ValueError("Invalid hand-joint shape or chunk anchor")
    if not all(np.all(np.isfinite(a)) for a in (poses, tags, joints)):
        raise ValueError("Action inputs must be finite; clean missing tag poses first")
    for array in (poses, tags):
        if not np.allclose(array[:, 3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("Invalid homogeneous transforms")
        rotations = array[:, :3, :3]
        if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
            raise ValueError("Rotation blocks must be in SO(3)")
    aligned = tags[anchor] @ np.linalg.inv(tags) @ poses
    relative = np.linalg.inv(aligned[anchor]) @ aligned
    return relative, joints.copy()
