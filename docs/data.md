# Data interfaces

Zarr 2 is required. Translations are metres, joint angles radians, timestamps
seconds. RGB is uint8 in RGB order; aligned depth is metric float data.

## Aligned input

`preprocess/interpolate_process_double.py` aligns raw sensor streams.
New acquisition systems can produce this aligned interface directly:

```text
intrinsic                                (3,3)
data/frame_00000/
  rgb                                    (H,W,3)
  depth                                  (H,W)
  realsense_timestamp                    scalar
  mvs_left_rgb, mvs_right_rgb             optional wrist RGB
  T_left/right_cube_to_cam                (4,4)
  T_left/right_finger_to_wrist             (5,4,4)
  T_left/right_finger_to_cam               written by calibration
  T_left/right_sensor_to_cam               written by calibration
  inpaint_rgb, inpaint_mask                written by ProPainter
data/frame_00001/...
left_tactile/, right_tactile/
  timestamps
  timestamps_realsense_aligned             optional
  distribution_force_json                 finger -> taxel XYZ mappings
  wrench                                  (Nt,5,6)
T_apriltag_10_to_cam                      (T,4,4)
apriltag_10_pose_valid                    (T,)
apriltag_10_pose_filled                   (T,)
apriltag_10_pose_available                (T,)
apriltag_10_reprojection_error_px         (T,)
apriltag_10_corners_px                    (T,4,2)
apriltag_10_timestamps                    (T,)
```

Finger order is thumb, index, middle, ring, pinky; each has 52 taxels. Tactile groups
carry `sensor_names_json` metadata, or the root has `left_tactile_sensor_names_json`.
Calibration records `T_sensor_to_finger_json` and wrist/cube metadata. Strict export
requires full-rate tactile groups, even when the input also contains per-frame
tactile fields.

The `apriltag_10` arrays contain the scene-tag observations. With another
`--scene-tag-id`, these arrays contain the selected tag's pose;
`table_apriltag_id` records its actual ID.

## Masks and final output

`mask_N/sequence.npz` contains bool `masks` of shape `(T,H,W)`; true means remove.
SAM2 adds per-object masks and annotation JSON; SAM3 records the prompt/backend.

Final Zarr structure:

```text
data/
  inpaint_rgb, rgb_o3d, depth
  mvs_left_rgb, mvs_right_rgb
  mvs_left_rgb_available, mvs_right_rgb_available
  intrinsic, realsense_timestamp, source_frame_index
tactile_stream/
  left/right_timestamp                   (Nt,)
  left/right_tactile                      (Nt,5,52,3)
calibration/
  taxel_positions_sensor, taxel_rotations_sensor
  taxel_to_fingertip, finger_names, taxel_indices
retarget/
  legwaist, left_arm, right_arm, left_hand, right_hand
  wrist_left, wrist_right                 camera_t palm poses
  left_flange, right_flange               camera_t flange poses
  whole_body_joints, whole_body_joint_names
  base_pose_cam0                          fixed base pose
camera/
  camera_trajectory_cam0                  cleaned T_cam0_cam_t
  T_apriltag_10_to_cam                     raw selected T_camera_t_tag
```

Missing wrist images are zero-filled with explicit availability arrays. Full-rate
touch is not downsampled with video. Diagnostic masks, depths, losses, and optional
comparison branches stay under `_diagnostics`, outside the training Zarr.

## Action chunks

Exported trajectories are unchunked. The optional utility implements Eq. (7) and
relative end-effector poses after camera-motion compensation:

```python
import numpy as np
import zarr
from training.actions import camera_anchored_chunk

s = zarr.open('/path/to/robot_sequence.zarr', mode='r')
sl = slice(0, 16)
camera = s['camera/camera_trajectory_cam0'][sl]
# A stationary reference derived from cleaned camera motion.
reference_in_camera = np.linalg.inv(camera)
arm_actions, hand_targets = camera_anchored_chunk(
    s['retarget/left_flange'][sl], reference_in_camera,
    s['retarget/left_hand'][sl],
)
```

`arm_actions[0]` is identity. Arm poses are relative to the anchor end-effector
frame; hand targets remain absolute. Call separately for the right arm and each
chunk. Choose your policy's horizon, rotation encoding, normalization, and packing.
