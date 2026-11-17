# Method and configuration

TouchBridge combines tactile-conditioned retargeting, scene-relative robot
placement, and visual synthesis to convert human demonstrations into robot data.

## Graph and base regularization

The tactile graph contains fingertip and contact-patch nodes. Hand bases are
regularized separately. The reference and cached fast paths share one edge
constructor.

Graph edges: active tip–patch, intra-hand contacting tip pairs, cross-hand
contacting tip pairs, and cross-hand active patch pairs. Human-derived edge
weights are reused on the robot. Auxiliary losses constrain contact-pair distances.

Independent base-anchor regularization stabilizes hand placement. Kabsch alignment of neutral robot
fingertips to human sensor targets produces each hand's reference pose, used for
base regularization, anchor-local tip tracking, and initialization. This reference pose is distinct from the measured human wrist and is not a
graph node.

## Parameters

The default YAML uses Table IV's selected global coefficients:

| Quantity | Configuration | Value |
|---|---|---:|
| Contact / active-taxel threshold | `contact_threshold`, `active_taxel_force_threshold` | 0.02, 0.01 |
| Position / rotation | `lambda_tip_global`, `lambda_tip_rot` | 100, 1 |
| Baseline/contact multiplier | `beta_base`, `beta_contact` | 1, 2 |
| Contact pose scale | `contact_tip_tracking_scale` | 0.05 |
| Patch / taxel | `lambda_patch`, `lambda_taxel_cloud` | 20, 4 |
| Compliance | `lambda_compliance_target` | 16 |
| Response-to-displacement factor | `compliance_stiffness` | 20 |
| Maximum displacement | `compliance_max_depth` | 0.01 m |
| Graph | `lambda_tactile_lap` | 1 |
| Distance decay / response modulation | `kappa`, `tactile_gamma` | 20, 1 |
| Cross-hand / pair distance | `lambda_cross_hand`, `lambda_pair_dist` | 0.4, 0.25 |
| Joint smoothness | `lambda_joint_smooth` | 0.45 |
| Base position / rotation smoothness | `lambda_base_pos_smooth`, `lambda_base_rot_smooth` | 10, 5 |
| Base / joint acceleration | `lambda_base_acc`, `lambda_joint_acc` | 0.08, 0.08 |

Finger multipliers default to one. Additional engineering controls are specified
in the YAML: minimum active-taxel count, contact hold, local tip tracking,
base-anchor/base-step regularizers, joint rest, collision proxy, bounds, and solver
limits. Table values are global coefficients applied before contact-dependent
residual multipliers.

## Compliance

For sensor origin `p` and contact center `mu`, both in the camera frame:

```text
u = (mu - p) / ||mu - p||
d = min(0.01, max(0, response - minimum_response) / 20)
target = mu + d * u
```

`compliance_force_direction_sign = +1` extends beyond the measured patch toward
the object under this geometric approximation (appendix Eq. 15). The direction
is defined by the sensor-to-patch geometry, not by a reaction-force vector.
Response is the sum of taxel vector magnitudes in the input scale. The term
encourages deeper contact; it does not prescribe a physical force or guarantee a
particular final displacement when combined with other objectives.

## Appendix calibration

Transforms act on column vectors; translations are metres. With camera C, cube Q,
glove wrist W, fingertip F, and sensor S:

```text
T_C_F = T_C_Q @ T_Q_W @ T_W_F
T_C_S = T_C_F @ T_F_S

T_Q_W = [[1,  0,  0,  0],
         [0, -1,  0, -0.065],
         [0,  0, -1, -0.015],
         [0,  0,  0,  1]]

T_F_S = [[-1,  0,         0,        0],
         [ 0, -0.275637,  0.961262, 0.02477],
         [ 0,  0.961262,  0.275637, 0.00186],
         [ 0,  0,         0,        1]]
```

These are Eqs. (10) and (11), at printed precision. Both hands use them by default.
The solver tracks `T_C_S`; the bundled robot fingertip link frames match the sensor
frames, so taxel positions need no additional frame correction.

## Robot base and rendering

The user supplies `T_tag_base`, and the first selected cleaned tag observation gives:

```text
T_cam0_base = T_cam0_tag @ T_tag_base
```

Replace the identity placeholder with a measured base pose in the stationary
scene-tag frame. Optical-view mounting calibration and full-robot kinematics are
configured separately; adapt them when changing robots or camera mounts.

Segmentation supports SAM3 text prompts and SAM2 desktop annotations. The paper
describes the SAM2 workflow. Desktop polygons become SAM2 mask prompts, and point
coordinates are normalized from image pixels before model inference.

The renderer composites robot meshes onto ProPainter RGB. Original depth is used
for occlusion only outside `inpaint_mask`, because depth inside the mask belongs
to the removed human/hardware. Two camera-enclosing head meshes are excluded from
the first-person render. Wrist-view images and measured tactile streams are kept.
