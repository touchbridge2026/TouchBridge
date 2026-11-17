# Validation

The pipeline was validated on an NVIDIA RTX 3090. The checks below document
tested functionality and the limits of this validation.

## Automated behavior checks

`python -m unittest discover -s tests -v`: **11 tests passed**.

- Reference and cached contact-graph residuals ignore base positions, retain
  sensitivity to contact geometry, and agree numerically.
- Positive compliance moves beyond the patch in a rotated camera frame and
  respects the 10 mm cap.
- Appendix transforms are written in the correct multiplication order.
- Scene-tag/base configuration preserves transform direction and rejects
  reflections; identity is the default.
- Paper coefficients and both segmentation command routes are checked.
- SAM3 combines instance masks and rejects incomplete propagation without
  writing a partial result.
- Fast forward kinematics and SO(3) logarithms match reference calculations.
- Camera compensation removes camera-induced motion while retaining actual
  end-effector motion and absolute hand targets.

## Real-data smoke checks

The integration check used a three-frame aligned recording at 640 x 480.

| Stage | Result |
|---|---|
| Standalone wrist-cube detector | Ran on a real RGB/depth frame without acquisition drivers |
| Appendix calibration | Wrote transforms to the working Zarr copy |
| Native SAM3 | Default `glove and its attachments` prompt produced 3 frame masks |
| Native SAM2.1 | Saved desktop-format polygon annotation propagated to 3 frame masks |
| ProPainter | Flow estimation, flow completion, propagation and inpainting completed; working Zarr updated |
| Contact retarget + full-body IK | Completed with the default configuration and identity base placeholder; 3 x 66 joints |
| Open3D CPU rendering | Completed and produced a three-frame MP4 |
| Strict Zarr export | RGB, wrist views, poses, joints and 950 full-rate tactile samples per hand exported |

The retarget/render check used SAM3 masks and ProPainter inpainting. Mask, inpaint,
and composite images were inspected. The text prompt did **not** fully remove all
glove/hardware regions in this excerpt; default text segmentation needs visual
review and, when necessary, prompt adjustment or desktop annotation. Identity
base placement also caused substantial IK/alignment error, as expected for an
uncalibrated scene. This validates execution and file interfaces, not visual
quality, robot accuracy, or reproduction of the paper's benchmarks.

The SAM2 run exercised saved polygon input and real model propagation. Interactive
mouse drawing was not manually exercised on a desktop in this validation. SAM2's
optional compiled hole-filling extension was unavailable; inference completed
with its upstream fallback. Raw timestamp alignment was not rerun across a full
recording, and the EGL/CUDA renderers were not exercised.

## Bundled light-bulb example

The complete bundled demo was additionally tested through
`examples/retarget_pointcloud.py --no-view`: 115 visual frames, 66 robot joints,
and 2,079 tactile samples per hand. Every frame produced finite joints and target
transforms; all full-body IK solves reported success. The default URDF and all
85 referenced meshes were verified against the selected hardware model.

The Open3D scene loaded without missing meshes. Frame 60 was exported as a colored
PLY and rendered with Open3D's headless EGL renderer for visual inspection. The
scene contains the measured point cloud, full-body robot geometry, and contact
targets. Interactive keyboard/mouse operation was not manually exercised.

The demo archive checksum, all frame dimensions, calibration transforms, and
portable metadata were checked. The 11 automated tests also passed after adding
the demo. This example uses included preprocessing results; it does not rerun
SAM or ProPainter.

Environment: Python 3.10.19, PyTorch 2.7.0+cu126, SciPy 1.15.3, Zarr 2.18.3,
Open3D 0.19.0. Tests used an existing compatible environment; installation into
a fresh environment has not been validated. The bundled light-bulb input is
included; model checkpoints and generated retarget outputs are not committed.
