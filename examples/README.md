# Light-bulb retargeting in a point cloud

Run from the repository root after installing `requirements.txt`:

```bash
python examples/retarget_pointcloud.py
```

The example verifies and extracts the bundled archive into `work/bulb_demo/input`,
computes tactile retargeting and full-body IK on the default MOZ + dual Wuji model,
and opens an interactive Open3D window. Results are written to
`work/bulb_demo/retarget`. CPU computation can take several minutes. SAM and
ProPainter do not run in this example; their preprocessing results are included.

The scene cloud is reconstructed from calibrated RGB-D data. Points inside the
human-removal mask are excluded because their depth belongs to the human/hardware.
The retargeted robot and measured contact targets share the stationary scene
coordinate frame. Empty areas behind the removed hands have no measured depth
and remain empty. The camera housing is hidden for the initial first-person view;
the rest of the robot remains available when orbiting the scene.

## Controls and overlays

| Control | Action |
|---|---|
| Space | Play / pause |
| Left / Right | Previous / next frame |
| Left drag | Orbit |
| Mouse wheel | Zoom |
| Q / Esc | Close |

Human taxels are cyan, active human taxels yellow, robot taxels magenta, active
robot taxels red, contact patch centers lime, and human sensor origins white.
The robot meshes show the actual full-body IK solution; target markers show the
measured human contact geometry, so tracking error remains visible.

## Replay and headless use

```bash
# Reopen computed results without optimization.
python examples/retarget_pointcloud.py --view-only

# Compute on a machine without a desktop display.
python examples/retarget_pointcloud.py --no-view

# Save one frame's scene and robot as a colored point cloud without a window.
python examples/retarget_pointcloud.py --view-only --no-view \
  --frame 60 --export-frame work/bulb_demo/frame_60.ply

# Show the original human RGB-D points alongside the robot.
python examples/retarget_pointcloud.py --view-only --raw-cloud

# Short computation check, isolated from the complete trajectory.
python examples/retarget_pointcloud.py --work-dir work/bulb_quick \
  --max-frames 3 --no-view
```

Use a desktop session for the interactive window. A PLY snapshot can be opened in
Open3D or another point-cloud viewer. `--frame` selects the initial displayed/exported
frame. Use a fresh `--work-dir` when changing the data, robot, or optimization
configuration to avoid reusing cached intermediate results.

## Included data and calibration

`data/screw_bulb.zarr.zip` contains one light-bulb screwing sequence spanning
22.8 seconds. It preserves the sequence's time span while retaining every sixth
visual frame (115 of 686) and resizing RGB-D from 640 x 480 to 320 x 240. Camera
intrinsics and tag pixel coordinates are scaled accordingly. Tactile streams
retain all 2,079 samples per hand. Timestamps share a zero-based reference;
wrist-camera images and acquisition payloads are omitted.

The archive contains RGB, metric depth, removal masks, inpainted RGB, scene-tag
observations, hand poses, and tactile measurements. Sensor transforms use the
paper's appendix matrices. `data/manifest.json` records shapes, sampling, and the
archive SHA-256 checksum.

`data/base_in_scene_tag.json` supplies the robot-base placement for this scene.
It maps base coordinates into the stationary tag frame and is used only by this
example. Calibrate this transform for your own recordings; the global default
in `configs/base_in_scene_tag.json` is still identity.
