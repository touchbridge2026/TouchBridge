# Default hardware

TouchBridge uses the MOZ platform with two Wuji hands. The default model is
[`urdf/moz_wuji_7_3_dual_hands.urdf`](urdf/moz_wuji_7_3_dual_hands.urdf), with
66 active joints: 6 leg/waist, 7 per arm, and 20 per hand. The `meshes/` directory
contains all 85 referenced mesh files with the relative paths used by the URDF.

The same model is used for retargeting, forward kinematics, point-cloud
visualization, and robot-image rendering. Joint ordering is read from the URDF
and saved with computed trajectories.

`configs/robot_camera.json` defines the camera mount. The robot base pose in the
stationary scene-tag frame is a separate, scene-specific calibration supplied
through `--base-in-tag`; its general default is identity. The bundled demo
provides its own scene calibration.
