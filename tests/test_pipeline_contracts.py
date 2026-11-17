"""Behavioral checks for the TouchBridge pipeline."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.config import load_retarget_config
from retarget.optimization.free_wrist_solver import FreeWristRetargetSolver, _graph_edges, _laplacian_residual
from preprocess.write_final_t_finger_to_cam import load_calibration, write_one_zarr
from preprocess.segment_sam3 import merge_frame_masks, segment_sequence
from whole_body_retarget.io_utils import load_base_in_tag
from run_human_to_robot import build_parser, segmentation_command, retarget_command

ROOT = Path(__file__).resolve().parents[1]


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.config = load_retarget_config(ROOT / "configs/retarget_free_wrist.yaml")

    def test_graph_no_base_influence_and_contact_geometry_still_matters(self):
        edges = _graph_edges(["thumb", "index"], ["thumb"])
        self.assertFalse(any("base" in node for edge in edges for node in edge))
        self.assertIn(("left_tip_thumb", "left_tip_index"), edges)
        self.assertIn(("left_tip_thumb", "right_tip_thumb"), edges)
        self.assertIn(("left_tip_thumb", "left_patch_thumb"), edges)
        rng = np.random.default_rng(7)
        names = sorted({node for edge in edges for node in edge} | {"left_base", "right_base"})
        human = {name: rng.normal(size=3) * .03 for name in names}
        robot = {name: value.copy() for name, value in human.items()}
        force = {name: 1. for name in names}
        robot["left_base"] += [100, 200, 300]
        robot["right_base"] -= [100, 200, 300]
        np.testing.assert_allclose(_laplacian_residual(human, robot, force, self.config), 0, atol=1e-12)
        robot["left_tip_thumb"] += [.01, 0, 0]
        reference = _laplacian_residual(human, robot, force, self.config)
        self.assertGreater(np.linalg.norm(reference), 0.)
        solver = object.__new__(FreeWristRetargetSolver)
        solver.config = self.config
        solver._laplacian_cache = {}
        fast = solver._fast_laplacian_residual(object(), human, robot, force)
        np.testing.assert_allclose(fast, reference, atol=1e-12)
        self.assertEqual(_laplacian_residual({"left_base": np.zeros(3)}, {"left_base": np.ones(3)}, {}, self.config).size, 0)

    def test_compliance_goes_beyond_patch_in_rotated_camera_frame(self):
        solver = object.__new__(FreeWristRetargetSolver)
        solver.config = self.config
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [.3, -.7, 1.2]).as_matrix()
        pose[:3, 3] = [.4, -.2, .8]
        patch = pose[:3, 3] + pose[:3, :3] @ np.array([0., 0., .015])
        unit = (patch - pose[:3, 3]) / np.linalg.norm(patch - pose[:3, 3])
        for response, expected_depth in [(0.1, .005), (1., .01)]:
            f = SimpleNamespace(contact_label=True, patch_center_cam=patch, total_force=response)
            target, depth = solver._compliance_target(f, pose)
            self.assertAlmostEqual(depth, expected_depth)
            np.testing.assert_allclose(target - patch, expected_depth * unit)
        f.contact_label = False
        self.assertIsNone(solver._compliance_target(f, pose))

    def test_appendix_transforms_written_in_correct_order(self):
        import zarr
        right, left, sensor = load_calibration()
        np.testing.assert_allclose(right, [[1,0,0,0],[0,-1,0,-.065],[0,0,-1,-.015],[0,0,0,1]])
        np.testing.assert_array_equal(right, left)
        self.assertAlmostEqual(sensor[1, 2], .961262)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "human.zarr"
            root = zarr.open(str(path), mode="w")
            root.attrs["n_frames"] = 1
            frame = root.create_group("data/frame_00000")
            cube = np.eye(4)
            cube[:3, :3] = Rotation.from_euler("z", .8).as_matrix()
            cube[:3, 3] = [.2, .1, .7]
            fingers = np.repeat(np.eye(4)[None], 5, axis=0)
            fingers[:, 0, 3] = np.arange(5) * .02
            for side in ["left", "right"]:
                frame.create_dataset(f"T_{side}_cube_to_cam", data=cube)
                frame.create_dataset(f"T_{side}_finger_to_wrist", data=fingers)
            write_one_zarr(path, right, left, sensor)
            for side in ["left", "right"]:
                np.testing.assert_allclose(frame[f"T_{side}_sensor_to_cam"][:], cube @ right @ fingers @ sensor, atol=1e-7)

    def test_scene_base_matrix_convention_and_validation(self):
        np.testing.assert_array_equal(load_base_in_tag(ROOT / "configs/base_in_scene_tag.json"), np.eye(4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.json"
            base = np.eye(4); base[:3, 3] = [1., 2., 3.]
            path.write_text(json.dumps({"T_tag_base": base.tolist()}))
            camera_tag = np.eye(4)
            camera_tag[:3, :3] = Rotation.from_euler("z", np.pi/2).as_matrix()
            np.testing.assert_allclose((camera_tag @ load_base_in_tag(path))[:3, 3], [-2, 1, 3])
            base[0, 0] = -1
            path.write_text(json.dumps({"T_tag_base": base.tolist()}))
            with self.assertRaises(ValueError): load_base_in_tag(path)

    def test_paper_coefficients(self):
        o = self.config.optimization
        for name, value in dict(lambda_tip_global=100, lambda_tip_rot=1, lambda_patch=20,
                                lambda_taxel_cloud=4, lambda_compliance_target=16,
                                compliance_stiffness=20, compliance_max_depth=.01,
                                compliance_force_direction_sign=1, lambda_tactile_lap=1,
                                lambda_cross_hand=.4, lambda_pair_dist=.25,
                                lambda_joint_smooth=.45, lambda_base_pos_smooth=10,
                                lambda_base_rot_smooth=5, lambda_base_acc=.08, lambda_joint_acc=.08).items():
            self.assertEqual(getattr(o, name), value, name)

    def test_mask_modes_and_base_configuration_reach_subcommands(self):
        args = build_parser().parse_args(["--data-root", "/input", "--work-root", "/work",
                                         "--output-root", "/output", "--base-in-tag", "/custom/base.json"])
        cmd = list(map(str, segmentation_command(args, Path("data"), Path("masks"), Path("ann"))))
        self.assertIn("glove and its attachments", cmd)
        self.assertTrue(any(s.endswith("segment_sam3.py") for s in cmd))
        args.mask_mode = "manual"; args.sam_no_interactive = True
        cmd = list(map(str, segmentation_command(args, Path("data"), Path("masks"), Path("ann"))))
        self.assertIn("--no-interactive", cmd)
        self.assertNotIn("--prompt", cmd)
        cmd = list(map(str, retarget_command(args, Path("data"))))
        self.assertEqual(cmd[cmd.index("--base-in-tag") + 1], "/custom/base.json")

    def test_sam3_union_and_incomplete_sequence_cleanup(self):
        import zarr
        masks = np.zeros((2, 8, 8), bool); masks[0, 1, 1] = True; masks[1, 6, 6] = True
        self.assertEqual(merge_frame_masks({"out_binary_masks": masks}, 8, 8).sum(), 2)
        class FakePredictor:
            closed = False
            def handle_request(self, request):
                if request["type"] == "start_session": return {"session_id": "test"}
                if request["type"] == "add_prompt": return {"outputs": {"out_binary_masks": masks}}
                if request["type"] == "close_session": self.closed = True
            def handle_stream_request(self, request):
                return iter([])  # Missing frame 1 must fail, not silently become background.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "human.zarr"
            root = zarr.open(str(path), mode="w")
            for i in range(2): root.create_dataset(f"data/frame_{i:05d}/rgb", data=np.zeros((8,8,3), np.uint8))
            predictor = FakePredictor(); output = Path(tmp) / "mask.npz"
            with self.assertRaises(RuntimeError): segment_sequence(predictor, path, output)
            self.assertTrue(predictor.closed)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
