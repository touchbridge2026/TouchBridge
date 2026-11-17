from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.robot.fk import CompiledHandKinematics, HandKinematics
from retarget.robot.urdf_model import parse_urdf_model
from retarget.utils import rotation_log_so3
from whole_body_retarget.hand_models import dual_hand_submodels_from_full


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_URDF = (
    PROJECT_ROOT
    / "assets" / "robot"
    / "urdf"
    / "moz_wuji_7_3_dual_hands.urdf"
)


class FastRetargetMathTest(unittest.TestCase):
    def test_so3_log_matches_scipy(self) -> None:
        rng = np.random.default_rng(20260816)
        matrices = list(Rotation.random(2000, random_state=rng).as_matrix())
        for magnitude in (0.0, 1.0e-12, 1.0e-9, 1.0e-7, 1.0e-5):
            for _ in range(10):
                matrices.append(
                    Rotation.from_rotvec(rng.normal(size=3) * magnitude).as_matrix()
                )
        for matrix in matrices:
            expected = Rotation.from_matrix(matrix).as_rotvec()
            actual = rotation_log_so3(matrix)
            np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-10)

    def test_compiled_hand_fk_matches_reference(self) -> None:
        full = parse_urdf_model(FULL_URDF, expected_dof=66)
        models = dual_hand_submodels_from_full(full, expected_dof_per_hand=20)
        rng = np.random.default_rng(20260816)
        for model in models:
            reference = HandKinematics(model)
            compiled = CompiledHandKinematics(model)
            for _ in range(50):
                q = rng.uniform(model.joint_limits_lower, model.joint_limits_upper)
                expected = reference.fk_all_links(q)
                actual = compiled.fk_all_links(q)
                self.assertEqual(actual.keys(), expected.keys())
                for link_name in expected:
                    np.testing.assert_allclose(
                        actual[link_name], expected[link_name], rtol=1.0e-13, atol=1.0e-13
                    )


if __name__ == "__main__":
    unittest.main()
