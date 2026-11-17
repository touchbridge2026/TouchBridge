import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from training.actions import camera_anchored_chunk


class CameraActionsTest(unittest.TestCase):
    def test_camera_motion_does_not_create_robot_motion(self):
        tags = np.repeat(np.eye(4)[None], 3, axis=0)
        tags[:, :3, :3] = Rotation.from_euler("z", [0, .3, -.2]).as_matrix()
        tags[:, :3, 3] = [[0,0,1],[.1,.2,1],[-.2,0,.9]]
        fixed_ee = np.eye(4); fixed_ee[:3, 3] = [.3, .2, .1]
        joints = np.arange(60).reshape(3, 20)
        actions, targets = camera_anchored_chunk(tags @ fixed_ee, tags, joints)
        np.testing.assert_allclose(actions, np.repeat(np.eye(4)[None], 3, axis=0), atol=1e-12)
        np.testing.assert_array_equal(targets, joints)

    def test_true_motion_remains_after_camera_compensation(self):
        tags = np.repeat(np.eye(4)[None], 3, axis=0)
        tags[:, 2, 3] = [.8, .9, 1.]
        trajectory = np.repeat(np.eye(4)[None], 3, axis=0)
        trajectory[:, 0, 3] = [0, .02, .04]
        actions, _ = camera_anchored_chunk(tags @ trajectory, tags, np.zeros((3, 20)), anchor=1)
        np.testing.assert_allclose(actions[:, 0, 3], [-.02, 0, .02])
