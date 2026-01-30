import unittest

import numpy as np

from sim.morph.geometry import compute_rotor_poses


def _is_unit(v: np.ndarray, tol: float = 1e-9) -> bool:
    v = np.asarray(v, dtype=float).reshape(3)
    return abs(float(np.linalg.norm(v)) - 1.0) <= tol


class TestMorphGeometry(unittest.TestCase):
    def test_zero_angles_rotor_normals_are_up(self):
        poses = compute_rotor_poses(phi_deg=0.0, psi_deg=0.0, theta_deg=0.0, symmetry="mirror_xy")
        self.assertEqual(len(poses), 4)
        for p in poses:
            self.assertTrue(_is_unit(p.arm_dir))
            self.assertTrue(_is_unit(p.rotor_normal))
            # zero angles => rotor normal should be +z
            self.assertTrue(np.allclose(p.rotor_normal, np.array([0.0, 0.0, 1.0]), atol=1e-9))

    def test_mirror_xy_symmetry_hinges(self):
        cx, cy = 0.035, 0.035
        poses = compute_rotor_poses(phi_deg=10.0, psi_deg=20.0, theta_deg=-15.0, cx=cx, cy=cy, symmetry="mirror_xy")
        hinges = np.stack([p.hinge for p in poses], axis=0)
        expected = np.array(
            [
                [+cx, +cy, 0.0],
                [-cx, +cy, 0.0],
                [-cx, -cy, 0.0],
                [+cx, -cy, 0.0],
            ],
            dtype=float,
        )
        self.assertTrue(np.allclose(hinges, expected, atol=1e-12))

    def test_mirror_xy_symmetry_normals_are_mirrored(self):
        poses = compute_rotor_poses(phi_deg=30.0, psi_deg=-10.0, theta_deg=45.0, symmetry="mirror_xy")

        n0 = poses[0].rotor_normal
        n1 = poses[1].rotor_normal
        n2 = poses[2].rotor_normal
        n3 = poses[3].rotor_normal

        M_x = np.diag([-1.0, 1.0, 1.0])
        M_y = np.diag([1.0, -1.0, 1.0])
        M_xy = np.diag([-1.0, -1.0, 1.0])

        self.assertTrue(np.allclose(n1, M_x @ n0, atol=1e-9))
        self.assertTrue(np.allclose(n2, M_xy @ n0, atol=1e-9))
        self.assertTrue(np.allclose(n3, M_y @ n0, atol=1e-9))

    def test_rotor_centers_shift_with_arm_length(self):
        p_short = compute_rotor_poses(phi_deg=0.0, psi_deg=0.0, theta_deg=0.0, arm_length_m=0.10, symmetry="mirror_xy")[0]
        p_long = compute_rotor_poses(phi_deg=0.0, psi_deg=0.0, theta_deg=0.0, arm_length_m=0.20, symmetry="mirror_xy")[0]
        # arm_dir is diagonal in XY, so center distance from hinge should scale with L
        d_short = float(np.linalg.norm(p_short.rotor_center - p_short.hinge))
        d_long = float(np.linalg.norm(p_long.rotor_center - p_long.hinge))
        self.assertAlmostEqual(d_short, 0.10, places=12)
        self.assertAlmostEqual(d_long, 0.20, places=12)


if __name__ == "__main__":
    unittest.main()





