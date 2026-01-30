import unittest

import numpy as np

from sim.control.mixer import build_allocation_matrix, solve_mixer_with_fallback


class TestMixerFallback(unittest.TestCase):
    def test_fz_priority_when_saturated(self):
        # Upright: all normals +z, positions give roll/pitch authority but not needed here.
        r = np.array(
            [
                [+0.1, +0.1, 0.0],
                [-0.1, +0.1, 0.0],
                [-0.1, -0.1, 0.0],
                [+0.1, -0.1, 0.0],
            ],
            dtype=float,
        )
        n = np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (4, 1))
        A = build_allocation_matrix(r_body=r, n_body=n, C_T=1.0, C_Q=0.0, spin_dir=np.array([1, -1, 1, -1], float))

        # Ask for large Fz but cap omega2 so it cannot be achieved.
        u = np.array([10.0, 0.0, 0.0, 0.0], dtype=float)
        res = solve_mixer_with_fallback(A=A, wrench_target=u, omega2_min=0.0, omega2_max=1.0)

        # Should be saturated and in fz_priority mode (since pinv will request >1.0)
        self.assertTrue(bool(res.saturated))
        # New PX4-like constrained desaturation solver
        self.assertEqual(str(res.mode), "desat")

        # Fz achieved should be as large as possible under cap:
        # A0 = [1,1,1,1], omega2_max=1 => max Fz = 4
        self.assertAlmostEqual(float(res.wrench_achieved[0]), 4.0, places=9)


if __name__ == "__main__":
    unittest.main()


