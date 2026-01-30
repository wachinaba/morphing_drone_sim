import unittest

import numpy as np

from sim.control.mixer import build_allocation_matrix, solve_mixer_pinv


class TestMixerYaw(unittest.TestCase):
    def test_yaw_requires_CQ(self):
        # Simple upright geometry: 4 rotors at corners, normals +z.
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
        s = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)

        # Target pure yaw torque
        u = np.array([0.0, 0.0, 0.0, 0.05], dtype=float)

        # CQ=0 -> tau_z is not achievable (A row 3 is 0 for upright, lever arm doesn't produce z torque)
        A0 = build_allocation_matrix(r_body=r, n_body=n, C_T=1.0, C_Q=0.0, spin_dir=s)
        mix0 = solve_mixer_pinv(A=A0, wrench_target=u, omega2_min=0.0)
        self.assertAlmostEqual(float(mix0.wrench_achieved[3]), 0.0, places=12)

        # CQ>0 -> tau_z achievable by alternating spin direction
        A1 = build_allocation_matrix(r_body=r, n_body=n, C_T=1.0, C_Q=0.1, spin_dir=s)
        mix1 = solve_mixer_pinv(A=A1, wrench_target=u, omega2_min=0.0)
        self.assertGreater(abs(float(mix1.wrench_achieved[3])), 1e-3)


if __name__ == "__main__":
    unittest.main()





