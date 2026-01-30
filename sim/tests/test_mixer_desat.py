import unittest

import numpy as np

from sim.control.mixer import build_allocation_matrix, solve_mixer_with_fallback


class TestMixerDesat(unittest.TestCase):
    def test_desat_handles_nonnegativity(self):
        # Standard quad geometry
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

        # Command large pitch torque with small Fz so unconstrained solution would require negative thrust on some rotors.
        u = np.array([0.2, 0.0, 0.08, 0.0], dtype=float)  # [Fz, tau_x, tau_y, tau_z]
        res = solve_mixer_with_fallback(A=A, wrench_target=u, omega2_min=0.0, omega2_max=1.0, torque_weights=(1.0, 1.0, 0.0))

        self.assertIn(res.mode, ["pinv", "desat"])
        # Must respect bounds
        self.assertTrue(np.all(res.omega2_cmd >= -1e-12))
        self.assertTrue(np.all(res.omega2_cmd <= 1.0 + 1e-12))
        # Should not produce runaway values
        self.assertLess(float(np.max(res.omega2_cmd)), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()




