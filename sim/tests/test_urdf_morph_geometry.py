import unittest

import numpy as np

from sim.env.pybullet_env import PyBulletEnv
from sim.morph.geometry import compute_rotor_poses


class TestUrdfMorphGeometry(unittest.TestCase):
    def _compare_once(
        self,
        *,
        phi: float,
        psi: float,
        theta: float,
        symmetry: str,
        tol_pos: float = 2e-4,
        tol_dir: float = 2e-4,
    ):
        """
        URDF（PyBullet FK）で得た rotor center/normal (body frame) と、
        `sim/morph/geometry.py` の幾何計算が一致することを確認する。

        注意:
        symmetry:
        - "none": URDFへ各アーム同一の(phi,psi,theta)を入れる（幾何側 symmetry="none" と一致）
        - "mirror_xy": URDFへ象限ごとにphi/theta符号を切り替えて入れる（幾何側 symmetry="mirror_xy" と一致）
        """
        env = PyBulletEnv(gui=False)
        try:
            env.load_body_urdf("assets/urdf/morphing_drone.urdf", base_pos=(0.0, 0.0, 0.3))
            env.configure_morphing_drone()

            env.set_morph_angles(phi_deg=float(phi), psi_deg=float(psi), theta_deg=float(theta), symmetry=str(symmetry))
            r_urdf, n_urdf = env.rotor_geometry_body()

            poses = compute_rotor_poses(
                phi_deg=float(phi),
                psi_deg=float(psi),
                theta_deg=float(theta),
                cx=0.035,
                cy=0.035,
                arm_length_m=0.18,
                symmetry=str(symmetry),
            )
            r_geo = np.stack([p.rotor_center for p in poses], axis=0)
            n_geo = np.stack([p.rotor_normal for p in poses], axis=0)

            # Unit vectors sanity
            for i in range(4):
                self.assertAlmostEqual(float(np.linalg.norm(n_urdf[i])), 1.0, places=6)
                self.assertAlmostEqual(float(np.linalg.norm(n_geo[i])), 1.0, places=6)

            # Compare with tolerances (URDF FK numerical errors are small but allow some slack)
            self.assertTrue(np.allclose(r_urdf, r_geo, atol=float(tol_pos)), msg=f"r mismatch\nurdf={r_urdf}\ngeo={r_geo}")
            self.assertTrue(np.allclose(n_urdf, n_geo, atol=float(tol_dir)), msg=f"n mismatch\nurdf={n_urdf}\ngeo={n_geo}")
        finally:
            env.disconnect()

    def test_match_zero_angles(self):
        self._compare_once(phi=0.0, psi=0.0, theta=0.0, symmetry="none")
        self._compare_once(phi=0.0, psi=0.0, theta=0.0, symmetry="mirror_xy")

    def test_match_moderate_angles(self):
        self._compare_once(phi=30.0, psi=20.0, theta=45.0, symmetry="none")
        self._compare_once(phi=30.0, psi=20.0, theta=45.0, symmetry="mirror_xy")

    def test_match_negative_angles(self):
        self._compare_once(phi=-60.0, psi=-15.0, theta=-90.0, symmetry="none")
        self._compare_once(phi=-60.0, psi=-15.0, theta=-90.0, symmetry="mirror_xy")


if __name__ == "__main__":
    unittest.main()


