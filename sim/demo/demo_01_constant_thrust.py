import argparse
import time

import numpy as np

from sim.env.pybullet_env import PyBulletEnv
from sim.morph.geometry import allocation_inputs_from_poses, compute_rotor_poses


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    PyBullet quaternion (x,y,z,w) -> rotation matrix (3x3)
    """
    q = np.asarray(q, dtype=float).reshape(4)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Constant thrust demo: apply 4 rotor forces based on morph geometry (phi/psi/theta)."
    )
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run until Ctrl+C (ignore --seconds). Useful to debug GUI interactions like mouse clicks.",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="Sim duration [s]. Default: 8.0")
    parser.add_argument("--hz", type=float, default=240.0, help="Sim frequency [Hz]. Default: 240")

    parser.add_argument("--phi", type=float, default=0.0, help="Fold angle phi [deg].")
    parser.add_argument("--psi", type=float, default=0.0, help="Slant angle psi [deg].")
    parser.add_argument("--theta", type=float, default=0.0, help="Tilt angle theta [deg].")
    parser.add_argument("--cx", type=float, default=0.035, help="Hinge x offset [m].")
    parser.add_argument("--cy", type=float, default=0.035, help="Hinge y offset [m].")
    parser.add_argument("--arm-length", type=float, default=0.18, help="Arm length [m].")
    parser.add_argument(
        "--symmetry",
        type=str,
        default="mirror_xy",
        choices=["mirror_xy", "none"],
        help="How to build 4 arms. Default: mirror_xy",
    )

    parser.add_argument(
        "--body-urdf",
        type=str,
        default="assets/urdf/drone_visual.urdf",
        help="URDF path for the body. Default: assets/urdf/drone_visual.urdf (visual-only drone body)",
    )
    parser.add_argument(
        "--body-z",
        type=float,
        default=0.30,
        help="Initial body z position [m]. Default: 0.30",
    )
    parser.add_argument(
        "--total-thrust",
        type=float,
        default=None,
        help="Total thrust [N]. If omitted, uses hover thrust ~= mass*9.81.",
    )
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2]. Default: 9.81")
    args = parser.parse_args()

    hz = float(args.hz)
    dt = 1.0 / hz

    env = PyBulletEnv(gui=bool(args.gui), time_step=dt, gravity=float(args.gravity))
    try:
        p = env.p
        env.load_plane()
        env.load_body_urdf(str(args.body_urdf), base_pos=(0.0, 0.0, float(args.body_z)))

        mass = float(p.getDynamicsInfo(env.body_id, -1)[0])
        total_thrust = float(args.total_thrust) if args.total_thrust is not None else (mass * float(args.gravity))
        thrust_per = total_thrust / 4.0

        poses = compute_rotor_poses(
            phi_deg=float(args.phi),
            psi_deg=float(args.psi),
            theta_deg=float(args.theta),
            cx=float(args.cx),
            cy=float(args.cy),
            arm_length_m=float(args.arm_length),
            symmetry=str(args.symmetry),
        )
        r_body, n_body = allocation_inputs_from_poses(poses)  # (4,3), (4,3)

        if bool(args.interactive) and bool(args.gui):
            print("[info] interactive mode: running until Ctrl+C")
            print("[info] if the window closes on right-click, please paste the last ~50 lines of console output")
            try:
                while True:
                    st = env.get_state()
                    R = _quat_xyzw_to_rotmat(st.quat)
                    for i in range(4):
                        world_pos = st.pos + (R @ r_body[i])
                        world_force = (R @ n_body[i]) * thrust_per
                        env.apply_wrench_world(world_pos=world_pos, world_force=world_force, world_torque=None)
                    env.step(1)
                    time.sleep(dt)
            except KeyboardInterrupt:
                print("[info] received Ctrl+C; exiting")
        else:
            n_steps = max(1, int(float(args.seconds) * hz))
            print(f"[info] running for {float(args.seconds):.1f}s ({n_steps} steps @ {hz:.1f}Hz)")
            for _ in range(n_steps):
                st = env.get_state()
                R = _quat_xyzw_to_rotmat(st.quat)

                # Apply 4 forces at rotor centers (world positions), along world thrust direction.
                for i in range(4):
                    world_pos = st.pos + (R @ r_body[i])
                    world_force = (R @ n_body[i]) * thrust_per
                    env.apply_wrench_world(world_pos=world_pos, world_force=world_force, world_torque=None)

                env.step(1)
                if bool(args.gui):
                    time.sleep(dt)
    finally:
        env.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


