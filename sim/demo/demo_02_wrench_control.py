import argparse
import time

import numpy as np

from sim.control.mixer import build_allocation_matrix, solve_mixer_pinv
from sim.control.motor_model import MotorModel
from sim.env.pybullet_env import PyBulletEnv
from sim.morph.geometry import allocation_inputs_from_poses, compute_rotor_poses


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Wrench input demo: (Fz, tau) -> mixer -> 4 point forces.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Sim duration [s]. Default: 8.0")
    parser.add_argument("--hz", type=float, default=240.0, help="Sim frequency [Hz]. Default: 240")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2]. Default: 9.81")

    # Morph geometry
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.035)
    parser.add_argument("--cy", type=float, default=0.035)
    parser.add_argument("--arm-length", type=float, default=0.18)
    parser.add_argument("--symmetry", type=str, default="mirror_xy", choices=["mirror_xy", "none"])

    # Body
    parser.add_argument("--body-urdf", type=str, default="assets/urdf/drone_visual.urdf")
    parser.add_argument("--body-z", type=float, default=0.30)

    # Target wrench in BODY frame: [Fz, tau_x, tau_y, tau_z]
    parser.add_argument(
        "--Fz",
        type=float,
        default=None,
        help="Target Fz [N] in body frame. If omitted, uses hover ~= mass*gravity.",
    )
    parser.add_argument("--tau-x", type=float, default=0.0, help="Target tau_x [N*m] in body frame.")
    parser.add_argument("--tau-y", type=float, default=0.0, help="Target tau_y [N*m] in body frame.")
    parser.add_argument("--tau-z", type=float, default=0.0, help="Target tau_z [N*m] in body frame.")

    # Mixer params
    parser.add_argument(
        "--CT",
        type=float,
        default=1.0,
        help="Thrust coefficient CT in T = CT * omega^2. Default: 1.0",
    )
    parser.add_argument(
        "--CQ",
        type=float,
        default=0.0,
        help="Reaction torque coefficient CQ in tau_drag = s*CQ*omega^2*n. Default: 0 (no reaction torque applied).",
    )
    parser.add_argument("--omega2-max", type=float, default=None, help="Upper bound for omega^2 (or thrust if CT=1).")
    parser.add_argument("--motor-tau", type=float, default=0.05, help="Motor first-order time constant tau [s]. Default: 0.05")
    parser.add_argument(
        "--omega2-rate",
        type=float,
        default=None,
        help="Optional omega^2 rate limit [omega2/s]. Example: 50.0",
    )
    parser.add_argument("--log-every", type=int, default=60, help="Print residual every N steps. Default: 60")
    args = parser.parse_args()

    hz = float(args.hz)
    dt = 1.0 / hz

    env = PyBulletEnv(gui=bool(args.gui), time_step=dt, gravity=float(args.gravity))
    try:
        p = env.p
        env.load_plane()
        env.load_body_urdf(str(args.body_urdf), base_pos=(0.0, 0.0, float(args.body_z)))

        mass = float(p.getDynamicsInfo(env.body_id, -1)[0])
        Fz_target = float(args.Fz) if args.Fz is not None else (mass * float(args.gravity))
        u = np.array([Fz_target, float(args.tau_x), float(args.tau_y), float(args.tau_z)], dtype=float)

        poses = compute_rotor_poses(
            phi_deg=float(args.phi),
            psi_deg=float(args.psi),
            theta_deg=float(args.theta),
            cx=float(args.cx),
            cy=float(args.cy),
            arm_length_m=float(args.arm_length),
            symmetry=str(args.symmetry),
        )
        r_body, n_body = allocation_inputs_from_poses(poses)

        spin_dir = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)
        A = build_allocation_matrix(r_body=r_body, n_body=n_body, C_T=float(args.CT), C_Q=float(args.CQ), spin_dir=spin_dir)
        mix = solve_mixer_pinv(
            A=A,
            wrench_target=u,
            omega2_min=0.0,
            omega2_max=(None if args.omega2_max is None else float(args.omega2_max)),
        )

        motor = MotorModel(
            tau=float(args.motor_tau),
            omega2_min=0.0,
            omega2_max=(None if args.omega2_max is None else float(args.omega2_max)),
            omega2_rate_limit=(None if args.omega2_rate is None else float(args.omega2_rate)),
        )
        motor.reset(mix.omega2_cmd)

        if args.gui:
            print("[info] GUI mode: running in real-time-ish (sleep)")
        print(f"[info] u_target=[Fz={u[0]:.3f}N, tx={u[1]:.4f}, ty={u[2]:.4f}, tz={u[3]:.4f}]")
        print(f"[info] omega2_cmd(init)={mix.omega2_cmd}")
        print(f"[info] achieved={mix.wrench_achieved} residual={mix.residual}")

        n_steps = max(1, int(float(args.seconds) * hz))
        for k in range(n_steps):
            st = env.get_state()
            R = _quat_xyzw_to_rotmat(st.quat)

            # motor dynamics
            omega2_actual = motor.step(mix.omega2_cmd, dt)

            for i in range(4):
                world_pos = st.pos + (R @ r_body[i])
                thrust_i = float(args.CT) * float(omega2_actual[i])
                force_body = n_body[i] * thrust_i
                world_force = R @ force_body
                # reaction torque (optional)
                if abs(float(args.CQ)) > 0.0:
                    torque_body = n_body[i] * (float(spin_dir[i]) * float(args.CQ) * float(omega2_actual[i]))
                    world_torque = R @ torque_body
                else:
                    world_torque = None
                env.apply_wrench_world(world_pos=world_pos, world_force=world_force, world_torque=world_torque)

            env.step(1)
            if bool(args.gui):
                time.sleep(dt)

            if int(args.log_every) > 0 and (k % int(args.log_every) == 0):
                # Recompute achieved wrench from current CT/n/r (static in this demo)
                mix_k = solve_mixer_pinv(
                    A=A,
                    wrench_target=u,
                    omega2_min=0.0,
                    omega2_max=(None if args.omega2_max is None else float(args.omega2_max)),
                )
                print(
                    f"[step {k:06d}] residual={mix_k.residual} omega2_cmd={mix_k.omega2_cmd} omega2_act={omega2_actual}",
                )
    finally:
        env.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


