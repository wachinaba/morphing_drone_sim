import argparse
import math

import numpy as np

from sim.control.attitude_pd import quat_xyzw_to_rotmat
from sim.env.pybullet_env import PyBulletEnv


def _estimate_axis_inertia(
    *,
    env: PyBulletEnv,
    axis: int,
    tau_mag: float,
    steps: int,
    warmup_steps: int,
    dt: float,
) -> float:
    """
    Estimate diagonal inertia I_axis using alpha = d(omega)/dt and tau = I*alpha.
    Apply +tau and -tau to cancel bias.
    axis: 0=x, 1=y, 2=z (body frame)
    """
    assert axis in (0, 1, 2)
    tau_mag = float(abs(tau_mag))
    steps = int(steps)
    warmup_steps = int(warmup_steps)
    dt = float(dt)

    def run(sign: float) -> float:
        # gather samples of alpha in BODY frame along 'axis'
        alphas = []
        st = env.get_state()
        R_bw = quat_xyzw_to_rotmat(st.quat)
        w_prev_b = (R_bw.T @ np.asarray(st.ang_vel, dtype=float).reshape(3)).copy()

        for k in range(steps):
            # Apply constant torque in BODY frame; convert to WORLD
            tau_b = np.zeros((3,), dtype=float)
            tau_b[axis] = float(sign) * tau_mag
            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)
            tau_w = R_bw @ tau_b
            env.apply_body_wrench_world(torque_world=(float(tau_w[0]), float(tau_w[1]), float(tau_w[2])))
            env.step(1)

            st2 = env.get_state()
            R_bw2 = quat_xyzw_to_rotmat(st2.quat)
            w_b = (R_bw2.T @ np.asarray(st2.ang_vel, dtype=float).reshape(3)).copy()
            alpha_b = (w_b - w_prev_b) / max(1e-12, dt)
            w_prev_b = w_b
            if k >= warmup_steps:
                alphas.append(float(alpha_b[axis]))

        # use robust mean (trim 10%) to reduce spikes
        if not alphas:
            return float("nan")
        a = np.array(alphas, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 10:
            a_mean = float(np.mean(a)) if a.size > 0 else float("nan")
        else:
            a_sorted = np.sort(a)
            lo = int(0.1 * a_sorted.size)
            hi = int(0.9 * a_sorted.size)
            a_mean = float(np.mean(a_sorted[lo:hi]))
        # I = tau/alpha
        return float((float(sign) * tau_mag) / max(1e-9, a_mean))

    I_pos = run(+1.0)
    I_neg = run(-1.0)
    # Average, but guard against sign issues
    Is = [x for x in [I_pos, I_neg] if np.isfinite(x) and x > 0.0]
    if not Is:
        return float("nan")
    return float(np.mean(Is))


def main() -> int:
    ap = argparse.ArgumentParser(description="PX4-like simple autotune: estimate inertia and suggest PD gains.")
    ap.add_argument("--urdf", type=str, default="assets/urdf/morphing_drone.urdf")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--physics-hz", type=float, default=240.0)
    ap.add_argument("--tau", type=float, default=0.02, help="Torque magnitude [N*m] for excitation. Default: 0.02")
    ap.add_argument("--steps", type=int, default=480, help="Steps per sign (+/-) per axis. Default: 480 (~2s @240Hz)")
    ap.add_argument("--warmup-steps", type=int, default=60, help="Ignore first N steps for alpha averaging. Default: 60")
    ap.add_argument("--zeta", type=float, default=0.7, help="Desired damping ratio for PD design. Default: 0.7")
    ap.add_argument("--wn", type=float, default=10.0, help="Desired natural frequency [rad/s] for PD design. Default: 10")

    # Optional morph configuration to tune at a specific geometry
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--psi", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--symmetry", type=str, default="mirror_xy", choices=["mirror_xy", "none"])

    args = ap.parse_args()

    physics_hz = float(args.physics_hz)
    dt = 1.0 / max(1.0, physics_hz)

    # gravity=0 to isolate rotational dynamics
    env = PyBulletEnv(gui=bool(args.gui), time_step=dt, gravity=0.0)
    try:
        env.load_plane()
        env.load_body_urdf(str(args.urdf), base_pos=(0.0, 0.0, 0.3))
        env.configure_morphing_drone()
        env.set_damping_all(linear=0.0, angular=0.0)

        # set morph angles (kinematic)
        env.set_morph_angles(phi_deg=float(args.phi), psi_deg=float(args.psi), theta_deg=float(args.theta), symmetry=str(args.symmetry))

        tau = float(args.tau)
        steps = int(args.steps)
        warm = int(args.warmup_steps)

        Ixx = _estimate_axis_inertia(env=env, axis=0, tau_mag=tau, steps=steps, warmup_steps=warm, dt=dt)
        Iyy = _estimate_axis_inertia(env=env, axis=1, tau_mag=tau, steps=steps, warmup_steps=warm, dt=dt)
        Izz = _estimate_axis_inertia(env=env, axis=2, tau_mag=tau, steps=steps, warmup_steps=warm, dt=dt)

        zeta = float(args.zeta)
        wn = float(args.wn)

        def gains(I: float) -> tuple[float, float]:
            # For theta_ddot + (kd/I) theta_dot + (kp/I) theta = 0:
            # kp = I*wn^2, kd = 2*zeta*I*wn
            kp = float(I) * float(wn) * float(wn)
            kd = 2.0 * float(zeta) * float(I) * float(wn)
            return kp, kd

        kp_x, kd_x = gains(Ixx)
        kp_y, kd_y = gains(Iyy)
        kp_z, kd_z = gains(Izz)

        print("[autotune] inertia estimate (diag approx) [kg*m^2]")
        print(f"  Ixx={Ixx:.6g}  Iyy={Iyy:.6g}  Izz={Izz:.6g}")
        print("[autotune] suggested attitude PD gains for tau_b = -kp*e_R - kd*w_b")
        print(f"  kp_xyz=({kp_x:.6g}, {kp_y:.6g}, {kp_z:.6g})")
        print(f"  kd_xyz=({kd_x:.6g}, {kd_y:.6g}, {kd_z:.6g})")
        print("[autotune] CLI form (axis-wise):")
        print(f"  --kp-att-xyz {kp_x:.6g} {kp_y:.6g} {kp_z:.6g} --kd-att-xyz {kd_x:.6g} {kd_y:.6g} {kd_z:.6g}")

        # Backward-compatible scalar suggestion: use min of roll/pitch to stay conservative.
        kp_scalar = float(min(kp_x, kp_y))
        kd_scalar = float(min(kd_x, kd_y))
        print("[autotune] conservative scalar suggestion (for current --kp-att/--kd-att):")
        print(f"  --kp-att {kp_scalar:.6g} --kd-att {kd_scalar:.6g}")

        return 0
    finally:
        env.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())


