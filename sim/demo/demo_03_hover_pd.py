import argparse
import time

import numpy as np

from sim.control.attitude_pd import (
    AltitudePDGains,
    AttitudePDGains,
    altitude_pd_Fz_body,
    attitude_pd_torque_body,
    quat_xyzw_to_rotmat,
    rotation_from_z_and_yaw,
    rot_z_deg,
)
from sim.control.mixer import build_allocation_matrix, solve_mixer_with_fallback
from sim.control.motor_model import MotorModel
from sim.env.pybullet_env import PyBulletEnv
from sim.logger.csv_logger import CsvLogger
from sim.morph.geometry import allocation_inputs_from_poses, compute_rotor_poses


def _yaw_deg_from_R_bw(R_bw: np.ndarray) -> float:
    """
    body->world 回転行列からyaw角（deg）を取り出す（ZYXのZ成分）。
    """
    R = np.asarray(R_bw, dtype=float).reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.degrees(yaw))

def _xy_circle_target(*, t: float, cx: float, cy: float, radius: float, freq_hz: float, phase_deg: float) -> tuple[float, float]:
    w = 2.0 * np.pi * float(freq_hz)
    ph = np.deg2rad(float(phase_deg))
    x = float(cx) + float(radius) * float(np.cos(w * float(t) + ph))
    y = float(cy) + float(radius) * float(np.sin(w * float(t) + ph))
    return x, y


def _yaw_circle_tangent_deg(*, t: float, radius: float, freq_hz: float, phase_deg: float, yaw_offset_deg: float) -> float:
    w = 2.0 * np.pi * float(freq_hz)
    ph = np.deg2rad(float(phase_deg))
    vx = -float(radius) * w * float(np.sin(w * float(t) + ph))
    vy = +float(radius) * w * float(np.cos(w * float(t) + ph))
    yaw = float(np.degrees(np.arctan2(vy, vx))) + float(yaw_offset_deg)
    return yaw

def _wrap_deg(d: float) -> float:
    x = (float(d) + 180.0) % 360.0 - 180.0
    return float(x)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hover-ish demo: altitude PD + attitude PD + pinv mixer.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI.")
    parser.add_argument("--seconds", type=float, default=12.0, help="Sim duration [s]. Default: 12.0")
    parser.add_argument("--hz", type=float, default=240.0, help="Sim frequency [Hz]. Default: 240")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2]. Default: 9.81")

    # Body
    parser.add_argument("--body-urdf", type=str, default="assets/urdf/drone_visual.urdf")
    parser.add_argument("--body-z", type=float, default=0.30)
    parser.add_argument("--z-des", type=float, default=None, help="Desired altitude z [m]. Default: initial z.")

    # Morph geometry
    parser.add_argument("--phi", type=float, default=0.0, help="Base fold angle phi [deg].")
    parser.add_argument("--psi", type=float, default=0.0, help="Base slant angle psi [deg].")
    parser.add_argument("--theta", type=float, default=0.0, help="Base tilt angle theta [deg].")
    # Optional time-varying morph (sin wave): angle(t) = base + amp * sin(2*pi*f*t)
    parser.add_argument("--phi-amp", type=float, default=0.0, help="Phi sine amplitude [deg]. Default: 0 (static).")
    parser.add_argument("--phi-freq", type=float, default=0.0, help="Phi sine frequency [Hz]. Default: 0")
    parser.add_argument("--psi-amp", type=float, default=0.0, help="Psi sine amplitude [deg]. Default: 0 (static).")
    parser.add_argument("--psi-freq", type=float, default=0.0, help="Psi sine frequency [Hz]. Default: 0")
    parser.add_argument("--theta-amp", type=float, default=0.0, help="Theta sine amplitude [deg]. Default: 0 (static).")
    parser.add_argument("--theta-freq", type=float, default=0.0, help="Theta sine frequency [Hz]. Default: 0")
    parser.add_argument("--cx", type=float, default=0.035)
    parser.add_argument("--cy", type=float, default=0.035)
    parser.add_argument("--arm-length", type=float, default=0.18)
    parser.add_argument("--symmetry", type=str, default="mirror_xy", choices=["mirror_xy", "none"])

    # Mixer params (MVP)
    parser.add_argument("--CT", type=float, default=1.0, help="Thrust coefficient CT. Default: 1.0")
    parser.add_argument(
        "--CQ",
        type=float,
        default=0.0,
        help="Reaction torque coefficient CQ for yaw control. Default: 0 (yaw not controllable). Try 0.02..0.2.",
    )
    parser.add_argument("--omega2-max", type=float, default=None, help="Upper bound for omega^2 (or thrust if CT=1).")
    parser.add_argument(
        "--tw-ratio",
        type=float,
        default=3.0,
        help="Design thrust-to-weight ratio (total max thrust / weight). Typical: 2..4. Default: 3.0",
    )
    parser.add_argument(
        "--max-thrust-per-rotor",
        type=float,
        default=None,
        help="Optional override: max thrust per rotor [N]. If set, omega2_max is derived from this and CT.",
    )
    # Motor model
    parser.add_argument("--motor-tau", type=float, default=0.05, help="Motor first-order time constant tau [s]. Default: 0.05")
    parser.add_argument(
        "--omega2-rate",
        type=float,
        default=None,
        help="Optional omega^2 rate limit [omega2/s]. Example: 50.0",
    )

    # Gains
    parser.add_argument("--kp-att", type=float, default=0.15)
    parser.add_argument("--kd-att", type=float, default=0.02)
    parser.add_argument("--kp-att-xyz", type=float, nargs=3, default=None, metavar=("KPX", "KPY", "KPZ"), help="Axis-wise attitude P gains (roll,pitch,yaw). Overrides --kp-att.")
    parser.add_argument("--kd-att-xyz", type=float, nargs=3, default=None, metavar=("KDX", "KDY", "KDZ"), help="Axis-wise attitude D gains (roll,pitch,yaw). Overrides --kd-att.")
    parser.add_argument("--kp-z", type=float, default=6.0)
    parser.add_argument("--kd-z", type=float, default=3.5)

    # Yaw target (deg). Roll/pitch are kept level; yaw is tracked if CQ!=0.
    parser.add_argument("--yaw-des", type=float, default=0.0, help="Desired yaw angle [deg]. Default: 0")
    parser.add_argument("--yaw-amp", type=float, default=0.0, help="Yaw sine amplitude [deg]. Default: 0")
    parser.add_argument("--yaw-freq", type=float, default=0.0, help="Yaw sine frequency [Hz]. Default: 0")
    parser.add_argument("--yaw-circle", action="store_true", help="When tracking xy-circle, set yaw to circle tangent heading.")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Yaw offset [deg] for yaw-circle. Default: 0")
    parser.add_argument("--yaw-tau", type=float, default=0.4, help="Yaw command first-order smoothing time constant [s]. Default: 0.4")
    parser.add_argument("--yaw-rate", type=float, default=120.0, help="Yaw command rate limit [deg/s]. Default: 120")
    parser.add_argument(
        "--torque-priority",
        type=str,
        default="rpy",
        choices=["rpy", "rp"],
        help="Torque priority under saturation. 'rp' deprioritizes yaw torque. Default: rpy",
    )
    parser.add_argument("--fallback-ridge", type=float, default=1e-6, help="Fallback ridge regularization. Default: 1e-6")
    parser.add_argument(
        "--fallback-auto-omega2-factor",
        type=float,
        default=10.0,
        help="If omega2_max is not set, fallback uses omega2_max_eff ~= factor*|Fz|/max|A0|. Default: 10",
    )

    # XY position hold (world frame)
    parser.add_argument("--xy-hold", action="store_true", help="Enable XY position hold (PD) toward (x_des,y_des).")
    parser.add_argument("--x-des", type=float, default=0.0, help="Desired x position [m]. Default: 0")
    parser.add_argument("--y-des", type=float, default=0.0, help="Desired y position [m]. Default: 0")
    parser.add_argument("--kp-xy", type=float, default=1.0, help="XY position P gain. Default: 1.0")
    parser.add_argument("--kd-xy", type=float, default=1.2, help="XY velocity D gain. Default: 1.2")
    parser.add_argument("--max-tilt-deg", type=float, default=25.0, help="Max tilt angle [deg] for XY hold. Default: 25")
    parser.add_argument("--xy-circle", action="store_true", help="Track circular XY trajectory (enables xy-hold implicitly).")
    parser.add_argument("--xy-circle-radius", type=float, default=0.5, help="Circle radius [m]. Default: 0.5")
    parser.add_argument("--xy-circle-freq", type=float, default=0.1, help="Circle frequency [Hz]. Default: 0.1")
    parser.add_argument("--xy-circle-cx", type=float, default=0.0, help="Circle center x [m]. Default: 0")
    parser.add_argument("--xy-circle-cy", type=float, default=0.0, help="Circle center y [m]. Default: 0")
    parser.add_argument("--xy-circle-phase", type=float, default=0.0, help="Circle phase [deg]. Default: 0")

    parser.add_argument("--log-every", type=int, default=120, help="Print status every N steps. Default: 120")
    parser.add_argument("--log-csv", type=str, default=None, help="Write per-step log to CSV (e.g. logs/run.csv).")
    parser.add_argument("--log-flush-every", type=int, default=10, help="CSV flush interval (rows). Default: 10")
    # Disturbance (WORLD) on body only (URDF版はロータ個別も対応)
    # --dist-body t0 dur fx fy fz tx ty tz
    parser.add_argument(
        "--dist-body",
        type=float,
        nargs=8,
        default=None,
        metavar=("T0", "DUR", "FX", "FY", "FZ", "TX", "TY", "TZ"),
        help="Apply constant disturbance wrench to body in WORLD frame.",
    )
    parser.add_argument(
        "--dist-body-frame",
        type=str,
        default="world",
        choices=["world", "body"],
        help="Frame for --dist-body vectors. Default: world",
    )
    # White-noise disturbance (Gaussian, per-step) on body
    parser.add_argument("--noise-seed", type=int, default=0, help="RNG seed for disturbance noise. Default: 0")
    parser.add_argument("--dist-body-noise-t0", type=float, default=0.0, help="Body noise start time [s]. Default: 0")
    parser.add_argument("--dist-body-noise-dur", type=float, default=1e9, help="Body noise duration [s]. Default: very long")
    parser.add_argument(
        "--dist-body-noise",
        type=float,
        nargs=6,
        default=None,
        metavar=("SFX", "SFY", "SFZ", "STX", "STY", "STZ"),
        help="Body white-noise disturbance stddev (WORLD) per step. Units: N and N*m.",
    )
    parser.add_argument(
        "--dist-body-noise-frame",
        type=str,
        default="world",
        choices=["world", "body"],
        help="Frame for --dist-body-noise vectors. Default: world",
    )
    # Thrust fluctuation noise (per-rotor, multiplicative on thrust/drag)
    parser.add_argument("--thrust-noise-t0", type=float, default=0.0, help="Thrust noise start time [s]. Default: 0")
    parser.add_argument("--thrust-noise-dur", type=float, default=1e9, help="Thrust noise duration [s]. Default: very long")
    parser.add_argument(
        "--thrust-noise-sigma",
        type=float,
        default=0.0,
        help="Per-rotor thrust multiplicative noise stddev (relative, 1-sigma). Example: 0.05 = 5%%. Default: 0",
    )
    args = parser.parse_args()

    hz = float(args.hz)
    dt = 1.0 / hz

    env = PyBulletEnv(gui=bool(args.gui), time_step=dt, gravity=float(args.gravity))
    try:
        p = env.p
        env.load_plane()
        env.load_body_urdf(str(args.body_urdf), base_pos=(0.0, 0.0, float(args.body_z)))

        mass = float(p.getDynamicsInfo(env.body_id, -1)[0])

        def morph_angles(t: float) -> tuple[float, float, float]:
            phi = float(args.phi) + float(args.phi_amp) * float(np.sin(2.0 * np.pi * float(args.phi_freq) * float(t)))
            psi = float(args.psi) + float(args.psi_amp) * float(np.sin(2.0 * np.pi * float(args.psi_freq) * float(t)))
            theta = float(args.theta) + float(args.theta_amp) * float(np.sin(2.0 * np.pi * float(args.theta_freq) * float(t)))
            return phi, psi, theta

        morph_dynamic = any(
            abs(float(x)) > 1e-12
            for x in (
                args.phi_amp,
                args.phi_freq,
                args.psi_amp,
                args.psi_freq,
                args.theta_amp,
                args.theta_freq,
            )
        )

        # Rotor spin direction (+1/-1) for reaction torque. Pattern: + - + - (typical quad)
        # Note: rotor ordering is consistent with mirror_xy: [id, Mx, Mxy, My]
        spin_dir = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)

        motor = MotorModel(
            tau=float(args.motor_tau),
            omega2_min=0.0,
            omega2_max=None,
            omega2_rate_limit=(None if args.omega2_rate is None else float(args.omega2_rate)),
        )

        # If morph is static, compute r,n,A once; otherwise update per step.
        if not morph_dynamic:
            poses0 = compute_rotor_poses(
                phi_deg=float(args.phi),
                psi_deg=float(args.psi),
                theta_deg=float(args.theta),
                cx=float(args.cx),
                cy=float(args.cy),
                arm_length_m=float(args.arm_length),
                symmetry=str(args.symmetry),
            )
            r_body0, n_body0 = allocation_inputs_from_poses(poses0)
            A0 = build_allocation_matrix(
                r_body=r_body0,
                n_body=n_body0,
                C_T=float(args.CT),
                C_Q=float(args.CQ),
                spin_dir=spin_dir,
            )
        else:
            r_body0, n_body0, A0 = None, None, None

        # Gains
        kp_att = (tuple(float(x) for x in args.kp_att_xyz) if args.kp_att_xyz is not None else float(args.kp_att))
        kd_att = (tuple(float(x) for x in args.kd_att_xyz) if args.kd_att_xyz is not None else float(args.kd_att))
        g_att = AttitudePDGains(kp=kp_att, kd=kd_att)
        g_z = AltitudePDGains(kp_z=float(args.kp_z), kd_z=float(args.kd_z))

        # Desired altitude
        st0 = env.get_state()
        z_des_alt = float(st0.pos[2]) if args.z_des is None else float(args.z_des)

        if args.gui:
            print("[info] GUI mode: running in real-time-ish (sleep)")
        print(f"[info] mass={mass:.4f} kg, z_des={z_des_alt:.3f} m")
        if morph_dynamic:
            print(
                "[info] morph dynamic: "
                f"phi={float(args.phi):.1f}+{float(args.phi_amp):.1f}*sin(2pi*{float(args.phi_freq):.2f}t), "
                f"psi={float(args.psi):.1f}+{float(args.psi_amp):.1f}*sin(2pi*{float(args.psi_freq):.2f}t), "
                f"theta={float(args.theta):.1f}+{float(args.theta_amp):.1f}*sin(2pi*{float(args.theta_freq):.2f}t), "
                f"symmetry={args.symmetry}"
            )
        else:
            print(f"[info] morph static: phi={float(args.phi):.1f} psi={float(args.psi):.1f} theta={float(args.theta):.1f} symmetry={args.symmetry}")
        print(f"[info] gains att(kp,kd)=({g_att.kp},{g_att.kd}) z(kp,kd)=({g_z.kp_z},{g_z.kd_z}) yaw_des={float(args.yaw_des):.1f}deg CQ={float(args.CQ):.3f}")

        # Optional CSV logger
        csv_logger = None
        if args.log_csv:
            fieldnames = [
                "step",
                "t",
                "phi_deg",
                "psi_deg",
                "theta_deg",
                "x",
                "y",
                "z",
                "x_des",
                "y_des",
                "vx",
                "vy",
                "vz",
                "yaw_deg",
                "wx",
                "wy",
                "wz",
                "Fz_body",
                "tau_x_body",
                "tau_y_body",
                "tau_z_body",
                "Fz_ach",
                "tau_x_ach",
                "tau_y_ach",
                "tau_z_ach",
                "Fz_err",
                "tau_x_err",
                "tau_y_err",
                "tau_z_err",
                "res_Fz",
                "res_tau_x",
                "res_tau_y",
                "res_tau_z",
                "mix_mode",
                "mix_saturated",
                "omega2_max_eff",
                "torque_wx",
                "torque_wy",
                "torque_wz",
                "fallback_ridge",
                "fallback_auto_omega2_factor",
                "omega2_cmd_0",
                "omega2_cmd_1",
                "omega2_cmd_2",
                "omega2_cmd_3",
                "omega2_act_0",
                "omega2_act_1",
                "omega2_act_2",
                "omega2_act_3",
            ]
            csv_logger = CsvLogger(path=args.log_csv, fieldnames=fieldnames, flush_every=int(args.log_flush_every))
            csv_logger.open()
            print(f"[info] CSV logging enabled: {args.log_csv}")

        # Derive omega2_max from design thrust margin if not explicitly provided.
        if args.max_thrust_per_rotor is not None:
            omega2_max_auto = float(args.max_thrust_per_rotor) / max(1e-12, float(args.CT))
        else:
            tw = max(1.0, float(args.tw_ratio))
            weight = float(mass) * float(args.gravity)
            thrust_total_max = tw * weight
            thrust_per_rotor_max = thrust_total_max / 4.0
            omega2_max_auto = thrust_per_rotor_max / max(1e-12, float(args.CT))

        omega2_max_eff = float(args.omega2_max) if args.omega2_max is not None else float(omega2_max_auto)
        # Apply to motor model (and keep motor state consistent)
        motor.omega2_max = float(omega2_max_eff)

        n_steps = max(1, int(float(args.seconds) * hz))
        rng = np.random.default_rng(int(args.noise_seed))
        for k in range(n_steps):
            st = env.get_state()
            R = quat_xyzw_to_rotmat(st.quat)

            t_now = float(k) * dt
            # yaw smoothing state (deg)
            if k == 0:
                yaw_filt_deg = _yaw_deg_from_R_bw(R)
            if bool(args.xy_circle):
                x_des, y_des = _xy_circle_target(
                    t=t_now,
                    cx=float(args.xy_circle_cx),
                    cy=float(args.xy_circle_cy),
                    radius=float(args.xy_circle_radius),
                    freq_hz=float(args.xy_circle_freq),
                    phase_deg=float(args.xy_circle_phase),
                )
                xy_hold = True
            else:
                x_des, y_des = float(args.x_des), float(args.y_des)
                xy_hold = bool(args.xy_hold)

            # Morph -> r,n,A (body frame)
            if morph_dynamic:
                phi_k, psi_k, theta_k = morph_angles(t_now)
                poses = compute_rotor_poses(
                    phi_deg=phi_k,
                    psi_deg=psi_k,
                    theta_deg=theta_k,
                    cx=float(args.cx),
                    cy=float(args.cy),
                    arm_length_m=float(args.arm_length),
                    symmetry=str(args.symmetry),
                )
                r_body, n_body = allocation_inputs_from_poses(poses)
                A = build_allocation_matrix(
                    r_body=r_body,
                    n_body=n_body,
                    C_T=float(args.CT),
                    C_Q=float(args.CQ),
                    spin_dir=spin_dir,
                )
            else:
                r_body, n_body, A = r_body0, n_body0, A0

            # Desired attitude: keep level, track yaw (about world z)
            if bool(args.yaw_circle) and bool(args.xy_circle):
                yaw_cmd_deg = _yaw_circle_tangent_deg(
                    t=t_now,
                    radius=float(args.xy_circle_radius),
                    freq_hz=float(args.xy_circle_freq),
                    phase_deg=float(args.xy_circle_phase),
                    yaw_offset_deg=float(args.yaw_offset),
                )
            else:
                yaw_cmd_deg = float(args.yaw_des) + float(args.yaw_amp) * float(np.sin(2.0 * np.pi * float(args.yaw_freq) * float(t_now)))

            # Smooth + rate-limit yaw command.
            dy = _wrap_deg(float(yaw_cmd_deg) - float(yaw_filt_deg))
            tau = max(1e-6, float(args.yaw_tau))
            yaw_dot = dy / tau  # [deg/s]
            yaw_rate = max(0.0, float(args.yaw_rate))
            if yaw_rate > 0.0:
                yaw_dot = float(np.clip(yaw_dot, -yaw_rate, +yaw_rate))
            yaw_filt_deg = float(yaw_filt_deg) + float(yaw_dot) * float(dt)
            yaw_cmd_deg = float(yaw_filt_deg)
            if xy_hold:
                # XY PD in world -> desired acceleration
                ex = float(x_des) - float(st.pos[0])
                ey = float(y_des) - float(st.pos[1])
                ax_cmd = float(args.kp_xy) * ex + float(args.kd_xy) * (0.0 - float(st.vel[0]))
                ay_cmd = float(args.kp_xy) * ey + float(args.kd_xy) * (0.0 - float(st.vel[1]))
                # Desired body-z direction in world. Small-angle: a_xy ≈ g * z_des_xy
                g = float(args.gravity)
                z_dir_des = np.array([ax_cmd / max(1e-6, g), ay_cmd / max(1e-6, g), 1.0], dtype=float)
                # Clamp tilt
                max_tilt = np.deg2rad(float(args.max_tilt_deg))
                xy_norm = float(np.linalg.norm(z_dir_des[:2]))
                if xy_norm > 1e-12:
                    # tan(tilt) ~= ||z_xy|| / z_z
                    tan_tilt = xy_norm / max(1e-6, float(z_dir_des[2]))
                    if tan_tilt > np.tan(max_tilt):
                        scale = np.tan(max_tilt) / tan_tilt
                        z_dir_des[0] *= scale
                        z_dir_des[1] *= scale
                z_dir_des = z_dir_des / max(1e-12, float(np.linalg.norm(z_dir_des)))
                R_des = rotation_from_z_and_yaw(z_world=z_dir_des, yaw_deg=float(yaw_cmd_deg))
            else:
                R_des = rot_z_deg(float(yaw_cmd_deg))

            # PD attitude torque in BODY frame
            tau_b = attitude_pd_torque_body(R_bw=R, ang_vel_world=st.ang_vel, R_des_bw=R_des, gains=g_att)

            # Altitude PD -> Fz in BODY frame
            Fz_b = altitude_pd_Fz_body(
                z_world=float(st.pos[2]),
                vz_world=float(st.vel[2]),
                z_des_world=z_des_alt,
                mass=mass,
                gravity=float(args.gravity),
                R_bw=R,
                gains=g_z,
                Fz_min=0.0,
                Fz_max=None,
            )

            u = np.array([Fz_b, float(tau_b[0]), float(tau_b[1]), float(tau_b[2])], dtype=float)
            mix = solve_mixer_with_fallback(
                A=A,
                wrench_target=u,
                omega2_min=0.0,
                omega2_max=float(omega2_max_eff),
                torque_weights=((1.0, 1.0, 1.0) if str(args.torque_priority) == "rpy" else (1.0, 1.0, 0.0)),
                fallback_ridge=float(args.fallback_ridge),
                fallback_auto_omega2_factor=float(args.fallback_auto_omega2_factor),
            )

            omega2_actual = motor.step(mix.omega2_cmd, dt)

            # Apply forces at rotor centers in world coordinates
            thrust_sigma = max(0.0, float(args.thrust_noise_sigma))
            tn_t0 = float(args.thrust_noise_t0)
            tn_dur = float(args.thrust_noise_dur)
            for i in range(4):
                world_pos = st.pos + (R @ r_body[i])
                thrust_i = float(args.CT) * float(omega2_actual[i])
                if thrust_sigma > 0.0 and (tn_t0 <= float(t_now) < tn_t0 + tn_dur):
                    m = 1.0 + float(rng.normal(loc=0.0, scale=thrust_sigma))
                    m = max(0.0, m)
                    thrust_i *= m
                force_body = n_body[i] * thrust_i
                world_force = R @ force_body
                # reaction torque (optional, enabled when CQ!=0)
                if abs(float(args.CQ)) > 0.0:
                    tau_react_i = float(spin_dir[i]) * float(args.CQ) * float(omega2_actual[i])
                    if thrust_sigma > 0.0 and (tn_t0 <= float(t_now) < tn_t0 + tn_dur):
                        # keep torque correlated with thrust fluctuation
                        tau_react_i *= (thrust_i / (float(args.CT) * float(omega2_actual[i]) + 1e-12))
                    torque_body = n_body[i] * tau_react_i
                    world_torque = R @ torque_body
                else:
                    world_torque = None
                env.apply_wrench_world(world_pos=world_pos, world_force=world_force, world_torque=world_torque)

            # Disturbance (WORLD) on body
            if args.dist_body is not None:
                t0, dur, fx, fy, fz, tx, ty, tz = [float(x) for x in args.dist_body]
                if float(t0) <= float(t_now) < float(t0) + float(dur):
                    f = np.array([fx, fy, fz], dtype=float)
                    tau = np.array([tx, ty, tz], dtype=float)
                    if str(args.dist_body_frame) == "body":
                        f = R @ f
                        tau = R @ tau
                    env.apply_body_wrench_world(force_world=(float(f[0]), float(f[1]), float(f[2])), torque_world=(float(tau[0]), float(tau[1]), float(tau[2])))
            if args.dist_body_noise is not None:
                t0 = float(args.dist_body_noise_t0)
                dur = float(args.dist_body_noise_dur)
                if t0 <= float(t_now) < t0 + dur:
                    sfx, sfy, sfz, stx, sty, stz = [float(x) for x in args.dist_body_noise]
                    f = rng.normal(loc=0.0, scale=[sfx, sfy, sfz])
                    tau = rng.normal(loc=0.0, scale=[stx, sty, stz])
                    if str(args.dist_body_noise_frame) == "body":
                        f = R @ np.asarray(f, dtype=float).reshape(3)
                        tau = R @ np.asarray(tau, dtype=float).reshape(3)
                    env.apply_body_wrench_world(force_world=(float(f[0]), float(f[1]), float(f[2])), torque_world=(float(tau[0]), float(tau[1]), float(tau[2])))

            # CSV log (one row per step)
            if csv_logger is not None:
                yaw_deg = _yaw_deg_from_R_bw(R)
                if morph_dynamic:
                    phi_log, psi_log, theta_log = phi_k, psi_k, theta_k
                else:
                    phi_log, psi_log, theta_log = float(args.phi), float(args.psi), float(args.theta)
                row = {
                    "step": int(k),
                    "t": float(t_now),
                    "phi_deg": float(phi_log),
                    "psi_deg": float(psi_log),
                    "theta_deg": float(theta_log),
                    "x": float(st.pos[0]),
                    "y": float(st.pos[1]),
                    "z": float(st.pos[2]),
                    "x_des": float(x_des),
                    "y_des": float(y_des),
                    "vx": float(st.vel[0]),
                    "vy": float(st.vel[1]),
                    "vz": float(st.vel[2]),
                    "yaw_deg": float(yaw_deg),
                    "wx": float(st.ang_vel[0]),
                    "wy": float(st.ang_vel[1]),
                    "wz": float(st.ang_vel[2]),
                    "Fz_body": float(Fz_b),
                    "tau_x_body": float(tau_b[0]),
                    "tau_y_body": float(tau_b[1]),
                    "tau_z_body": float(tau_b[2]),
                    "Fz_ach": float(mix.wrench_achieved[0]),
                    "tau_x_ach": float(mix.wrench_achieved[1]),
                    "tau_y_ach": float(mix.wrench_achieved[2]),
                    "tau_z_ach": float(mix.wrench_achieved[3]),
                    "Fz_err": float(mix.wrench_target[0] - mix.wrench_achieved[0]),
                    "tau_x_err": float(mix.wrench_target[1] - mix.wrench_achieved[1]),
                    "tau_y_err": float(mix.wrench_target[2] - mix.wrench_achieved[2]),
                    "tau_z_err": float(mix.wrench_target[3] - mix.wrench_achieved[3]),
                    "res_Fz": float(mix.residual[0]),
                    "res_tau_x": float(mix.residual[1]),
                    "res_tau_y": float(mix.residual[2]),
                    "res_tau_z": float(mix.residual[3]),
                    "mix_mode": str(mix.mode),
                    "mix_saturated": int(bool(mix.saturated)),
                    "omega2_max_eff": ("" if mix.omega2_max_eff is None else float(mix.omega2_max_eff)),
                    "torque_wx": float(mix.torque_weights[0]),
                    "torque_wy": float(mix.torque_weights[1]),
                    "torque_wz": float(mix.torque_weights[2]),
                    "fallback_ridge": float(mix.fallback_ridge),
                    "fallback_auto_omega2_factor": float(mix.fallback_auto_omega2_factor),
                    "omega2_cmd_0": float(mix.omega2_cmd[0]),
                    "omega2_cmd_1": float(mix.omega2_cmd[1]),
                    "omega2_cmd_2": float(mix.omega2_cmd[2]),
                    "omega2_cmd_3": float(mix.omega2_cmd[3]),
                    "omega2_act_0": float(omega2_actual[0]),
                    "omega2_act_1": float(omega2_actual[1]),
                    "omega2_act_2": float(omega2_actual[2]),
                    "omega2_act_3": float(omega2_actual[3]),
                }
                csv_logger.write(row)

            env.step(1)
            if bool(args.gui):
                time.sleep(dt)

            if int(args.log_every) > 0 and (k % int(args.log_every) == 0):
                if morph_dynamic:
                    phi_k, psi_k, theta_k = morph_angles(float(k) * dt)
                    morph_str = f"phi/psi/theta=({phi_k:+.1f},{psi_k:+.1f},{theta_k:+.1f})"
                else:
                    morph_str = "phi/psi/theta=(static)"
                print(
                    f"[step {k:06d}] z={float(st.pos[2]):+.3f} vz={float(st.vel[2]):+.3f} "
                    f"{morph_str} Fz_b={Fz_b:+.3f} tau_b={tau_b} residual={mix.residual} omega2_act={omega2_actual}"
                )
    finally:
        try:
            if "csv_logger" in locals() and csv_logger is not None:
                csv_logger.close()
        except Exception:
            pass
        env.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


