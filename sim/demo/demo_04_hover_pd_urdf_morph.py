import argparse
import time

import numpy as np

from sim.control.attitude_pd import (
    AltitudePDGains,
    AttitudePDGains,
    altitude_pd_Fz_world,
    attitude_pd_torque_body,
    quat_xyzw_to_rotmat,
    rotation_from_z_and_yaw,
    rot_z_deg,
)
from sim.control.mixer import build_allocation_matrix, solve_mixer_with_fallback
from sim.control.motor_model import MotorModel
from sim.env.pybullet_env import PyBulletEnv
from sim.logger.csv_logger import CsvLogger


def _xy_circle_target(*, t: float, cx: float, cy: float, radius: float, freq_hz: float, phase_deg: float) -> tuple[float, float]:
    w = 2.0 * np.pi * float(freq_hz)
    ph = np.deg2rad(float(phase_deg))
    x = float(cx) + float(radius) * float(np.cos(w * float(t) + ph))
    y = float(cy) + float(radius) * float(np.sin(w * float(t) + ph))
    return x, y


def _yaw_deg_from_R_bw(R_bw: np.ndarray) -> float:
    # yaw from rotation matrix (body->world), ZYX convention
    R = np.asarray(R_bw, dtype=float).reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.degrees(yaw))

def _yaw_circle_tangent_deg(*, t: float, radius: float, freq_hz: float, phase_deg: float, yaw_offset_deg: float) -> float:
    """
    円軌道の接線方向へyawを向ける（進行方向）。
    x=cx+R cos(wt+ph), y=cy+R sin(wt+ph)
    v=[-R w sin, R w cos] => yaw = atan2(vy, vx)
    """
    w = 2.0 * np.pi * float(freq_hz)
    ph = np.deg2rad(float(phase_deg))
    vx = -float(radius) * w * float(np.sin(w * float(t) + ph))
    vy = +float(radius) * w * float(np.cos(w * float(t) + ph))
    yaw = float(np.degrees(np.arctan2(vy, vx))) + float(yaw_offset_deg)
    return yaw

def _wrap_deg(d: float) -> float:
    # wrap to [-180, 180)
    x = (float(d) + 180.0) % 360.0 - 180.0
    return float(x)


def main() -> int:
    parser = argparse.ArgumentParser(description="URDF morph demo: hover PD + pinv mixer using URDF rotor links.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI.")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--hz", type=float, default=240.0)
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=240.0,
        help="PyBullet physics stepping rate [Hz]. Control runs at --hz. Use physics-hz>=hz and preferably an integer multiple. Default: 240",
    )
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--lin-damping", type=float, default=0.05, help="PyBullet linear damping applied to all links. Default: 0.05")
    parser.add_argument("--ang-damping", type=float, default=0.05, help="PyBullet angular damping applied to all links. Default: 0.05")

    # URDF
    parser.add_argument("--urdf", type=str, default="assets/urdf/morphing_drone.urdf")
    parser.add_argument("--body-z", type=float, default=0.30)
    parser.add_argument("--z-des", type=float, default=None)

    # Morph angles (base + optional sin)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--symmetry", type=str, default="mirror_xy", choices=["mirror_xy", "none"], help="Morph symmetry semantics. Default matches visualize_morph_drone mirror_xy.")
    # Morph servo (joint dynamics)
    parser.add_argument("--morph-tau", type=float, default=0.08, help="Morph joint time constant tau [s]. Default: 0.08")
    parser.add_argument("--morph-rate", type=float, default=None, help="Morph joint max speed [rad/s]. Example: 3.0")
    parser.add_argument("--phi-amp", type=float, default=0.0)
    parser.add_argument("--phi-freq", type=float, default=0.0)
    parser.add_argument("--psi-amp", type=float, default=0.0)
    parser.add_argument("--psi-freq", type=float, default=0.0)
    parser.add_argument("--theta-amp", type=float, default=0.0)
    parser.add_argument("--theta-freq", type=float, default=0.0)

    # Mixer/coefficients
    parser.add_argument("--CT", type=float, default=1.0, help="T = CT * omega^2")
    parser.add_argument("--CQ", type=float, default=0.0, help="tau_drag = s * CQ * omega^2 (around rotor normal)")
    parser.add_argument("--omega2-max", type=float, default=None)
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

    # Motor
    parser.add_argument("--motor-tau", type=float, default=0.05)
    parser.add_argument("--omega2-rate", type=float, default=None)

    # Control
    parser.add_argument("--kp-att", type=float, default=0.15)
    parser.add_argument("--kd-att", type=float, default=0.02)
    parser.add_argument("--kp-att-xyz", type=float, nargs=3, default=None, metavar=("KPX", "KPY", "KPZ"), help="Axis-wise attitude P gains (roll,pitch,yaw). Overrides --kp-att.")
    parser.add_argument("--kd-att-xyz", type=float, nargs=3, default=None, metavar=("KDX", "KDY", "KDZ"), help="Axis-wise attitude D gains (roll,pitch,yaw). Overrides --kd-att.")
    parser.add_argument("--ki-att", type=float, default=0.0, help="Attitude integral gain (SO(3) error). Default: 0")
    parser.add_argument("--att-int-limit", type=float, default=1.0, help="Attitude integrator clamp (per-axis). Default: 1.0")
    parser.add_argument("--att-int-leak", type=float, default=0.0, help="Attitude integrator leak [1/s]. Default: 0")
    parser.add_argument("--kp-z", type=float, default=6.0)
    parser.add_argument("--kd-z", type=float, default=3.5)
    parser.add_argument("--yaw-des", type=float, default=0.0)
    parser.add_argument("--yaw-amp", type=float, default=0.0, help="Yaw sine amplitude [deg]. Default: 0")
    parser.add_argument("--yaw-freq", type=float, default=0.0, help="Yaw sine frequency [Hz]. Default: 0")
    parser.add_argument("--yaw-circle", action="store_true", help="When tracking xy-circle, set yaw to circle tangent heading.")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Yaw offset [deg] for yaw-circle. Default: 0")
    parser.add_argument("--yaw-tau", type=float, default=0.4, help="Yaw command first-order smoothing time constant [s]. Default: 0.4")
    parser.add_argument("--yaw-rate", type=float, default=120.0, help="Yaw command rate limit [deg/s]. Default: 120")
    parser.add_argument(
        "--tilt-abort-deg",
        type=float,
        default=80.0,
        help="If current tilt exceeds this, disable xy/yaw tracking and command level attitude (recovery). Default: 80",
    )
    parser.add_argument(
        "--torque-priority",
        type=str,
        default="rpy",
        choices=["rpy", "rp"],
        help="Torque priority under saturation. 'rp' deprioritizes yaw torque. Default: rpy",
    )
    parser.add_argument("--fallback-ridge", type=float, default=1e-6)
    parser.add_argument("--fallback-auto-omega2-factor", type=float, default=10.0)
    parser.add_argument("--fz-weight", type=float, default=10.0, help="Mixer weight for world-Z force tracking in desat solver. Default: 10")
    parser.add_argument("--tau-max", type=float, default=None, help="Optional clamp for attitude torque per-axis [N*m]. If omitted, auto-clamp from authority.")
    parser.add_argument(
        "--tau-max-factor",
        type=float,
        default=0.8,
        help="When --tau-max is omitted, tau_limit_axis = factor * sum_i |A_axis,i| * omega2_max. Default: 0.8",
    )
    # XY position hold (world frame)
    parser.add_argument("--xy-hold", action="store_true", help="Enable XY position hold (PD) toward (x_des,y_des).")
    parser.add_argument("--x-des", type=float, default=0.0)
    parser.add_argument("--y-des", type=float, default=0.0)
    parser.add_argument("--kp-xy", type=float, default=1.0)
    parser.add_argument("--kd-xy", type=float, default=1.2)
    parser.add_argument("--ki-xy", type=float, default=0.0, help="XY position integrator gain. Default: 0")
    parser.add_argument("--xy-int-limit", type=float, default=2.0, help="XY integrator clamp [m*s] per axis. Default: 2.0")
    parser.add_argument("--xy-int-leak", type=float, default=0.0, help="XY integrator leak [1/s]. Default: 0")
    parser.add_argument("--max-tilt-deg", type=float, default=25.0)
    parser.add_argument("--xy-circle", action="store_true", help="Track circular XY trajectory (enables xy-hold implicitly).")
    parser.add_argument("--xy-circle-radius", type=float, default=0.5)
    parser.add_argument("--xy-circle-freq", type=float, default=0.1)
    parser.add_argument("--xy-circle-cx", type=float, default=0.0)
    parser.add_argument("--xy-circle-cy", type=float, default=0.0)
    parser.add_argument("--xy-circle-phase", type=float, default=0.0)

    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--log-csv", type=str, default=None, help="Write per-step log to CSV (e.g. logs/urdf.csv).")
    parser.add_argument("--log-flush-every", type=int, default=10, help="CSV flush interval (rows). Default: 10")
    # Disturbances (world frame)
    # --dist-body t0 dur fx fy fz tx ty tz
    # --dist-rotor rotor_idx t0 dur fx fy fz tx ty tz   (can be repeated)
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
    parser.add_argument(
        "--dist-rotor",
        action="append",
        nargs=9,
        type=float,
        default=[],
        metavar=("IDX", "T0", "DUR", "FX", "FY", "FZ", "TX", "TY", "TZ"),
        help="Apply constant disturbance wrench to rotor IDX in WORLD frame. Repeatable.",
    )
    # White-noise disturbances (Gaussian, per-step)
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
    parser.add_argument(
        "--dist-rotor-noise",
        action="append",
        nargs=9,
        type=float,
        default=[],
        metavar=("IDX", "T0", "DUR", "SFX", "SFY", "SFZ", "STX", "STY", "STZ"),
        help="Rotor IDX white-noise disturbance stddev (WORLD) per step. Repeatable.",
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
    dt = 1.0 / hz  # control dt
    physics_hz = float(args.physics_hz)
    dt_phys = 1.0 / max(1.0, physics_hz)
    substeps = int(max(1, round(float(dt) / float(dt_phys))))

    env = PyBulletEnv(gui=bool(args.gui), time_step=dt_phys, gravity=float(args.gravity))
    try:
        p = env.p
        env.load_plane()
        env.load_body_urdf(str(args.urdf), base_pos=(0.0, 0.0, float(args.body_z)))
        env.configure_morphing_drone()
        env.set_damping_all(linear=float(args.lin_damping), angular=float(args.ang_damping))

        # rotor spin directions (+ - + -)
        spin_dir = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)

        mass = float(env.total_mass())
        st0 = env.get_state()
        z_des_alt = float(st0.pos[2]) if args.z_des is None else float(args.z_des)

        kp_att = (tuple(float(x) for x in args.kp_att_xyz) if args.kp_att_xyz is not None else float(args.kp_att))
        kd_att = (tuple(float(x) for x in args.kd_att_xyz) if args.kd_att_xyz is not None else float(args.kd_att))
        g_att = AttitudePDGains(kp=kp_att, kd=kd_att)
        g_z = AltitudePDGains(kp_z=float(args.kp_z), kd_z=float(args.kd_z))

        # Derive omega2_max from a typical drone thrust margin if user didn't specify it.
        # omega2 is ω^2, thrust = CT * omega2.
        if args.max_thrust_per_rotor is not None:
            omega2_max_auto = float(args.max_thrust_per_rotor) / max(1e-12, float(args.CT))
        else:
            tw = max(1.0, float(args.tw_ratio))
            weight = float(mass) * float(args.gravity)
            thrust_total_max = tw * weight
            thrust_per_rotor_max = thrust_total_max / 4.0
            omega2_max_auto = thrust_per_rotor_max / max(1e-12, float(args.CT))

        omega2_max_eff = (
            float(args.omega2_max)
            if args.omega2_max is not None
            else float(omega2_max_auto)
        )

        motor = MotorModel(
            tau=float(args.motor_tau),
            omega2_min=0.0,
            omega2_max=float(omega2_max_eff),
            omega2_rate_limit=(None if args.omega2_rate is None else float(args.omega2_rate)),
        )
        # Initialize motor state near hover to avoid free-fall at t=0 due to motor lag.
        try:
            r0_body, n0_body = env.rotor_geometry_body()
            st_init = env.get_state()
            R0_bw = quat_xyzw_to_rotmat(st_init.quat)
            n0_world = (R0_bw @ n0_body.T).T
            n0z_sum = float(np.sum(n0_world[:, 2]))
            n0z_sum = max(1e-6, n0z_sum)
            weight = float(mass) * float(args.gravity)
            omega2_hover = weight / (max(1e-12, float(args.CT)) * n0z_sum)
            motor.reset(np.full((4,), float(omega2_hover), dtype=float))
        except Exception:
            # Fallback: safe default
            motor.reset(np.zeros((4,), dtype=float))

        def morph_angles(t: float) -> tuple[float, float, float]:
            phi = float(args.phi) + float(args.phi_amp) * float(np.sin(2.0 * np.pi * float(args.phi_freq) * float(t)))
            psi = float(args.psi) + float(args.psi_amp) * float(np.sin(2.0 * np.pi * float(args.psi_freq) * float(t)))
            theta = float(args.theta) + float(args.theta_amp) * float(np.sin(2.0 * np.pi * float(args.theta_freq) * float(t)))
            return phi, psi, theta

        if args.gui:
            print("[info] GUI mode: running in real-time-ish (sleep)")
        print(f"[info] urdf={args.urdf} mass_total={mass:.4f}kg z_des={z_des_alt:.3f} yaw_des={float(args.yaw_des):.1f}deg")
        print(f"[info] gains att(kp,kd)=({g_att.kp},{g_att.kd}) z(kp,kd)=({g_z.kp_z},{g_z.kd_z})")
        print(
            f"[info] coeff CT={float(args.CT):.3f} CQ={float(args.CQ):.3f} motor_tau={float(args.motor_tau):.3f} "
            f"omega2_max_eff={omega2_max_eff:.3g}"
        )

        csv_logger = None
        if args.log_csv:
            fieldnames = [
                "step",
                "t",
                "phi_cmd_deg",
                "psi_cmd_deg",
                "theta_cmd_deg",
                "phi_applied_deg",
                "psi_applied_deg",
                "theta_applied_deg",
                "x",
                "y",
                "z",
                "yaw_deg",
                "yaw_cmd_deg",
                "x_des",
                "y_des",
                "vx",
                "vy",
                "vz",
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
                "nz_min",
                "nz_sum",
                "tilt_deg",
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

        n_steps = max(1, int(float(args.seconds) * hz))
        rng = np.random.default_rng(int(args.noise_seed))
        # Yaw command smoothing state (deg)
        yaw_filt_deg = float(args.yaw_des)
        # Attitude integrator state (body frame)
        e_int = np.zeros((3,), dtype=float)
        # XY integrator state (world frame, integrates position error)
        xy_int = np.zeros((2,), dtype=float)
        for k in range(n_steps):
            t = float(k) * dt
            # snapshot integrators for anti-windup rollback
            e_int_prev = e_int.copy()
            xy_int_prev = xy_int.copy()
            phi_k, psi_k, theta_k = morph_angles(t)
            phi_a, psi_a, theta_a = env.set_morph_angles_servo(
                phi_deg=phi_k,
                psi_deg=psi_k,
                theta_deg=theta_k,
                dt=dt,
                symmetry=str(args.symmetry),
                tau=float(args.morph_tau),
                rate_limit=(None if args.morph_rate is None else float(args.morph_rate)),
            )

            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)
            if k == 0:
                yaw_filt_deg = _yaw_deg_from_R_bw(R_bw)

            if bool(args.xy_circle):
                x_des, y_des = _xy_circle_target(
                    t=t,
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

            # URDFからr,nを取得して allocation を作る。
            # NOTE: rotor tilt morphingでは、wrenchとallocationのフレームを混ぜると破綻しやすい。
            # ここでは WORLD フレームで統一する:
            #   u_world = [Fz_world, tau_x_world, tau_y_world, tau_z_world]
            r_body, n_body = env.rotor_geometry_body()
            r_world = (R_bw @ r_body.T).T
            n_world = (R_bw @ n_body.T).T
            # Physical guard: for world-Z force tracking, only rotors with positive world-Z component can contribute
            # to upward force. If a rotor points downward (n_z < 0), its "upward" contribution is 0.
            n_world_for_fz = n_world.copy()
            n_world_for_fz[:, 2] = np.maximum(0.0, n_world_for_fz[:, 2])
            A = build_allocation_matrix(r_body=r_world, n_body=n_world_for_fz, C_T=float(args.CT), C_Q=float(args.CQ), spin_dir=spin_dir)
            # diagnostics
            nz_min = float(np.min(n_world[:, 2]))
            nz_sum = float(np.sum(n_world[:, 2]))
            # tilt: angle between body-z and world-z
            tilt_deg = float(np.rad2deg(np.arccos(np.clip(float(R_bw[2, 2]), -1.0, 1.0))))

            # Control: desired attitude (level + yaw)
            # Yaw command: fixed + optional sine, or circle tangent heading.
            if bool(args.yaw_circle) and bool(args.xy_circle):
                yaw_cmd_deg = _yaw_circle_tangent_deg(
                    t=t,
                    radius=float(args.xy_circle_radius),
                    freq_hz=float(args.xy_circle_freq),
                    phase_deg=float(args.xy_circle_phase),
                    yaw_offset_deg=float(args.yaw_offset),
                )
            else:
                yaw_cmd_deg = float(args.yaw_des) + float(args.yaw_amp) * float(np.sin(2.0 * np.pi * float(args.yaw_freq) * float(t)))

            # If CQ==0, yaw is effectively uncontrollable; avoid injecting yaw error into SO(3) PD
            # by tracking current yaw (reduced attitude control).
            if abs(float(args.CQ)) <= 0.0:
                yaw_cmd_deg = _yaw_deg_from_R_bw(R_bw)

            # Smooth + rate-limit yaw command to avoid large instantaneous attitude errors.
            # Use a consistent formulation: yaw_dot = (yaw_cmd - yaw_filt)/tau, then clamp yaw_dot by yaw_rate.
            dy = _wrap_deg(float(yaw_cmd_deg) - float(yaw_filt_deg))
            tau = max(1e-6, float(args.yaw_tau))
            yaw_dot = dy / tau  # [deg/s]
            yaw_rate = max(0.0, float(args.yaw_rate))
            if yaw_rate > 0.0:
                yaw_dot = float(np.clip(yaw_dot, -yaw_rate, +yaw_rate))
            yaw_filt_deg = float(yaw_filt_deg) + float(yaw_dot) * float(dt)
            yaw_cmd_deg = float(yaw_filt_deg)
            # Safety: if tilt is already large, stop tracking xy/yaw and try to recover to level.
            tilt_abort = max(0.0, float(args.tilt_abort_deg))
            if tilt_abort > 0.0 and float(tilt_deg) > float(tilt_abort):
                xy_hold = False
                e_int[:] = 0.0
                xy_int[:] = 0.0
                yaw_cmd_deg = _yaw_deg_from_R_bw(R_bw)
            if xy_hold:
                ex = float(x_des) - float(st.pos[0])
                ey = float(y_des) - float(st.pos[1])
                ki_xy = float(args.ki_xy)
                # leak
                leak_xy = max(0.0, float(args.xy_int_leak))
                if leak_xy > 0.0:
                    xy_int = xy_int * max(0.0, 1.0 - leak_xy * float(dt))
                if abs(ki_xy) > 0.0:
                    xy_int = xy_int + np.array([ex, ey], dtype=float) * float(dt)
                    lim = max(0.0, float(args.xy_int_limit))
                    if lim > 0.0:
                        xy_int = np.clip(xy_int, -lim, +lim)
                else:
                    xy_int[:] = 0.0
                ax_cmd = float(args.kp_xy) * ex + float(args.kd_xy) * (0.0 - float(st.vel[0]))
                ay_cmd = float(args.kp_xy) * ey + float(args.kd_xy) * (0.0 - float(st.vel[1]))
                if abs(ki_xy) > 0.0:
                    ax_cmd += ki_xy * float(xy_int[0])
                    ay_cmd += ki_xy * float(xy_int[1])
                g = float(args.gravity)
                z_dir_des = np.array([ax_cmd / max(1e-6, g), ay_cmd / max(1e-6, g), 1.0], dtype=float)
                max_tilt = np.deg2rad(float(args.max_tilt_deg))
                xy_norm = float(np.linalg.norm(z_dir_des[:2]))
                if xy_norm > 1e-12:
                    tan_tilt = xy_norm / max(1e-6, float(z_dir_des[2]))
                    if tan_tilt > np.tan(max_tilt):
                        scale = np.tan(max_tilt) / tan_tilt
                        z_dir_des[0] *= scale
                        z_dir_des[1] *= scale
                z_dir_des = z_dir_des / max(1e-12, float(np.linalg.norm(z_dir_des)))
                R_des = rotation_from_z_and_yaw(z_world=z_dir_des, yaw_deg=yaw_cmd_deg)
            else:
                R_des = rot_z_deg(yaw_cmd_deg)
            # Attitude PI(D): PD from existing function + optional integral on SO(3) error.
            tau_b = attitude_pd_torque_body(R_bw=R_bw, ang_vel_world=st.ang_vel, R_des_bw=R_des, gains=g_att)
            ki_att = float(args.ki_att)
            if abs(ki_att) > 0.0:
                # leak
                leak_att = max(0.0, float(args.att_int_leak))
                if leak_att > 0.0:
                    e_int = e_int * max(0.0, 1.0 - leak_att * float(dt))
                # Recompute e_R consistent with attitude_pd_torque_body
                R_err = R_des.T @ R_bw
                e_R = 0.5 * np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]], dtype=float)
                e_int = e_int + e_R * float(dt)
                lim = max(0.0, float(args.att_int_limit))
                if lim > 0.0:
                    e_int = np.clip(e_int, -lim, +lim)
                tau_b = np.asarray(tau_b, dtype=float) - ki_att * e_int
            # Clamp torque command to what the current geometry + omega2_max can deliver (per-axis, conservative).
            if args.tau_max is None:
                tau_cap = float(args.tau_max_factor) * float(omega2_max_eff) * np.sum(np.abs(A[1:4, :]), axis=1)
                tau_b = np.clip(np.asarray(tau_b, dtype=float), -tau_cap, tau_cap)
            else:
                tau_lim = float(args.tau_max)
                tau_b = np.clip(np.asarray(tau_b, dtype=float), -tau_lim, tau_lim)
            # Use world-Z force command to remain valid under morphing (rotor normals tilt).
            Fz_b = altitude_pd_Fz_world(
                z_world=float(st.pos[2]),
                vz_world=float(st.vel[2]),
                z_des_world=z_des_alt,
                mass=mass,
                gravity=float(args.gravity),
                gains=g_z,
                Fz_min=0.0,
                Fz_max=None,
            )
            tau_w = R_bw @ np.asarray(tau_b, dtype=float).reshape(3)
            u = np.array([float(Fz_b), float(tau_w[0]), float(tau_w[1]), float(tau_w[2])], dtype=float)

            mix = solve_mixer_with_fallback(
                A=A,
                wrench_target=u,
                omega2_min=0.0,
                omega2_max=float(omega2_max_eff),
                torque_weights=((1.0, 1.0, 1.0) if str(args.torque_priority) == "rpy" else (1.0, 1.0, 0.0)),
                fz_weight=float(args.fz_weight),
                fallback_ridge=float(args.fallback_ridge),
                fallback_auto_omega2_factor=float(args.fallback_auto_omega2_factor),
            )
            # Anti-windup: if any rotor hits bounds, rollback this step's integrator accumulation.
            hit_bound = bool(np.any(mix.omega2_cmd <= 1e-9) or np.any(mix.omega2_cmd >= float(omega2_max_eff) - 1e-9))
            if hit_bound:
                e_int = e_int_prev
                xy_int = xy_int_prev
            # --- Disturbances/noise are applied every physics step (PyBullet clears external forces each step) ---
            body_force_w = np.zeros((3,), dtype=float)
            body_tau_w = np.zeros((3,), dtype=float)
            if args.dist_body is not None:
                t0, dur, fx, fy, fz, tx, ty, tz = [float(x) for x in args.dist_body]
                if float(t0) <= float(t) < float(t0) + float(dur):
                    f = np.array([fx, fy, fz], dtype=float)
                    tau = np.array([tx, ty, tz], dtype=float)
                    if str(args.dist_body_frame) == "body":
                        f = R_bw @ f
                        tau = R_bw @ tau
                    body_force_w += f
                    body_tau_w += tau

            # Body white-noise (sampled at control rate, held over substeps)
            body_noise_f = np.zeros((3,), dtype=float)
            body_noise_tau = np.zeros((3,), dtype=float)
            if args.dist_body_noise is not None:
                t0 = float(args.dist_body_noise_t0)
                dur = float(args.dist_body_noise_dur)
                if t0 <= float(t) < t0 + dur:
                    sfx, sfy, sfz, stx, sty, stz = [float(x) for x in args.dist_body_noise]
                    f = rng.normal(loc=0.0, scale=[sfx, sfy, sfz])
                    tau = rng.normal(loc=0.0, scale=[stx, sty, stz])
                    if str(args.dist_body_noise_frame) == "body":
                        f = R_bw @ np.asarray(f, dtype=float).reshape(3)
                        tau = R_bw @ np.asarray(tau, dtype=float).reshape(3)
                    body_noise_f = np.asarray(f, dtype=float).reshape(3)
                    body_noise_tau = np.asarray(tau, dtype=float).reshape(3)

            # Rotor disturbances/noise (sample at control rate, apply each physics step)
            rotor_force_w = np.zeros((4, 3), dtype=float)
            rotor_tau_w = np.zeros((4, 3), dtype=float)
            for spec in list(args.dist_rotor):
                idx, t0, dur, fx, fy, fz, tx, ty, tz = [float(x) for x in spec]
                if float(t0) <= float(t) < float(t0) + float(dur):
                    rotor_force_w[int(idx), :] += np.array([fx, fy, fz], dtype=float)
                    rotor_tau_w[int(idx), :] += np.array([tx, ty, tz], dtype=float)
            for spec in list(args.dist_rotor_noise):
                idx, t0, dur, sfx, sfy, sfz, stx, sty, stz = [float(x) for x in spec]
                if float(t0) <= float(t) < float(t0) + float(dur):
                    f = rng.normal(loc=0.0, scale=[sfx, sfy, sfz])
                    tau = rng.normal(loc=0.0, scale=[stx, sty, stz])
                    rotor_force_w[int(idx), :] += np.asarray(f, dtype=float).reshape(3)
                    rotor_tau_w[int(idx), :] += np.asarray(tau, dtype=float).reshape(3)

            # Thrust fluctuation noise (sampled at control rate, held over substeps)
            thrust_sigma = max(0.0, float(args.thrust_noise_sigma))
            tn_t0 = float(args.thrust_noise_t0)
            tn_dur = float(args.thrust_noise_dur)
            thrust_mul = np.ones((4,), dtype=float)
            if thrust_sigma > 0.0 and (tn_t0 <= float(t) < tn_t0 + tn_dur):
                for i in range(4):
                    m = 1.0 + float(rng.normal(loc=0.0, scale=thrust_sigma))
                    thrust_mul[i] = max(0.0, m)

            # Physics substeps: apply forces each step
            omega2_act = np.asarray(mix.omega2_cmd, dtype=float).reshape(4)
            for _ in range(int(substeps)):
                # Keep URDF joint angles "hard-set" at physics rate.
                # (resetJointState is kinematic; without re-applying each physics tick, joints can drift under thrust,
                # which makes behavior depend strongly on control rate vs physics rate.)
                env.set_morph_angles(phi_deg=phi_a, psi_deg=psi_a, theta_deg=theta_a, symmetry=str(args.symmetry))
                omega2_act = motor.step(mix.omega2_cmd, dt_phys)

                for i in range(4):
                    thrust = float(args.CT) * float(omega2_act[i]) * float(thrust_mul[i])
                    tau_react = float(spin_dir[i]) * float(args.CQ) * float(omega2_act[i]) * float(thrust_mul[i])
                    env.apply_rotor_thrust_link_frame(rotor_idx=i, thrust=thrust, reaction_torque=tau_react)

                    if np.linalg.norm(rotor_force_w[i, :]) > 0.0 or np.linalg.norm(rotor_tau_w[i, :]) > 0.0:
                        env.apply_rotor_wrench_world(
                            rotor_idx=int(i),
                            force_world=(float(rotor_force_w[i, 0]), float(rotor_force_w[i, 1]), float(rotor_force_w[i, 2])),
                            torque_world=(float(rotor_tau_w[i, 0]), float(rotor_tau_w[i, 1]), float(rotor_tau_w[i, 2])),
                        )

                f = body_force_w + body_noise_f
                tau = body_tau_w + body_noise_tau
                if np.linalg.norm(f) > 0.0 or np.linalg.norm(tau) > 0.0:
                    env.apply_body_wrench_world(force_world=(float(f[0]), float(f[1]), float(f[2])), torque_world=(float(tau[0]), float(tau[1]), float(tau[2])))

                env.step(1)

            # Refresh state after physics substeps (for logging/printing)
            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)
            # refresh diagnostics for log
            _r_body, _n_body = env.rotor_geometry_body()
            _n_world = (R_bw @ _n_body.T).T
            nz_min = float(np.min(_n_world[:, 2]))
            nz_sum = float(np.sum(_n_world[:, 2]))
            tilt_deg = float(np.rad2deg(np.arccos(np.clip(float(R_bw[2, 2]), -1.0, 1.0))))

            if csv_logger is not None:
                row = {
                    "step": int(k),
                    "t": float(t),
                    "phi_cmd_deg": float(phi_k),
                    "psi_cmd_deg": float(psi_k),
                    "theta_cmd_deg": float(theta_k),
                    "phi_applied_deg": float(phi_a),
                    "psi_applied_deg": float(psi_a),
                    "theta_applied_deg": float(theta_a),
                    "x": float(st.pos[0]),
                    "y": float(st.pos[1]),
                    "z": float(st.pos[2]),
                    "yaw_deg": float(_yaw_deg_from_R_bw(R_bw)),
                    "yaw_cmd_deg": float(yaw_cmd_deg),
                    "x_des": float(x_des),
                    "y_des": float(y_des),
                    "vx": float(st.vel[0]),
                    "vy": float(st.vel[1]),
                    "vz": float(st.vel[2]),
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
                    "nz_min": float(nz_min),
                    "nz_sum": float(nz_sum),
                    "tilt_deg": float(tilt_deg),
                    "omega2_cmd_0": float(mix.omega2_cmd[0]),
                    "omega2_cmd_1": float(mix.omega2_cmd[1]),
                    "omega2_cmd_2": float(mix.omega2_cmd[2]),
                    "omega2_cmd_3": float(mix.omega2_cmd[3]),
                    "omega2_act_0": float(omega2_act[0]),
                    "omega2_act_1": float(omega2_act[1]),
                    "omega2_act_2": float(omega2_act[2]),
                    "omega2_act_3": float(omega2_act[3]),
                }
                csv_logger.write(row)

            if bool(args.gui):
                time.sleep(dt)

            if int(args.log_every) > 0 and (k % int(args.log_every) == 0):
                print(
                    f"[step {k:06d}] t={t:.2f} phi/psi/theta=({phi_k:+.1f},{psi_k:+.1f},{theta_k:+.1f}) "
                    f"z={float(st.pos[2]):+.3f} vz={float(st.vel[2]):+.3f} "
                    f"Fz_b={Fz_b:+.3f} tau_b={tau_b} res={mix.residual}"
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


