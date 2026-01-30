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
)
from sim.control.mixer import build_allocation_matrix, solve_mixer_with_fallback
from sim.control.motor_model import MotorModel
from sim.env.pybullet_env import PyBulletEnv
from sim.logger.csv_logger import CsvLogger


def _wrap_deg(d: float) -> float:
    return float((float(d) + 180.0) % 360.0 - 180.0)


def _yaw_from_vel_world_deg(vx: float, vy: float) -> float | None:
    v2 = float(vx) * float(vx) + float(vy) * float(vy)
    if v2 <= 1e-12:
        return None
    return float(np.degrees(np.arctan2(float(vy), float(vx))))


def _has_contact(*, p, body_a: int, body_b: int | None = None, min_normal_force: float = 0.0) -> bool:
    """
    Return True if there is any contact point between body_a and (optionally) body_b
    with normalForce >= min_normal_force.
    """
    min_nf = max(0.0, float(min_normal_force))
    pts = p.getContactPoints(bodyA=int(body_a)) if body_b is None else p.getContactPoints(bodyA=int(body_a), bodyB=int(body_b))
    if not pts:
        return False
    if min_nf <= 0.0:
        return True
    # PyBullet contact tuple: normalForce is at index 9
    for pt in pts:
        try:
            if float(pt[9]) >= min_nf:
                return True
        except Exception:
            continue
    return False


def _create_wall(*, p, wall_x: float, half_extents: tuple[float, float, float], rgba=(0.7, 0.7, 0.7, 1.0)):
    hx, hy, hz = (float(half_extents[0]), float(half_extents[1]), float(half_extents[2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(rgba))
    wall_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[float(wall_x), 0.0, float(hz)])
    return int(wall_id)


def _create_side_wall(
    *,
    p,
    x0: float,
    x1: float,
    y: float,
    half_thickness_y: float,
    half_height_z: float,
    rgba=(0.7, 0.7, 0.7, 1.0),
):
    """
    Create a wall parallel to +X (a side wall at y=const), spanning x in [x0,x1].
    """
    x0 = float(min(x0, x1))
    x1 = float(max(x0, x1))
    cx = 0.5 * (x0 + x1)
    hx = 0.5 * (x1 - x0)
    hy = float(half_thickness_y)
    hz = float(half_height_z)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(rgba))
    wall_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[cx, float(y), float(hz)])
    return int(wall_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="Wall effect demo: forward flight into a wall area, apply wall-induced moment as disturbance.")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--win-width", type=int, default=None, help="(GUI) Window width in pixels. Default: unset")
    ap.add_argument("--win-height", type=int, default=None, help="(GUI) Window height in pixels. Default: unset")
    ap.add_argument("--record-mp4", type=str, default=None, help="Record an MP4 video to this path using PyBullet state logging.")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--hz", type=float, default=240.0)
    ap.add_argument("--physics-hz", type=float, default=240.0)

    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--lin-damping", type=float, default=0.05)
    ap.add_argument("--ang-damping", type=float, default=0.05)

    ap.add_argument("--urdf", type=str, default="assets/urdf/morphing_drone.urdf")
    ap.add_argument("--body-z", type=float, default=0.30)
    ap.add_argument("--z-des", type=float, default=None)

    # Initial morphing (URDF)
    ap.add_argument(
        "--morph-seconds",
        type=float,
        default=0.0,
        help="Ramp morph angles from start->target over this many seconds at the beginning. 0 disables. Default: 0",
    )
    ap.add_argument(
        "--morph-start",
        type=float,
        default=0.0,
        help="Start time [s] of the morph ramp. Before this, hold start angles. Default: 0",
    )
    ap.add_argument("--phi-start", type=float, default=0.0, help="Initial fold angle phi [deg]. Default: 0")
    ap.add_argument("--psi-start", type=float, default=0.0, help="Initial slant angle psi [deg]. Default: 0")
    ap.add_argument("--theta-start", type=float, default=0.0, help="Initial tilt angle theta [deg]. Default: 0")
    ap.add_argument("--phi-deg", type=float, default=0.0, help="Target fold angle phi [deg]. Default: 0")
    ap.add_argument("--psi-deg", type=float, default=0.0, help="Target slant angle psi [deg]. Default: 0")
    ap.add_argument("--theta-deg", type=float, default=0.0, help="Target tilt angle theta [deg]. Default: 0")
    ap.add_argument(
        "--morph-symmetry",
        type=str,
        default="mirror_xy",
        choices=["none", "mirror_xy"],
        help="Morph symmetry mapping for arms. Default: mirror_xy",
    )

    # Flight plan: forward ramp in +X (world)
    ap.add_argument("--x-vel", type=float, default=0.5, help="Forward desired speed [m/s]. Default: 0.5")
    ap.add_argument("--x-max", type=float, default=6.0, help="Max x_des [m] (clamp). Default: 6.0")
    ap.add_argument("--y-des", type=float, default=0.0)
    ap.add_argument("--max-tilt-deg", type=float, default=20.0)
    ap.add_argument("--kp-xy", type=float, default=1.0)
    ap.add_argument("--kd-xy", type=float, default=1.2)

    # Yaw (keep forward)
    ap.add_argument("--yaw-des", type=float, default=0.0)
    ap.add_argument("--yaw-tau", type=float, default=0.4)
    ap.add_argument("--yaw-rate", type=float, default=120.0)

    # Attitude gains (axis-wise supported)
    ap.add_argument("--kp-att", type=float, default=0.15)
    ap.add_argument("--kd-att", type=float, default=0.02)
    ap.add_argument("--kp-att-xyz", type=float, nargs=3, default=None, metavar=("KPX", "KPY", "KPZ"))
    ap.add_argument("--kd-att-xyz", type=float, nargs=3, default=None, metavar=("KDX", "KDY", "KDZ"))
    ap.add_argument("--kp-z", type=float, default=6.0)
    ap.add_argument("--kd-z", type=float, default=3.5)

    # Mixer/coefficients
    ap.add_argument("--CT", type=float, default=1.0)
    ap.add_argument("--CQ", type=float, default=0.08)
    ap.add_argument("--tw-ratio", type=float, default=4.0)
    ap.add_argument("--omega2-max", type=float, default=None)
    ap.add_argument("--torque-priority", type=str, default="rp", choices=["rpy", "rp"])
    ap.add_argument("--fallback-ridge", type=float, default=1e-6)
    ap.add_argument("--fallback-auto-omega2-factor", type=float, default=10.0)
    ap.add_argument("--fz-weight", type=float, default=10.0)

    # Motor
    ap.add_argument("--motor-tau", type=float, default=0.05)
    ap.add_argument("--omega2-rate", type=float, default=None)

    # Wall geometry
    ap.add_argument(
        "--wall-orient",
        type=str,
        default="side",
        choices=["side", "front"],
        help="Wall orientation relative to flight direction +X. 'side' is parallel to motion (y=const). Default: side",
    )
    # legacy/front wall params (plane normal +X)
    ap.add_argument("--wall-x", type=float, default=5.0, help="(front wall) wall position x [m]. Default: 5.0")
    ap.add_argument("--wall-half-x", type=float, default=0.05, help="(front wall) thickness in x [m]. Default: 0.05")
    ap.add_argument("--wall-half-y", type=float, default=2.0, help="(front wall) half width in y [m]. Default: 2.0")
    ap.add_argument("--wall-half-z", type=float, default=1.0, help="Wall half height in z [m]. Default: 1.0")
    # side wall params (parallel to +X)
    ap.add_argument("--wall-y", type=float, default=1.0, help="(side wall) wall position y [m]. Default: 1.0")
    ap.add_argument("--wall-x0", type=float, default=None, help="(side wall) wall start x [m]. Default: wall-zone-x")
    ap.add_argument("--wall-len-x", type=float, default=4.0, help="(side wall) wall length along x [m]. Default: 4.0")
    ap.add_argument("--wall-thickness", type=float, default=0.1, help="(side wall) wall thickness in y [m]. Default: 0.1")

    # Wall effect "area" + moment model
    ap.add_argument("--wall-zone-x", type=float, default=3.0, help="Enable wall-effect model once x >= this. Default: 3.0")
    ap.add_argument("--wall-range", type=float, default=2.0, help="Effect active when distance d < range. Default: 2.0")
    ap.add_argument("--wall-d0", type=float, default=0.3, help="Distance softening [m] in denom: d^2+d0^2. Default: 0.3")
    ap.add_argument("--wall-k", type=float, default=0.02, help="Wall moment gain. tau ~ k*v^2/(d^2+d0^2). Default: 0.02")
    ap.add_argument(
        "--wall-axis",
        type=str,
        default="roll",
        choices=["roll", "pitch", "yaw"],
        help="Which body-axis torque to apply as wall-effect moment. Default: pitch",
    )
    ap.add_argument("--wall-tau-max", type=float, default=0.2, help="Clamp |tau_wall| [N*m]. Default: 0.2")

    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--log-flush-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=120)

    # Camera (GUI)
    ap.add_argument("--cam-follow", action="store_true", help="In GUI, keep camera following the drone base.")
    ap.add_argument("--cam-dist", type=float, default=2.5, help="Camera distance. Default: 2.5")
    ap.add_argument("--cam-yaw", type=float, default=60.0, help="Camera yaw [deg]. Default: 60")
    ap.add_argument(
        "--cam-yaw-mode",
        type=str,
        default="fixed",
        choices=["fixed", "behind", "behind_vel"],
        help=(
            "Camera yaw mode when --cam-follow is enabled. "
            "fixed: use --cam-yaw (world). "
            "behind: follow drone yaw + 180. "
            "behind_vel: follow velocity heading + 180 (behind direction of travel). "
            "Default: fixed"
        ),
    )
    ap.add_argument(
        "--cam-yaw-offset",
        type=float,
        default=0.0,
        help="Additional yaw offset [deg] when --cam-yaw-mode behind/behind_vel. Default: 0",
    )
    ap.add_argument(
        "--cam-vel-min",
        type=float,
        default=0.15,
        help="(behind_vel) Min XY speed [m/s] to use velocity heading; otherwise fall back to drone yaw. Default: 0.15",
    )
    ap.add_argument(
        "--cam-heading-tau",
        type=float,
        default=0.25,
        help="(behind_vel) Low-pass time constant [s] for camera heading to reduce jitter. Default: 0.25",
    )
    ap.add_argument("--cam-pitch", type=float, default=-25.0, help="Camera pitch [deg]. Default: -25")
    ap.add_argument("--cam-target-z", type=float, default=0.3, help="Camera target z offset [m] when following. Default: 0.3")
    ap.add_argument(
        "--cam-target-body",
        type=float,
        nargs=3,
        default=None,
        metavar=("TX", "TY", "TZ"),
        help="If set (and --cam-follow), use a body-frame target offset [m] instead of (x,y,cam-target-z).",
    )
    ap.add_argument(
        "--cam-freeze-on-contact",
        action="store_true",
        help="If set (and --cam-follow), stop updating the camera once the drone contacts wall/ground (prevents camera jitter after a crash).",
    )
    ap.add_argument(
        "--cam-contact-min-force",
        type=float,
        default=1.0,
        help="Min normal contact force [N] to trigger camera freeze. Default: 1.0",
    )

    args = ap.parse_args()

    hz = float(args.hz)
    dt = 1.0 / max(1.0, hz)
    physics_hz = float(args.physics_hz)
    dt_phys = 1.0 / max(1.0, physics_hz)
    substeps = int(max(1, round(float(dt) / float(dt_phys))))

    env = PyBulletEnv(
        gui=bool(args.gui),
        time_step=dt_phys,
        gravity=float(args.gravity),
        gui_width=args.win_width,
        gui_height=args.win_height,
    )
    csv_logger = None
    video_log_id = None
    try:
        p = env.p
        plane_id = int(env.load_plane())
        env.load_body_urdf(str(args.urdf), base_pos=(0.0, 0.0, float(args.body_z)))
        env.configure_morphing_drone()
        env.set_damping_all(linear=float(args.lin_damping), angular=float(args.ang_damping))

        # Video recording (MP4)
        if args.record_mp4:
            video_log_id = env.start_video_recording(str(args.record_mp4))
            if video_log_id is None:
                print("[warn] MP4 recording not supported in this PyBullet build/backend.")
            else:
                print(f"[info] Recording MP4: {args.record_mp4}")

        # Set initial morph pose (also helps avoid an initial 'snap' later)
        env.set_morph_angles(
            phi_deg=float(args.phi_start),
            psi_deg=float(args.psi_start),
            theta_deg=float(args.theta_start),
            symmetry=str(args.morph_symmetry),
        )

        wall_id = (
            _create_wall(
                p=p,
                wall_x=float(args.wall_x),
                half_extents=(float(args.wall_half_x), float(args.wall_half_y), float(args.wall_half_z)),
            )
            if str(args.wall_orient) == "front"
            else _create_side_wall(
                p=p,
                x0=(float(args.wall_zone_x) if args.wall_x0 is None else float(args.wall_x0)),
                x1=(float(args.wall_zone_x) if args.wall_x0 is None else float(args.wall_x0)) + float(args.wall_len_x),
                y=float(args.wall_y),
                half_thickness_y=0.5 * float(args.wall_thickness),
                half_height_z=float(args.wall_half_z),
            )
        )

        spin_dir = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)

        mass = float(env.total_mass())
        st0 = env.get_state()
        z_des = float(st0.pos[2]) if args.z_des is None else float(args.z_des)

        kp_att = (tuple(float(x) for x in args.kp_att_xyz) if args.kp_att_xyz is not None else float(args.kp_att))
        kd_att = (tuple(float(x) for x in args.kd_att_xyz) if args.kd_att_xyz is not None else float(args.kd_att))
        g_att = AttitudePDGains(kp=kp_att, kd=kd_att)
        g_z = AltitudePDGains(kp_z=float(args.kp_z), kd_z=float(args.kd_z))

        # omega2_max
        if args.omega2_max is not None:
            omega2_max_eff = float(args.omega2_max)
        else:
            tw = max(1.0, float(args.tw_ratio))
            weight = float(mass) * float(args.gravity)
            omega2_max_eff = (tw * weight / 4.0) / max(1e-12, float(args.CT))

        motor = MotorModel(
            tau=float(args.motor_tau),
            omega2_min=0.0,
            omega2_max=float(omega2_max_eff),
            omega2_rate_limit=(None if args.omega2_rate is None else float(args.omega2_rate)),
        )

        # initialize motor near hover
        try:
            r0_body, n0_body = env.rotor_geometry_body()
            st_init = env.get_state()
            R0_bw = quat_xyzw_to_rotmat(st_init.quat)
            n0_world = (R0_bw @ n0_body.T).T
            n0z_sum = max(1e-6, float(np.sum(n0_world[:, 2])))
            weight = float(mass) * float(args.gravity)
            omega2_hover = weight / (max(1e-12, float(args.CT)) * n0z_sum)
            motor.reset(np.full((4,), float(omega2_hover), dtype=float))
        except Exception:
            motor.reset(np.zeros((4,), dtype=float))

        if args.log_csv:
            fieldnames = [
                "step",
                "t",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "yaw_deg",
                "yaw_cmd_deg",
                "x_des",
                "y_des",
                "Fz_cmd",
                "tau_x_cmd_b",
                "tau_y_cmd_b",
                "tau_z_cmd_b",
                "mix_mode",
                "omega2_cmd_0",
                "omega2_cmd_1",
                "omega2_cmd_2",
                "omega2_cmd_3",
                "omega2_act_0",
                "omega2_act_1",
                "omega2_act_2",
                "omega2_act_3",
                "d_wall",
                "wall_active",
                "tau_wall_x_b",
                "tau_wall_y_b",
                "tau_wall_z_b",
            ]
            csv_logger = CsvLogger(path=str(args.log_csv), fieldnames=fieldnames, flush_every=int(args.log_flush_every))
            csv_logger.open()
            print(f"[info] CSV logging enabled: {args.log_csv}")

        n_steps = max(1, int(float(args.seconds) * hz))
        yaw_filt_deg = float(args.yaw_des)

        if bool(args.gui):
            print("[info] GUI mode: running in real-time-ish (sleep)")
        if str(args.wall_orient) == "front":
            print(f"[info] mass_total={mass:.4f}kg z_des={z_des:.3f} wall(front) x={float(args.wall_x):.2f} wall_zone_x={float(args.wall_zone_x):.2f}")
        else:
            wall_x0_eff = float(args.wall_zone_x) if args.wall_x0 is None else float(args.wall_x0)
            print(
                f"[info] mass_total={mass:.4f}kg z_des={z_des:.3f} wall(side) y={float(args.wall_y):.2f} "
                f"x∈[{wall_x0_eff:.2f},{(wall_x0_eff+float(args.wall_len_x)):.2f}] wall_zone_x={float(args.wall_zone_x):.2f}"
            )
        if float(args.morph_seconds) > 0.0:
            print(
                f"[info] initial morph ramp: {float(args.morph_seconds):.2f}s "
                f"(start t={float(args.morph_start):.2f}s, "
                f"(phi {float(args.phi_start):+.1f}->{float(args.phi_deg):+.1f} deg, "
                f"psi {float(args.psi_start):+.1f}->{float(args.psi_deg):+.1f} deg, "
                f"theta {float(args.theta_start):+.1f}->{float(args.theta_deg):+.1f} deg, "
                f"sym={str(args.morph_symmetry)})"
            )

        # Camera heading filter (deg)
        cam_yaw_filt_deg = float(args.cam_yaw)
        cam_frozen = False

        for k in range(n_steps):
            t = float(k) * dt
            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)

            if bool(args.gui) and bool(args.cam_follow):
                if bool(args.cam_freeze_on_contact) and (not cam_frozen):
                    # Freeze camera after first meaningful contact with ground or wall.
                    if _has_contact(p=p, body_a=env.body_id, body_b=plane_id, min_normal_force=float(args.cam_contact_min_force)) or _has_contact(
                        p=p, body_a=env.body_id, body_b=wall_id, min_normal_force=float(args.cam_contact_min_force)
                    ):
                        cam_frozen = True

                if cam_frozen:
                    # Stop updating camera (leave it at the last set position) to avoid jitter.
                    pass
                else:
                # camera target
                    if args.cam_target_body is not None:
                        tb = np.array([float(args.cam_target_body[0]), float(args.cam_target_body[1]), float(args.cam_target_body[2])], dtype=float)
                        tw = np.asarray(st.pos, dtype=float).reshape(3) + (R_bw @ tb.reshape(3))
                        target = (float(tw[0]), float(tw[1]), float(tw[2]))
                    else:
                        target = (float(st.pos[0]), float(st.pos[1]), float(args.cam_target_z))

                # camera yaw
                    cam_yaw = float(args.cam_yaw)
                    if str(args.cam_yaw_mode) == "behind":
                        yaw_deg = float(np.degrees(np.arctan2(float(R_bw[1, 0]), float(R_bw[0, 0]))))
                        # PyBullet cameraYaw convention: yaw=0 looks along +Y (see getDebugVisualizerCamera()).
                        # Convert heading (0=+X) to cameraYaw so camera looks along heading:
                        #   cameraYaw = heading - 90 deg
                        # "behind" means the CAMERA is behind the drone, looking forward along the drone heading.
                        cam_yaw = float(yaw_deg) - 90.0 + float(args.cam_yaw_offset)
                    elif str(args.cam_yaw_mode) == "behind_vel":
                        # Prefer velocity heading; fall back to drone yaw when too slow.
                        vxy = float(np.hypot(float(st.vel[0]), float(st.vel[1])))
                        yaw_vel = _yaw_from_vel_world_deg(float(st.vel[0]), float(st.vel[1])) if vxy >= float(args.cam_vel_min) else None
                        if yaw_vel is None:
                            yaw_vel = float(np.degrees(np.arctan2(float(R_bw[1, 0]), float(R_bw[0, 0]))))
                        cam_yaw_target = float(yaw_vel) - 90.0 + float(args.cam_yaw_offset)
                        # low-pass filter in wrapped angle space
                        tau_h = max(1e-6, float(args.cam_heading_tau))
                        d = _wrap_deg(cam_yaw_target - cam_yaw_filt_deg)
                        cam_yaw_filt_deg = float(cam_yaw_filt_deg) + float(dt) * float(d) / float(tau_h)
                        cam_yaw = float(cam_yaw_filt_deg)

                    env.set_camera(
                        distance=float(args.cam_dist),
                        yaw=float(cam_yaw),
                        pitch=float(args.cam_pitch),
                        target=target,
                    )

            # Desired forward motion: ramp x_des
            x_des = min(float(args.x_max), float(args.x_vel) * float(t))
            y_des = float(args.y_des)

            # Yaw command smoothing (fixed yaw)
            dy = _wrap_deg(float(args.yaw_des) - float(yaw_filt_deg))
            tau = max(1e-6, float(args.yaw_tau))
            yaw_dot = dy / tau
            yaw_rate = max(0.0, float(args.yaw_rate))
            if yaw_rate > 0.0:
                yaw_dot = float(np.clip(yaw_dot, -yaw_rate, +yaw_rate))
            yaw_filt_deg = float(yaw_filt_deg) + float(yaw_dot) * float(dt)
            yaw_cmd_deg = float(yaw_filt_deg)

            # XY hold -> desired body-z direction
            ex = float(x_des) - float(st.pos[0])
            ey = float(y_des) - float(st.pos[1])
            ax_cmd = float(args.kp_xy) * ex + float(args.kd_xy) * (0.0 - float(st.vel[0]))
            ay_cmd = float(args.kp_xy) * ey + float(args.kd_xy) * (0.0 - float(st.vel[1]))
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

            # Control outputs
            tau_b = attitude_pd_torque_body(R_bw=R_bw, ang_vel_world=st.ang_vel, R_des_bw=R_des, gains=g_att)
            Fz_cmd = altitude_pd_Fz_world(
                z_world=float(st.pos[2]),
                vz_world=float(st.vel[2]),
                z_des_world=float(z_des),
                mass=float(mass),
                gravity=float(args.gravity),
                gains=g_z,
                Fz_min=0.0,
                Fz_max=None,
            )

            # Build allocation matrix in WORLD frame
            r_body, n_body = env.rotor_geometry_body()
            r_world = (R_bw @ r_body.T).T
            n_world = (R_bw @ n_body.T).T
            n_world_for_fz = n_world.copy()
            n_world_for_fz[:, 2] = np.maximum(0.0, n_world_for_fz[:, 2])
            A = build_allocation_matrix(r_body=r_world, n_body=n_world_for_fz, C_T=float(args.CT), C_Q=float(args.CQ), spin_dir=spin_dir)

            # Target wrench: [Fz_world, tau_world]
            tau_w = R_bw @ np.asarray(tau_b, dtype=float).reshape(3)
            u = np.array([float(Fz_cmd), float(tau_w[0]), float(tau_w[1]), float(tau_w[2])], dtype=float)

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

            # Wall effect torque (computed at control rate, applied at physics substeps)
            # Distance-to-wall and activation
            if str(args.wall_orient) == "front":
                d_signed = float(args.wall_x) - float(st.pos[0])  # + if wall is ahead
                d_wall = float(d_signed)
                wall_active = int(bool((float(st.pos[0]) >= float(args.wall_zone_x)) and (0.0 < d_signed < float(args.wall_range))))
            else:
                wall_x0_eff = float(args.wall_zone_x) if args.wall_x0 is None else float(args.wall_x0)
                wall_x1_eff = wall_x0_eff + float(args.wall_len_x)
                # Signed lateral distance: + if drone is above wall_y, - if below.
                d_signed = float(st.pos[1]) - float(args.wall_y)
                d_wall = float(abs(d_signed))
                wall_active = int(bool((wall_x0_eff <= float(st.pos[0]) <= wall_x1_eff) and (float(abs(d_signed)) < float(args.wall_range))))
            v_b = R_bw.T @ np.asarray(st.vel, dtype=float).reshape(3)
            v_forward = max(0.0, float(v_b[0]))
            tau_wall_mag = 0.0
            if wall_active:
                denom = float(d_wall) * float(d_wall) + float(args.wall_d0) * float(args.wall_d0)
                tau_wall_mag = float(args.wall_k) * float(v_forward) * float(v_forward) / max(1e-6, denom)
                tau_wall_mag = float(np.clip(tau_wall_mag, -abs(float(args.wall_tau_max)), +abs(float(args.wall_tau_max))))
            tau_wall_b = np.zeros((3,), dtype=float)
            # Restore sign for side wall: direction depends on which side the wall is on.
            # Convention: d_signed = y - y_wall. For default wall_y=+1 and y≈0 => d_signed<0, torque becomes negative.
            # (This matches the user's requested sign flip.)
            sign_dir = 1.0
            if str(args.wall_orient) == "side":
                sign_dir = float(np.sign(d_signed))
                if abs(sign_dir) < 1e-12:
                    sign_dir = 1.0
            if str(args.wall_axis) == "roll":
                tau_wall_b[0] = float(sign_dir) * tau_wall_mag
            elif str(args.wall_axis) == "pitch":
                tau_wall_b[1] = float(sign_dir) * tau_wall_mag
            else:
                tau_wall_b[2] = float(sign_dir) * tau_wall_mag

            omega2_act = np.asarray(mix.omega2_cmd, dtype=float).reshape(4)

            # Morph update (apply every physics substep to prevent joint drift)
            morph_T = max(0.0, float(args.morph_seconds))
            morph_t0 = max(0.0, float(args.morph_start))
            if morph_T > 0.0:
                # s=0 until t>=morph_t0, then ramp to 1 over morph_T
                s = float(np.clip((t - morph_t0) / max(1e-9, morph_T), 0.0, 1.0))
                phi_cmd = float(args.phi_start) + s * (float(args.phi_deg) - float(args.phi_start))
                psi_cmd = float(args.psi_start) + s * (float(args.psi_deg) - float(args.psi_start))
                theta_cmd = float(args.theta_start) + s * (float(args.theta_deg) - float(args.theta_start))
            else:
                # No ramp: hold target
                phi_cmd = float(args.phi_deg)
                psi_cmd = float(args.psi_deg)
                theta_cmd = float(args.theta_deg)

            for _ in range(int(substeps)):
                env.set_morph_angles(phi_deg=phi_cmd, psi_deg=psi_cmd, theta_deg=theta_cmd, symmetry=str(args.morph_symmetry))
                omega2_act = motor.step(mix.omega2_cmd, dt_phys)
                for i in range(4):
                    thrust = float(args.CT) * float(omega2_act[i])
                    tau_react = float(spin_dir[i]) * float(args.CQ) * float(omega2_act[i])
                    env.apply_rotor_thrust_link_frame(rotor_idx=i, thrust=thrust, reaction_torque=tau_react)

                if wall_active and float(abs(tau_wall_mag)) > 0.0:
                    R_bw_now = quat_xyzw_to_rotmat(env.get_state().quat)
                    tau_wall_w = R_bw_now @ tau_wall_b
                    env.apply_body_wrench_world(torque_world=(float(tau_wall_w[0]), float(tau_wall_w[1]), float(tau_wall_w[2])))

                env.step(1)

            # refresh state for log/print
            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)
            yaw_deg = float(np.degrees(np.arctan2(float(R_bw[1, 0]), float(R_bw[0, 0]))))

            if csv_logger is not None:
                row = {
                    "step": int(k),
                    "t": float(t),
                    "x": float(st.pos[0]),
                    "y": float(st.pos[1]),
                    "z": float(st.pos[2]),
                    "vx": float(st.vel[0]),
                    "vy": float(st.vel[1]),
                    "vz": float(st.vel[2]),
                    "yaw_deg": float(yaw_deg),
                    "yaw_cmd_deg": float(yaw_cmd_deg),
                    "x_des": float(x_des),
                    "y_des": float(y_des),
                    "Fz_cmd": float(Fz_cmd),
                    "tau_x_cmd_b": float(tau_b[0]),
                    "tau_y_cmd_b": float(tau_b[1]),
                    "tau_z_cmd_b": float(tau_b[2]),
                    "mix_mode": str(mix.mode),
                    "omega2_cmd_0": float(mix.omega2_cmd[0]),
                    "omega2_cmd_1": float(mix.omega2_cmd[1]),
                    "omega2_cmd_2": float(mix.omega2_cmd[2]),
                    "omega2_cmd_3": float(mix.omega2_cmd[3]),
                    "omega2_act_0": float(omega2_act[0]),
                    "omega2_act_1": float(omega2_act[1]),
                    "omega2_act_2": float(omega2_act[2]),
                    "omega2_act_3": float(omega2_act[3]),
                    "d_wall": float(d_wall),
                    "wall_active": int(wall_active),
                    "tau_wall_x_b": float(tau_wall_b[0]),
                    "tau_wall_y_b": float(tau_wall_b[1]),
                    "tau_wall_z_b": float(tau_wall_b[2]),
                }
                csv_logger.write(row)

            if int(args.log_every) > 0 and (k % int(args.log_every) == 0):
                print(f"[step {k:06d}] t={t:.2f} x={float(st.pos[0]):+.2f} z={float(st.pos[2]):+.2f} d_wall={d_wall:+.2f} wall={wall_active} tau_wall_b={tau_wall_b}")

            if bool(args.gui):
                time.sleep(dt)

        return 0
    finally:
        try:
            if csv_logger is not None:
                csv_logger.close()
        except Exception:
            pass
        try:
            env.stop_video_recording(video_log_id)
        except Exception:
            pass
        env.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())


