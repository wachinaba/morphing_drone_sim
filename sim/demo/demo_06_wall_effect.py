import argparse
import time
from pathlib import Path

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


def _wrap_01(x: float) -> float:
    x = float(x)
    return float(x - np.floor(x))


def _traj_ramp(*, t: float, x_vel: float, x_max: float, y_des: float, xy_vel_ff: bool) -> tuple[float, float, float, float]:
    """
    Legacy ramp along +X with optional derivative feed-forward.
    Returns: (x_des, y_des, xdot_des, ydot_des)
    """
    t = float(max(0.0, t))
    x_des = min(float(x_max), float(x_vel) * t)
    y_des = float(y_des)
    # ramp-only compatibility: only enable xdot_des when --xy-vel-ff is set
    if bool(xy_vel_ff) and (float(x_des) < float(x_max) - 1e-9):
        xdot_des = float(x_vel)
    else:
        xdot_des = 0.0
    ydot_des = 0.0
    return float(x_des), float(y_des), float(xdot_des), float(ydot_des)


def _traj_circle(
    *,
    t: float,
    radius: float,
    freq_hz: float,
    cx: float,
    cy: float,
    phase: float,
) -> tuple[float, float, float, float]:
    """
    Circle in XY plane with analytic velocity.
    phase: fraction of cycle in [0,1).
    Returns: (x_des, y_des, xdot_des, ydot_des)
    """
    r = float(radius)
    f = float(freq_hz)
    w = 2.0 * float(np.pi) * f
    th0 = 2.0 * float(np.pi) * _wrap_01(float(phase))
    th = w * float(t) + th0
    c = float(np.cos(th))
    s = float(np.sin(th))
    x_des = float(cx) + r * c
    y_des = float(cy) + r * s
    xdot_des = -r * w * s
    ydot_des = +r * w * c
    return float(x_des), float(y_des), float(xdot_des), float(ydot_des)


def _traj_rect(
    *,
    t: float,
    width: float,
    height: float,
    speed: float,
    cx: float,
    cy: float,
    phase: float,
) -> tuple[float, float, float, float]:
    """
    Rectangle loop with constant speed along edges (piecewise-constant velocity).
    phase: fraction of cycle in [0,1).
    Returns: (x_des, y_des, xdot_des, ydot_des)
    """
    w = float(max(1e-9, width))
    h = float(max(1e-9, height))
    v = float(max(0.0, speed))
    L = 2.0 * (w + h)  # perimeter
    # distance traveled along the perimeter
    s = (float(_wrap_01(float(phase))) * L + v * float(t)) % L

    # local rectangle centered at origin, starting at bottom edge (-h/2) moving +x
    x0 = -0.5 * w
    x1 = +0.5 * w
    y0 = -0.5 * h
    y1 = +0.5 * h

    if s < w:
        # bottom edge: (-w/2,-h/2) -> (+w/2,-h/2)
        x = x0 + s
        y = y0
        vx, vy = v, 0.0
    elif s < w + h:
        # right edge: (+w/2,-h/2) -> (+w/2,+h/2)
        x = x1
        y = y0 + (s - w)
        vx, vy = 0.0, v
    elif s < (2.0 * w + h):
        # top edge: (+w/2,+h/2) -> (-w/2,+h/2)
        x = x1 - (s - (w + h))
        y = y1
        vx, vy = -v, 0.0
    else:
        # left edge: (-w/2,+h/2) -> (-w/2,-h/2)
        x = x0
        y = y1 - (s - (2.0 * w + h))
        vx, vy = 0.0, -v

    x_des = float(cx) + float(x)
    y_des = float(cy) + float(y)
    return float(x_des), float(y_des), float(vx), float(vy)


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


def _default_plot_path(*, args) -> Path:
    """
    Pick a reasonable default output path for plots.
    Priority:
      1) --plot-path
      2) --log-csv stem + "_plots.png"
      3) out/demo_06_wall_effect_plots_<timestamp>.png
    """
    if getattr(args, "plot_path", None):
        return Path(str(args.plot_path))
    if getattr(args, "log_csv", None):
        p = Path(str(args.log_csv))
        return p.with_name(p.stem + "_plots.png")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"demo_06_wall_effect_plots_{ts}.png"


def _save_timeseries_plots(
    *,
    plot_path: Path,
    t: np.ndarray,  # (K,)
    pos: np.ndarray,  # (K,3)
    vel: np.ndarray,  # (K,3)
    euler_deg: np.ndarray,  # (K,3) roll,pitch,yaw [deg]
    ang_vel_w: np.ndarray,  # (K,3)
    ang_vel_b: np.ndarray,  # (K,3)
    omega2_cmd: np.ndarray,  # (K,4)
    omega2_act: np.ndarray,  # (K,4)
    thrust_N: np.ndarray,  # (K,4)
    power_W: np.ndarray,  # (K,4) proxy (aero drag) power
    d_wall: np.ndarray,  # (K,)
    wall_active: np.ndarray,  # (K,)
    tau_wall_b: np.ndarray,  # (K,3)
    dpi: int = 160,
):
    # Lazy import so demo runs without matplotlib unless plotting is requested.
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(t, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    vel = np.asarray(vel, dtype=float).reshape(-1, 3)
    euler_deg = np.asarray(euler_deg, dtype=float).reshape(-1, 3)
    ang_vel_w = np.asarray(ang_vel_w, dtype=float).reshape(-1, 3)
    ang_vel_b = np.asarray(ang_vel_b, dtype=float).reshape(-1, 3)
    omega2_cmd = np.asarray(omega2_cmd, dtype=float).reshape(-1, 4)
    omega2_act = np.asarray(omega2_act, dtype=float).reshape(-1, 4)
    thrust_N = np.asarray(thrust_N, dtype=float).reshape(-1, 4)
    power_W = np.asarray(power_W, dtype=float).reshape(-1, 4)
    d_wall = np.asarray(d_wall, dtype=float).reshape(-1)
    wall_active = np.asarray(wall_active, dtype=float).reshape(-1)
    tau_wall_b = np.asarray(tau_wall_b, dtype=float).reshape(-1, 3)

    fig, axs = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)

    # Position
    ax = axs[0, 0]
    ax.plot(t, pos[:, 0], label="x [m]")
    ax.plot(t, pos[:, 1], label="y [m]")
    ax.plot(t, pos[:, 2], label="z [m]")
    ax.set_title("Position")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Velocity
    ax = axs[0, 1]
    ax.plot(t, vel[:, 0], label="vx [m/s]")
    ax.plot(t, vel[:, 1], label="vy [m/s]")
    ax.plot(t, vel[:, 2], label="vz [m/s]")
    ax.set_title("Velocity (world)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Attitude (Euler ZYX)
    ax = axs[1, 0]
    ax.plot(t, euler_deg[:, 0], label="roll [deg]")
    ax.plot(t, euler_deg[:, 1], label="pitch [deg]")
    ax.plot(t, euler_deg[:, 2], label="yaw [deg]")
    ax.set_title("Attitude (Euler ZYX)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Angular velocity (body)
    ax = axs[1, 1]
    ax.plot(t, ang_vel_b[:, 0], label="p [rad/s]")
    ax.plot(t, ang_vel_b[:, 1], label="q [rad/s]")
    ax.plot(t, ang_vel_b[:, 2], label="r [rad/s]")
    ax.set_title("Angular velocity (body)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Angular velocity (world)
    ax = axs[2, 0]
    ax.plot(t, ang_vel_w[:, 0], label="wx [rad/s]")
    ax.plot(t, ang_vel_w[:, 1], label="wy [rad/s]")
    ax.plot(t, ang_vel_w[:, 2], label="wz [rad/s]")
    ax.set_title("Angular velocity (world)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Rotor omega^2 cmd vs act
    ax = axs[2, 1]
    for i in range(4):
        ax.plot(t, omega2_act[:, i], label=f"ω²_act{i}", linewidth=1.6)
        ax.plot(t, omega2_cmd[:, i], label=f"ω²_cmd{i}", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("Rotor ω² (cmd/act)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=4, fontsize=8)

    # Rotor thrusts
    ax = axs[3, 0]
    for i in range(4):
        ax.plot(t, thrust_N[:, i], label=f"T{i} [N]")
    ax.plot(t, np.sum(thrust_N, axis=1), label="T_total [N]", linewidth=2.0, alpha=0.9)
    ax.set_title("Rotor thrusts")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    # Power proxy
    ax = axs[3, 1]
    for i in range(4):
        ax.plot(t, power_W[:, i], label=f"P{i} [W]")
    ax.plot(t, np.sum(power_W, axis=1), label="P_total [W]", linewidth=2.0, alpha=0.9)
    ax.set_title("Power (proxy: |CQ|*omega^3)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    # Additional wall plot in a second figure (keeps the main figure readable)
    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    ax = axs2[0]
    ax.plot(t, d_wall, label="d_wall [m]")
    ax.plot(t, wall_active, label="wall_active", alpha=0.8)
    ax.set_title("Wall distance / activation")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axs2[1]
    ax.plot(t, tau_wall_b[:, 0], label="tau_wall_x_b [N*m]")
    ax.plot(t, tau_wall_b[:, 1], label="tau_wall_y_b [N*m]")
    ax.plot(t, tau_wall_b[:, 2], label="tau_wall_z_b [N*m]")
    ax.set_title("Wall torque (body)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("demo_06_wall_effect: time series (states/actuators)", fontsize=14)
    fig2.suptitle("demo_06_wall_effect: time series (wall)", fontsize=14)
    fig.savefig(str(plot_path), dpi=int(max(80, int(dpi))))
    wall_path = plot_path.with_name(plot_path.stem + "_wall" + plot_path.suffix)
    fig2.savefig(str(wall_path), dpi=int(max(80, int(dpi))))
    plt.close(fig)
    plt.close(fig2)
    print(f"[info] Saved plots: {str(plot_path)}")
    print(f"[info] Saved plots: {str(wall_path)}")


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
    ap.add_argument("--record-mp4", type=str, default=None, help="Record an MP4 video to this path (GUI recorder).")
    ap.add_argument("--record-fps", type=int, default=60, help="(record-mp4) Output video FPS. Default: 60")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--hz", type=float, default=240.0)
    ap.add_argument("--physics-hz", type=float, default=240.0)

    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--lin-damping", type=float, default=0.08)
    ap.add_argument("--ang-damping", type=float, default=0.08)

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
    # Optional time-varying morph (sin wave): angle(t) = base + amp * sin(2*pi*f*t)
    # NOTE: This stacks with the initial morph ramp (if enabled).
    ap.add_argument("--phi-amp", type=float, default=0.0, help="Phi sine amplitude [deg]. Default: 0 (off).")
    ap.add_argument("--phi-freq", type=float, default=0.0, help="Phi sine frequency [Hz]. Default: 0")
    ap.add_argument("--psi-amp", type=float, default=0.0, help="Psi sine amplitude [deg]. Default: 0 (off).")
    ap.add_argument("--psi-freq", type=float, default=0.0, help="Psi sine frequency [Hz]. Default: 0")
    ap.add_argument("--theta-amp", type=float, default=0.0, help="Theta sine amplitude [deg]. Default: 0 (off).")
    ap.add_argument("--theta-freq", type=float, default=0.0, help="Theta sine frequency [Hz]. Default: 0")

    # Flight plan / XY trajectory
    ap.add_argument(
        "--traj",
        type=str,
        default="ramp",
        choices=["ramp", "circle", "rect"],
        help="XY trajectory type. ramp: legacy x ramp (+X). circle/rect: periodic tracking with analytic desired velocity. Default: ramp",
    )
    ap.add_argument("--x-vel", type=float, default=0.5, help="Forward desired speed [m/s]. Default: 0.5")
    ap.add_argument("--x-max", type=float, default=6.0, help="Max x_des [m] (clamp). Default: 6.0")
    ap.add_argument("--y-des", type=float, default=0.0)
    ap.add_argument("--max-tilt-deg", type=float, default=20.0)
    ap.add_argument("--kp-xy", type=float, default=1.0)
    ap.add_argument("--kd-xy", type=float, default=1.2)
    ap.add_argument(
        "--xy-vel-ff",
        action="store_true",
        help=(
            "Use XY desired velocity feed-forward in the derivative term. "
            "Useful to reduce steady tracking error during x_des ramp (x_vel). "
            "NOTE: ramp-only compatibility option; circle/rect always use analytic xdot_des/ydot_des. Default: off"
        ),
    )
    ap.add_argument("--ki-xy", type=float, default=0.0, help="XY integral gain for acceleration command (anti steady-state bias). Default: 0")
    ap.add_argument("--i-xy-max", type=float, default=2.0, help="Clamp for XY integrator state |∫e dt| [m*s]. Default: 2.0")

    # Trajectory params: circle
    ap.add_argument("--circle-radius", type=float, default=0.8, help="(traj=circle) Radius [m]. Default: 0.8")
    ap.add_argument("--circle-freq", type=float, default=0.15, help="(traj=circle) Frequency [Hz]. Default: 0.15")
    ap.add_argument("--circle-cx", type=float, default=5.0, help="(traj=circle) Center x [m]. Default: 5.0")
    ap.add_argument("--circle-cy", type=float, default=0.0, help="(traj=circle) Center y [m]. Default: 0.0")
    ap.add_argument("--circle-phase", type=float, default=0.0, help="(traj=circle) Phase as fraction of cycle [0,1). Default: 0")

    # Trajectory params: rectangle
    ap.add_argument("--rect-width", type=float, default=1.6, help="(traj=rect) Rectangle width in x [m]. Default: 1.6")
    ap.add_argument("--rect-height", type=float, default=1.2, help="(traj=rect) Rectangle height in y [m]. Default: 1.2")
    ap.add_argument("--rect-speed", type=float, default=0.8, help="(traj=rect) Speed along edges [m/s]. Default: 0.8")
    ap.add_argument("--rect-cx", type=float, default=5.0, help="(traj=rect) Center x [m]. Default: 5.0")
    ap.add_argument("--rect-cy", type=float, default=0.0, help="(traj=rect) Center y [m]. Default: 0.0")
    ap.add_argument("--rect-phase", type=float, default=0.0, help="(traj=rect) Phase as fraction of cycle [0,1). Default: 0")

    # Yaw (keep forward)
    ap.add_argument("--yaw-des", type=float, default=0.0)
    ap.add_argument("--yaw-amp", type=float, default=0.0, help="Yaw sine amplitude [deg]. Default: 0")
    ap.add_argument("--yaw-freq", type=float, default=0.0, help="Yaw sine frequency [Hz]. Default: 0")
    ap.add_argument("--yaw-spin-deg-s", type=float, default=0.0, help="Yaw spin rate [deg/s]. Positive = CCW. Default: 0")
    ap.add_argument("--yaw-tau", type=float, default=0.4)
    ap.add_argument("--yaw-rate", type=float, default=120.0)

    # Attitude gains (tuned for 1.8 kg morphing_drone; axis-wise override supported)
    ap.add_argument("--kp-att", type=float, default=1.4)
    ap.add_argument("--kd-att", type=float, default=0.22)
    ap.add_argument("--kp-att-xyz", type=float, nargs=3, default=None, metavar=("KPX", "KPY", "KPZ"))
    ap.add_argument("--kd-att-xyz", type=float, nargs=3, default=None, metavar=("KDX", "KDY", "KDZ"))
    ap.add_argument("--kp-z", type=float, default=10.0)
    ap.add_argument("--kd-z", type=float, default=6.5)

    # Mixer/coefficients
    ap.add_argument("--CT", type=float, default=1.0)
    ap.add_argument("--CQ", type=float, default=0.08)
    ap.add_argument("--tw-ratio", type=float, default=5.5, help="Thrust-to-weight ratio. 5.5 gives ~2 kgf/rotor at 1.8 kg. Default: 5.5")
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

    # Wall effect model (selectable)
    ap.add_argument("--wall-zone-x", type=float, default=3.0, help="Enable wall-effect model once x >= this. Default: 3.0")
    ap.add_argument("--wall-range", type=float, default=2.0, help="Effect active when distance d < range. Default: 2.0")
    ap.add_argument("--wall-d0", type=float, default=0.3, help="Distance softening [m] in denom: d^2+d0^2. Default: 0.3")
    ap.add_argument(
        "--wall-model",
        type=str,
        default="v2_over_d2",
        choices=["v2_over_d2", "fixed_tau", "off"],
        help=(
            "Wall-effect disturbance model. "
            "v2_over_d2: tau ~ k*v^2/(d^2+d0^2). "
            "fixed_tau: apply constant torque when active. "
            "off: disable wall-effect torque. "
            "Default: v2_over_d2"
        ),
    )
    ap.add_argument("--wall-k", type=float, default=0.02, help="(wall-model=v2_over_d2) Wall moment gain k. Default: 0.02")
    ap.add_argument("--wall-tau-fixed", type=float, default=0.05, help="(wall-model=fixed_tau) Constant |tau_wall| [N*m] before sign/clamp. Default: 0.05")
    ap.add_argument(
        "--wall-axis",
        type=str,
        default="roll",
        choices=["roll", "pitch", "yaw"],
        help="Which body-axis torque to apply as wall-effect moment. Default: roll",
    )
    ap.add_argument("--wall-tau-max", type=float, default=0.2, help="Clamp |tau_wall| [N*m]. Default: 0.2")

    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--log-flush-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=120)

    # Offline plots (saved after simulation)
    ap.add_argument("--plot", action="store_true", help="Save time-series plots (PNG) after the run. Default: off")
    ap.add_argument("--plot-path", type=str, default=None, help="Output path for the PNG. Default: derived from --log-csv or out/...")
    ap.add_argument("--plot-dpi", type=int, default=160, help="PNG DPI. Default: 160")

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
        "--cam-target-mode",
        type=str,
        default="follow",
        choices=["follow", "traj_center", "world"],
        help=(
            "Camera target mode. "
            "'follow': target follows the drone (requires --cam-follow). "
            "'traj_center': target fixed at trajectory center (circle/rect centers). "
            "'world': target fixed at --cam-target-world. "
            "Default: follow"
        ),
    )
    ap.add_argument(
        "--cam-target-world",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="(cam-target-mode=world) Fixed world target position [m]. If omitted, uses (0,0,cam-target-z).",
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
    ap.add_argument(
        "--draw-trajectory",
        action="store_true",
        help="In GUI, draw target trajectory (green) and actual trajectory (blue) as lines. Default: off",
    )
    ap.add_argument(
        "--traj-line-width-des",
        type=float,
        default=2.5,
        help="(GUI, --draw-trajectory) Line width for desired trajectory (green). Default: 2.5",
    )
    ap.add_argument(
        "--traj-line-width-act",
        type=float,
        default=2.5,
        help="(GUI, --draw-trajectory) Line width for actual trajectory (blue). Default: 2.5",
    )
    ap.add_argument(
        "--draw-trajectory-every",
        type=int,
        default=15,
        help="When --draw-trajectory: record and draw every N steps (default 15). Larger = lighter. Default: 15",
    )
    ap.add_argument(
        "--draw-target-marker",
        action="store_true",
        help="In GUI, draw a marker at the *current* target (x_des,y_des,z_des). Helps when trajectory lines overlap. Default: off",
    )
    ap.add_argument(
        "--target-marker-every",
        type=int,
        default=2,
        help="When --draw-target-marker: update marker every N control steps. Default: 2",
    )
    ap.add_argument(
        "--target-marker-size",
        type=float,
        default=0.12,
        help="When --draw-target-marker: marker half-size [m] (cross arms length). Default: 0.12",
    )
    ap.add_argument(
        "--target-marker-line-width",
        type=float,
        default=4.0,
        help="When --draw-target-marker: marker line width. Default: 4.0",
    )
    ap.add_argument(
        "--target-marker-text",
        action="store_true",
        help="When --draw-target-marker: also draw a small 'des' text label. Default: off",
    )

    args = ap.parse_args()

    hz = float(args.hz)
    dt = 1.0 / max(1.0, hz)
    physics_hz = float(args.physics_hz)
    dt_phys = 1.0 / max(1.0, physics_hz)
    substeps = int(max(1, round(float(dt) / float(dt_phys))))

    # MP4 recording: for correct 1x playback, use the GUI recorder configured at connect-time.
    # If the user requested recording but forgot --gui, enable GUI automatically.
    gui_enabled = bool(args.gui) or (args.record_mp4 is not None)
    if (args.record_mp4 is not None) and (not bool(args.gui)):
        print("[info] --record-mp4 specified without --gui; enabling GUI for recording.")

    record_fps = int(args.record_fps)
    if record_fps <= 0:
        record_fps = 60
    if (args.record_mp4 is not None) and (record_fps > int(round(float(args.physics_hz)))):
        # We can't reliably record more unique frames than physics steps.
        record_fps = int(round(float(args.physics_hz)))
        print(f"[warn] --record-fps capped to physics rate: record_fps={record_fps} (physics_hz={float(args.physics_hz):.1f})")

    env = PyBulletEnv(
        gui=bool(gui_enabled),
        time_step=dt_phys,
        gravity=float(args.gravity),
        gui_width=args.win_width,
        gui_height=args.win_height,
        record_mp4=(str(args.record_mp4) if (args.record_mp4 is not None) else None),
        record_fps=(int(record_fps) if (args.record_mp4 is not None) else None),
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
        # Preferred: GUI recorder enabled via connect options (record_mp4/record_fps in PyBulletEnv).
        if args.record_mp4:
            print(f"[info] Recording MP4: {args.record_mp4} (fps={int(record_fps)})")

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
                "traj",
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
                "xdot_des",
                "ydot_des",
                "ex",
                "ey",
                "ax_cmd",
                "ay_cmd",
                "ixy_x",
                "ixy_y",
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
                "wall_model",
                "wall_active",
                "tau_wall_x_b",
                "tau_wall_y_b",
                "tau_wall_z_b",
            ]
            csv_logger = CsvLogger(path=str(args.log_csv), fieldnames=fieldnames, flush_every=int(args.log_flush_every))
            csv_logger.open()
            print(f"[info] CSV logging enabled: {args.log_csv}")

        # Time-series buffers for offline plots (control-rate samples)
        plot_enabled = bool(args.plot)
        if plot_enabled:
            t_log: list[float] = []
            pos_log: list[np.ndarray] = []
            vel_log: list[np.ndarray] = []
            euler_deg_log: list[np.ndarray] = []
            ang_w_log: list[np.ndarray] = []
            ang_b_log: list[np.ndarray] = []
            omega2_cmd_log: list[np.ndarray] = []
            omega2_act_log: list[np.ndarray] = []
            thrust_log: list[np.ndarray] = []
            power_log: list[np.ndarray] = []
            d_wall_log: list[float] = []
            wall_active_log: list[float] = []
            tau_wall_b_log: list[np.ndarray] = []

        n_steps = max(1, int(float(args.seconds) * hz))
        yaw_filt_deg = float(args.yaw_des)
        ixy_x = 0.0
        ixy_y = 0.0

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
        morph_sine = any(
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
        if morph_sine:
            print(
                "[info] morph sine (stacked on base/ramp): "
                f"phi += {float(args.phi_amp):+.1f}*sin(2pi*{float(args.phi_freq):.2f}t) deg, "
                f"psi += {float(args.psi_amp):+.1f}*sin(2pi*{float(args.psi_freq):.2f}t) deg, "
                f"theta += {float(args.theta_amp):+.1f}*sin(2pi*{float(args.theta_freq):.2f}t) deg"
            )

        # Camera heading filter (deg)
        cam_yaw_filt_deg = float(args.cam_yaw)
        cam_frozen = False

        # If camera target is fixed (traj_center/world), set it once here.
        if bool(args.gui) and str(args.cam_target_mode) in {"traj_center", "world"}:
            if str(args.cam_target_mode) == "world":
                if args.cam_target_world is not None:
                    target_fixed = (float(args.cam_target_world[0]), float(args.cam_target_world[1]), float(args.cam_target_world[2]))
                else:
                    target_fixed = (0.0, 0.0, float(args.cam_target_z))
            else:
                # traj_center
                traj = str(args.traj)
                if traj == "circle":
                    target_fixed = (float(args.circle_cx), float(args.circle_cy), float(args.cam_target_z))
                elif traj == "rect":
                    target_fixed = (float(args.rect_cx), float(args.rect_cy), float(args.cam_target_z))
                else:
                    # ramp has no single "center"; use a sensible fixed point.
                    target_fixed = (0.5 * float(args.x_max), float(args.y_des), float(args.cam_target_z))
            env.set_camera(
                distance=float(args.cam_dist),
                yaw=float(args.cam_yaw),
                pitch=float(args.cam_pitch),
                target=target_fixed,
            )

        # Trajectory: target=green, actual=blue. addUserDebugLine per segment (can hitch each time).
        traj_des_pts: list[tuple[float, float, float]] = []
        traj_act_pts: list[tuple[float, float, float]] = []
        # Current target marker (replace-in-place to avoid accumulating debug items)
        target_marker_ids = {"x": -1, "y": -1, "z": -1, "text": -1}

        # Deterministic video: render exactly one frame per (1/record_fps) simulated seconds,
        # independent of wall-clock performance. Achieved by enabling/disabling GUI rendering per physics step.
        dt_frame = (1.0 / float(record_fps)) if (args.record_mp4 is not None) else None
        next_frame_t = 0.0
        k_phys_total = 0  # counts physics steps (dt_phys)

        for k in range(n_steps):
            t = float(k) * dt
            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)

            if bool(args.gui) and bool(args.cam_follow) and (str(args.cam_target_mode) == "follow"):
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

            # Desired XY trajectory (position + analytic velocity for D term)
            traj = str(args.traj)
            if traj == "ramp":
                x_des, y_des, xdot_des, ydot_des = _traj_ramp(
                    t=float(t),
                    x_vel=float(args.x_vel),
                    x_max=float(args.x_max),
                    y_des=float(args.y_des),
                    xy_vel_ff=bool(args.xy_vel_ff),
                )
            elif traj == "circle":
                x_des, y_des, xdot_des, ydot_des = _traj_circle(
                    t=float(t),
                    radius=float(args.circle_radius),
                    freq_hz=float(args.circle_freq),
                    cx=float(args.circle_cx),
                    cy=float(args.circle_cy),
                    phase=float(args.circle_phase),
                )
            else:
                x_des, y_des, xdot_des, ydot_des = _traj_rect(
                    t=float(t),
                    width=float(args.rect_width),
                    height=float(args.rect_height),
                    speed=float(args.rect_speed),
                    cx=float(args.rect_cx),
                    cy=float(args.rect_cy),
                    phase=float(args.rect_phase),
                )

            # Marker for current target (helps when trajectory lines overlap)
            if bool(args.gui) and bool(args.draw_target_marker):
                every = max(1, int(args.target_marker_every))
                if (k % every) == 0:
                    z_t = float(z_des)
                    size = max(0.01, float(args.target_marker_size))
                    a = (float(x_des) - size, float(y_des), z_t)
                    b = (float(x_des) + size, float(y_des), z_t)
                    c = (float(x_des), float(y_des) - size, z_t)
                    d = (float(x_des), float(y_des) + size, z_t)
                    e = (float(x_des), float(y_des), z_t - size)
                    f = (float(x_des), float(y_des), z_t + size)
                    # Default marker color: match desired trajectory line (green).
                    col = [0.0, 1.0, 0.0]
                    lw = max(0.5, float(args.target_marker_line_width))
                    target_marker_ids["x"] = int(
                        p.addUserDebugLine(
                            a,
                            b,
                            lineColorRGB=col,
                            lineWidth=float(lw),
                            lifeTime=0,
                            replaceItemUniqueId=int(target_marker_ids["x"]),
                        )
                    )
                    target_marker_ids["y"] = int(
                        p.addUserDebugLine(
                            c,
                            d,
                            lineColorRGB=col,
                            lineWidth=float(lw),
                            lifeTime=0,
                            replaceItemUniqueId=int(target_marker_ids["y"]),
                        )
                    )
                    target_marker_ids["z"] = int(
                        p.addUserDebugLine(
                            e,
                            f,
                            lineColorRGB=col,
                            lineWidth=float(lw),
                            lifeTime=0,
                            replaceItemUniqueId=int(target_marker_ids["z"]),
                        )
                    )
                    if bool(args.target_marker_text):
                        target_marker_ids["text"] = int(
                            p.addUserDebugText(
                                "des",
                                [float(x_des), float(y_des), float(z_t + 1.5 * size)],
                                textColorRGB=col,
                                textSize=1.2,
                                lifeTime=0,
                                replaceItemUniqueId=int(target_marker_ids["text"]),
                            )
                        )

            # Record and draw trajectory in real time (lines; addUserDebugLine can hitch each time)
            if bool(args.gui) and bool(args.draw_trajectory):
                every = max(1, int(args.draw_trajectory_every))
                if (k % every) == 0:
                    traj_des_pts.append((float(x_des), float(y_des), float(z_des)))
                    traj_act_pts.append((float(st.pos[0]), float(st.pos[1]), float(st.pos[2])))
                    if len(traj_des_pts) >= 2:
                        lw_des = max(0.5, float(args.traj_line_width_des))
                        lw_act = max(0.5, float(args.traj_line_width_act))
                        p.addUserDebugLine(
                            traj_des_pts[-2],
                            traj_des_pts[-1],
                            lineColorRGB=[0.0, 1.0, 0.0],
                            lineWidth=float(lw_des),
                            lifeTime=0,
                        )
                        p.addUserDebugLine(
                            traj_act_pts[-2],
                            traj_act_pts[-1],
                            lineColorRGB=[0.0, 0.4, 1.0],
                            lineWidth=float(lw_act),
                            lifeTime=0,
                        )

            # Yaw command smoothing (base + optional sine/spin)
            yaw_target_deg = float(args.yaw_des)
            if abs(float(args.yaw_spin_deg_s)) > 0.0:
                yaw_target_deg = float(yaw_target_deg) + float(args.yaw_spin_deg_s) * float(t)
            if abs(float(args.yaw_amp)) > 0.0 and abs(float(args.yaw_freq)) > 0.0:
                yaw_target_deg = float(yaw_target_deg) + float(args.yaw_amp) * float(np.sin(2.0 * np.pi * float(args.yaw_freq) * float(t)))
            dy = _wrap_deg(float(yaw_target_deg) - float(yaw_filt_deg))
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

            # Simple integral action to reject steady biases (windup-limited)
            ki_xy = float(args.ki_xy)
            if abs(ki_xy) > 0.0:
                ixy_x = float(np.clip(float(ixy_x) + float(ex) * float(dt), -abs(float(args.i_xy_max)), +abs(float(args.i_xy_max))))
                ixy_y = float(np.clip(float(ixy_y) + float(ey) * float(dt), -abs(float(args.i_xy_max)), +abs(float(args.i_xy_max))))

            ax_cmd = float(args.kp_xy) * ex + float(args.kd_xy) * (float(xdot_des) - float(st.vel[0])) + float(ki_xy) * float(ixy_x)
            ay_cmd = float(args.kp_xy) * ey + float(args.kd_xy) * (float(ydot_des) - float(st.vel[1])) + float(ki_xy) * float(ixy_y)
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

            wall_model = str(args.wall_model)
            tau_wall_mag = 0.0
            if wall_active and (wall_model != "off"):
                if wall_model == "v2_over_d2":
                    denom = float(d_wall) * float(d_wall) + float(args.wall_d0) * float(args.wall_d0)
                    tau_wall_mag = float(args.wall_k) * float(v_forward) * float(v_forward) / max(1e-6, denom)
                elif wall_model == "fixed_tau":
                    tau_wall_mag = float(args.wall_tau_fixed)
                else:
                    # argparse choices should prevent this, but keep safe fallback
                    tau_wall_mag = 0.0
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
            if morph_sine:
                # angle(t) = base(t) + amp * sin(2*pi*f*t)
                phi_cmd = float(phi_cmd) + float(args.phi_amp) * float(np.sin(2.0 * np.pi * float(args.phi_freq) * float(t)))
                psi_cmd = float(psi_cmd) + float(args.psi_amp) * float(np.sin(2.0 * np.pi * float(args.psi_freq) * float(t)))
                theta_cmd = float(theta_cmd) + float(args.theta_amp) * float(np.sin(2.0 * np.pi * float(args.theta_freq) * float(t)))

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

                if args.record_mp4 is not None:
                    # Decide whether THIS physics step should produce a video frame.
                    # We base the decision on the *next* sim time after stepping.
                    t_next = float(k_phys_total + 1) * float(dt_phys)
                    want_frame = bool((t_next + 1e-12) >= float(next_frame_t)) and bool(float(next_frame_t) < float(args.seconds) - 1e-12)
                    env.set_rendering_enabled(bool(want_frame))
                env.step(1)
                if args.record_mp4 is not None:
                    if want_frame:
                        next_frame_t = float(next_frame_t) + float(dt_frame)
                    k_phys_total += 1

            # refresh state for log/print
            st = env.get_state()
            R_bw = quat_xyzw_to_rotmat(st.quat)
            yaw_deg = float(np.degrees(np.arctan2(float(R_bw[1, 0]), float(R_bw[0, 0]))))

            if plot_enabled:
                # Rotor thrust and (proxy) power from omega2_act.
                omega2_act_clip = np.maximum(0.0, np.asarray(omega2_act, dtype=float).reshape(4))
                thrust_N = float(args.CT) * omega2_act_clip
                omega = np.sqrt(omega2_act_clip)
                # Proxy mechanical power due to drag torque: P = |CQ| * omega^3 (W)
                power_W = abs(float(args.CQ)) * omega2_act_clip * omega
                ang_w = np.asarray(st.ang_vel, dtype=float).reshape(3)
                ang_b = (R_bw.T @ ang_w.reshape(3)).reshape(3)
                # Euler angles (ZYX yaw-pitch-roll) from R_bw (body->world)
                pitch = float(np.arcsin(float(np.clip(-float(R_bw[2, 0]), -1.0, 1.0))))
                roll = float(np.arctan2(float(R_bw[2, 1]), float(R_bw[2, 2])))
                yaw = float(np.arctan2(float(R_bw[1, 0]), float(R_bw[0, 0])))
                euler_deg = np.degrees(np.array([roll, pitch, yaw], dtype=float))

                t_log.append(float(t))
                pos_log.append(np.asarray(st.pos, dtype=float).reshape(3).copy())
                vel_log.append(np.asarray(st.vel, dtype=float).reshape(3).copy())
                euler_deg_log.append(np.asarray(euler_deg, dtype=float).reshape(3).copy())
                ang_w_log.append(ang_w.copy())
                ang_b_log.append(ang_b.copy())
                omega2_cmd_log.append(np.asarray(mix.omega2_cmd, dtype=float).reshape(4).copy())
                omega2_act_log.append(omega2_act_clip.copy())
                thrust_log.append(np.asarray(thrust_N, dtype=float).reshape(4).copy())
                power_log.append(np.asarray(power_W, dtype=float).reshape(4).copy())
                d_wall_log.append(float(d_wall))
                wall_active_log.append(float(wall_active))
                tau_wall_b_log.append(np.asarray(tau_wall_b, dtype=float).reshape(3).copy())

            if csv_logger is not None:
                row = {
                    "step": int(k),
                    "t": float(t),
                    "traj": str(traj),
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
                    "xdot_des": float(xdot_des),
                    "ydot_des": float(ydot_des),
                    "ex": float(ex),
                    "ey": float(ey),
                    "ax_cmd": float(ax_cmd),
                    "ay_cmd": float(ay_cmd),
                    "ixy_x": float(ixy_x),
                    "ixy_y": float(ixy_y),
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
                    "wall_model": str(wall_model),
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

        if plot_enabled:
            plot_path = _default_plot_path(args=args)
            _save_timeseries_plots(
                plot_path=plot_path,
                t=np.asarray(t_log, dtype=float),
                pos=np.stack(pos_log, axis=0) if pos_log else np.zeros((0, 3), dtype=float),
                vel=np.stack(vel_log, axis=0) if vel_log else np.zeros((0, 3), dtype=float),
                euler_deg=np.stack(euler_deg_log, axis=0) if euler_deg_log else np.zeros((0, 3), dtype=float),
                ang_vel_w=np.stack(ang_w_log, axis=0) if ang_w_log else np.zeros((0, 3), dtype=float),
                ang_vel_b=np.stack(ang_b_log, axis=0) if ang_b_log else np.zeros((0, 3), dtype=float),
                omega2_cmd=np.stack(omega2_cmd_log, axis=0) if omega2_cmd_log else np.zeros((0, 4), dtype=float),
                omega2_act=np.stack(omega2_act_log, axis=0) if omega2_act_log else np.zeros((0, 4), dtype=float),
                thrust_N=np.stack(thrust_log, axis=0) if thrust_log else np.zeros((0, 4), dtype=float),
                power_W=np.stack(power_log, axis=0) if power_log else np.zeros((0, 4), dtype=float),
                d_wall=np.asarray(d_wall_log, dtype=float),
                wall_active=np.asarray(wall_active_log, dtype=float),
                tau_wall_b=np.stack(tau_wall_b_log, axis=0) if tau_wall_b_log else np.zeros((0, 3), dtype=float),
                dpi=int(args.plot_dpi),
            )

        return 0
    finally:
        try:
            if csv_logger is not None:
                csv_logger.close()
        except Exception:
            pass
        env.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())


