from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vec3(x: float | np.ndarray | tuple[float, float, float] | list[float], *, name: str) -> np.ndarray:
    """
    Accept scalar or 3-vector and return (3,) float array.
    """
    if isinstance(x, (int, float, np.floating)):
        return np.full((3,), float(x), dtype=float)
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"{name} must be scalar or length-3, got shape={a.shape}")
    return a.reshape(3)


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    PyBullet quaternion (x,y,z,w) -> rotation matrix (3x3).
    R maps body -> world.
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


def rot_z_rad(yaw_rad: float) -> np.ndarray:
    """
    z軸回り回転（body->world の目標姿勢用）
    """
    c = float(np.cos(float(yaw_rad)))
    s = float(np.sin(float(yaw_rad)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot_z_deg(yaw_deg: float) -> np.ndarray:
    return rot_z_rad(np.deg2rad(float(yaw_deg)))


def rotation_from_z_and_yaw(*, z_world: np.ndarray, yaw_deg: float) -> np.ndarray:
    """
    目標の機体z軸（world座標）とyawから、body->world の目標回転行列 R_des を構成する。
    - z_world: 目標の body_z が向く方向（world, unit）
    - yaw_deg: worldの+z軸回りの方位（deg）

    構成（一般的な方法）:
      x_c = [cos(yaw), sin(yaw), 0]
      y_des = normalize(z_des x x_c)
      x_des = y_des x z_des
    """
    z_des = np.asarray(z_world, dtype=float).reshape(3)
    nz = float(np.linalg.norm(z_des))
    if nz < 1e-12:
        raise ValueError("z_world is zero vector")
    z_des = z_des / nz

    yaw = np.deg2rad(float(yaw_deg))
    x_c = np.array([float(np.cos(yaw)), float(np.sin(yaw)), 0.0], dtype=float)
    y_des = np.cross(z_des, x_c)
    ny = float(np.linalg.norm(y_des))
    if ny < 1e-9:
        # z_des is almost parallel to x_c; choose a different reference
        x_c = np.array([0.0, 1.0, 0.0], dtype=float)
        y_des = np.cross(z_des, x_c)
        ny = float(np.linalg.norm(y_des))
        if ny < 1e-9:
            return np.eye(3, dtype=float)
    y_des = y_des / ny
    x_des = np.cross(y_des, z_des)
    x_des = x_des / max(1e-12, float(np.linalg.norm(x_des)))

    R = np.stack([x_des, y_des, z_des], axis=1)  # columns
    return R

def vee_so3(S: np.ndarray) -> np.ndarray:
    """
    so(3)の反対称行列 S からベクトルへ:
      [  0 -z  y]
      [  z  0 -x]  -> [x,y,z]
      [ -y  x  0]
    """
    S = np.asarray(S, dtype=float).reshape(3, 3)
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


@dataclass(frozen=True)
class AttitudePDGains:
    # Scalar gains are applied uniformly to (roll,pitch,yaw).
    # You can also pass a length-3 vector to set axis-wise gains: (kp_x,kp_y,kp_z).
    kp: float | tuple[float, float, float] = 0.15
    kd: float | tuple[float, float, float] = 0.02


@dataclass(frozen=True)
class AltitudePDGains:
    kp_z: float = 6.0
    kd_z: float = 3.5


def attitude_pd_torque_body(
    *,
    R_bw: np.ndarray,
    ang_vel_world: np.ndarray,
    R_des_bw: np.ndarray | None = None,
    gains: AttitudePDGains = AttitudePDGains(),
) -> np.ndarray:
    """
    姿勢PD（ボディ座標トルク）.
    - R_bw: body->world 回転行列
    - ang_vel_world: world角速度 [rad/s]
    - R_des_bw: 目標姿勢（body->world）. None の場合は水平（identity）.
    """
    R = np.asarray(R_bw, dtype=float).reshape(3, 3)
    w_w = np.asarray(ang_vel_world, dtype=float).reshape(3)

    R_des = np.eye(3, dtype=float) if R_des_bw is None else np.asarray(R_des_bw, dtype=float).reshape(3, 3)

    # Standard SO(3) error: e_R = 0.5 * vee(R_des^T R - R^T R_des)
    R_err = R_des.T @ R
    e_R = 0.5 * vee_so3(R_err - R_err.T)

    # Convert angular velocity to body frame, desired w=0
    w_b = R.T @ w_w

    # Negative feedback (axis-wise)
    kp = _as_vec3(gains.kp, name="kp")
    kd = _as_vec3(gains.kd, name="kd")
    tau_b = -kp * e_R - kd * w_b
    return tau_b


def altitude_pd_Fz_body(
    *,
    z_world: float,
    vz_world: float,
    z_des_world: float,
    mass: float,
    gravity: float,
    R_bw: np.ndarray,
    gains: AltitudePDGains = AltitudePDGains(),
    Fz_min: float = 0.0,
    Fz_max: float | None = None,
) -> float:
    """
    高度PDで必要な「世界Z方向の力」を作り、ボディz推力(Fz_body)に変換する。

    mixer は u=[Fz_body, tau_x, tau_y, tau_z] を想定するため、
    worldのZ成分 Fz_world を、body_z が世界zへ投影される分で割って Fz_body にする:
      Fz_world = (R_bw @ [0,0,Fz_body])_z = R_bw[2,2] * Fz_body
      => Fz_body = Fz_world / R_bw[2,2]

    ただし大きく傾くと R_bw[2,2] が小さくなり発散するため、下限を設ける。
    """
    R = np.asarray(R_bw, dtype=float).reshape(3, 3)
    z = float(z_world)
    vz = float(vz_world)
    z_des = float(z_des_world)

    # Desired vertical acceleration
    az_cmd = float(gains.kp_z) * (z_des - z) + float(gains.kd_z) * (0.0 - vz)
    Fz_world = float(mass) * (float(gravity) + az_cmd)

    # Convert to body Fz using the projection of body-z onto world-z
    c = float(R[2, 2])
    c_eff = max(0.25, c)  # avoid blow-up when tilted
    Fz_body = Fz_world / c_eff

    # Clamp
    Fz_body = max(float(Fz_min), float(Fz_body))
    if Fz_max is not None:
        Fz_body = min(float(Fz_max), float(Fz_body))
    return float(Fz_body)


def altitude_pd_Fz_world(
    *,
    z_world: float,
    vz_world: float,
    z_des_world: float,
    mass: float,
    gravity: float,
    gains: AltitudePDGains = AltitudePDGains(),
    Fz_min: float = 0.0,
    Fz_max: float | None = None,
) -> float:
    """
    高度PDで必要な「世界Z方向の力」Fz_world を返す。

    モーフィング等で推力方向がbody zと一致しない場合は、
    mixer側の1行目も「world-Z成分」をターゲットにする設計（推奨）。
    """
    z = float(z_world)
    vz = float(vz_world)
    z_des = float(z_des_world)

    az_cmd = float(gains.kp_z) * (z_des - z) + float(gains.kd_z) * (0.0 - vz)
    Fz_world = float(mass) * (float(gravity) + az_cmd)

    Fz_world = max(float(Fz_min), float(Fz_world))
    if Fz_max is not None:
        Fz_world = min(float(Fz_max), float(Fz_world))
    return float(Fz_world)


