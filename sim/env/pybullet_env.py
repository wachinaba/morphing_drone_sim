from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidBodyState:
    pos: np.ndarray      # (3,) world [m]
    quat: np.ndarray     # (4,) world (x,y,z,w) in PyBullet convention
    vel: np.ndarray      # (3,) world [m/s]
    ang_vel: np.ndarray  # (3,) world [rad/s]


class PyBulletEnv:
    """
    PyBullet の最小ラッパ.
    - connect/disconnect
    - load URDF
    - get_state (base link)
    - apply external force/torque at a world position
    - step
    """

    def __init__(
        self,
        *,
        gui: bool = False,
        time_step: float = 1.0 / 240.0,
        gravity: float = 9.81,
        gui_width: int | None = None,
        gui_height: int | None = None,
    ):
        import pybullet as p
        import pybullet_data

        self._p = p
        if bool(gui) and (gui_width is not None or gui_height is not None):
            w = None if gui_width is None else int(gui_width)
            h = None if gui_height is None else int(gui_height)
            opts = []
            if w is not None and w > 0:
                opts.append(f"--width={w}")
            if h is not None and h > 0:
                opts.append(f"--height={h}")
            options = " ".join(opts) if opts else None
            self._cid = self._p.connect(self._p.GUI, options=options)
        else:
            self._cid = self._p.connect(self._p.GUI if bool(gui) else self._p.DIRECT)
        if self._cid < 0:
            raise RuntimeError("Failed to connect to PyBullet.")

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0.0, 0.0, -float(gravity))
        self._p.setTimeStep(float(time_step))

        self._body_id: int | None = None
        self._plane_id: int | None = None

    def disconnect(self):
        if self._cid is not None:
            try:
                self._p.disconnect()
            finally:
                self._cid = None

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    @property
    def p(self):
        return self._p

    @property
    def body_id(self) -> int:
        if self._body_id is None:
            raise RuntimeError("Body not loaded yet. Call load_body_urdf() first.")
        return int(self._body_id)

    def load_plane(self):
        if self._plane_id is None:
            self._plane_id = int(self._p.loadURDF("plane.urdf"))
        return int(self._plane_id)

    def load_body_urdf(self, urdf_path: str, *, base_pos=(0.0, 0.0, 0.2), base_quat=(0.0, 0.0, 0.0, 1.0)) -> int:
        self._body_id = int(self._p.loadURDF(str(urdf_path), basePosition=base_pos, baseOrientation=base_quat, useFixedBase=False))
        return int(self._body_id)

    def set_camera(self, *, distance: float, yaw: float, pitch: float, target: tuple[float, float, float]):
        """
        PyBullet GUI のカメラを移動する。
        - distance: カメラ距離
        - yaw: ヨー角 [deg]
        - pitch: ピッチ角 [deg]
        - target: 注視点（world）
        """
        self._p.resetDebugVisualizerCamera(
            cameraDistance=float(distance),
            cameraYaw=float(yaw),
            cameraPitch=float(pitch),
            cameraTargetPosition=[float(target[0]), float(target[1]), float(target[2])],
        )

    def start_video_recording(self, path: str) -> int | None:
        """
        Start MP4 video logging via PyBullet state logging.
        Returns a log id that should be passed to stop_video_recording().
        If the backend doesn't support MP4 logging, returns None.
        """
        try:
            log_id = self._p.startStateLogging(self._p.STATE_LOGGING_VIDEO_MP4, str(path))
            return int(log_id)
        except Exception:
            return None

    def stop_video_recording(self, log_id: int | None):
        if log_id is None:
            return
        try:
            self._p.stopStateLogging(int(log_id))
        except Exception:
            pass

    def _joint_name_to_index(self) -> dict[str, int]:
        n = int(self._p.getNumJoints(self.body_id))
        out: dict[str, int] = {}
        for j in range(n):
            info = self._p.getJointInfo(self.body_id, j)
            name = info[1].decode("utf-8")
            out[name] = int(j)
        return out

    def _link_name_to_index(self) -> dict[str, int]:
        n = int(self._p.getNumJoints(self.body_id))
        out: dict[str, int] = {"base_link": -1}
        for j in range(n):
            info = self._p.getJointInfo(self.body_id, j)
            link_name = info[12].decode("utf-8")
            out[link_name] = int(j)  # in PyBullet, link index == joint index
        return out

    def configure_morphing_drone(self):
        """
        morphing_drone.urdf 用の関節/リンクindexを解決してキャッシュする。
        """
        self._joint_map = self._joint_name_to_index()
        self._link_map = self._link_name_to_index()

        # joints: arm{0..3}_{fold,slant,tilt}_joint
        self._morph_joints: dict[str, list[int]] = {"fold": [], "slant": [], "tilt": []}
        for i in range(4):
            for kind in ("fold", "slant", "tilt"):
                key = f"arm{i}_{kind}_joint"
                if key not in self._joint_map:
                    raise RuntimeError(f"Expected joint {key!r} in URDF but not found.")
                self._morph_joints[kind].append(int(self._joint_map[key]))

        # rotor links: rotor{0..3}_link
        self._rotor_links: list[int] = []
        for i in range(4):
            lname = f"rotor{i}_link"
            if lname not in self._link_map:
                raise RuntimeError(f"Expected link {lname!r} in URDF but not found.")
            self._rotor_links.append(int(self._link_map[lname]))

        # Optional morph servo state (initialized lazily)
        self._morph_servo = None

    def set_damping_all(self, *, linear: float = 0.0, angular: float = 0.0):
        """
        Set linear/angular damping for base and all links.
        Useful to emulate aerodynamic damping and avoid long-horizon energy build-up.
        """
        lin = max(0.0, float(linear))
        ang = max(0.0, float(angular))
        # base (-1) and all joints/links
        self._p.changeDynamics(self.body_id, -1, linearDamping=lin, angularDamping=ang)
        for li in range(int(self._p.getNumJoints(self.body_id))):
            self._p.changeDynamics(self.body_id, int(li), linearDamping=lin, angularDamping=ang)

    def set_morph_angles_servo(
        self,
        *,
        phi_deg: float,
        psi_deg: float,
        theta_deg: float,
        dt: float,
        symmetry: str = "mirror_xy",
        tau: float = 0.08,
        rate_limit: float | None = None,
    ) -> tuple[float, float, float]:
        """
        morph角を「サーボ（一次遅れ＋速度制限）」で追従させて関節へ反映する。
        Returns: (phi_applied_deg, psi_applied_deg, theta_applied_deg)
        """
        if not hasattr(self, "_morph_joints"):
            self.configure_morphing_drone()

        # Lazy init servo with 3 states (phi,psi,theta) in radians
        if self._morph_servo is None:
            from sim.control.servo_model import ServoModel

            self._morph_servo = ServoModel(tau=float(tau), rate_limit=(None if rate_limit is None else float(rate_limit)))
            self._morph_servo.reset(np.zeros((3,), dtype=float))

        q_cmd = np.array([np.deg2rad(float(phi_deg)), np.deg2rad(float(psi_deg)), np.deg2rad(float(theta_deg))], dtype=float)
        q = self._morph_servo.step(q_cmd, float(dt))
        phi_a, psi_a, th_a = float(np.rad2deg(q[0])), float(np.rad2deg(q[1])), float(np.rad2deg(q[2]))

        # Apply to joints using the same symmetry logic
        self.set_morph_angles(phi_deg=phi_a, psi_deg=psi_a, theta_deg=th_a, symmetry=str(symmetry))
        return phi_a, psi_a, th_a

    def set_morph_angles(self, *, phi_deg: float, psi_deg: float, theta_deg: float, symmetry: str = "none"):
        """
        morphing_drone の各アームに (phi,psi,theta) を設定する。

        symmetry:
          - "none": 各アームに同一の (phi,psi,theta) を入れる（URDFの素直な解釈）
          - "mirror_xy": `visualize_morph_drone.py` の mirror_xy と一致するよう、
                         象限ごとに phi/theta の符号を切り替える（実測で一致を確認済み）

        符号規約（visualize_morph_drone準拠）:
          fold  = -phi
          slant = -psi
          tilt  = +theta
        """
        if not hasattr(self, "_morph_joints"):
            self.configure_morphing_drone()

        symmetry = str(symmetry)
        if symmetry not in {"none", "mirror_xy"}:
            raise ValueError(f"Unknown symmetry: {symmetry!r}")

        # Per-arm sign for mirror_xy equivalence (arm order: 0:+,+ 1:-,+ 2:-,- 3:+,-)
        # Empirically verified to match visualize_morph_drone.py symmetry="mirror_xy":
        #   phi_i   = s_i * phi
        #   psi_i   = psi
        #   theta_i = s_i * theta
        # where s = [+1,-1,+1,-1] (i.e., sign(x*y) per quadrant).
        s = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float) if symmetry == "mirror_xy" else np.ones((4,), dtype=float)

        phi_base = float(phi_deg)
        psi_base = float(psi_deg)
        th_base = float(theta_deg)

        # resetJointState is simplest/stable for kinematic morphing in MVP
        for i in range(4):
            phi_i = float(s[i]) * phi_base
            psi_i = psi_base
            th_i = float(s[i]) * th_base

            fold = -np.deg2rad(phi_i)
            slant = -np.deg2rad(psi_i)
            tilt = +np.deg2rad(th_i)

            self._p.resetJointState(self.body_id, self._morph_joints["fold"][i], targetValue=float(fold))
            self._p.resetJointState(self.body_id, self._morph_joints["slant"][i], targetValue=float(slant))
            self._p.resetJointState(self.body_id, self._morph_joints["tilt"][i], targetValue=float(tilt))

    def rotor_link_indices(self) -> list[int]:
        if not hasattr(self, "_rotor_links"):
            self.configure_morphing_drone()
        return list(self._rotor_links)

    def apply_rotor_thrust_link_frame(
        self,
        *,
        rotor_idx: int,
        thrust: float,
        reaction_torque: float = 0.0,
    ):
        """
        rotorリンク座標系で推力（+Z）と反トルク（+Z）を適用する。
        - thrust: [N]  +Z 方向
        - reaction_torque: [N*m] +Z 軸回り
        """
        links = self.rotor_link_indices()
        li = int(links[int(rotor_idx)])
        f_local = [0.0, 0.0, float(thrust)]
        t_local = [0.0, 0.0, float(reaction_torque)]
        # Apply at link COM (pos=(0,0,0) in link frame)
        self._p.applyExternalForce(self.body_id, li, f_local, [0.0, 0.0, 0.0], self._p.LINK_FRAME)
        if abs(float(reaction_torque)) > 0.0:
            self._p.applyExternalTorque(self.body_id, li, t_local, self._p.LINK_FRAME)

    def apply_body_wrench_world(self, *, force_world: tuple[float, float, float] = (0.0, 0.0, 0.0), torque_world: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        base_link（linkIndex=-1）に、world座標で外力/外トルクを加える。
        """
        fx, fy, fz = (float(force_world[0]), float(force_world[1]), float(force_world[2]))
        tx, ty, tz = (float(torque_world[0]), float(torque_world[1]), float(torque_world[2]))
        self._p.applyExternalForce(self.body_id, -1, [fx, fy, fz], [0.0, 0.0, 0.0], self._p.WORLD_FRAME)
        if abs(tx) > 0.0 or abs(ty) > 0.0 or abs(tz) > 0.0:
            self._p.applyExternalTorque(self.body_id, -1, [tx, ty, tz], self._p.WORLD_FRAME)

    def apply_link_wrench_world(
        self,
        *,
        link_index: int,
        force_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
        torque_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        任意リンクに、world座標で外力/外トルクを加える。
        link_index は PyBullet の link index（baseは -1）。
        """
        li = int(link_index)
        fx, fy, fz = (float(force_world[0]), float(force_world[1]), float(force_world[2]))
        tx, ty, tz = (float(torque_world[0]), float(torque_world[1]), float(torque_world[2]))
        self._p.applyExternalForce(self.body_id, li, [fx, fy, fz], [0.0, 0.0, 0.0], self._p.WORLD_FRAME)
        if abs(tx) > 0.0 or abs(ty) > 0.0 or abs(tz) > 0.0:
            self._p.applyExternalTorque(self.body_id, li, [tx, ty, tz], self._p.WORLD_FRAME)

    def apply_rotor_wrench_world(
        self,
        *,
        rotor_idx: int,
        force_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
        torque_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        rotor_idx(0..3) の rotor_link に、world座標で外力/外トルクを加える。
        """
        links = self.rotor_link_indices()
        li = int(links[int(rotor_idx)])
        self.apply_link_wrench_world(link_index=li, force_world=force_world, torque_world=torque_world)

    def rotor_geometry_body(self) -> tuple[np.ndarray, np.ndarray]:
        """
        現在のURDF姿勢から、ロータ中心 r_i と推力方向 n_i をボディ座標で返す。
        - r: (4,3) base->rotor_center in body frame
        - n: (4,3) rotor normal (+Z axis) in body frame
        """
        links = self.rotor_link_indices()
        # base world pose
        pos_b, quat_b = self._p.getBasePositionAndOrientation(self.body_id)
        R_bw = np.asarray(self._p.getMatrixFromQuaternion(quat_b), dtype=float).reshape(3, 3)
        p_b = np.asarray(pos_b, dtype=float).reshape(3)

        rs = []
        ns = []
        for li in links:
            st = self._p.getLinkState(self.body_id, int(li), computeForwardKinematics=True)
            p_w = np.asarray(st[4], dtype=float).reshape(3)  # worldLinkFramePosition
            q_w = np.asarray(st[5], dtype=float).reshape(4)  # worldLinkFrameOrientation (x,y,z,w)
            R_lw = np.asarray(self._p.getMatrixFromQuaternion(q_w), dtype=float).reshape(3, 3)
            n_w = R_lw @ np.array([0.0, 0.0, 1.0], dtype=float)
            # body frame vectors
            r_b = R_bw.T @ (p_w - p_b)
            n_b = R_bw.T @ n_w
            rs.append(r_b)
            ns.append(n_b)
        return np.stack(rs, axis=0), np.stack(ns, axis=0)

    def get_state(self) -> RigidBodyState:
        pos, quat = self._p.getBasePositionAndOrientation(self.body_id)
        vel, ang = self._p.getBaseVelocity(self.body_id)
        return RigidBodyState(
            pos=np.asarray(pos, dtype=float),
            quat=np.asarray(quat, dtype=float),
            vel=np.asarray(vel, dtype=float),
            ang_vel=np.asarray(ang, dtype=float),
        )

    def total_mass(self) -> float:
        """
        マルチボディ全体の質量（base + 全リンク）を返す。
        """
        m = float(self._p.getDynamicsInfo(self.body_id, -1)[0])
        n = int(self._p.getNumJoints(self.body_id))
        for li in range(n):
            m += float(self._p.getDynamicsInfo(self.body_id, li)[0])
        return float(m)

    def apply_wrench_world(self, *, world_pos: np.ndarray, world_force: np.ndarray, world_torque: np.ndarray | None = None):
        """
        ボディ（base）に対して、ワールド座標で外力を適用する.
        - world_pos: 作用点（world）
        - world_force: 力（world）
        - world_torque: トルク（world, 任意）
        """
        wp = np.asarray(world_pos, dtype=float).reshape(3).tolist()
        wf = np.asarray(world_force, dtype=float).reshape(3).tolist()
        self._p.applyExternalForce(self.body_id, -1, wf, wp, self._p.WORLD_FRAME)
        if world_torque is not None:
            wt = np.asarray(world_torque, dtype=float).reshape(3).tolist()
            self._p.applyExternalTorque(self.body_id, -1, wt, self._p.WORLD_FRAME)

    def step(self, n: int = 1):
        for _ in range(int(n)):
            self._p.stepSimulation()


