## demo_06_wall_effect 制御系（概略）

以下のMermaid図は `sim/demo/demo_06_wall_effect.py` の制御ループ構造を「信号の流れ」と「更新周期（制御 dt / 物理 dt_phys）」の観点でまとめたものです。

```mermaid
flowchart TD
  %% Params / Inputs
  P["CLI params\ntraj, gains, yaw, morph, wall\nhz / physics-hz"] --> TG
  P --> XY
  P --> YAW
  P --> MORPH_CMD
  P --> WALL
  P --> VIZ

  %% Physics / State
  PHY["PyBullet physics\n(dt_phys)"] --> S["State st\npos, vel, quat, ang_vel"]
  S --> TG
  S --> XY
  S --> YAW
  S --> ATT
  S --> ALT
  S --> GEOM
  S --> VIZ

  %% Trajectory generator
  TG["Trajectory generator\nramp / circle / rect\nx_des, y_des, xdot_des, ydot_des"] --> XY

  %% XY -> desired attitude (z direction)
  XY["XY controller (world)\nex, ey\nax_cmd, ay_cmd\n(tilt clamp)"] --> ZDIR["z_dir_des (world)"]
  ZDIR --> RDES["R_des = rotation_from_z_and_yaw\n(z_dir_des + yaw_cmd)"]

  %% Yaw command (filtered)
  YAW["Yaw target\nbase + spin + sine\nfilter (yaw_tau) + rate limit (yaw_rate)"] --> RDES

  %% Attitude/Altitude PD
  ATT["Attitude PD\n-> tau_b (body)"] --> WRENCH
  ALT["Altitude PD\n-> Fz_cmd (world Z)"] --> WRENCH
  WRENCH["Target wrench u (world)\n[Fz_cmd, tau_world]"] --> MIX

  %% Allocation / Mixer
  GEOM["Rotor geometry from URDF\nr_i, n_i (body) -> world"] --> A["Allocation matrix A (world)"]
  A --> MIX["Mixer solve\nomega2_cmd (fallback/desat)"]

  %% Motor + Apply forces
  MIX --> MOTOR["MotorModel\nomega2_act"]
  MOTOR --> APPLY["Apply thrust/torque\nper-rotor (link frame)"]
  APPLY --> PHY

  %% Morphing joints (kinematic/servo)
  MORPH_CMD["Morph command\nramp + sine\nphi/psi/theta"] --> MORPH["set_morph_angles\n(every physics substep)"]
  MORPH --> PHY

  %% Wall disturbance (optional)
  WALL["Wall model\n(active region, distance)\n-> tau_wall"] --> DIST["Apply disturbance torque (world)"]
  DIST --> PHY

  %% Camera / debug draw (optional)
  VIZ["GUI camera + debug draw\ntrajectory lines / target marker\n(deterministic video option)"] --> PHY
```

## demo_06_wall_effect 制御器ブロック図（誤差 -> 指令 -> 推力 -> 機体 -> フィードバック）

「軌道追従・姿勢・高度・ヨー」の**制御器としての中身**だけを抜き出した図です（壁外乱やカメラ描画などは除外）。

```mermaid
flowchart TD
  %% --- References ---
  REF["Reference\nx_des, y_des, xdot_des, ydot_des, z_des"] --> E_XY
  REF --> E_Z
  REF --> YAW_REF

  %% --- Feedback state ---
  PLANT["Rigid body + rotors\n(PyBullet)"] --> ST["State\npos, vel, quat, ang_vel"]
  ST --> E_XY
  ST --> E_Z
  ST --> ATT_PD
  ST --> ALLOC

  %% --- XY (world) -> desired z direction ---
  E_XY["XY error\nex = x_des - x\n ey = y_des - y"] --> XY_PID
  ST -->|"vx, vy"| XY_PID
  REF -->|"xdot_des, ydot_des"| XY_PID
  XY_PID["XY controller (world)\nax_cmd, ay_cmd\nP + D + I\n(tilt clamp)"] --> ZDIR["Desired z axis (world)\nz_dir_des"]

  %% --- Yaw command (target -> filter) ---
  YAW_REF["Yaw target\nbase + spin + sine"] --> YAW_F
  ST -->|"yaw_filt"| YAW_F
  YAW_F["Yaw filter\n(yaw_tau) + rate limit\n(yaw_rate)"] --> YAW_CMD["yaw_cmd"]

  %% --- Attitude reference from z_dir + yaw ---
  ZDIR --> RDES
  YAW_CMD --> RDES
  RDES["R_des from\nz_dir_des + yaw_cmd"] --> ATT_PD

  %% --- Attitude PD -> torque (body) ---
  ATT_PD["Attitude PD\n(R_des, R, w)\n-> tau_b"] --> TAU_W["tau_world"]

  %% --- Altitude PD -> Fz (world Z) ---
  E_Z["Altitude error\nz_des - z"] --> ALT_PD
  ST -->|"z, vz"| ALT_PD
  ALT_PD["Altitude PD\n-> Fz_cmd (world Z)"] --> FZ["Fz_cmd"]

  %% --- Allocation / mixing ---
  FZ --> MIX
  TAU_W --> MIX
  ALLOC["Rotor geometry -> A\n(world frame)"] --> MIX
  MIX["Mixer\nsolve omega2_cmd\n(desat / fallback)"] --> MOTOR
  MOTOR["MotorModel\n(1st order)\n-> omega2_act"] --> APPLY
  APPLY["Apply thrust + reaction torque\n(per rotor)"] --> PLANT
```

