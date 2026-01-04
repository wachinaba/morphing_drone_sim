# 変形ドローン・シミュレーション 実装計画書

## 1. コンセプト
**「厳密な剛体演算は物理エンジンに任せ、ドローンの挙動特性（空力・制御）は自前で精密に実装する」**

* **物理演算:** PyBullet (剛体力学、拘束条件、衝突判定)
* **空力モデル:** 自前実装 (計測値ベースの推力・抗力計算)
* **制御システム:** PID制御 + 動的コントロールアロケーション (Mixing)
* **言語:** Python (数値計算に NumPy を使用)

---

## 2. システムアーキテクチャ

シミュレーションのメインループにおける処理フローと役割分担です。

| コンポーネント   | 担当ライブラリ      | 役割・詳細                                                                                                            |
| :--------------- | :------------------ | :-------------------------------------------------------------------------------------------------------------------- |
| **物理環境**     | **PyBullet**        | ・$F=ma$、$\tau=I\dot{\omega}$ の積分<br>・変形機構（ジョイント）の連動計算<br>・重力、コリオリ力、ジャイロ効果の演算 |
| **機体モデル**   | **URDF**            | ・リンク長、質量、慣性モーメントの定義<br>・可変機構（サーボ関節）の定義                                              |
| **空力モデル**   | **Python (Custom)** | ・回転数 $\to$ 推力・反トルクへの変換<br>・機体速度 $\to$ 空気抵抗（Drag）の計算                                      |
| **制御ロジック** | **Python (NumPy)**  | ・姿勢制御 (PID)<br>・**動的ミキシング (可変形状対応)**                                                               |

---

## 3. 実装詳細（数式モデル）

### A. 空力モデル (Aerodynamics)
物理エンジンの空気抵抗は使わず、自前で計算して物理エンジンに外力として与えます。
各ロータ $i$ において、回転数 $\omega_i$ (rad/s) から推力 $T_i$ と反トルク $Q_i$ を求めます。

$$
T_i = C_T \cdot \omega_i^2
$$

$$
Q_i = C_Q \cdot \omega_i^2
$$

* $C_T$: 推力係数 (Thrust Coefficient)
* $C_Q$: トルク係数 (Torque Coefficient)
* **注意:** 推力ベクトル $\vec{F}_i$ の向きは、変形ドローンの場合、ロータのチルト角によって変化します。

### B. 制御則：動的ミキシング (Dynamic Mixing)
PID制御が出力する「機体に必要な力とトルク」を、その瞬間の変形状態に合わせて各モータへ分配します。

#### 1. 関係式の定義
機体全体に働く力 $F_{total}$ とトルク $\tau$ は、アロケーション行列（Mixing Matrix）$\mathbf{A}$ と、各モータの推力（回転数の二乗ベクトル $\mathbf{\Omega}$）の積で表されます。

$$
\begin{bmatrix} 
F_{total} \\ 
\tau_x \\ 
\tau_y \\ 
\tau_z 
\end{bmatrix} 
= \mathbf{A}(\alpha) 
\begin{bmatrix} 
\omega_1^2 \\ 
\omega_2^2 \\ 
\vdots \\ 
\omega_n^2 
\end{bmatrix}
$$

ここで、$\mathbf{A}(\alpha)$ は **変形角度 $\alpha$ の関数** です。

#### 2. モータ指令値の逆算
制御入力 $\mathbf{u} = [F, \tau_x, \tau_y, \tau_z]^T$ を実現するための各モータ回転数は、行列 $\mathbf{A}$ の **擬似逆行列 (Pseudo-inverse)** $\mathbf{A}^{\dagger}$ を用いて計算します。

$$
\mathbf{\Omega}_{cmd} = \mathbf{A}(\alpha)^{\dagger} \cdot \mathbf{u}_{pid}
$$

Pythonでは `numpy.linalg.pinv` を使用します。

---

## 4. シミュレーションループ (擬似コード)

```python
# 初期化
env = PyBulletEnv(urdf="morphing_drone.urdf")
controller = PIDController()
mixer = DynamicMixer()

while simulation_running:
    # 1. センサ情報の取得 (PyBulletから真値を取得 + ノイズ付加も可)
    pos, quat, vel, ang_vel = env.get_state()
    joint_angles = env.get_servo_angles() # 変形状態

    # 2. 制御演算
    # PIDで必要な力とトルクを計算
    target_wrench = controller.update(target_pos, pos, quat, vel, ang_vel)
    
    # 現在の変形状態(joint_angles)に基づいて各モータ出力を配分
    # ここで行列 A(alpha) の再計算と逆行列演算を行う
    motor_cmds = mixer.solve(target_wrench, joint_angles)

    # 3. アクチュエータモデル (一次遅れ)
    real_motor_rpms = apply_motor_delay(motor_cmds, previous_rpms)

    # 4. 物理エンジンへの適用
    for i, rpm in enumerate(real_motor_rpms):
        # 推力・反トルク計算
        force, torque = calculate_aerodynamics(rpm)
        
        # PyBulletの各ロータリンクへ外力として適用
        # リンクの姿勢(回転行列)を取得して、ロータ座標系の力をワールド座標系へ変換
        env.apply_force(link_index=i, force=force, torque=torque)
        
        # プロペラ自体も視覚的・物理的に回す
        env.spin_propeller(link_index=i, speed=rpm)

    # 5. 時間を進める
    p.stepSimulation()