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

| コンポーネント | 担当ライブラリ | 役割・詳細      |
| -------------- | -------------- | --------------- |
| **物理環境**   | **PyBullet**   | ・、 の積分<br> |

<br>・変形機構（ジョイント）の連動計算<br>

<br>・重力、コリオリ力、ジャイロ効果の演算<br>

<br>・地面との衝突判定 |
| **機体モデル** | **URDF** | ・リンク長、質量、慣性モーメントの定義<br>

<br>・可変機構（サーボ関節）の定義 |
| **空力モデル** | **Python (Custom)** | ・回転数  推力・反トルクへの変換<br>

<br>・機体速度  空気抵抗（Drag）の計算 |
| **アクチュエータ** | **Python (Custom)** | ・モータの一次遅れ（応答速度）の模擬<br>

<br>・サーボモータの動作速度制限 |
| **制御ロジック** | **Python (NumPy)** | ・姿勢制御 (PID)<br>

<br>・**動的ミキシング (可変形状対応)** |

---

## 3. 実装詳細

### A. 機体モデリング (URDF)

通常のドローンと異なり、アームやロータが動く「マルチボディ」として定義します。

* **Base:** 機体中心部（バッテリー、FC等）
* **Joints:** チルト機構（Revolute Joint / Servo）
* **Links:** 可動アーム、プロペラ
* *Point:* プロペラ自体も回転ジョイント（Continuous）として定義し、PyBullet上で実際に回転させることで、ジャイロ効果を物理エンジンに自動計算させます。



### B. 空力モデル (自前実装)

PyBulletの標準空気抵抗は使用せず、毎ステップ以下の計算を行い `applyExternalForce` で適用します。

* **座標変換:** ロータ  は変形により傾いているため、ロータ座標系での力  を、その瞬間のリンク姿勢（クォータニオン）を用いてワールド座標系のベクトルに変換して適用します。

### C. 制御則 (PID + Mixing)

変形ドローンの核心部分です。

1. **PIDコントローラ:**
* 目標姿勢と現在姿勢の偏差から、必要な「機体トルク ()`」と「総推力 ($F_{total}$)`」を算出。


2. **動的ミキシング (Control Allocation):**
* その瞬間の変形角度  に基づき、推力発生方向が変化します。
* 関係式: 
* : [総推力, ロールトルク, ピッチトルク, ヨートルク]
* : アロケーション行列（変形角  の関数）
* : 各モータの推力（または回転数の二乗）


* **解法:** 擬似逆行列を用いて、各モータへの指令値を逆算します。





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
    motor_cmds = mixer.solve(target_wrench, joint_angles)

    # 3. アクチュエータモデル (一次遅れ)
    real_motor_rpms = apply_motor_delay(motor_cmds, previous_rpms)

    # 4. 物理エンジンへの適用
    for i, rpm in enumerate(real_motor_rpms):
        # 推力・反トルク計算
        force, torque = calculate_aerodynamics(rpm)
        # PyBulletの各ロータリンクへ外力として適用
        env.apply_force(link_index=i, force=force, torque=torque)
        
        # プロペラ自体も視覚的・物理的に回す
        env.spin_propeller(link_index=i, speed=rpm)

    # 5. 時間を進める
    p.stepSimulation()

```

---

## 5. 開発ロードマップ

1. **Phase 1: 環境構築とベースモデル**
* PyBulletの導入。
* シンプルな「変形しない」クアドコプターをURDFで定義。
* 自前空力モデルで浮上させる（制御なし、ただ上に飛ぶだけ）。


2. **Phase 2: 姿勢制御の実装**
* PID制御の実装。
* ホバリング安定化の確認。


3. **Phase 3: 変形機構の導入**
* URDFにサーボジョイント（チルト機構）を追加。
* PyBullet上でサーボを動かせるようにする。


4. **Phase 4: 動的ミキシングの実装**
* 変形角度に応じたアロケーション行列の実装。
* 変形しながらの飛行（ホバリング中にアーム形状を変えても位置を維持できるか）のテスト。

