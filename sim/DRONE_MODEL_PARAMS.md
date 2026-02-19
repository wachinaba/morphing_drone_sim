# ドローンモデルパラメータ整理

シミュレーションで使うドローンの物理パラメータと、重さ・慣性の調整方法をまとめています。

---

## 1. 使用するURDFと役割

| URDF | 用途 | 質量の扱い |
|------|------|------------|
| `assets/urdf/morphing_drone.urdf` | demo_04, demo_05, demo_06（モーフィング機体） | 全リンクの質量を合算（`env.total_mass()`） |
| `assets/urdf/drone_visual.urdf` | demo_01, demo_02, demo_03（単一リンク・見た目用） | base_link のみ（1リンク） |

---

## 2. 質量・慣性（URDF内）

### 2.1 morphing_drone.urdf

**base_link（機体中心）**

| パラメータ | 現在値 | 単位 | 場所（行） |
|------------|--------|------|------------|
| mass | **1.4** | kg | 23 |
| inertia (Ixx, Iyy) | 6.1833333e-4 | kg·m² | 25-27 |
| inertia (Izz) | 1.1433333e-3 | kg·m² | 25-27 |

- 慣性は base 質量 1.4 kg の箱（0.07×0.07×0.02）を仮定した近似。

**アーム・ロータ各リンク**

- **arm0〜3_link**（アーム棒）: 各 **mass = 0.1** kg → アーム合計 **0.4 kg**。
- その他（yaw/fold/slant/rotor）: 各 **mass = 0.0001** kg、**inertia = 1e-9**（ほぼ点質量）。

**合計質量（morphing_drone）**

- **1.4（base）+ 0.4（アーム4本）+ その他 ≈ 1.8 kg**

※ `sim.env.pybullet_env.PyBulletEnv.total_mass()` が base(-1) + 全ジョイント対応リンクの質量を足して返します。

### 2.2 drone_visual.urdf

- **base_link のみ**: mass = **0.1** kg。慣性は見た目用の近似（Ixx=Iyy≈0.00103, Izz≈0.00206）。
- アーム・ロータは visual のみで質量なし。

---

## 3. 機体まわり幾何（URDF）

| パラメータ | 値 | 説明・場所 |
|------------|-----|------------|
| アーム取り付け位置 (cx, cy) | 0.035 m | base 中心から各アーム根元。joint origin xyz（例: 49行目 `0.035 0.035 0`） |
| アーム長さ (arm_length) | **0.12** m | ヒンジ〜ロータ中心。joint origin xyz（例: tilt_joint `0.12 0 0`） |
| ロータ半径（表示） | 0.10 m | rotor_link の cylinder radius（collision/visual） |

※ 制御・ミキサーで使う「ロータ位置」は、**URDFの現在姿勢から** `env.rotor_geometry_body()` で取得しています。ジョイント角で変わるため、幾何の意味での「アーム長」は 0.12 m です。

---

## 4. シミュレーション・制御まわり（デモ起動引数など）

重力・推力係数・高度などは **URDF ではなく** デモのコマンドライン引数（またはコード内デフォルト）で指定します。

| パラメータ | 典型デフォルト | 説明 | 主な登場箇所 |
|------------|----------------|------|--------------|
| gravity | 9.81 | m/s² | PyBulletEnv, 各 demo |
| CT | 1.0 | 推力係数 T = CT × ω² | demo_04, demo_06, mixer |
| CQ | 0.0 | 反トルク係数 | mixer |
| thrust-to-weight (tw-ratio) | 5.5 | 全推力最大 / 重量。1.8 kg 時はロータあたり約 2 kgf（約24 N） | demo_04, demo_06 |
| body_z / z_des | 0.2〜0.3 m | 初期高度・目標高度 | 各 demo |
| arm_length（幾何のみ） | 0.12 | ミキサー用 pose 計算（demo_01/02/03 の --arm-length デフォルト） | morph.geometry |

質量は **URDF から** PyBullet が読み、`env.total_mass()` で取得。  
ホバ推力は `weight = mass * gravity`、`omega2_hover = weight / (CT * n0z_sum)` などで計算されています。

---

## 5. 重さ・慣性を変えたいとき

### 5.1 質量だけ変える（機体を重く/軽くする）

1. **morphing_drone.urdf を使う場合**
   - **base_link の質量**: `assets/urdf/morphing_drone.urdf` の base_link の `<mass value="1.4"/>` を希望の値（kg）に変更。
   - アーム質量: 各 `arm0_link`〜`arm3_link` の `<mass value="0.1"/>` を変更（現在はアーム合計 0.4 kg）。
   - ロータ等は 0.0001 kg のままなので、base と arm*_link だけで合計質量を決めます。

2. **drone_visual.urdf を使う場合**
   - `assets/urdf/drone_visual.urdf` の 13 行付近 `<mass value="0.1"/>` を変更。

※ シミュレーション上の質量は **URDF のみ** から取られるため、デモの引数で「質量」を直接指定する項目はありません。必ず URDF を編集してください。

### 5.2 慣性テンソルを変える

- **base_link の慣性**: 同じく `morphing_drone.urdf` の base_link の `<inertial>` 内、
  - `<inertia ixx="..." ixy="..." ixz="..." iyy="..." iyz="..." izz="..."/>`
  の数値を変更。対角近似なら `ixy=ixz=iyz="0"` のまま `ixx`, `iyy`, `izz` だけ変えます。
- **drone_visual.urdf**: 上と同様に base_link の `<inertia .../>` を編集。

慣性を大きくすると「回しにくく」なり、小さくすると「回しやすく」なります。オートチューンで目安を出す場合は `demo_05_autotune_inertia.py` を利用できます。

### 5.3 推力・重量バランス（重さを変えたあと）

- 質量を増やすと「同じ omega2_max では推力不足」になりやすいです。
- デモでは **thrust-to-weight** で `omega2_max` を決めているため、`--tw-ratio` を大きくする（例: 3.0 → 4.0）か、`--max-thrust-per-rotor` でロータあたり最大推力 [N] を直接指定すると安定しやすいです。
- 例: 質量を 0.2 kg にした場合  
  `python -m sim.demo.demo_04_hover_pd_urdf_morph --tw-ratio 3.5`  
  のように tw-ratio を少し上げるか、`--max-thrust-per-rotor` で調整。

---

## 6. パラメータの参照まとめ

| 目的 | 編集する場所 |
|------|----------------|
| 機体全体の重さ | URDF の base_link `<mass value="..."/>`（＋必要ならアーム/ロータ各 link の mass） |
| 機体の回りにくさ（慣性） | URDF の base_link `<inertia ixx=... iyy=... izz=.../>` |
| 重力 | デモの `--gravity`（既定 9.81） |
| 推力の強さ | `--CT`, `--tw-ratio`, `--max-thrust-per-rotor` |
| アーム長さ（幾何） | URDF の joint origin（例: tilt_joint の `xyz="0.12 0 0"`）と arm_link の box/origin。デモの --arm-length は 0.12 がデフォルト |

---

## 7. 制御パラメータ（1.8 kg 用デフォルト）

demo_04 / demo_06 では 1.8 kg 機体向けに以下がデフォルトです。

| パラメータ | 値 | 説明 |
|------------|-----|------|
| kp-att | 1.4 | 姿勢 P ゲイン |
| kd-att | 0.22 | 姿勢 D ゲイン |
| kp-z | 10.0 | 高度 P ゲイン |
| kd-z | 6.5 | 高度 D ゲイン |
| lin-damping / ang-damping | 0.08 | リンクの線形・角減衰 |
| physics-hz | 960 | 物理ステップ [Hz]（制御 240 Hz で 4 substep） |
| body-z | 0.35 | 初期高度 [m] |
| motor-tau | 0.04 | モータ一次遅れ [s] |

Fz はミキサー飽和を避けるため、`min(Fz_cmd, max(1.05*weight, 0.92*CT*nz_sum*omega2_max))` でクランプされます。ホバー確認は `run_hover.bat --gui` または `python -m sim.demo.demo_04_hover_pd_urdf_morph --gui` で実行してください。

---

## 8. demo_06 用 XY ゲイン（オートチューニング推奨値）

壁効果デモ（demo_06）の XY 追従ゲインは `sim/tune/demo_06_autotune_xy.py` でオートチューニングできます。モーフなし／あり両方で破綻しないように探索した推奨値（1.8 kg 機体）:

| パラメータ | 推奨値 | 説明 |
|------------|--------|------|
| kp-xy | 2.76 | XY 位置 P ゲイン |
| kd-xy | 2.4 | XY 速度 D ゲイン |
| ki-xy | 0.425 | XY 積分ゲイン（定常偏差低減） |

**チューニング手順（再実行したい場合）**

1. モーフなしのみ:  
   `python -m sim.tune.demo_06_autotune_xy --operating-point baseline --seconds 5 --grid-n 3 --out-csv out/demo06_xy_coarse.csv`
2. モーフなし＋モーフあり両方:  
   `python -m sim.tune.demo_06_autotune_xy --operating-point both --seconds 5 --grid-n 3 --out-csv out/demo06_xy_both.csv`
3. ベストゲインは CLI 形式で出力されるので、`sim_example.bash` の demo_06 コマンドに `--kp-xy ... --kd-xy ... --ki-xy ...` として追記する。

**demo_06 で軌道追従（circle/rect）を使うとき**: 壁は物理衝突ありです。円軌道（中心 (5,0)、半径 0.8）だと y∈[-0.8, 0.8] で、デフォルト壁 y=1.0（内側≈0.95 m）だと隙間が約 0.15 m しかなく、追従誤差で壁に当たることがあります。**衝突を避けるには** `--wall-y 1.5` など壁を遠ざけるか、`--circle-radius 0.5` で円を小さくしてください。`sim_example.bash` 末尾に壁衝突なしの円軌道例（コメント）を載せています。

このファイルは `sim/DRONE_MODEL_PARAMS.md` にあり、重さや慣性を変えるときの参照用にしてください。
