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
| mass | **0.1** | kg | 23 |
| inertia (Ixx, Iyy) | 0.0010316666667 | kg·m² | 24-25 |
| inertia (Izz) | 0.0020606666667 | kg·m² | 26 |

- 慣性は「0.05×0.05×0.05 m の立方体」を仮定した近似値。

**アーム・ロータ各リンク（arm0〜3 の yaw/fold/slant/link + rotor0〜3_link）**

- 各リンク: **mass = 0.0001** kg、**inertia = 1e-9**（ほぼ点質量）。
- アーム4本 × 5リンク + ロータ4 = 合計 24 リンク → 追加質量 24×0.0001 = **0.0024 kg**。

**合計質量（morphing_drone）**

- **0.1 + 0.0024 = 0.1024 kg**

※ `sim.env.pybullet_env.PyBulletEnv.total_mass()` が base(-1) + 全ジョイント対応リンクの質量を足して返します。

### 2.2 drone_visual.urdf

- **base_link のみ**: mass = **0.1** kg、慣性は morphing_drone の base と同じ値（Ixx=Iyy≈0.00103, Izz≈0.00206）。
- アーム・ロータは visual のみで質量なし。

---

## 3. 機体まわり幾何（URDF）

| パラメータ | 値 | 説明・場所 |
|------------|-----|------------|
| アーム取り付け位置 (cx, cy) | 0.035 m | base 中心から各アーム根元。joint origin xyz（例: 49行目 `0.035 0.035 0`） |
| アーム長さ (arm_length) | **0.18** m | ヒンジ〜ロータ中心。joint origin xyz（例: 108行目 `0.18 0 0`） |
| ロータ半径（表示） | 0.10 m | rotor_link の cylinder radius（collision/visual） |

※ 制御・ミキサーで使う「ロータ位置」は、**URDFの現在姿勢から** `env.rotor_geometry_body()` で取得しています。ジョイント角で変わるため、幾何の意味での「アーム長」は 0.18 m です。

---

## 4. シミュレーション・制御まわり（デモ起動引数など）

重力・推力係数・高度などは **URDF ではなく** デモのコマンドライン引数（またはコード内デフォルト）で指定します。

| パラメータ | 典型デフォルト | 説明 | 主な登場箇所 |
|------------|----------------|------|--------------|
| gravity | 9.81 | m/s² | PyBulletEnv, 各 demo |
| CT | 1.0 | 推力係数 T = CT × ω² | demo_04, demo_06, mixer |
| CQ | 0.0 | 反トルク係数 | mixer |
| thrust-to-weight (tw-ratio) | 3.0 | 全推力最大 / 重量 | omega2_max の算出に使用 |
| body_z / z_des | 0.2〜0.3 m | 初期高度・目標高度 | 各 demo |
| arm_length（幾何のみ） | 0.18 | ミキサー用 pose 計算（demo_03 等で --arm-length） | morph.geometry |

質量は **URDF から** PyBullet が読み、`env.total_mass()` で取得。  
ホバ推力は `weight = mass * gravity`、`omega2_hover = weight / (CT * n0z_sum)` などで計算されています。

---

## 5. 重さ・慣性を変えたいとき

### 5.1 質量だけ変える（機体を重く/軽くする）

1. **morphing_drone.urdf を使う場合**
   - **base_link の質量**: `assets/urdf/morphing_drone.urdf` の 20〜27 行付近。
   - `<mass value="0.1"/>` の `0.1` を希望の値（kg）に変更。
   - アーム・ロータの 0.0001 kg はそのままでよければ、ここだけ変えれば全体質量が変わります。
   - アーム質量も変える場合: 各 `<link name="arm...">` / `<link name="rotor..._link">` 内の `<mass value="0.0001"/>` を一括で変更。

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
| アーム長さ（幾何） | URDF の joint origin（例: tilt_joint の `xyz="0.18 0 0"`）と、geometry の arm_length を一致させる想定 |

このファイルは `sim/DRONE_MODEL_PARAMS.md` にあり、重さや慣性を変えるときの参照用にしてください。
