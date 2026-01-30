from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixResult:
    """
    動的ミキシング結果.

    omega2_cmd:
      - (N,) の ω^2 コマンド
      - CT=1 の場合は「推力[N]」として扱ってOK（MVP用の簡略）
    """

    omega2_cmd: np.ndarray  # (N,)
    omega2_raw: np.ndarray  # (N,) before clipping / fallback
    saturated: bool
    mode: str  # "pinv" | "fz_priority"
    # Diagnostics / tuning knobs actually used
    omega2_max_eff: float | None
    torque_weights: tuple[float, float, float]
    fallback_ridge: float
    fallback_auto_omega2_factor: float
    wrench_achieved: np.ndarray  # (4,)
    wrench_target: np.ndarray  # (4,)
    residual: np.ndarray  # (4,)
    A: np.ndarray  # (4,N)


def build_allocation_matrix(
    *,
    r_body: np.ndarray,
    n_body: np.ndarray,
    C_T: float = 1.0,
    C_Q: float = 0.0,
    spin_dir: np.ndarray | None = None,
) -> np.ndarray:
    """
    アロケーション行列 A を構築する。

    定義（`impl_plan_detailed.md` 準拠）:
      u = [Fz, tau_x, tau_y, tau_z]^T
      omega2_i = ω_i^2  (>=0)
      F_i = (C_T * omega2_i) * n_i
      tau_arm_i = r_i x F_i = C_T * omega2_i * (r_i x n_i)

    反トルク（ロータ回転によるdrag torque）も入れる場合:
      tau_drag_i = s_i * (C_Q * omega2_i) * n_i
      ここで s_i は回転方向（+1/-1）

    よって列 i:
      A[:,i] = [C_T*nz, C_T*(r×n)_x, C_T*(r×n)_y, C_T*(r×n)_z]^T
      （さらに反トルクを加える場合は tau 成分に + s_i*(C_Q*n) を加算）
    """
    r = np.asarray(r_body, dtype=float)
    n = np.asarray(n_body, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r_body must be (N,3), got {r.shape}")
    if n.ndim != 2 or n.shape[1] != 3 or n.shape[0] != r.shape[0]:
        raise ValueError(f"n_body must be (N,3) and match r_body, got {n.shape} vs {r.shape}")

    N = int(r.shape[0])
    if spin_dir is None:
        s = np.ones((N,), dtype=float)
    else:
        s = np.asarray(spin_dir, dtype=float).reshape(N)
    A = np.zeros((4, N), dtype=float)
    rn = np.cross(r, n)  # (N,3)
    A[0, :] = float(C_T) * n[:, 2]
    A[1, :] = float(C_T) * rn[:, 0] + float(C_Q) * s[:] * n[:, 0]
    A[2, :] = float(C_T) * rn[:, 1] + float(C_Q) * s[:] * n[:, 1]
    A[3, :] = float(C_T) * rn[:, 2] + float(C_Q) * s[:] * n[:, 2]
    return A


def solve_mixer_pinv(
    *,
    A: np.ndarray,
    wrench_target: np.ndarray,
    omega2_min: float = 0.0,
    omega2_max: float | None = None,
) -> MixResult:
    """
    擬似逆行列で omega^2 を解き、非負/上限でクリップする。
    """
    A = np.asarray(A, dtype=float)
    u = np.asarray(wrench_target, dtype=float).reshape(4)
    if A.ndim != 2 or A.shape[0] != 4:
        raise ValueError(f"A must be (4,N), got {A.shape}")

    omega2_raw = np.linalg.pinv(A) @ u  # (N,)
    omega2_raw = np.asarray(omega2_raw, dtype=float).reshape(-1)

    lo = float(omega2_min)
    hi = float(omega2_max) if omega2_max is not None else None
    if hi is None:
        omega2_clip = np.maximum(omega2_raw, lo)
    else:
        omega2_clip = np.clip(omega2_raw, lo, hi)

    saturated = bool(np.any(np.abs(omega2_clip - omega2_raw) > 1e-12))
    u_hat = A @ omega2_clip
    res = u - u_hat
    return MixResult(
        omega2_cmd=omega2_clip,
        omega2_raw=omega2_raw,
        saturated=saturated,
        mode="pinv",
        omega2_max_eff=(None if omega2_max is None else float(omega2_max)),
        torque_weights=(1.0, 1.0, 1.0),
        fallback_ridge=0.0,
        fallback_auto_omega2_factor=0.0,
        wrench_achieved=u_hat,
        wrench_target=u,
        residual=res,
        A=A,
    )


def _solve_bounded_wls_active_set(
    *,
    A: np.ndarray,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    W: np.ndarray,
    ridge: float,
    x0: np.ndarray,
    max_iter: int = 20,
    eps: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """
    Bound-constrained weighted least squares via simple active-set iterations.
      minimize || W (A x - b) ||^2 + ridge ||x||^2
      s.t. lo <= x <= hi

    Returns (x, saturated_flag) where saturated_flag is True if any actuator is on a bound.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(A.shape[0])
    lo = np.asarray(lo, float).reshape(A.shape[1])
    hi = np.asarray(hi, float).reshape(A.shape[1])
    W = np.asarray(W, float).reshape(A.shape[0], A.shape[0])
    x = np.asarray(x0, float).reshape(A.shape[1])

    x = np.clip(x, lo, hi)
    lam = max(0.0, float(ridge))
    x_ref = np.asarray(x0, float).reshape(A.shape[1])

    for _ in range(int(max_iter)):
        at_lo = x <= (lo + eps)
        at_hi = x >= (hi - eps)
        fixed = at_lo | at_hi
        free = ~fixed

        if not np.any(free):
            break

        # Solve for free variables given fixed ones.
        Af = A[:, free]
        xf = x[free]
        rhs = b - A[:, fixed] @ x[fixed]

        WAf = W @ Af
        Wrhs = W @ rhs

        if Af.shape[1] == 0:
            break

        if lam > 0.0:
            # Tikhonov regularization toward x_ref (not toward 0):
            # minimize ||W(Ax-b)||^2 + lam||x-x_ref||^2
            H = WAf.T @ WAf + lam * np.eye(Af.shape[1], dtype=float)
            g = WAf.T @ Wrhs + lam * x_ref[free]
            xf_new = np.linalg.solve(H, g)
        else:
            xf_new = np.linalg.pinv(WAf) @ Wrhs

        x_new = x.copy()
        x_new[free] = xf_new
        x_new = np.clip(x_new, lo, hi)

        if float(np.linalg.norm(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new

    saturated = bool(np.any(x <= (lo + eps)) or np.any(x >= (hi - eps)))
    return x, saturated


def _allocate_fz_greedy(
    *,
    A0: np.ndarray,
    Fz_target: float,
    omega2_min: float,
    omega2_max: float | None,
) -> np.ndarray:
    """
    Fz（u[0]）を最優先で満たすための簡易配分（貪欲）。
    A0: (N,) where Fz = sum_i A0[i] * omega2[i]
    """
    A0 = np.asarray(A0, dtype=float).reshape(-1)
    N = int(A0.shape[0])
    out = np.full((N,), float(omega2_min), dtype=float)
    rem = float(Fz_target) - float(np.dot(A0, out))
    if rem <= 0.0:
        return out

    hi = None if omega2_max is None else float(omega2_max)
    # Prefer large A0 (more Fz per omega2)
    order = list(np.argsort(-A0))
    for i in order:
        a = float(A0[i])
        if a <= 1e-12:
            continue
        cap = float("inf") if hi is None else max(0.0, hi - out[i])
        if cap <= 0.0:
            continue
        d = min(cap, rem / a)
        if d <= 0.0:
            continue
        out[i] += d
        rem -= a * d
        if rem <= 1e-9:
            break
    return out


def solve_mixer_with_fallback(
    *,
    A: np.ndarray,
    wrench_target: np.ndarray,
    omega2_min: float = 0.0,
    omega2_max: float | None = None,
    torque_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fz_weight: float = 10.0,
    fallback_ridge: float = 1e-6,
    fallback_auto_omega2_factor: float = 10.0,
) -> MixResult:
    """
    まず pinv で解き、飽和した場合は Fz優先フォールバックを行う。

    フォールバック方針:
    - PX4ライクに「制約付きWLS + 逐次デサチュレーション（active-set）」で解き直す
      （lo<=omega2<=hi を守りつつ、Fzを強く優先し、次にroll/pitch、最後にyawを追う）
    """
    A = np.asarray(A, dtype=float)
    u = np.asarray(wrench_target, dtype=float).reshape(4)
    if A.ndim != 2 or A.shape[0] != 4:
        raise ValueError(f"A must be (4,N), got {A.shape}")

    # 1) pinv solution
    pinv_res = solve_mixer_pinv(A=A, wrench_target=u, omega2_min=omega2_min, omega2_max=omega2_max)
    # If not saturated, keep pinv result.
    if not pinv_res.saturated:
        return pinv_res

    N = int(A.shape[1])
    A0 = A[0, :].reshape(N)

    # 2) Fz-first allocation
    # If omega2_max is not provided, apply a soft cap in fallback to avoid runaway solutions.
    if omega2_max is None:
        amax = float(np.max(np.abs(A0)))
        amax = max(1e-6, amax)
        omega2_max_eff = float(fallback_auto_omega2_factor) * abs(float(u[0])) / amax
        omega2_max_eff = max(omega2_max_eff, float(omega2_min))
    else:
        omega2_max_eff = float(omega2_max)

    omega2_base = _allocate_fz_greedy(A0=A0, Fz_target=float(u[0]), omega2_min=omega2_min, omega2_max=omega2_max_eff)

    # 3) Constrained weighted solve (PX4-like desaturation)
    # Build wrench weights: strongly prioritize Fz; torque weights come from torque_weights.
    tw = np.asarray(torque_weights, dtype=float).reshape(3)
    tw = np.maximum(tw, 0.0)
    # Heuristic priority: Fz is important but must not starve attitude torque.
    wFz = max(0.0, float(fz_weight))
    W4 = np.diag([wFz, float(tw[0]), float(tw[1]), float(tw[2])]).astype(float)

    lo = np.full((N,), float(omega2_min), dtype=float)
    hi = np.full((N,), float(omega2_max_eff), dtype=float)
    omega2_cmd, sat2 = _solve_bounded_wls_active_set(
        A=A,
        b=u,
        lo=lo,
        hi=hi,
        W=W4,
        ridge=float(fallback_ridge),
        x0=omega2_base,
        max_iter=20,
    )

    u_hat = A @ omega2_cmd
    res = u - u_hat
    return MixResult(
        omega2_cmd=omega2_cmd,
        omega2_raw=pinv_res.omega2_raw,
        saturated=True,
        mode="desat",
        omega2_max_eff=(None if omega2_max is None else float(omega2_max_eff)),
        torque_weights=tuple(float(x) for x in torque_weights),
        fallback_ridge=float(fallback_ridge),
        fallback_auto_omega2_factor=float(fallback_auto_omega2_factor),
        wrench_achieved=u_hat,
        wrench_target=u,
        residual=res,
        A=A,
    )


