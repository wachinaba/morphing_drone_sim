from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ServoModel:
    """
    関節角（rad）の簡易サーボモデル（MVP）.

    - 一次遅れ: d(q)/dt = (q_cmd - q)/tau
    - 速度制限: |dq/dt| <= rate_limit [rad/s]（任意）
    """

    tau: float = 0.08  # [s]
    rate_limit: float | None = None  # [rad/s]
    q: np.ndarray | None = None  # (N,)

    def reset(self, q_init: np.ndarray):
        self.q = np.asarray(q_init, dtype=float).reshape(-1).copy()

    def step(self, q_cmd: np.ndarray, dt: float) -> np.ndarray:
        cmd = np.asarray(q_cmd, dtype=float).reshape(-1)
        if self.q is None:
            self.reset(np.zeros_like(cmd))

        q = np.asarray(self.q, dtype=float).reshape(-1)
        if q.shape != cmd.shape:
            raise ValueError(f"shape mismatch: q {q.shape} vs cmd {cmd.shape}")

        dt = float(dt)
        if dt <= 0.0:
            return q.copy()

        tau = max(1e-6, float(self.tau))
        alpha = dt / (tau + dt)
        q_next = q + alpha * (cmd - q)

        if self.rate_limit is not None:
            vmax = max(0.0, float(self.rate_limit))
            dq_max = vmax * dt
            dq = np.clip(q_next - q, -dq_max, dq_max)
            q_next = q + dq

        self.q = q_next
        return q_next.copy()





