from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorModel:
    """
    モータ（回転数）の簡易モデル（MVP）.

    状態は omega2 = ω^2 として扱う（>=0）。
    - 一次遅れ: d(omega2)/dt = (cmd - omega2)/tau
    - レート制限: |d(omega2)/dt| <= rate_limit（任意）
    - 飽和: omega2_min <= omega2 <= omega2_max（任意）
    """

    tau: float = 0.05  # [s]
    omega2_min: float = 0.0
    omega2_max: float | None = None
    omega2_rate_limit: float | None = None  # [omega2/s]

    omega2: np.ndarray | None = None  # (N,) current state

    def reset(self, omega2_init: np.ndarray):
        x = np.asarray(omega2_init, dtype=float).reshape(-1)
        x = np.maximum(x, float(self.omega2_min))
        if self.omega2_max is not None:
            x = np.minimum(x, float(self.omega2_max))
        self.omega2 = x

    def step(self, omega2_cmd: np.ndarray, dt: float) -> np.ndarray:
        cmd = np.asarray(omega2_cmd, dtype=float).reshape(-1)
        if self.omega2 is None:
            self.reset(np.zeros_like(cmd))

        x = np.asarray(self.omega2, dtype=float).reshape(-1)
        if x.shape != cmd.shape:
            raise ValueError(f"shape mismatch: omega2 state {x.shape} vs cmd {cmd.shape}")

        dt = float(dt)
        if dt <= 0.0:
            return x.copy()

        # Clamp command (also enforce non-negative)
        cmd = np.maximum(cmd, float(self.omega2_min))
        if self.omega2_max is not None:
            cmd = np.minimum(cmd, float(self.omega2_max))

        # First-order lag
        tau = max(1e-6, float(self.tau))
        alpha = dt / (tau + dt)  # stable discretization
        x_next = x + alpha * (cmd - x)

        # Rate limit (optional)
        if self.omega2_rate_limit is not None:
            r = max(0.0, float(self.omega2_rate_limit))
            dx_max = r * dt
            dx = np.clip(x_next - x, -dx_max, dx_max)
            x_next = x + dx

        # Saturation again
        x_next = np.maximum(x_next, float(self.omega2_min))
        if self.omega2_max is not None:
            x_next = np.minimum(x_next, float(self.omega2_max))

        self.omega2 = x_next
        return x_next.copy()





