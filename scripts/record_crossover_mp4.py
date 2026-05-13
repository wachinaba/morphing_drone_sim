"""
Record MP4 at a chosen nominal wall tau.
Trade-off design: baseline gets nominal_tau, morph gets nominal_tau/4.
Default (toughest): tau=1.0 — highest nominal tau where baseline typically completes without crash in GUI; use --tau 1.5 for max severity (baseline may crash).
Output: out/sweep/<prefix>_baseline.mp4, <prefix>_fixed_morph.mp4, <prefix>_inflight_morph.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Default: toughest condition where all three (baseline, fixed_morph, inflight_morph) complete without crash
NOMINAL_TAU_DEFAULT = 1.0
WALL_EFFECT_RATIO_MORPH = 0.25  # morph experiences 1/4

BASE = [
    sys.executable, "-m", "sim.demo.demo_06_wall_effect",
    "--gui",
    "--seconds", "15",
    "--hz", "240",
    "--physics-hz", "240",
    "--wall-orient", "side",
    "--wall-y", "0.7",
    "--wall-zone-x", "3.0",
    "--wall-len-x", "4.0",
    "--wall-range", "1.5",
    "--x-vel", "0.8",
    "--wall-axis", "roll",
    "--wall-model", "fixed_tau",
    "--wall-ff",
    "--kp-xy", "2.76",
    "--kd-xy", "2.4",
    "--ki-xy", "0.425",
    "--z-des", "0.5",
    "--morph-symmetry", "mirror_xy",
    "--cam-follow",
    "--cam-yaw-mode", "behind",
    "--cam-yaw-offset", "0",
    "--cam-dist", "1.2",
    "--cam-pitch", "-10",
    "--cam-target-body", "0.3", "0", "0.35",
    "--record-fps", "60",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Record MP4 at given nominal wall tau (default: toughest=1.5).")
    ap.add_argument("--tau", type=float, default=NOMINAL_TAU_DEFAULT, help=f"Nominal wall tau. Default: {NOMINAL_TAU_DEFAULT}")
    ap.add_argument("--out-dir", type=str, default="out/sweep", help="Output directory. Default: out/sweep")
    ap.add_argument("--prefix", type=str, default="toughest", help="MP4 filename prefix. Default: toughest")
    args = ap.parse_args()
    nominal_tau = float(args.tau)
    applied_baseline = nominal_tau
    applied_morph = nominal_tau * WALL_EFFECT_RATIO_MORPH
    wall_tau_max = max(nominal_tau * 1.05, 0.25)

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("baseline", applied_baseline, ["--morph-seconds", "0", "--phi-deg", "0", "--psi-deg", "0", "--theta-deg", "0"]),
        ("fixed_morph", applied_morph, ["--morph-seconds", "0", "--phi-deg", "0", "--psi-deg", "11", "--theta-deg", "11"]),
        ("inflight_morph", applied_morph, [
            "--morph-start", "2.0", "--morph-seconds", "1.5",
            "--phi-start", "0", "--psi-start", "0", "--theta-start", "0",
            "--phi-deg", "0", "--psi-deg", "11", "--theta-deg", "11",
        ]),
    ]
    base_with_max = BASE + ["--wall-tau-max", str(wall_tau_max)]
    for name, wall_tau, morph_args in runs:
        mp4_path = out_dir / f"{args.prefix}_{name}.mp4"
        cmd = base_with_max + ["--wall-tau-fixed", str(wall_tau), "--record-mp4", str(mp4_path)] + morph_args
        print(f"[record] {name}: nominal_tau={nominal_tau} applied_tau={wall_tau} -> {mp4_path}")
        subprocess.run(cmd, cwd=root, check=True, timeout=180)
    print(f"[info] Done. MP4s in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
