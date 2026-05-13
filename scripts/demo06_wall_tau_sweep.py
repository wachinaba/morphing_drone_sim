"""
Sweep nominal wall strength to evaluate trade-off: morphing reduces wall effect to 1/4 vs rotor-tilt energy penalty.

Experiment: same wall scenario (nominal tau). Compare
  - Baseline: wall moment = 1  → apply wall_tau_fixed = tau
  - Morphing: wall moment = 1/4 → apply wall_tau_fixed = tau/4

So we compare baseline under full disturbance vs morphing under 1/4 disturbance. Crossover (if any)
is where the energy saving from reduced disturbance outweighs the tilt-induced efficiency loss.

Usage: python scripts/demo06_wall_tau_sweep.py [--tau-min 0.05] [--tau-max 0.8] [--tau-step 0.05] [--out-dir out/sweep]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_demo(
    *,
    wall_tau: float,
    condition: str,
    out_dir: Path,
    wall_tau_max: float,
    seconds: float = 15.0,
    hz: float = 240.0,
    no_plot: bool = True,
) -> dict | None:
    """Run demo_06_wall_effect for one (wall_tau, condition). Return energy summary dict or None on failure."""
    stem = f"{condition}_{wall_tau:.3f}".replace(".", "_")
    log_csv = out_dir / f"{stem}.csv"
    # Energy JSON is written to same dir, stem + "_energy.json"
    energy_json_path = out_dir / f"{stem}_energy.json"

    base = [
        sys.executable, "-m", "sim.demo.demo_06_wall_effect",
        "--seconds", str(seconds),
        "--hz", str(hz),
        "--physics-hz", str(hz),
        "--wall-orient", "side",
        "--wall-y", "0.7",
        "--wall-zone-x", "3.0",
        "--wall-len-x", "4.0",
        "--wall-range", "1.5",
        "--x-vel", "0.8",
        "--wall-axis", "roll",
        "--wall-model", "fixed_tau",
        "--wall-tau-fixed", str(wall_tau),
        "--wall-tau-max", str(wall_tau_max),
        "--kp-xy", "2.76",
        "--kd-xy", "2.4",
        "--ki-xy", "0.425",
        "--z-des", "0.5",
        "--morph-symmetry", "mirror_xy",
        "--log-csv", str(log_csv),
        "--energy",
        "--energy-t0", "0.0",
        "--wall-ff",  # Feedforward wall torque so controller counters disturbance (avoids crash at high tau)
    ]
    if no_plot:
        pass  # omit --plot
    else:
        base.append("--plot")

    if condition == "baseline":
        base.extend(["--morph-seconds", "0", "--phi-deg", "0", "--psi-deg", "0", "--theta-deg", "0"])
    elif condition == "fixed_morph":
        base.extend(["--morph-seconds", "0", "--phi-deg", "0", "--psi-deg", "11", "--theta-deg", "11"])
    elif condition == "inflight_morph":
        base.extend([
            "--morph-start", "2.0", "--morph-seconds", "1.5",
            "--phi-start", "0", "--psi-start", "0", "--theta-start", "0",
            "--phi-deg", "0", "--psi-deg", "11", "--theta-deg", "11",
        ])
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            base,
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[warn] Run failed tau={wall_tau} cond={condition}: {e}")
        return None

    if not energy_json_path.exists():
        print(f"[warn] Missing {energy_json_path}")
        return None
    with open(energy_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep wall_tau_fixed to find morphing advantage boundary.")
    ap.add_argument("--tau-min", type=float, default=0.05, help="Min nominal wall tau. Default: 0.05")
    ap.add_argument("--tau-max", type=float, default=2.5, help="Max nominal wall tau (wall moment ~ thrust; wider range). Default: 2.5")
    ap.add_argument("--tau-step", type=float, default=0.05, help="Step for nominal tau. Default: 0.05")
    ap.add_argument("--out-dir", type=str, default="out/sweep", help="Output dir for CSV/JSON and results. Default: out/sweep")
    ap.add_argument("--plot-runs", action="store_true", help="Add --plot to each run (slower). Default: off")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taus = []
    t = float(args.tau_min)
    while t <= float(args.tau_max) + 1e-9:
        taus.append(round(t, 4))
        t += float(args.tau_step)
    conditions = ["baseline", "fixed_morph", "inflight_morph"]

    # Trade-off experiment: baseline experiences full wall (moment=1), morph experiences 1/4 (moment=1/4).
    WALL_EFFECT_RATIO_MORPH = 0.25  # morph reduces wall effect to 1/4
    # wall_tau_max must allow at least tau for baseline (morph uses tau/4).
    wall_tau_max = max(float(args.tau_max) * 1.05, 0.25)

    print(f"[sweep] nominal wall_tau (x-axis) in {taus}")
    print(f"[sweep] conditions: {conditions}")
    print(f"[sweep] baseline: wall moment = 1 (apply tau). morph: wall moment = 1/4 (apply tau/4). Trade-off: wall reduction vs tilt energy penalty.")
    results = []  # list of {tau, condition, applied_wall_tau, ...}
    for nominal_tau in taus:
        for cond in conditions:
            # Baseline gets full tau; morph gets tau/4 (they experience 1 vs 1/4 in same scenario).
            applied_tau = nominal_tau if cond == "baseline" else nominal_tau * WALL_EFFECT_RATIO_MORPH
            print(f"  nominal_tau={nominal_tau:.3f} {cond} (applied wall_tau={applied_tau:.3f}) ...")
            data = run_demo(
                wall_tau=applied_tau,
                condition=cond,
                out_dir=out_dir,
                wall_tau_max=wall_tau_max,
                no_plot=not args.plot_runs,
            )
            if data is None:
                continue
            total = data.get("total", {})
            crash_info = data.get("crash", {})
            crashed = bool(crash_info.get("crashed", False))
            results.append({
                "tau": nominal_tau,
                "condition": cond,
                "applied_wall_tau": applied_tau,
                "crashed": crashed,
                "COT_total": total.get("COT"),
                "Wh_per_m_total": total.get("Wh_per_m"),
                "E_mech_Wh": total.get("E_mech_Wh"),
                "distance_m": total.get("distance_m"),
                "P_avg_W": total.get("P_avg_W"),
            })

    if not results:
        print("[error] No results.")
        return 1

    # Build per-tau lookup (COT and crashed flag). For crossover, use None for crashed runs so they are excluded.
    by_tau: dict[float, dict[str, float | None]] = {}
    crashed_by_tau: dict[float, dict[str, bool]] = {}
    for r in results:
        tau = r["tau"]
        if tau not in by_tau:
            by_tau[tau] = {}
            crashed_by_tau[tau] = {}
        by_tau[tau][r["condition"]] = None if r.get("crashed") else r["COT_total"]
        crashed_by_tau[tau][r["condition"]] = r.get("crashed", False)

    # Crossover: smallest tau where morph has lower COT than baseline (crashed runs excluded)
    taus_sorted = sorted(by_tau.keys())
    crossover_fixed = None
    crossover_inflight = None
    for tau in taus_sorted:
        row = by_tau[tau]
        cb = row.get("baseline")
        cf = row.get("fixed_morph")
        ci = row.get("inflight_morph")
        if cb is not None and cf is not None and cf < cb and crossover_fixed is None:
            crossover_fixed = tau
        if cb is not None and ci is not None and ci < cb and crossover_inflight is None:
            crossover_inflight = tau

    # Summary table CSV
    csv_path = out_dir / "wall_tau_sweep_results.csv"
    headers = ["tau", "condition", "applied_wall_tau", "crashed", "COT_total", "Wh_per_m_total", "E_mech_Wh", "distance_m", "P_avg_W"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print(f"[info] Wrote {csv_path}")

    # Summary markdown
    md_path = out_dir / "wall_tau_sweep_summary.md"
    md_lines = [
        "# Wall tau sweep: trade-off (wall reduction 1/4 vs rotor-tilt energy penalty)",
        "",
        "**Experiment:** Same wall scenario (nominal tau). Baseline: wall moment = 1 (apply `wall_tau_fixed = tau`). Morphing: wall moment = 1/4 (apply `wall_tau_fixed = tau/4`). We compare energy (COT) to evaluate whether the benefit of reduced disturbance outweighs the tilt-induced efficiency loss.",
        "",
        "**Control:** `--wall-ff` is enabled: wall torque is fed forward and subtracted from the attitude command so the controller explicitly counters the disturbance (avoids crash at high tau and allows full-range evaluation).",
        "",
        "## Crash",
        "Runs that crash (ground/wall contact) are detected and stopped; crossover is computed excluding crashed runs.",
        "",
        f"Nominal tau range: {args.tau_min} .. {args.tau_max} step {args.tau_step}",
        f"Conditions: baseline (no morph), fixed_morph (phi=0, psi=11, theta=11), inflight_morph (ramp to same).",
        "",
        "## Crossover (first nominal tau where morph COT < baseline COT, non-crashed only)",
        f"- **fixed_morph**: " + (f"tau >= {crossover_fixed}" if crossover_fixed is not None else "no crossover in range"),
        f"- **inflight_morph**: " + (f"tau >= {crossover_inflight}" if crossover_inflight is not None else "no crossover in range"),
        "",
        "## COT_total by nominal wall tau",
        "| nominal_tau | baseline | fixed_morph | inflight_morph |",
        "|-------------|----------|-------------|----------------|",
    ]
    for tau in taus_sorted:
        row = by_tau[tau]
        crash_row = crashed_by_tau.get(tau, {})
        cb = row.get("baseline")
        cf = row.get("fixed_morph")
        ci = row.get("inflight_morph")
        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"
        def cell(v, c):
            s = fmt(v)
            if crash_row.get(c, False):
                s += " (crash)"
            return s
        md_lines.append(f"| {tau:.3f} | {cell(cb, 'baseline')} | {cell(cf, 'fixed_morph')} | {cell(ci, 'inflight_morph')} |")
    md_lines.append("")
    md_lines.append("Full data (including `crashed` column): wall_tau_sweep_results.csv")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[info] Wrote {md_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not available, skipping plot.")
        return 0

    def _to_float(v):
        return float(v) if v is not None else np.nan

    fig, ax = plt.subplots(layout="constrained", figsize=(9, 5))
    taus_arr = np.array(taus_sorted, dtype=float)
    cot_b = np.array([_to_float(by_tau[t].get("baseline")) for t in taus_sorted])
    cot_f = np.array([_to_float(by_tau[t].get("fixed_morph")) for t in taus_sorted])
    cot_i = np.array([_to_float(by_tau[t].get("inflight_morph")) for t in taus_sorted])
    mask_b = np.isfinite(cot_b)
    mask_f = np.isfinite(cot_f)
    mask_i = np.isfinite(cot_i)
    ax.plot(taus_arr[mask_b], cot_b[mask_b], "o-", label="baseline", color="C0")
    ax.plot(taus_arr[mask_f], cot_f[mask_f], "s-", label="fixed_morph", color="C1")
    ax.plot(taus_arr[mask_i], cot_i[mask_i], "^-", label="inflight_morph", color="C2")
    if crossover_fixed is not None:
        ax.axvline(crossover_fixed, color="C1", linestyle="--", alpha=0.7, label=f"crossover fixed @ {crossover_fixed}")
    if crossover_inflight is not None:
        ax.axvline(crossover_inflight, color="C2", linestyle="--", alpha=0.7, label=f"crossover inflight @ {crossover_inflight}")
    ax.set_xlabel("Nominal wall tau [N·m] (baseline: applied tau, morph: applied tau/4)")
    ax.set_ylabel("COT (total)")
    ax.set_title("Trade-off: wall 1/4 reduction vs tilt penalty — COT vs nominal wall tau")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "wall_tau_sweep.png", dpi=150)
    plt.close(fig)
    print(f"[info] Wrote {out_dir / 'wall_tau_sweep.png'}")

    print("")
    print("--- Summary ---")
    print(f"Crossover (morph COT < baseline): fixed_morph tau>={crossover_fixed}, inflight_morph tau>={crossover_inflight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
