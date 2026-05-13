"""
Compare energy/COT across different vehicle masses (mass-scale).
Runs demo_06_wall_effect with --mass-scale for baseline, fixed_morph, inflight_morph at one or more nominal tau.
Output: out/mass_compare/mass_compare_results.csv, mass_compare_summary.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WALL_EFFECT_RATIO_MORPH = 0.25


def run_demo(
    *,
    wall_tau: float,
    condition: str,
    mass_scale: float,
    out_dir: Path,
    wall_tau_max: float,
    seconds: float = 15.0,
    hz: float = 240.0,
) -> dict | None:
    stem = f"{condition}_tau{wall_tau:.2f}_m{mass_scale:.2f}".replace(".", "_")
    log_csv = out_dir / f"{stem}.csv"
    energy_json_path = out_dir / f"{stem}_energy.json"

    base = [
        sys.executable, "-m", "sim.demo.demo_06_wall_effect",
        "--seconds", str(seconds),
        "--hz", str(hz),
        "--physics-hz", str(hz),
        "--mass-scale", str(mass_scale),
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
        "--wall-ff",
    ]
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
            timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[warn] Run failed mass_scale={mass_scale} cond={condition} tau={wall_tau}: {e}")
        return None

    if not energy_json_path.exists():
        print(f"[warn] Missing {energy_json_path}")
        return None
    with open(energy_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare energy/COT across vehicle mass scales.")
    ap.add_argument("--mass-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25],
                    help="Mass scale factors (default: 0.75 1.0 1.25)")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 1.0],
                    help="Nominal wall tau values to run (default: 0.5 1.0)")
    ap.add_argument("--out-dir", type=str, default="out/mass_compare", help="Output directory")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["baseline", "fixed_morph", "inflight_morph"]
    results = []
    for mass_scale in args.mass_scales:
        for nominal_tau in args.tau:
            applied_baseline = nominal_tau
            applied_morph = nominal_tau * WALL_EFFECT_RATIO_MORPH
            wall_tau_max = max(nominal_tau * 1.05, 0.25)
            for cond in conditions:
                wall_tau = applied_baseline if cond == "baseline" else applied_morph
                print(f"  mass_scale={mass_scale:.2f} nominal_tau={nominal_tau} {cond} ...")
                data = run_demo(
                    wall_tau=wall_tau,
                    condition=cond,
                    mass_scale=mass_scale,
                    out_dir=out_dir,
                    wall_tau_max=wall_tau_max,
                )
                if data is None:
                    continue
                total = data.get("total", {})
                crash_info = data.get("crash", {})
                crashed = bool(crash_info.get("crashed", False))
                mass_kg = data.get("meta", {}).get("mass_kg")
                if mass_kg is None:
                    mass_kg = 1.8 * mass_scale  # assume default 1.8 kg
                results.append({
                    "mass_scale": mass_scale,
                    "mass_kg": mass_kg,
                    "nominal_tau": nominal_tau,
                    "condition": cond,
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

    # CSV
    csv_path = out_dir / "mass_compare_results.csv"
    headers = ["mass_scale", "mass_kg", "nominal_tau", "condition", "crashed", "COT_total", "Wh_per_m_total", "E_mech_Wh", "distance_m", "P_avg_W"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print(f"[info] Wrote {csv_path}")

    # Markdown: table by (mass_scale, nominal_tau) with conditions as columns
    md_path = out_dir / "mass_compare_summary.md"
    md_lines = [
        "# Mass comparison: energy/COT vs mass scale",
        "",
        "Same scenario (nominal tau, baseline vs morph 1/4); vehicle mass scaled by --mass-scale (default URDF ~1.8 kg).",
        "",
        "## COT_total by mass_scale and nominal_tau",
        "",
    ]
    for nominal_tau in sorted({r["nominal_tau"] for r in results}):
        md_lines.append(f"### nominal_tau = {nominal_tau}")
        md_lines.append("| mass_scale | mass_kg | baseline | fixed_morph | inflight_morph |")
        md_lines.append("|------------|---------|----------|--------------|----------------|")
        for mass_scale in sorted({r["mass_scale"] for r in results}):
            row_res = [r for r in results if r["nominal_tau"] == nominal_tau and r["mass_scale"] == mass_scale]
            mass_kg = row_res[0]["mass_kg"] if row_res else ""
            def cot(c):
                r = next((x for x in row_res if x["condition"] == c), None)
                if r is None:
                    return "—"
                if r.get("crashed"):
                    return "crash"
                return f"{r['COT_total']:.4f}" if r.get("COT_total") is not None else "—"
            md_lines.append(f"| {mass_scale:.2f} | {mass_kg:.2f} | {cot('baseline')} | {cot('fixed_morph')} | {cot('inflight_morph')} |")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[info] Wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
