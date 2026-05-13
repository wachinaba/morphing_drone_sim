"""
Compare energy-evaluation results from demo_06_wall_effect (3 conditions).
Reads *_energy.json from out/, writes a summary table and optional bar chart.
Usage: python scripts/demo06_energy_compare.py [--out-dir out] [--no-plot]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare demo06 energy JSONs and write summary.")
    ap.add_argument("--out-dir", type=str, default="out", help="Directory containing *_energy.json and for outputs. Default: out")
    ap.add_argument("--no-plot", action="store_true", help="Skip generating the bar chart PNG.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("baseline", "Baseline (wall_tau=0.2, no morph)"),
        ("fixed_morph", "Fixed morph (wall_tau=0.2, phi/psi/theta=0/11/11)"),
        ("inflight_morph", "Inflight morph (wall_tau=0.2, ramp 0→0/11/11)"),
    ]
    rows = []
    for key, label in conditions:
        path = out_dir / f"demo06_{key}_energy.json"
        if not path.exists():
            print(f"[warn] Missing {path}, skipping.")
            continue
        data = load_json(path)
        total = data.get("total", {})
        wall = data.get("wall_active", {})
        rows.append({
            "condition": key,
            "label": label,
            "Wh_per_m_total": total.get("Wh_per_m"),
            "COT_total": total.get("COT"),
            "E_Wh_total": total.get("E_mech_Wh"),
            "dist_m_total": total.get("distance_m"),
            "Wh_per_m_wall": wall.get("Wh_per_m"),
            "COT_wall": wall.get("COT"),
            "E_Wh_wall": wall.get("E_mech_Wh"),
            "dist_m_wall": wall.get("distance_m"),
        })

    if not rows:
        print("[error] No energy JSONs found.")
        return 1

    # Summary Markdown
    md_path = out_dir / "demo06_energy_comparison.md"
    lines = [
        "# demo_06_wall_effect エネルギー評価 比較",
        "",
        "| Condition | Wh/m (total) | COT (total) | Wh/m (wall_active) | COT (wall_active) |",
        "|-----------|---------------|-------------|--------------------|-------------------|",
    ]
    for r in rows:
        wpm_t = r["Wh_per_m_total"] if r["Wh_per_m_total"] is not None else "—"
        cot_t = r["COT_total"] if r["COT_total"] is not None else "—"
        wpm_w = r["Wh_per_m_wall"] if r["Wh_per_m_wall"] is not None else "—"
        cot_w = r["COT_wall"] if r["COT_wall"] is not None else "—"
        if isinstance(wpm_t, float):
            wpm_t = f"{wpm_t:.6f}"
        if isinstance(cot_t, float):
            cot_t = f"{cot_t:.4f}"
        if isinstance(wpm_w, float):
            wpm_w = f"{wpm_w:.6f}"
        if isinstance(cot_w, float):
            cot_w = f"{cot_w:.4f}"
        lines.append(f"| {r['condition']} | {wpm_t} | {cot_t} | {wpm_w} | {cot_w} |")
    lines.append("")
    lines.append("Energy: mechanical proxy P_mech = |CQ| ω³. Source: *_energy.json from demo_06_wall_effect.")
    lines.append("Note: When morphing, wall effect influence is 1/4 of baseline. If all runs use the same wall_tau_fixed, morph conditions experience lower effective disturbance; for fair comparison (same effective tau), use scripts/demo06_wall_tau_sweep.py.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Wrote {md_path}")

    # CSV summary
    csv_path = out_dir / "demo06_energy_comparison.csv"
    headers = ["condition", "Wh_per_m_total", "COT_total", "Wh_per_m_wall_active", "COT_wall_active"]
    csv_lines = [",".join(headers)]
    for r in rows:
        csv_lines.append(",".join([
            r["condition"],
            str(r["Wh_per_m_total"]) if r["Wh_per_m_total"] is not None else "",
            str(r["COT_total"]) if r["COT_total"] is not None else "",
            str(r["Wh_per_m_wall"]) if r["Wh_per_m_wall"] is not None else "",
            str(r["COT_wall"]) if r["COT_wall"] is not None else "",
        ]))
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"[info] Wrote {csv_path}")

    # Bar chart
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[warn] matplotlib not available, skipping plot.")
        else:
            labels = [r["condition"] for r in rows]
            x = np.arange(len(labels))
            width = 0.35
            cot_total = [r["COT_total"] if r["COT_total"] is not None else 0.0 for r in rows]
            cot_wall = [r["COT_wall"] if r["COT_wall"] is not None else 0.0 for r in rows]
            fig, ax = plt.subplots(layout="constrained", figsize=(8, 5))
            bars1 = ax.bar(x - width / 2, cot_total, width, label="COT (total)")
            bars2 = ax.bar(x + width / 2, cot_wall, width, label="COT (wall_active)")
            ax.set_ylabel("COT")
            ax.set_title("demo_06 energy comparison: COT (total) vs COT (wall_active)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            fig.savefig(out_dir / "demo06_energy_comparison.png", dpi=150)
            plt.close(fig)
            print(f"[info] Wrote {out_dir / 'demo06_energy_comparison.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
