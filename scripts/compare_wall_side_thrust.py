"""
Compare wall-side vs opposite-side rotor thrust from demo_06_wall_effect CSV.

Rotor layout (body frame, URDF): rotor0 +x+y, rotor1 -x+y, rotor2 -x-y, rotor3 +x-y.
Wall at y=0.7 → wall side = +y = rotor0, rotor1. Opposite = -y = rotor2, rotor3.
Thrust T_i = CT * omega2_act_i (from CSV we have omega2_act_0..3; CT from args or default 1.0).

Usage: python scripts/compare_wall_side_thrust.py <path_to_csv> [--CT 1.0]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare wall-side vs opposite-side thrust from wall_effect CSV.")
    ap.add_argument("csv_path", type=str, help="Path to CSV from demo_06_wall_effect (with omega2_act_0..3, wall_active)")
    ap.add_argument("--CT", type=float, default=1.0, help="Thrust coefficient for T = CT*omega2. Default: 1.0")
    args = ap.parse_args()
    path = Path(args.csv_path)
    if not path.exists():
        print(f"[error] File not found: {path}")
        return 1
    CT = float(args.CT)

    rows_wall = []
    rows_no_wall = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                o0 = float(row["omega2_act_0"])
                o1 = float(row["omega2_act_1"])
                o2 = float(row["omega2_act_2"])
                o3 = float(row["omega2_act_3"])
                wa = int(float(row.get("wall_active", 0)))
            except (KeyError, ValueError):
                continue
            t_wall = CT * (o0 + o1)  # wall side (+y): rotor0, rotor1
            t_opp = CT * (o2 + o3)   # opposite (-y): rotor2, rotor3
            t_total = t_wall + t_opp
            rec = {"t": float(row.get("t", 0)), "T_wall_side": t_wall, "T_opp_side": t_opp, "T_total": t_total}
            if wa == 1:
                rows_wall.append(rec)
            else:
                rows_no_wall.append(rec)

    if not rows_wall:
        print("[info] No rows with wall_active=1 in CSV. Run with wall in zone and --log-csv.")
        return 0

    n_wall = len(rows_wall)
    avg_t_wall = sum(r["T_wall_side"] for r in rows_wall) / n_wall
    avg_t_opp = sum(r["T_opp_side"] for r in rows_wall) / n_wall
    avg_total = sum(r["T_total"] for r in rows_wall) / n_wall
    diff = avg_t_wall - avg_t_opp
    print(f"CSV: {path.name}")
    print(f"CT = {CT}")
    print(f"Wall-active rows: {n_wall}")
    print(f"  Mean thrust wall-side (rotor0+1): {avg_t_wall:.4f} N")
    print(f"  Mean thrust opposite (rotor2+3):  {avg_t_opp:.4f} N")
    print(f"  Mean total:                       {avg_total:.4f} N")
    print(f"  Difference (wall - opp):          {diff:+.4f} N  (positive => wall side higher)")
    if rows_no_wall:
        n_no = len(rows_no_wall)
        avg_t_wall_no = sum(r["T_wall_side"] for r in rows_no_wall) / n_no
        avg_t_opp_no = sum(r["T_opp_side"] for r in rows_no_wall) / n_no
        print(f"No-wall rows: {n_no}")
        print(f"  Mean T_wall_side: {avg_t_wall_no:.4f} N  T_opp_side: {avg_t_opp_no:.4f} N (should be ~equal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
