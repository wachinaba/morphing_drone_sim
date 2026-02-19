from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    args: list[str]
    use_wall_window: bool
    weight: float = 1.0


def _parse_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.DictReader(fp)
        return list(r)


def _f(x: str | None, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: str | None, default: int = 0) -> int:
    if x is None:
        return int(default)
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / float(len(xs)))


def _scenario_score(
    *,
    rows: list[dict[str, str]],
    use_wall_window: bool,
    z_min_ok: float,
    xy_abs_max_ok: float,
    empty_window_penalty: float,
    crash_penalty: float,
) -> tuple[float, dict[str, float]]:
    # Basic signals
    errs: list[float] = []
    errs_wall: list[float] = []
    min_z = float("inf")
    max_abs_x = 0.0
    max_abs_y = 0.0

    for row in rows:
        ex = _f(row.get("ex"), 0.0)
        ey = _f(row.get("ey"), 0.0)
        e = float(math.hypot(ex, ey))
        errs.append(e)

        wall_active = _i(row.get("wall_active"), 0)
        if wall_active:
            errs_wall.append(e)

        z = _f(row.get("z"), float("inf"))
        x = _f(row.get("x"), 0.0)
        y = _f(row.get("y"), 0.0)
        min_z = min(min_z, z)
        max_abs_x = max(max_abs_x, abs(x))
        max_abs_y = max(max_abs_y, abs(y))

    # Window selection
    if use_wall_window:
        sel = errs_wall
        if not sel:
            base = float(empty_window_penalty)
        else:
            base = _mean(sel)
    else:
        sel = errs
        base = _mean(sel)

    # Penalties (simple, robust)
    pen = 0.0
    if not math.isfinite(base):
        pen += 10.0 * float(crash_penalty)
        base = 0.0

    if math.isfinite(min_z) and (min_z < float(z_min_ok)):
        pen += float(crash_penalty) * (float(z_min_ok) - float(min_z))
    if max(max_abs_x, max_abs_y) > float(xy_abs_max_ok):
        pen += float(crash_penalty) * (max(max_abs_x, max_abs_y) - float(xy_abs_max_ok))

    score = float(base) + float(pen)
    details = {
        "base": float(base),
        "pen": float(pen),
        "min_z": float(min_z if math.isfinite(min_z) else 0.0),
        "max_abs_x": float(max_abs_x),
        "max_abs_y": float(max_abs_y),
        "n": float(len(rows)),
        "n_wall": float(len(errs_wall)),
    }
    return float(score), details


def _run_demo06_to_csv(*, argv: list[str], csv_path: Path, timeout_s: float | None) -> int:
    cmd = [sys.executable, "-m", "sim.demo.demo_06_wall_effect", *argv, "--log-csv", str(csv_path), "--log-every", "0"]
    p = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=timeout_s)
    return int(p.returncode)


def _csv_path_in_dir(dir_path: Path, stem: str) -> Path:
    # Avoid Windows "?" ":" etc in filenames
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in stem)
    return dir_path / f"{safe}.csv"


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("empty list")
    return out


def _linspace(a: float, b: float, n: int) -> list[float]:
    n = int(n)
    if n <= 1:
        return [float(a)]
    a = float(a)
    b = float(b)
    return [a + (b - a) * (i / (n - 1)) for i in range(n)]


def _unique_sorted(xs: list[float]) -> list[float]:
    s = sorted(set(float(x) for x in xs))
    return s


def _refine_around(best: list[tuple[float, float, float]], *, scales: list[float]) -> list[tuple[float, float, float]]:
    cand: set[tuple[float, float, float]] = set()
    for kp, kd, ki in best:
        for skp in scales:
            for skd in scales:
                for ski in scales:
                    cand.add((float(kp) * float(skp), float(kd) * float(skd), float(ki) * float(ski)))
    # Keep non-negative, and clamp tiny negatives due to float
    out = [(max(0.0, kp), max(0.0, kd), max(0.0, ki)) for (kp, kd, ki) in cand]
    return out


def _build_scenarios(
    *,
    seconds: float,
    hz: float,
    physics_hz: float,
    morph_args: list[str] | None = None,
    name_prefix: str = "",
) -> list[Scenario]:
    base = [
        "--seconds",
        str(float(seconds)),
        "--hz",
        str(float(hz)),
        "--physics-hz",
        str(float(physics_hz)),
        "--wall-y",
        "1.5",
    ]
    if morph_args:
        base = base + list(morph_args)

    # Scenario design: wall-y 1.5 so circle (y∈[-0.8,0.8]) and rect (y∈[-0.6,0.6]) don't collide
    # - wall scenarios: keep default wall params + wall_model=v2_over_d2 (enabled)
    # - no wall scenarios: set wall_model=off (disturbance disabled)
    # - trajectories are centered around x≈5 within wall x-range [3,7] (default zone_x=3 len_x=4)
    # - side wall at y=+1, range=2 ⇒ y near 0 is still within active band (|y-1|<2)
    ramp_wall = Scenario(
        name=name_prefix + "ramp_wall",
        use_wall_window=True,
        args=[
            *base,
            "--traj",
            "ramp",
            "--x-vel",
            "0.8",
            "--x-max",
            "6.0",
            "--y-des",
            "0.0",
            "--wall-model",
            "v2_over_d2",
        ],
        weight=1.0,
    )
    circle_no_wall = Scenario(
        name=name_prefix + "circle_no_wall",
        use_wall_window=False,
        args=[
            *base,
            "--traj",
            "circle",
            "--circle-radius",
            "0.8",
            "--circle-freq",
            "0.20",
            "--circle-cx",
            "5.0",
            "--circle-cy",
            "0.0",
            "--wall-model",
            "off",
        ],
        weight=1.0,
    )
    circle_wall = Scenario(
        name=name_prefix + "circle_wall",
        use_wall_window=True,
        args=[
            *base,
            "--traj",
            "circle",
            "--circle-radius",
            "0.8",
            "--circle-freq",
            "0.20",
            "--circle-cx",
            "5.0",
            "--circle-cy",
            "0.0",
            "--wall-model",
            "v2_over_d2",
        ],
        weight=1.0,
    )
    rect_no_wall = Scenario(
        name=name_prefix + "rect_no_wall",
        use_wall_window=False,
        args=[
            *base,
            "--traj",
            "rect",
            "--rect-width",
            "1.6",
            "--rect-height",
            "1.2",
            "--rect-speed",
            "0.9",
            "--rect-cx",
            "5.0",
            "--rect-cy",
            "0.0",
            "--wall-model",
            "off",
        ],
        weight=1.0,
    )
    rect_wall = Scenario(
        name=name_prefix + "rect_wall",
        use_wall_window=True,
        args=[
            *base,
            "--traj",
            "rect",
            "--rect-width",
            "1.6",
            "--rect-height",
            "1.2",
            "--rect-speed",
            "0.9",
            "--rect-cx",
            "5.0",
            "--rect-cy",
            "0.0",
            "--wall-model",
            "v2_over_d2",
        ],
        weight=1.0,
    )
    return [ramp_wall, circle_no_wall, circle_wall, rect_no_wall, rect_wall]


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-tune demo_06 XY gains by running multiple scenarios and scoring CSV logs.")
    ap.add_argument("--seconds", type=float, default=8.0, help="Sim duration per scenario [s]. Default: 8")
    ap.add_argument("--hz", type=float, default=240.0, help="Control rate [Hz]. Default: 240")
    ap.add_argument("--physics-hz", type=float, default=240.0, help="Physics rate [Hz]. Default: 240")
    ap.add_argument("--timeout", type=float, default=None, help="Timeout per scenario run [s]. Default: none")

    ap.add_argument("--kp-list", type=str, default=None, help="Comma-separated candidates for kp_xy (e.g. 0.6,1.0,1.6).")
    ap.add_argument("--kd-list", type=str, default=None, help="Comma-separated candidates for kd_xy (e.g. 0.6,1.0,1.6).")
    ap.add_argument("--ki-list", type=str, default=None, help="Comma-separated candidates for ki_xy (e.g. 0.0,0.2,0.5).")
    ap.add_argument("--kp-min", type=float, default=0.4)
    ap.add_argument("--kp-max", type=float, default=2.0)
    ap.add_argument("--kd-min", type=float, default=0.3)
    ap.add_argument("--kd-max", type=float, default=2.2)
    ap.add_argument("--ki-min", type=float, default=0.0)
    ap.add_argument("--ki-max", type=float, default=0.8)
    ap.add_argument("--grid-n", type=int, default=4, help="If *-list not given: linspace count for each axis. Default: 4")

    ap.add_argument("--refine", action="store_true", help="Refine around top-N using multiplicative scales.")
    ap.add_argument("--refine-top", type=int, default=5, help="Top-N from coarse grid to refine around. Default: 5")
    ap.add_argument("--refine-scales", type=str, default="0.7,1.0,1.3", help="Scales for refinement (comma-separated). Default: 0.7,1.0,1.3")

    ap.add_argument("--dry-run", action="store_true", help="Only evaluate a small subset of candidates (fast sanity).")
    ap.add_argument("--max-evals", type=int, default=None, help="Max number of candidate evaluations. Default: none")

    ap.add_argument("--out-csv", type=str, default=None, help="Write all evaluated candidates and scores to this CSV.")

    # scoring config
    ap.add_argument("--z-min-ok", type=float, default=0.08, help="Penalty if min z below this [m]. Default: 0.08")
    ap.add_argument("--xy-abs-max-ok", type=float, default=20.0, help="Penalty if |x| or |y| exceeds this [m]. Default: 20")
    ap.add_argument("--empty-wall-window-penalty", type=float, default=5.0, help="Base cost when wall window is empty. Default: 5")
    ap.add_argument("--crash-penalty", type=float, default=10.0, help="Penalty scale for crash/out-of-bounds. Default: 10")

    # morph (for operating-point morphing|both; demo_06 args)
    ap.add_argument("--operating-point", type=str, default="both", choices=["baseline", "morphing", "both"], help="Evaluate baseline (no morph), morphing (with morph), or both. Default: both")
    ap.add_argument("--morph-start", type=float, default=2.0, help="(morphing) Start time [s] of morph ramp. Default: 2.0")
    ap.add_argument("--morph-seconds", type=float, default=1.5, help="(morphing) Ramp duration [s]. Default: 1.5")
    ap.add_argument("--phi-deg", type=float, default=5.0, help="(morphing) Target fold angle [deg]. Default: 5")
    ap.add_argument("--psi-deg", type=float, default=7.0, help="(morphing) Target slant angle [deg]. Default: 7")
    ap.add_argument("--theta-deg", type=float, default=30.0, help="(morphing) Target tilt angle [deg]. Default: 30")
    ap.add_argument("--morph-symmetry", type=str, default="mirror_xy", choices=["none", "mirror_xy"], help="(morphing) Morph symmetry. Default: mirror_xy")

    args = ap.parse_args()

    if args.kp_list is not None:
        kp_vals = _unique_sorted(_parse_float_list(args.kp_list))
    else:
        kp_vals = _unique_sorted(_linspace(float(args.kp_min), float(args.kp_max), int(args.grid_n)))
    if args.kd_list is not None:
        kd_vals = _unique_sorted(_parse_float_list(args.kd_list))
    else:
        kd_vals = _unique_sorted(_linspace(float(args.kd_min), float(args.kd_max), int(args.grid_n)))
    if args.ki_list is not None:
        ki_vals = _unique_sorted(_parse_float_list(args.ki_list))
    else:
        ki_vals = _unique_sorted(_linspace(float(args.ki_min), float(args.ki_max), int(args.grid_n)))

    op = str(args.operating_point)
    morph_args_list = [
        "--morph-start", str(float(args.morph_start)),
        "--morph-seconds", str(float(args.morph_seconds)),
        "--phi-deg", str(float(args.phi_deg)),
        "--psi-deg", str(float(args.psi_deg)),
        "--theta-deg", str(float(args.theta_deg)),
        "--morph-symmetry", str(args.morph_symmetry),
    ]
    if op == "baseline":
        scenarios = _build_scenarios(seconds=float(args.seconds), hz=float(args.hz), physics_hz=float(args.physics_hz), morph_args=None, name_prefix="")
    elif op == "morphing":
        scenarios = _build_scenarios(seconds=float(args.seconds), hz=float(args.hz), physics_hz=float(args.physics_hz), morph_args=morph_args_list, name_prefix="morph_")
    else:
        base_scenarios = _build_scenarios(seconds=float(args.seconds), hz=float(args.hz), physics_hz=float(args.physics_hz), morph_args=None, name_prefix="base_")
        morph_scenarios = _build_scenarios(seconds=float(args.seconds), hz=float(args.hz), physics_hz=float(args.physics_hz), morph_args=morph_args_list, name_prefix="morph_")
        scenarios = base_scenarios + morph_scenarios

    # Candidate list
    candidates: list[tuple[float, float, float]] = [(kp, kd, ki) for kp in kp_vals for kd in kd_vals for ki in ki_vals]
    if bool(args.dry_run):
        candidates = candidates[: min(len(candidates), 4)]

    max_evals = None if args.max_evals is None else int(args.max_evals)
    if max_evals is not None:
        candidates = candidates[: max(0, max_evals)]

    results: list[dict[str, float | str]] = []

    with tempfile.TemporaryDirectory(prefix="demo06_autotune_") as td:
        tdir = Path(td)
        for idx, (kp, kd, ki) in enumerate(candidates):
            total = 0.0
            per: dict[str, float] = {}
            ok = True

            for sc in scenarios:
                csv_path = _csv_path_in_dir(tdir, f"{idx:05d}_{sc.name}")
                argv = [
                    *sc.args,
                    "--kp-xy",
                    str(float(kp)),
                    "--kd-xy",
                    str(float(kd)),
                    "--ki-xy",
                    str(float(ki)),
                ]

                rc = _run_demo06_to_csv(argv=argv, csv_path=csv_path, timeout_s=(None if args.timeout is None else float(args.timeout)))
                if rc != 0 or (not csv_path.exists()):
                    ok = False
                    sc_score = float(args.empty_wall_window_penalty) + float(args.crash_penalty) * 10.0
                    sc_det = {"base": sc_score, "pen": float(args.crash_penalty) * 10.0, "min_z": 0.0, "max_abs_x": 0.0, "max_abs_y": 0.0, "n": 0.0, "n_wall": 0.0}
                else:
                    rows = _parse_csv_rows(csv_path)
                    sc_score, sc_det = _scenario_score(
                        rows=rows,
                        use_wall_window=bool(sc.use_wall_window),
                        z_min_ok=float(args.z_min_ok),
                        xy_abs_max_ok=float(args.xy_abs_max_ok),
                        empty_window_penalty=float(args.empty_wall_window_penalty),
                        crash_penalty=float(args.crash_penalty),
                    )

                total += float(sc.weight) * float(sc_score)
                per[f"{sc.name}_score"] = float(sc_score)
                per[f"{sc.name}_base"] = float(sc_det["base"])
                per[f"{sc.name}_pen"] = float(sc_det["pen"])
                per[f"{sc.name}_n_wall"] = float(sc_det["n_wall"])

            rec: dict[str, float | str] = {
                "kp": float(kp),
                "kd": float(kd),
                "ki": float(ki),
                "score": float(total),
                "ok": str(bool(ok)),
                **per,
            }
            results.append(rec)

            # compact progress
            if (idx % 1) == 0:
                print(f"[{idx+1:4d}/{len(candidates):4d}] score={total:.4f} kp={kp:.3g} kd={kd:.3g} ki={ki:.3g}")

        # optional refinement
        if bool(args.refine) and results:
            results_sorted = sorted(results, key=lambda r: float(r["score"]))
            top = results_sorted[: max(1, int(args.refine_top))]
            top_gains = [(float(r["kp"]), float(r["kd"]), float(r["ki"])) for r in top]
            scales = _unique_sorted(_parse_float_list(str(args.refine_scales)))
            refined = _refine_around(top_gains, scales=scales)

            if bool(args.dry_run):
                refined = refined[: min(len(refined), 6)]

            if max_evals is not None:
                refined = refined[: max(0, max_evals)]

            print(f"[info] refinement candidates: {len(refined)} (top={len(top_gains)}, scales={scales})")

            # run refinement, append
            base_idx = len(results)
            for j, (kp, kd, ki) in enumerate(refined):
                total = 0.0
                per: dict[str, float] = {}
                ok = True
                for sc in scenarios:
                    csv_path = _csv_path_in_dir(tdir, f"{base_idx+j:05d}_{sc.name}")
                    argv = [
                        *sc.args,
                        "--kp-xy",
                        str(float(kp)),
                        "--kd-xy",
                        str(float(kd)),
                        "--ki-xy",
                        str(float(ki)),
                    ]
                    rc = _run_demo06_to_csv(argv=argv, csv_path=csv_path, timeout_s=(None if args.timeout is None else float(args.timeout)))
                    if rc != 0 or (not csv_path.exists()):
                        ok = False
                        sc_score = float(args.empty_wall_window_penalty) + float(args.crash_penalty) * 10.0
                        sc_det = {"base": sc_score, "pen": float(args.crash_penalty) * 10.0, "min_z": 0.0, "max_abs_x": 0.0, "max_abs_y": 0.0, "n": 0.0, "n_wall": 0.0}
                    else:
                        rows = _parse_csv_rows(csv_path)
                        sc_score, sc_det = _scenario_score(
                            rows=rows,
                            use_wall_window=bool(sc.use_wall_window),
                            z_min_ok=float(args.z_min_ok),
                            xy_abs_max_ok=float(args.xy_abs_max_ok),
                            empty_window_penalty=float(args.empty_wall_window_penalty),
                            crash_penalty=float(args.crash_penalty),
                        )
                    total += float(sc.weight) * float(sc_score)
                    per[f"{sc.name}_score"] = float(sc_score)
                    per[f"{sc.name}_base"] = float(sc_det["base"])
                    per[f"{sc.name}_pen"] = float(sc_det["pen"])
                    per[f"{sc.name}_n_wall"] = float(sc_det["n_wall"])

                rec = {
                    "kp": float(kp),
                    "kd": float(kd),
                    "ki": float(ki),
                    "score": float(total),
                    "ok": str(bool(ok)),
                    **per,
                }
                results.append(rec)
                print(f"[ref {j+1:4d}/{len(refined):4d}] score={total:.4f} kp={kp:.3g} kd={kd:.3g} ki={ki:.3g}")

    if not results:
        print("[error] no results")
        return 2

    results_sorted = sorted(results, key=lambda r: float(r["score"]))
    best = results_sorted[0]
    print("")
    print("[best]")
    print(f"score={float(best['score']):.6f}  kp={float(best['kp']):.6g}  kd={float(best['kd']):.6g}  ki={float(best['ki']):.6g}")
    print("cli:")
    print(f"--kp-xy {float(best['kp']):.6g} --kd-xy {float(best['kd']):.6g} --ki-xy {float(best['ki']):.6g}")

    if args.out_csv:
        outp = Path(str(args.out_csv))
        outp.parent.mkdir(parents=True, exist_ok=True)
        # stable field order
        fieldnames: list[str] = []
        # gather all keys
        keyset: set[str] = set()
        for r in results:
            keyset.update(str(k) for k in r.keys())
        # basic first
        for k in ["kp", "kd", "ki", "score", "ok"]:
            if k in keyset:
                fieldnames.append(k)
        for k in sorted(keyset):
            if k not in set(fieldnames):
                fieldnames.append(k)
        with outp.open("w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results_sorted:
                w.writerow(r)
        print(f"[info] wrote results: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

