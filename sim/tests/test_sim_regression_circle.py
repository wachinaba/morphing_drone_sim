import csv
import os
import subprocess
import sys
import tempfile
import unittest


def _run_module(args: list[str], timeout_s: float = 45.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _f(row: dict, key: str) -> float:
    return float(row[key])


@unittest.skipUnless(os.environ.get("RUN_SIM_REGRESSION", "0") == "1", "set RUN_SIM_REGRESSION=1 to run regression tests")
class TestSimRegressionCircle(unittest.TestCase):
    def _assert_circle_metrics(self, rows: list[dict], *, name: str, check_mix_saturated: bool):
        self.assertGreater(len(rows), 10, msg=f"{name}: too few rows")

        e_xy = []
        sat = 0
        bound_hit = 0
        for row in rows:
            x, y = _f(row, "x"), _f(row, "y")
            xd, yd = _f(row, "x_des"), _f(row, "y_des")
            dx, dy = x - xd, y - yd
            e = (dx * dx + dy * dy) ** 0.5
            e_xy.append(e)
            sat += int(row.get("mix_saturated", "0"))

            # Hard safety bounds to catch explosions quickly (regression signal)
            self.assertLess(abs(x), 5.0, msg=f"{name}: |x| exploded: {x}")
            self.assertLess(abs(y), 5.0, msg=f"{name}: |y| exploded: {y}")

            # Bound-hit ratio: more meaningful than mix_saturated when using desat solver.
            omega2 = [_f(row, f"omega2_cmd_{i}") for i in range(4)]
            mx = row.get("omega2_max_eff", "")
            if mx != "":
                mxf = float(mx)
                if any(o <= 1e-9 or o >= (mxf - 1e-6) for o in omega2):
                    bound_hit += 1

        e_mean = sum(e_xy) / len(e_xy)
        e_max = max(e_xy)
        sat_ratio = sat / len(rows)
        bound_hit_ratio = bound_hit / len(rows)

        # Conservative thresholds (avoid flakiness while still catching regressions)
        self.assertLess(e_mean, 1.0, msg=f"{name}: mean XY error too large: {e_mean:.3f} m")
        self.assertLess(e_max, 3.0, msg=f"{name}: max XY error too large: {e_max:.3f} m")
        if check_mix_saturated:
            self.assertLess(sat_ratio, 0.98, msg=f"{name}: mix_saturated ratio too high: {sat_ratio:.2%}")
        # We still want to catch pathological cases where every step is clamped.
        self.assertLess(bound_hit_ratio, 0.999, msg=f"{name}: bound-hit ratio too high: {bound_hit_ratio:.2%}")

    def test_demo03_circle_regression(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "demo03.csv")
            p = _run_module(
                [
                    "sim.demo.demo_03_hover_pd",
                    "--seconds",
                    "1.0",
                    "--xy-circle",
                    "--xy-circle-radius",
                    "0.2",
                    "--xy-circle-freq",
                    "0.5",
                    "--kp-xy",
                    "1.0",
                    "--kd-xy",
                    "1.2",
                    "--tw-ratio",
                    "3.0",
                    "--log-csv",
                    csv_path,
                    "--log-flush-every",
                    "1",
                    "--log-every",
                    "0",
                ],
                timeout_s=45.0,
            )
            self.assertEqual(
                p.returncode,
                0,
                msg=f"demo_03 failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}",
            )
            rows = _read_csv_rows(csv_path)
            self._assert_circle_metrics(rows, name="demo_03", check_mix_saturated=True)

    def test_demo04_urdf_circle_regression(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "demo04.csv")
            p = _run_module(
                [
                    "sim.demo.demo_04_hover_pd_urdf_morph",
                    "--seconds",
                    "1.0",
                    "--xy-circle",
                    "--xy-circle-radius",
                    "0.2",
                    "--xy-circle-freq",
                    "0.5",
                    "--kp-xy",
                    "1.0",
                    "--kd-xy",
                    "1.2",
                    "--tw-ratio",
                    "3.0",
                    "--log-csv",
                    csv_path,
                    "--log-flush-every",
                    "1",
                    "--log-every",
                    "0",
                ],
                timeout_s=45.0,
            )
            self.assertEqual(
                p.returncode,
                0,
                msg=f"demo_04 failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}",
            )
            rows = _read_csv_rows(csv_path)
            # URDF版はdesatへ頻繁に入るので mix_saturated は回帰指標にしない
            self._assert_circle_metrics(rows, name="demo_04", check_mix_saturated=False)


if __name__ == "__main__":
    unittest.main()


