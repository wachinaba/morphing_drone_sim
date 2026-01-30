import csv
import os
import subprocess
import sys
import tempfile
import unittest


def _run_module(args: list[str], timeout_s: float = 120.0) -> subprocess.CompletedProcess:
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


@unittest.skipUnless(os.environ.get("RUN_SIM_LONG", "0") == "1", "set RUN_SIM_LONG=1 to run long URDF regression")
class TestSimLongRegressionUrdfMorphCircle(unittest.TestCase):
    def test_urdf_morph_circle_yaw_noise_disturb(self):
        # Long-ish scenario, but keep Hz moderate for CI/local speed.
        # This regression focuses on long-horizon stability under morphing + noise + steady torque disturbance.
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "long.csv")
            p = _run_module(
                [
                    "sim.demo.demo_04_hover_pd_urdf_morph",
                    "--seconds",
                    "22.0",
                    "--hz",
                    "240",
                    "--CQ",
                    "0.08",
                    "--phi-amp",
                    "10",
                    "--phi-freq",
                    "0.2",
                    "--morph-tau",
                    "0.08",
                    "--morph-rate",
                    "3.0",
                    "--tw-ratio",
                    "4.0",
                    "--kp-att",
                    "0.6",
                    "--kd-att",
                    "0.08",
                    "--ki-att",
                    "0.02",
                    "--att-int-limit",
                    "1.0",
                    "--dist-body",
                    "10.0",
                    "10.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.01",
                    "0.0",
                    "0.0",
                    "--dist-body-frame",
                    "body",
                    "--thrust-noise-sigma",
                    "0.01",
                    "--noise-seed",
                    "42",
                    "--log-csv",
                    csv_path,
                    "--log-flush-every",
                    "1",
                    "--log-every",
                    "0",
                ],
                timeout_s=120.0,
            )
            self.assertEqual(
                p.returncode,
                0,
                msg=f"demo_04 failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}",
            )
            rows = _read_csv_rows(csv_path)
            self.assertGreater(len(rows), 100)

            # Safety checks: never free-fall due to all-zero omega2_cmd, and avoid ground contact.
            for row in rows:
                z = float(row["z"])
                self.assertGreater(z, 0.15, msg=f"z too low at t={row['t']}: z={z}")
                o = [float(row[f"omega2_cmd_{i}"]) for i in range(4)]
                self.assertGreater(max(o), 1e-8, msg=f"all omega2_cmd ~0 at t={row['t']}")

            # Position sanity: keep XY within reasonable bounds (conservative)
            max_abs_x = max(abs(float(r["x"])) for r in rows)
            max_abs_y = max(abs(float(r["y"])) for r in rows)
            self.assertLess(max_abs_x, 20.0)
            self.assertLess(max_abs_y, 20.0)


if __name__ == "__main__":
    unittest.main()


