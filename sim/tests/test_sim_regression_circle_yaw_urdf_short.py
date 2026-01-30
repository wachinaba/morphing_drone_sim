import csv
import os
import subprocess
import sys
import tempfile
import unittest


def _run_module(args: list[str], timeout_s: float = 60.0) -> subprocess.CompletedProcess:
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


@unittest.skipUnless(
    os.environ.get("RUN_SIM_REGRESSION_CIRCLE", "0") == "1",
    "set RUN_SIM_REGRESSION_CIRCLE=1 to run circle+yaw regression tests",
)
class TestSimRegressionCircleYawUrdfShort(unittest.TestCase):
    def test_circle_yaw_phi_morph_no_allzero(self):
        # Short test intended to catch early "omega2_cmd all-zero" regressions.
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "short.csv")
            p = _run_module(
                [
                    "sim.demo.demo_04_hover_pd_urdf_morph",
                    "--seconds",
                    "3.0",
                    "--hz",
                    "120",
                    "--xy-circle",
                    "--xy-circle-radius",
                    "0.2",
                    "--xy-circle-freq",
                    "0.5",
                    "--yaw-circle",
                    "--yaw-tau",
                    "0.6",
                    "--yaw-rate",
                    "60",
                    "--phi-amp",
                    "25",
                    "--phi-freq",
                    "0.5",
                    "--morph-tau",
                    "0.08",
                    "--morph-rate",
                    "3.0",
                    "--CQ",
                    "0.08",
                    "--tw-ratio",
                    "6.0",
                    "--kp-att",
                    "0.2",
                    "--kd-att",
                    "0.05",
                    "--torque-priority",
                    "rp",
                    "--log-csv",
                    csv_path,
                    "--log-flush-every",
                    "1",
                    "--log-every",
                    "0",
                ],
                timeout_s=60.0,
            )
            self.assertEqual(
                p.returncode,
                0,
                msg=f"demo_04 failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}",
            )
            rows = _read_csv_rows(csv_path)
            self.assertGreater(len(rows), 50)

            for row in rows:
                z = float(row["z"])
                self.assertGreater(z, 0.05, msg=f"z too low at t={row['t']}: z={z}")
                o = [float(row[f"omega2_cmd_{i}"]) for i in range(4)]
                self.assertGreater(max(o), 1e-8, msg=f"all omega2_cmd ~0 at t={row['t']}")


if __name__ == "__main__":
    unittest.main()


