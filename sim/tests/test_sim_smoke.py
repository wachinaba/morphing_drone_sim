import csv
import os
import subprocess
import sys
import tempfile
import unittest


def _run_module(args: list[str], timeout_s: float = 30.0) -> subprocess.CompletedProcess:
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


def _float(x: str) -> float:
    return float(x)


@unittest.skipUnless(os.environ.get("RUN_SIM_SMOKE", "0") == "1", "set RUN_SIM_SMOKE=1 to run PyBullet smoke tests")
class TestSimSmoke(unittest.TestCase):
    def test_demo03_circle_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "demo03.csv")
            p = _run_module(
                [
                    "sim.demo.demo_03_hover_pd",
                    "--seconds",
                    "0.3",
                    "--xy-circle",
                    "--xy-circle-radius",
                    "0.2",
                    "--xy-circle-freq",
                    "0.5",
                    "--tw-ratio",
                    "3.0",
                    "--log-csv",
                    csv_path,
                    "--log-flush-every",
                    "1",
                    "--log-every",
                    "0",
                ],
                timeout_s=30.0,
            )
            self.assertEqual(
                p.returncode,
                0,
                msg=f"demo_03 failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}",
            )
            rows = _read_csv_rows(csv_path)
            self.assertGreater(len(rows), 5)
            # Basic sanity: no NaN/Inf and omega2 non-negative
            for row in rows[:20]:
                for k in ["x", "y", "z", "Fz_body", "tau_x_body", "tau_y_body"]:
                    v = _float(row[k])
                    self.assertTrue(v == v)  # not NaN
                for k in ["omega2_cmd_0", "omega2_cmd_1", "omega2_cmd_2", "omega2_cmd_3"]:
                    self.assertGreaterEqual(_float(row[k]), -1e-9)

    def test_demo04_urdf_circle_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "demo04.csv")
            p = _run_module(
                [
                    "sim.demo.demo_04_hover_pd_urdf_morph",
                    "--seconds",
                    "0.3",
                    "--xy-circle",
                    "--xy-circle-radius",
                    "0.2",
                    "--xy-circle-freq",
                    "0.5",
                    "--tw-ratio",
                    "3.0",
                    "--log-csv",
                    csv_path,
                    "--log-flush-every",
                    "1",
                    "--log-every",
                    "0",
                ],
                timeout_s=30.0,
            )
            self.assertEqual(
                p.returncode,
                0,
                msg=f"demo_04 failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}",
            )
            rows = _read_csv_rows(csv_path)
            self.assertGreater(len(rows), 5)
            # Sanity: omega2 within [0, omega2_max_eff] (when present)
            for row in rows[:30]:
                mx = row.get("omega2_max_eff", "")
                omega2 = [_float(row[f"omega2_cmd_{i}"]) for i in range(4)]
                self.assertTrue(all(o >= -1e-9 for o in omega2))
                if mx != "":
                    mxf = _float(mx)
                    self.assertTrue(all(o <= mxf + 1e-6 for o in omega2))


if __name__ == "__main__":
    unittest.main()




