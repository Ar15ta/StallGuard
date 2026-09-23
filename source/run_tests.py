from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    unit = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, env=environment)
    if unit.returncode:
        return unit.returncode
    candidates = (ROOT.parent / "data" / "pressure_timestep.csv", ROOT.parent / "release_v022" / "pressure_timestep.csv")
    smoke_csv = next((path for path in candidates if path.is_file()), candidates[0])
    smoke = subprocess.run([sys.executable, "app.py", "--smoke-test", str(smoke_csv)], cwd=ROOT, env=environment)
    if smoke.returncode:
        return smoke.returncode
    simulator = subprocess.run(
        [sys.executable, "sensor_simulator_gui.py", "--smoke-test", str(smoke_csv)],
        cwd=ROOT,
        env=environment,
    )
    return simulator.returncode


if __name__ == "__main__":
    raise SystemExit(main())
