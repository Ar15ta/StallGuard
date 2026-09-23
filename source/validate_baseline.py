from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_defaults
from src.io_csv import load_cfd_csv
from src.models import DataSourceSettings
from src.online_engine import OnlineWarningEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用指定正常基线离线检查另一段工况数据的报警情况。")
    parser.add_argument("baseline_csv", type=Path, help="已知正常且稳定的数据段")
    parser.add_argument("monitor_csv", type=Path, help="待检查的连续实验数据")
    parser.add_argument("--method", choices=("autocorrelation", "entropy"), default="autocorrelation")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--minimum-probes", type=int, default=2)
    parser.add_argument("--known-stall-step", type=float, help="可选；用于区分失速前误报与失速后报警")
    return parser.parse_args()


def engine_params(method: str, threshold: float, minimum_probes: int) -> dict:
    defaults = load_defaults()["analysis"]
    window_size = int(defaults["window_size"])
    step_size = max(1, round(window_size * (100 - int(defaults["overlap_percent"])) / 100))
    return {
        "window_size": window_size,
        "step_size": step_size,
        "threshold": threshold,
        "direction": "high",
        "trigger_windows": int(defaults["trigger_windows"]),
        "release_windows": int(defaults["release_windows"]),
        "release_ratio": float(defaults["release_ratio"]),
        "minimum_probes": minimum_probes,
        "entropy_order": int(defaults["entropy_order"]),
        "entropy_delay": int(defaults["entropy_delay"]),
        "min_lag": int(defaults["min_lag"]),
        "max_lag": int(defaults["max_lag"]),
    }


def main() -> int:
    args = parse_args()
    settings = DataSourceSettings(first_row_zero=True)
    baseline = load_cfd_csv(args.baseline_csv, settings)
    monitor = load_cfd_csv(args.monitor_csv, settings)
    if baseline.probe_names != monitor.probe_names:
        raise ValueError("基线段与监测段的探针名称或顺序不一致。")
    if baseline.probe_count != monitor.probe_count:
        raise ValueError("基线段与监测段的探针数量不一致。")
    params = engine_params(args.method, args.threshold, min(args.minimum_probes, baseline.probe_count))
    available = 1 + max(0, (baseline.sample_count - params["window_size"]) // params["step_size"])
    calibration_windows = min(int(load_defaults()["online"]["calibration_windows"]), available)
    if calibration_windows < 10:
        raise ValueError("基线段不足以生成 10 个标定窗口。")
    commissioning_rows = params["window_size"] + (calibration_windows - 1) * params["step_size"]
    calibration = OnlineWarningEngine(baseline.probe_count, args.method, params, calibration_windows)
    calibration.update(baseline.time_s[:commissioning_rows], baseline.values[:commissioning_rows])
    profile = calibration.calibration_profile()
    if profile is None:
        raise RuntimeError("未能生成基线。")
    monitoring = OnlineWarningEngine(monitor.probe_count, args.method, params, calibration_windows, profile)
    snapshots = monitoring.update(monitor.time_s, monitor.values)
    alarm_snapshots = [snapshot for snapshot in snapshots if snapshot.system_alarm]
    print(f"判据：{args.method}")
    print(f"标定窗口：{calibration_windows}")
    print(f"监测窗口：{len(snapshots)}")
    print(f"报警窗口：{len(alarm_snapshots)}")
    if alarm_snapshots:
        first = alarm_snapshots[0]
        first_step = float(monitor.steps[min(first.sample_end - 1, monitor.sample_count - 1)])
        print(f"首次报警：t={first.time_s:.9f} s，step={first_step:g}")
    else:
        print("首次报警：无")
    if args.known_stall_step is not None:
        pre_stall = 0
        for snapshot in alarm_snapshots:
            step = monitor.steps[min(snapshot.sample_end - 1, monitor.sample_count - 1)]
            pre_stall += int(step < args.known_stall_step)
        print(f"已知失速步之前的报警窗口：{pre_stall}")
        return 2 if pre_stall else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
