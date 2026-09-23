from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

from src.config import load_defaults
from src.io_csv import load_cfd_csv
from src.models import DataSourceSettings


def default_csv_path() -> Path:
    """Return the demo CSV beside the packaged simulator or in the source tree."""
    frozen = bool(getattr(sys, "frozen", False))
    root = Path(sys.executable).resolve().parent.parent if frozen else Path(__file__).resolve().parent
    candidates = (root / "pressure_timestep.csv", root.parent / "release_v022" / "pressure_timestep.csv")
    return next((path for path in candidates if path.is_file()), candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟实时传感器，向 StallGuard 发送多探针数据。")
    parser.add_argument("csv", nargs="?", type=Path, default=default_csv_path())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=20.0, help="发送速率，行/秒；0 表示不限速")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-step", type=float)
    parser.add_argument("--end-step", type=float)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def send_message(connection: socket.socket, payload: dict) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    connection.sendall(encoded)


def main() -> int:
    args = parse_args()
    if args.rate < 0 or args.batch_size < 1 or (args.max_rows is not None and args.max_rows < 1):
        raise ValueError("rate 不能小于 0；batch-size 和 max-rows 必须大于等于 1。")
    defaults = load_defaults()["data_source"]
    settings = DataSourceSettings(
        output_interval_s=float(defaults["output_interval_s"]),
        step_origin=float(defaults["step_origin"]),
        time_origin_s=float(defaults["time_origin_s"]),
        first_row_zero=True,
        start_angle_deg=float(defaults["start_angle_deg"]),
        angle_direction=int(defaults["angle_direction"]),
    )
    dataset = load_cfd_csv(args.csv, settings)
    mask = dataset.steps >= args.start_step if args.start_step is not None else dataset.steps == dataset.steps
    if args.end_step is not None:
        mask &= dataset.steps <= args.end_step
    indices = mask.nonzero()[0]
    if args.max_rows is not None:
        indices = indices[: args.max_rows]
    if indices.size == 0:
        raise ValueError("指定范围内没有可发送的数据。")
    print(f"连接 {args.host}:{args.port}，准备发送 {indices.size:,} 行，{dataset.probe_count} 个探针。")
    with socket.create_connection((args.host, args.port), timeout=10.0) as connection:
        send_message(
            connection,
            {
                "type": "meta",
                "probe_names": dataset.probe_names,
                "output_interval_s": dataset.settings.output_interval_s,
                "total_rows": int(indices.size),
            },
        )
        for offset in range(0, indices.size, args.batch_size):
            selected = indices[offset : offset + args.batch_size]
            send_message(
                connection,
                {
                    "type": "samples",
                    "steps": dataset.steps[selected].tolist(),
                    "values": dataset.values[selected].tolist(),
                },
            )
            if args.rate > 0:
                time.sleep(selected.size / args.rate)
            if offset == 0 or (offset + selected.size) % 100 == 0:
                print(f"已发送 {offset + selected.size:,}/{indices.size:,} 行", flush=True)
    print("发送完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
