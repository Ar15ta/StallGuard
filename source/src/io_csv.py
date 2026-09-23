from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import DataSourceSettings, ProbeDataset

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin-1")


def _detect_encoding(path: Path) -> str:
    for encoding in ENCODINGS:
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _find_header(path: Path, encoding: str) -> tuple[int, str]:
    with path.open("r", encoding=encoding, newline="") as handle:
        for row_number, line in enumerate(handle):
            for delimiter in (",", "\t", ";"):
                cells = next(csv.reader([line], delimiter=delimiter))
                if len(cells) >= 2 and any("step" in cell.lower() or "time" in cell.lower() for cell in cells):
                    return row_number, delimiter
    raise ValueError("未找到包含时间步和探针列的 CSV 表头。")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simplify_probe_name(header: str, fallback: str) -> str:
    text = re.sub(r"\s*\[[^]]*]\s*$", "", str(header)).strip()
    match = re.search(r"Monitor Point:\s*(.*?)\s*\([^)]*\)\s*$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text or fallback


def _extract_unit(headers: list[str]) -> str:
    units = []
    for header in headers:
        match = re.search(r"\[([^]]+)]\s*$", str(header))
        if match:
            units.append(match.group(1).strip())
    unique = list(dict.fromkeys(units))
    return unique[0] if len(unique) == 1 else (", ".join(unique) if unique else "未指定")


def generate_angles(count: int, start_angle_deg: float = 0.0, direction: int = 1) -> np.ndarray:
    if count <= 0:
        raise ValueError("探针数必须大于 0。")
    if direction not in (-1, 1):
        raise ValueError("周向排列方向必须为递增或递减。")
    spacing = 360.0 / count
    return np.mod(start_angle_deg + direction * spacing * np.arange(count), 360.0)


def load_cfd_csv(path: str | Path, settings: DataSourceSettings) -> ProbeDataset:
    """Load a CFD Monitor wide CSV and derive physical time and probe angles."""
    settings.validate()
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"文件不存在：{source}")
    if source.stat().st_size == 0:
        raise ValueError("CSV 文件为空。")
    encoding = _detect_encoding(source)
    header_index, delimiter = _find_header(source, encoding)
    frame = pd.read_csv(source, sep=delimiter, skiprows=header_index, encoding=encoding, engine="python")
    if frame.shape[1] < 2:
        raise ValueError("CSV 至少需要一列累计时间步和一列探针数据。")
    step_header = str(frame.columns[0])
    probe_headers = [str(column) for column in frame.columns[1:]]
    frame["__step__"] = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    invalid_step_rows = int(frame["__step__"].isna().sum())
    frame = frame.dropna(subset=["__step__"])
    duplicate_rows = int(frame["__step__"].duplicated().sum())
    frame = frame.groupby("__step__", sort=True, as_index=False).mean(numeric_only=True)
    steps = frame.pop("__step__").to_numpy(dtype=float)
    if steps.size == 0:
        raise ValueError("CSV 中没有可用的数值时间步。")
    numeric = pd.DataFrame({header: pd.to_numeric(frame.get(header), errors="coerce") for header in probe_headers})
    all_missing = [header for header in probe_headers if numeric[header].isna().all()]
    if all_missing:
        raise ValueError("以下探针列没有任何有效数值：" + ", ".join(all_missing))
    values = numeric.to_numpy(dtype=float)
    if settings.first_row_zero:
        time_s = (steps - steps[0]) * settings.output_interval_s
    else:
        time_s = settings.time_origin_s + (steps - settings.step_origin) * settings.output_interval_s
    differences = np.diff(steps)
    quality = {
        "encoding": encoding,
        "delimiter": delimiter,
        "header_row": header_index + 1,
        "invalid_step_rows": invalid_step_rows,
        "duplicate_rows": duplicate_rows,
        "missing_values": int(np.count_nonzero(~np.isfinite(values))),
        "gap_count": int(np.count_nonzero(differences > 1)),
        "non_unit_step_intervals": int(np.count_nonzero(~np.isclose(differences, 1.0))),
        "first_step": float(steps[0]),
        "last_step": float(steps[-1]),
    }
    names = [_simplify_probe_name(header, f"探针 {index + 1}") for index, header in enumerate(probe_headers)]
    return ProbeDataset(
        path=source,
        file_hash=_file_sha256(source),
        steps=steps,
        time_s=time_s,
        probe_names=names,
        original_headers=probe_headers,
        angles_deg=generate_angles(len(names), settings.start_angle_deg, settings.angle_direction),
        values=values,
        unit=_extract_unit(probe_headers),
        settings=settings,
        quality=quality,
    )
