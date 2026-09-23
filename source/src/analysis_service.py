from __future__ import annotations

from typing import Any

import numpy as np

from .algorithms import autocorrelation_features, permutation_entropy_features, wavelet_energy_features
from .models import SystemAnalysisResult
from .warning_engine import aggregate_probe_alarms, apply_hysteresis, robust_risk_scores

METHOD_LABELS = {
    "wavelet": "小波分析",
    "entropy": "数据熵",
    "autocorrelation": "时序自相关",
}


def _single_feature(method: str, values: np.ndarray, time_s: np.ndarray, params: dict[str, Any]):
    common = {
        "values": values,
        "time_s": time_s,
        "window_size": int(params["window_size"]),
        "step_size": int(params["step_size"]),
    }
    if method == "wavelet":
        return wavelet_energy_features(
            **common,
            wavelet=params.get("wavelet", "db4"),
            level=int(params.get("wavelet_level", 4)),
            detail_level=int(params.get("detail_level", 2)),
        )
    if method == "entropy":
        return permutation_entropy_features(
            **common,
            order=int(params.get("entropy_order", 5)),
            delay=int(params.get("entropy_delay", 1)),
        )
    if method == "autocorrelation":
        return autocorrelation_features(
            **common,
            min_lag=int(params.get("min_lag", 2)),
            max_lag=int(params.get("max_lag", 200)),
        )
    raise ValueError(f"未知分析类型：{method}")


def analyze_signals(
    values: np.ndarray, time_s: np.ndarray, method: str, params: dict[str, Any]
) -> SystemAnalysisResult:
    threshold = float(params.get("threshold", 3.0))
    direction = str(params.get("direction", "high"))
    feature, times, starts, ends, auxiliary = _single_feature(method, values, time_s, params)
    risk_score, centers, scales = robust_risk_scores(
        feature,
        baseline_fraction=float(params.get("baseline_fraction", 0.2)),
        direction=direction,
    )
    probe_alarm = apply_hysteresis(
        risk_score,
        threshold=threshold,
        trigger_windows=int(params.get("trigger_windows", 3)),
        release_windows=int(params.get("release_windows", 5)),
        release_ratio=float(params.get("release_ratio", 0.7)),
    )
    minimum_probes = min(int(params.get("minimum_probes", 2)), values.shape[1])
    system_alarm = aggregate_probe_alarms(probe_alarm, minimum_probes)
    return SystemAnalysisResult(
        method=method,
        window_times=np.asarray(times),
        window_starts=np.asarray(starts),
        window_ends=np.asarray(ends),
        feature=feature,
        risk_score=risk_score,
        probe_alarm=probe_alarm,
        system_alarm=system_alarm,
        valid=np.isfinite(risk_score),
        threshold=threshold,
        centers=centers,
        scales=scales,
        metadata={"method_label": METHOD_LABELS[method], **params},
        auxiliary=auxiliary,
    )
