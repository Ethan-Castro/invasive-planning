from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

from communication_lab.schemas import CompareRequest, DivergenceWindow, PairwiseDelta, ProxyFamily

ROI_FAMILIES: dict[ProxyFamily, tuple[str, ...]] = {
    ProxyFamily.language_processing_proxy: ("A5", "45", "PGi", "TE1a", "STS*"),
    ProxyFamily.emotional_social_proxy: ("PGi", "TE1a", "STS*", "TPOJ*"),
    ProxyFamily.auditory_salience_proxy: ("A1", "A4", "A5", "LBelt", "MBelt", "PBelt", "RI", "STS*"),
    ProxyFamily.visual_salience_proxy: ("V1", "V2", "V3", "V4", "V4t", "MT", "MST", "LO*", "IPS*", "PH"),
}


@dataclass
class VariantAnalysis:
    roi_timeseries: pd.DataFrame
    proxy_timeseries: dict[str, np.ndarray]
    proxy_scores: dict[str, float]
    top_rois: list[str]
    mean_surface: np.ndarray


@lru_cache(maxsize=1)
def get_hcp_labels(mesh: str = "fsaverage5") -> dict[str, np.ndarray]:
    try:
        from tribev2.utils import get_hcp_labels as tribe_get_hcp_labels
    except ImportError as exc:
        raise RuntimeError(
            "TRIBE runtime dependencies are missing. Install with `uv sync --extra runtime`."
        ) from exc
    labels = tribe_get_hcp_labels(mesh=mesh, combine=False, hemi="both")
    return {name: np.asarray(indices, dtype=int) for name, indices in labels.items()}


def summarize_predictions(
    preds: np.ndarray,
    request: CompareRequest,
    mesh: str = "fsaverage5",
) -> VariantAnalysis:
    labels = get_hcp_labels(mesh=mesh)
    roi_frame = pd.DataFrame(
        {
            roi_name: preds[:, vertex_indices].mean(axis=1)
            for roi_name, vertex_indices in labels.items()
        }
    )
    abs_roi_frame = roi_frame.abs()
    top_rois = abs_roi_frame.max(axis=0).sort_values(ascending=False).head(request.top_k_rois).index.tolist()
    proxy_timeseries: dict[str, np.ndarray] = {}
    proxy_scores: dict[str, float] = {}
    for proxy_family, roi_patterns in ROI_FAMILIES.items():
        selected = _select_rois(roi_frame.columns.tolist(), roi_patterns)
        if not selected:
            proxy_timeseries[proxy_family.value] = np.zeros(len(roi_frame), dtype=float)
            proxy_scores[proxy_family.value] = 0.0
            continue
        series = abs_roi_frame[selected].mean(axis=1).to_numpy(dtype=float)
        proxy_timeseries[proxy_family.value] = series
        proxy_scores[proxy_family.value] = float(series.mean())
    spread_series = _compute_spread_proxy(abs_roi_frame.to_numpy(dtype=float))
    proxy_timeseries[ProxyFamily.cross_region_spread_proxy.value] = spread_series
    proxy_scores[ProxyFamily.cross_region_spread_proxy.value] = float(spread_series.mean())
    return VariantAnalysis(
        roi_timeseries=roi_frame,
        proxy_timeseries=proxy_timeseries,
        proxy_scores=proxy_scores,
        top_rois=top_rois,
        mean_surface=np.abs(preds).mean(axis=0),
    )


def build_pairwise_deltas(
    analyses: dict[str, VariantAnalysis],
    request: CompareRequest,
) -> list[PairwiseDelta]:
    results: list[PairwiseDelta] = []
    for left_id, right_id in combinations(analyses.keys(), 2):
        left = analyses[left_id]
        right = analyses[right_id]
        proxy_deltas = {
            proxy_name: right.proxy_scores[proxy_name] - left.proxy_scores[proxy_name]
            for proxy_name in left.proxy_scores
        }
        top_roi_deltas = _roi_delta_map(left.roi_timeseries, right.roi_timeseries, top_n=request.top_k_rois)
        divergence_windows = _find_divergence_windows(
            left.proxy_timeseries[request.primary_proxy_family.value],
            right.proxy_timeseries[request.primary_proxy_family.value],
        )
        results.append(
            PairwiseDelta(
                variant_a_id=left_id,
                variant_b_id=right_id,
                proxy_deltas=proxy_deltas,
                top_roi_deltas=top_roi_deltas,
                divergence_windows=divergence_windows,
            )
        )
    return results


def _select_rois(all_rois: list[str], patterns: tuple[str, ...]) -> list[str]:
    selected: set[str] = set()
    for pattern in patterns:
        if pattern.endswith("*"):
            selected.update(roi for roi in all_rois if roi.startswith(pattern[:-1]))
        elif pattern.startswith("*"):
            selected.update(roi for roi in all_rois if roi.endswith(pattern[1:]))
        else:
            selected.update(roi for roi in all_rois if roi == pattern)
    return sorted(selected)


def _compute_spread_proxy(abs_roi_values: np.ndarray) -> np.ndarray:
    if abs_roi_values.size == 0:
        return np.zeros(0, dtype=float)
    totals = abs_roi_values.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    probabilities = abs_roi_values / totals
    entropy = -(probabilities * np.log(probabilities + 1e-8)).sum(axis=1)
    return entropy / np.log(abs_roi_values.shape[1])


def _roi_delta_map(
    left: pd.DataFrame,
    right: pd.DataFrame,
    top_n: int,
) -> dict[str, float]:
    common = left.columns.intersection(right.columns)
    deltas = right[common].abs().mean(axis=0) - left[common].abs().mean(axis=0)
    top = deltas.abs().sort_values(ascending=False).head(top_n)
    return {roi: float(deltas[roi]) for roi in top.index}


def _find_divergence_windows(
    left_series: np.ndarray,
    right_series: np.ndarray,
    window_s: int = 3,
    max_windows: int = 3,
) -> list[DivergenceWindow]:
    length = min(len(left_series), len(right_series))
    if length == 0:
        return []
    delta = np.abs(right_series[:length] - left_series[:length])
    kernel = np.ones(window_s) / window_s
    smoothed = np.convolve(delta, kernel, mode="same")
    ranked = np.argsort(smoothed)[::-1]
    chosen: list[int] = []
    windows: list[DivergenceWindow] = []
    for index in ranked:
        if any(abs(index - prev) < window_s for prev in chosen):
            continue
        chosen.append(int(index))
        start = max(0.0, float(index - window_s / 2))
        end = float(min(length - 1, index + window_s / 2))
        windows.append(
            DivergenceWindow(
                start_time_s=start,
                end_time_s=end,
                peak_delta=float(smoothed[index]),
            )
        )
        if len(windows) >= max_windows:
            break
    return windows
