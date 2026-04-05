from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "TRIBE v2 Communication Lab"
DEFAULT_LAG_S = 5.0
DEFAULT_MAX_DURATION_S = 120
DEFAULT_TOP_K_ROIS = 8
DEFAULT_PROXY_FAMILY = "language_processing_proxy"
DEFAULT_CACHE_ROOT = Path(".communication_lab")
TRIBE_REPO_ID = "facebook/tribev2"
LLAMA_GATE_MODEL_ID = "meta-llama/Llama-3.2-3B"
REPORT_TEMPLATE_VERSION = "v1"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    cache: Path
    exports: Path
    uploads: Path
    raw_predictions: Path
    figures: Path
    trimmed: Path


def get_app_paths(root: Path | None = None) -> AppPaths:
    base = (root or DEFAULT_CACHE_ROOT).resolve()
    cache = base / "cache"
    exports = base / "exports"
    uploads = base / "uploads"
    raw_predictions = base / "raw_predictions"
    figures = base / "figures"
    trimmed = base / "trimmed"
    for path in [base, cache, exports, uploads, raw_predictions, figures, trimmed]:
        path.mkdir(parents=True, exist_ok=True)
    return AppPaths(
        root=base,
        cache=cache,
        exports=exports,
        uploads=uploads,
        raw_predictions=raw_predictions,
        figures=figures,
        trimmed=trimmed,
    )
