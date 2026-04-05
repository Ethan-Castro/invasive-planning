from __future__ import annotations

import json
import math
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from communication_lab.config import DEFAULT_LAG_S, TRIBE_REPO_ID, get_app_paths
from communication_lab.schemas import CompareRequest, InputVariant, Modality


@dataclass
class VariantPrediction:
    variant_id: str
    prepared_path: Path
    segment_times_s: list[float]
    stimulus_times_s: list[float]
    preds: np.ndarray
    warnings: list[str]


class TribeService:
    def __init__(
        self,
        cache_root: Path | None = None,
        repo_id: str = TRIBE_REPO_ID,
        device: str | None = None,
        hemodynamic_lag_s: float = DEFAULT_LAG_S,
    ) -> None:
        self.paths = get_app_paths(cache_root)
        self.repo_id = repo_id
        self.device = device or "auto"
        self.hemodynamic_lag_s = hemodynamic_lag_s
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from tribev2 import TribeModel
        except ImportError as exc:
            raise RuntimeError(
                "TRIBE runtime dependencies are missing. Install with "
                "`uv sync --extra runtime`."
            ) from exc
        chosen_device = self.device
        if chosen_device == "auto":
            try:
                import torch
            except ImportError:
                chosen_device = "cpu"
            else:
                chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = TribeModel.from_pretrained(
            self.repo_id,
            cache_folder=str(self.paths.cache),
            device=chosen_device,
        )
        return self._model

    def predict_variant(
        self,
        variant: InputVariant,
        request: CompareRequest,
    ) -> VariantPrediction:
        model = self._load_model()
        prepared_path, warnings = self._prepare_input(variant, request.max_duration_s)
        events = model.get_events_dataframe(
            **{f"{variant.modality.value}_path": str(prepared_path)}
        )
        preds, segments = model.predict(events=events, verbose=False)
        stimulus_times_s = [self._segment_start(segment, index) for index, segment in enumerate(segments)]
        segment_times_s = [time + self.hemodynamic_lag_s for time in stimulus_times_s]
        return VariantPrediction(
            variant_id=variant.id,
            prepared_path=prepared_path,
            segment_times_s=segment_times_s,
            stimulus_times_s=stimulus_times_s,
            preds=np.asarray(preds),
            warnings=warnings,
        )

    def _prepare_input(
        self,
        variant: InputVariant,
        max_duration_s: int,
    ) -> tuple[Path, list[str]]:
        warnings: list[str] = []
        if variant.modality == Modality.text:
            return self._prepare_text_input(variant, max_duration_s)
        duration = self._probe_media_duration(variant.file_path)
        if duration is None or duration <= max_duration_s:
            return variant.file_path, warnings
        trimmed_path = self.paths.trimmed / f"{variant.id}_{uuid.uuid4().hex}{variant.file_path.suffix.lower()}"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(variant.file_path),
            "-t",
            str(max_duration_s),
            "-c",
            "copy",
            str(trimmed_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to trim {variant.file_path}: {completed.stderr.strip()}"
            )
        warnings.append(
            f"{variant.label} exceeded the {max_duration_s}s limit and was trimmed for inference."
        )
        return trimmed_path, warnings

    def _prepare_text_input(
        self,
        variant: InputVariant,
        max_duration_s: int,
    ) -> tuple[Path, list[str]]:
        warnings: list[str] = []
        text = variant.file_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Text variant is empty: {variant.file_path}")
        estimated_seconds = self._estimate_text_duration_s(text)
        if estimated_seconds <= max_duration_s:
            return variant.file_path, warnings
        allowed_words = max(1, math.floor(max_duration_s / 60 * 150))
        trimmed_text = " ".join(text.split()[:allowed_words]).strip()
        trimmed_path = self.paths.trimmed / f"{variant.id}_{uuid.uuid4().hex}.txt"
        trimmed_path.write_text(trimmed_text, encoding="utf-8")
        warnings.append(
            f"{variant.label} exceeded the text length budget and was trimmed to roughly {max_duration_s}s of speech."
        )
        return trimmed_path, warnings

    @staticmethod
    def _estimate_text_duration_s(text: str) -> float:
        words = max(1, len(text.split()))
        return words / 150 * 60

    @staticmethod
    def _probe_media_duration(path: Path) -> float | None:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return None
        payload = json.loads(completed.stdout or "{}")
        try:
            return float(payload["format"]["duration"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _segment_start(segment: object, fallback_index: int) -> float:
        for attr in ("start", "offset"):
            value = getattr(segment, attr, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return float(fallback_index)
