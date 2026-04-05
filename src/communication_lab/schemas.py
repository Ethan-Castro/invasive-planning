from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from communication_lab.config import (
    DEFAULT_LAG_S,
    DEFAULT_MAX_DURATION_S,
    DEFAULT_PROXY_FAMILY,
    DEFAULT_TOP_K_ROIS,
)


class Modality(str, Enum):
    text = "text"
    audio = "audio"
    video = "video"


class DomainPack(str, Enum):
    teacher = "teacher"
    doctor = "doctor"
    marketing = "marketing"
    investor = "investor"


class ProxyFamily(str, Enum):
    language_processing_proxy = "language_processing_proxy"
    emotional_social_proxy = "emotional_social_proxy"
    auditory_salience_proxy = "auditory_salience_proxy"
    visual_salience_proxy = "visual_salience_proxy"
    cross_region_spread_proxy = "cross_region_spread_proxy"


SUPPORTED_SUFFIXES: dict[Modality, set[str]] = {
    Modality.text: {".txt"},
    Modality.audio: {".wav", ".mp3", ".flac", ".ogg"},
    Modality.video: {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}


class InputVariant(BaseModel):
    id: str
    label: str
    modality: Modality
    file_path: Path
    domain_pack: DomainPack
    notes: str = ""

    @field_validator("id", "label")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("file_path")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _validate_path(self) -> "InputVariant":
        if not self.file_path.exists():
            raise FileNotFoundError(f"Variant file does not exist: {self.file_path}")
        suffix = self.file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES[self.modality]:
            raise ValueError(
                f"{self.modality.value} expects one of "
                f"{sorted(SUPPORTED_SUFFIXES[self.modality])}, got {suffix!r}"
            )
        return self


class CompareRequest(BaseModel):
    variants: list[InputVariant]
    primary_proxy_family: ProxyFamily = Field(default=ProxyFamily(DEFAULT_PROXY_FAMILY))
    top_k_rois: int = Field(default=DEFAULT_TOP_K_ROIS, ge=1, le=25)
    max_duration_s: int = Field(default=DEFAULT_MAX_DURATION_S, ge=15, le=600)
    export_dir: Path

    @field_validator("export_dir")
    @classmethod
    def _expand_export_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _validate_variants(self) -> "CompareRequest":
        if not 1 <= len(self.variants) <= 5:
            raise ValueError("CompareRequest expects between 1 and 5 variants")
        ids = [variant.id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("Variant ids must be unique")
        return self


class DivergenceWindow(BaseModel):
    start_time_s: float
    end_time_s: float
    peak_delta: float


class PairwiseDelta(BaseModel):
    variant_a_id: str
    variant_b_id: str
    proxy_deltas: dict[str, float]
    top_roi_deltas: dict[str, float]
    divergence_windows: list[DivergenceWindow]


class CompareResult(BaseModel):
    segment_times_s: dict[str, list[float]]
    prediction_shapes: dict[str, list[int]] = Field(default_factory=dict)
    raw_vertex_preds_path: dict[str, str]
    roi_timeseries: dict[str, dict[str, list[float]]]
    top_rois: dict[str, list[str]]
    proxy_scores: dict[str, dict[str, float]]
    pairwise_deltas: list[PairwiseDelta]
    warnings: list[str]
    report_path: str
    cortical_image_paths: dict[str, str | None] = Field(default_factory=dict)
    hemodynamic_lag_s: float = DEFAULT_LAG_S
    average_subject_scope: bool = True
    llm_explanation: str | None = None
    llm_explanation_model: str | None = None
    model_scope_note: str = (
        "TRIBE v2 predictions represent average-subject fMRI-like responses and should "
        "be treated as neural proxies rather than direct behavioral predictions."
    )


class HumanRatingRecord(BaseModel):
    variant_id: str
    rater_id: str
    domain_pack: DomainPack
    clarity: float = Field(ge=0.0, le=1.0)
    emotional_load: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class CalibrationSchema(BaseModel):
    enabled: bool = False
    status: str = "reserved_for_phase_2"
    fields: tuple[str, ...] = ("clarity", "emotional_load")
