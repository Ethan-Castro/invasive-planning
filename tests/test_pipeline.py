from pathlib import Path

import numpy as np
import pytest

from communication_lab.pipeline import CommunicationLabPipeline
from communication_lab.schemas import CompareRequest, DomainPack, InputVariant, Modality, ProxyFamily
from communication_lab.tribe_service import VariantPrediction


class FakePredictionService:
    def __init__(self):
        self.vertex_count = 12

    def predict_variant(self, variant: InputVariant, request: CompareRequest) -> VariantPrediction:
        seed = sum(ord(ch) for ch in variant.id) + len(variant.label)
        rng = np.random.default_rng(seed)
        preds = rng.normal(loc=0.0, scale=1.0, size=(6, self.vertex_count))
        times = [float(index + 5) for index in range(len(preds))]
        return VariantPrediction(
            variant_id=variant.id,
            prepared_path=variant.file_path,
            segment_times_s=times,
            stimulus_times_s=[time - 5 for time in times],
            preds=preds,
            warnings=[f"fake warning for {variant.id}"],
        )


@pytest.fixture(autouse=True)
def patch_hcp_labels(monkeypatch):
    labels = {
        "A5": np.array([0, 1]),
        "45": np.array([2]),
        "PGi": np.array([3]),
        "TE1a": np.array([4]),
        "STSda": np.array([5]),
        "TPOJ1": np.array([6]),
        "A1": np.array([7]),
        "V1": np.array([8]),
        "LO1": np.array([9]),
        "PH": np.array([10]),
        "IPS1": np.array([11]),
    }
    monkeypatch.setattr("communication_lab.analysis.get_hcp_labels", lambda mesh="fsaverage5": labels)


def write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def build_variant(path: Path, variant_id: str, label: str, modality: Modality, domain_pack: DomainPack) -> InputVariant:
    if modality == Modality.text:
        path.write_text(label, encoding="utf-8")
    else:
        path.write_bytes(b"stub")
    return InputVariant(
        id=variant_id,
        label=label,
        modality=modality,
        file_path=path,
        domain_pack=domain_pack,
    )


def test_pipeline_generates_exports_for_multimodal_request(tmp_path: Path):
    variants = [
        build_variant(tmp_path / "a.txt", "text_variant", "Text Variant", Modality.text, DomainPack.teacher),
        build_variant(tmp_path / "b.wav", "audio_variant", "Audio Variant", Modality.audio, DomainPack.marketing),
        build_variant(tmp_path / "c.mp4", "video_variant", "Video Variant", Modality.video, DomainPack.investor),
    ]
    request = CompareRequest(
        variants=variants,
        primary_proxy_family=ProxyFamily.language_processing_proxy,
        export_dir=tmp_path / "exports",
    )
    pipeline = CommunicationLabPipeline(
        service=FakePredictionService(),
        cache_root=tmp_path / "cache",
        enforce_environment_checks=False,
    )
    artifacts = pipeline.compare(request)
    assert artifacts.json_path.exists()
    assert artifacts.report_path.exists()
    assert len(artifacts.result.pairwise_deltas) == 3
    assert set(artifacts.result.segment_times_s) == {variant.id for variant in variants}
    assert "language_processing_proxy" in artifacts.result.proxy_scores[variants[0].id]


@pytest.mark.parametrize("domain_pack", list(DomainPack))
def test_pipeline_supports_each_domain_pack(domain_pack: DomainPack, tmp_path: Path):
    variants = [
        build_variant(tmp_path / "left.txt", "left", f"{domain_pack.value} left", Modality.text, domain_pack),
        build_variant(tmp_path / "right.txt", "right", f"{domain_pack.value} right", Modality.text, domain_pack),
    ]
    request = CompareRequest(
        variants=variants,
        primary_proxy_family=ProxyFamily.cross_region_spread_proxy,
        export_dir=tmp_path / f"exports_{domain_pack.value}",
    )
    pipeline = CommunicationLabPipeline(
        service=FakePredictionService(),
        cache_root=tmp_path / "cache",
        enforce_environment_checks=False,
    )
    artifacts = pipeline.compare(request)
    assert artifacts.result.report_path.endswith("report.html")
    assert artifacts.result.top_rois["left"]
