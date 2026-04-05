from pathlib import Path

from fastapi.testclient import TestClient

from communication_lab.api import create_app
from communication_lab.schemas import CompareRequest, DomainPack, InputVariant, Modality
from communication_lab.tribe_service import VariantPrediction


class FakePredictionService:
    def predict_variant(self, variant: InputVariant, request: CompareRequest) -> VariantPrediction:
        import numpy as np

        preds = np.ones((4, 12), dtype=float)
        return VariantPrediction(
            variant_id=variant.id,
            prepared_path=variant.file_path,
            segment_times_s=[5.0, 6.0, 7.0, 8.0],
            stimulus_times_s=[0.0, 1.0, 2.0, 3.0],
            preds=preds,
            warnings=[],
        )


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        "communication_lab.api.build_environment_report",
        lambda: type(
            "FakeReport",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "warnings": [],
                    "ffmpeg": {"ok": True},
                    "openai_api_key": {"ok": False},
                }
            },
        )(),
    )
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "environment" in response.json()


def test_compare_endpoint(tmp_path: Path, monkeypatch):
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("left", encoding="utf-8")
    right.write_text("right", encoding="utf-8")

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def compare(self, request: CompareRequest):
            class FakeArtifacts:
                environment_report = type(
                    "Env",
                    (),
                    {"model_dump": lambda self, mode="json": {"warnings": []}},
                )()
                result = type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda self, mode="json": {
                            "warnings": [],
                            "report_path": str(request.export_dir / "report.html"),
                            "proxy_scores": {},
                        }
                    },
                )()
                json_path = request.export_dir / "compare_result.json"
                report_path = request.export_dir / "report.html"

            request.export_dir.mkdir(parents=True, exist_ok=True)
            FakeArtifacts.json_path.write_text("{}", encoding="utf-8")
            FakeArtifacts.report_path.write_text("<html></html>", encoding="utf-8")
            return FakeArtifacts()

    monkeypatch.setattr("communication_lab.api.CommunicationLabPipeline", FakePipeline)
    client = TestClient(create_app())
    response = client.post(
        "/api/compare",
        json={
            "variants": [
                {
                    "id": "left",
                    "label": "Left",
                    "domain_pack": "teacher",
                    "source_type": "path",
                    "modality": "text",
                    "file_path": str(left),
                },
                {
                    "id": "right",
                    "label": "Right",
                    "domain_pack": "teacher",
                    "source_type": "path",
                    "modality": "text",
                    "file_path": str(right),
                },
            ],
            "export_name": "api_test",
        },
    )
    assert response.status_code == 200
    assert "result" in response.json()


def test_infer_endpoint_accepts_text_input(monkeypatch):
    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def compare(self, request: CompareRequest):
            class FakeArtifacts:
                environment_report = type(
                    "Env",
                    (),
                    {"model_dump": lambda self, mode="json": {"warnings": []}},
                )()
                analyses = {}
                result = type(
                    "Result",
                    (),
                    {
                        "warnings": [],
                        "model_dump": lambda self, mode="json": {
                            "warnings": [],
                            "segment_times_s": {"uploaded_input": [5.0]},
                            "prediction_shapes": {"uploaded_input": [1, 12]},
                            "raw_vertex_preds_path": {"uploaded_input": "/tmp/raw.npy"},
                            "roi_timeseries": {"uploaded_input": {}},
                            "top_rois": {"uploaded_input": ["A5"]},
                            "proxy_scores": {"uploaded_input": {"language_processing_proxy": 0.5}},
                            "pairwise_deltas": [],
                            "report_path": str(request.export_dir / "report.html"),
                            "cortical_image_paths": {"uploaded_input": None},
                            "hemodynamic_lag_s": 5.0,
                            "llm_explanation": None,
                            "llm_explanation_model": None,
                            "model_scope_note": "scope note",
                        },
                    },
                )()
                json_path = request.export_dir / "compare_result.json"
                report_path = request.export_dir / "report.html"

            request.export_dir.mkdir(parents=True, exist_ok=True)
            FakeArtifacts.json_path.write_text("{}", encoding="utf-8")
            FakeArtifacts.report_path.write_text("<html></html>", encoding="utf-8")
            return FakeArtifacts()

    monkeypatch.setattr("communication_lab.api.CommunicationLabPipeline", FakePipeline)
    client = TestClient(create_app())
    response = client.post(
        "/api/infer",
        data={
            "label": "Uploaded text",
            "domain_pack": "teacher",
            "source_type": "text",
            "text_content": "hello world",
            "modality": "text",
            "primary_proxy_family": "language_processing_proxy",
            "top_k_rois": "8",
            "max_duration_s": "120",
            "export_name": "single_api_test",
            "explain_with_llm": "false",
            "explanation_audience": "general",
        },
    )
    assert response.status_code == 200
    assert response.json()["result"]["prediction_shapes"]["uploaded_input"] == [1, 12]
