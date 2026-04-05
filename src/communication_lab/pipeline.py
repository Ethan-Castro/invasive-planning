from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from communication_lab.analysis import VariantAnalysis, build_pairwise_deltas, summarize_predictions
from communication_lab.config import DEFAULT_LAG_S, get_app_paths
from communication_lab.environment import EnvironmentReport, build_environment_report
from communication_lab.reporting import export_html_report, export_json_report, render_cortical_image
from communication_lab.schemas import CompareRequest, CompareResult
from communication_lab.tribe_service import TribeService, VariantPrediction


@dataclass
class ComparisonArtifacts:
    result: CompareResult
    analyses: dict[str, VariantAnalysis]
    environment_report: EnvironmentReport
    json_path: Path
    report_path: Path


class CommunicationLabPipeline:
    def __init__(
        self,
        service: TribeService | None = None,
        cache_root: Path | None = None,
        enforce_environment_checks: bool = True,
    ) -> None:
        self.paths = get_app_paths(cache_root)
        self.service = service or TribeService(cache_root=self.paths.root)
        self.enforce_environment_checks = enforce_environment_checks

    def compare(self, request: CompareRequest) -> ComparisonArtifacts:
        environment_report = build_environment_report(
            skip_remote_checks=not self.enforce_environment_checks
        )
        if self.enforce_environment_checks and not environment_report.ready_for_inference:
            raise RuntimeError(
                "Runtime checks failed. Run `communication-lab-check` and fix the missing requirements before inference."
            )
        request.export_dir.mkdir(parents=True, exist_ok=True)
        variant_predictions: dict[str, VariantPrediction] = {}
        analyses: dict[str, VariantAnalysis] = {}
        warnings: list[str] = [
            "TRIBE v2 predictions represent average-subject fMRI-like responses, not direct human reaction predictions.",
            "TRIBE v2 uses a built-in 5-second hemodynamic lag and 1 Hz output resolution.",
            "The model treats the brain as a passive observer and does not model behavior directly.",
        ]
        raw_paths: dict[str, str] = {}
        prediction_shapes: dict[str, list[int]] = {}
        cortical_image_paths: dict[str, str | None] = {}
        segment_times: dict[str, list[float]] = {}
        roi_timeseries: dict[str, dict[str, list[float]]] = {}
        top_rois: dict[str, list[str]] = {}
        proxy_scores: dict[str, dict[str, float]] = {}

        for variant in request.variants:
            prediction = self.service.predict_variant(variant, request)
            variant_predictions[variant.id] = prediction
            warnings.extend(prediction.warnings)
            raw_path = request.export_dir / "raw_predictions" / f"{variant.id}.npy"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, prediction.preds)
            raw_paths[variant.id] = str(raw_path)
            prediction_shapes[variant.id] = [int(dimension) for dimension in prediction.preds.shape]
            segment_times[variant.id] = prediction.segment_times_s
            analysis = summarize_predictions(prediction.preds, request)
            analyses[variant.id] = analysis
            roi_timeseries[variant.id] = {
                roi: [float(value) for value in series]
                for roi, series in analysis.roi_timeseries.to_dict(orient="list").items()
            }
            top_rois[variant.id] = analysis.top_rois
            proxy_scores[variant.id] = {
                proxy_name: float(score) for proxy_name, score in analysis.proxy_scores.items()
            }
            image_path = render_cortical_image(
                analysis.mean_surface,
                request.export_dir / "figures" / f"{variant.id}_cortical.png",
                title=f"{variant.label} mean absolute cortical response",
            )
            cortical_image_paths[variant.id] = str(image_path) if image_path else None

        pairwise_deltas = build_pairwise_deltas(analyses, request)
        json_path = request.export_dir / "compare_result.json"
        provisional_result = CompareResult(
            segment_times_s=segment_times,
            prediction_shapes=prediction_shapes,
            raw_vertex_preds_path=raw_paths,
            roi_timeseries=roi_timeseries,
            top_rois=top_rois,
            proxy_scores=proxy_scores,
            pairwise_deltas=pairwise_deltas,
            warnings=sorted(set(warnings)),
            report_path=str(request.export_dir / "report.html"),
            cortical_image_paths=cortical_image_paths,
            hemodynamic_lag_s=DEFAULT_LAG_S,
        )
        export_json_report(provisional_result, json_path)
        report_path = export_html_report(
            request=request,
            result=provisional_result,
            analyses=analyses,
            environment=environment_report,
            output_path=Path(provisional_result.report_path),
        )
        final_result = provisional_result.model_copy(update={"report_path": str(report_path)})
        export_json_report(final_result, json_path)
        return ComparisonArtifacts(
            result=final_result,
            analyses=analyses,
            environment_report=environment_report,
            json_path=json_path,
            report_path=report_path,
        )
