from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
import uvicorn

from communication_lab.config import APP_NAME, DEFAULT_MAX_DURATION_S, DEFAULT_TOP_K_ROIS, get_app_paths
from communication_lab.domain_packs import DOMAIN_PACK_SPECS
from communication_lab.environment import build_environment_report
from communication_lab.pipeline import CommunicationLabPipeline
from communication_lab.explanations import generate_natural_language_explanation
from communication_lab.reporting import export_html_report, export_json_report
from communication_lab.schemas import CompareRequest, DomainPack, InputVariant, Modality, ProxyFamily


class VariantInput(BaseModel):
    id: str
    label: str
    domain_pack: DomainPack
    notes: str = ""
    source_type: Literal["text", "path"]
    modality: Modality | None = None
    text_content: str | None = None
    file_path: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "VariantInput":
        if self.source_type == "text":
            if not (self.text_content or "").strip():
                raise ValueError("text_content is required for text variants")
            self.modality = Modality.text
        else:
            if not (self.file_path or "").strip():
                raise ValueError("file_path is required for path variants")
            if self.modality is None:
                suffix = Path(self.file_path).suffix.lower()
                if suffix == ".txt":
                    self.modality = Modality.text
                elif suffix in {".wav", ".mp3", ".flac", ".ogg"}:
                    self.modality = Modality.audio
                elif suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
                    self.modality = Modality.video
                else:
                    raise ValueError(f"Unsupported file type: {suffix}")
        return self


class CompareRequestPayload(BaseModel):
    variants: list[VariantInput]
    primary_proxy_family: ProxyFamily = ProxyFamily.language_processing_proxy
    top_k_rois: int = Field(default=DEFAULT_TOP_K_ROIS, ge=1, le=25)
    max_duration_s: int = Field(default=DEFAULT_MAX_DURATION_S, ge=15, le=600)
    export_name: str = "latest_run"


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    paths = get_app_paths()

    @app.get("/api/health")
    def health():
        report = build_environment_report()
        return {
            "app_name": APP_NAME,
            "environment": report.model_dump(mode="json"),
        }

    @app.get("/api/meta")
    def meta():
        return {
            "domain_packs": {
                pack.value: {
                    "title": spec.title,
                    "description": spec.description,
                    "focus_proxies": [proxy.value for proxy in spec.focus_proxies],
                    "disclaimers": list(spec.disclaimers),
                    "interpretation_notes": list(spec.interpretation_notes),
                }
                for pack, spec in DOMAIN_PACK_SPECS.items()
            },
            "input_modes": ["text", "path", "upload"],
            "proxy_families": [proxy.value for proxy in ProxyFamily],
            "sample_files": {
                "teacher": [
                    str(Path("samples/teacher_variant_a.txt").resolve()),
                    str(Path("samples/teacher_variant_b.txt").resolve()),
                ],
                "doctor": [
                    str(Path("samples/doctor_variant_a.txt").resolve()),
                    str(Path("samples/doctor_variant_b.txt").resolve()),
                ],
                "marketing": [
                    str(Path("samples/marketing_variant_a.txt").resolve()),
                    str(Path("samples/marketing_variant_b.txt").resolve()),
                ],
                "investor": [
                    str(Path("samples/investor_variant_a.txt").resolve()),
                    str(Path("samples/investor_variant_b.txt").resolve()),
                ],
            },
        }

    @app.post("/api/infer")
    async def infer(
        label: str = Form(...),
        domain_pack: DomainPack = Form(...),
        notes: str = Form(""),
        source_type: Literal["text", "path", "upload"] = Form(...),
        primary_proxy_family: ProxyFamily = Form(ProxyFamily.language_processing_proxy),
        top_k_rois: int = Form(DEFAULT_TOP_K_ROIS),
        max_duration_s: int = Form(DEFAULT_MAX_DURATION_S),
        export_name: str = Form("single_run"),
        modality: Modality | None = Form(None),
        text_content: str | None = Form(None),
        file_path: str | None = Form(None),
        explain_with_llm: bool = Form(False),
        explanation_audience: str = Form("general"),
        uploaded_file: UploadFile | None = File(None),
    ):
        try:
            variant = await _materialize_single_variant(
                cache_root=paths.root,
                label=label,
                domain_pack=domain_pack,
                notes=notes,
                source_type=source_type,
                modality=modality,
                text_content=text_content,
                file_path=file_path,
                uploaded_file=uploaded_file,
            )
            request = CompareRequest(
                variants=[variant],
                primary_proxy_family=primary_proxy_family,
                top_k_rois=top_k_rois,
                max_duration_s=max_duration_s,
                export_dir=paths.exports / export_name,
            )
            pipeline = CommunicationLabPipeline(cache_root=paths.root)
            artifacts = pipeline.compare(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = artifacts.result
        if explain_with_llm:
            try:
                explanation, model_name = generate_natural_language_explanation(
                    result=result,
                    variant=variant,
                    primary_proxy_family=primary_proxy_family,
                    audience=explanation_audience,
                )
                result = result.model_copy(
                    update={
                        "llm_explanation": explanation,
                        "llm_explanation_model": model_name,
                    }
                )
            except Exception as exc:
                result = result.model_copy(
                    update={
                        "warnings": sorted(
                            set([*result.warnings, f"LLM explanation unavailable: {exc}"])
                        )
                    }
                )
            export_json_report(result, artifacts.json_path)
            export_html_report(
                request=request,
                result=result,
                analyses=artifacts.analyses,
                environment=artifacts.environment_report,
                output_path=artifacts.report_path,
            )

        return {
            "result": result.model_dump(mode="json"),
            "environment": artifacts.environment_report.model_dump(mode="json"),
            "json_path": str(artifacts.json_path),
            "report_path": str(artifacts.report_path),
        }

    @app.post("/api/compare")
    def compare(payload: CompareRequestPayload):
        try:
            request = _materialize_request(payload, paths.root)
            pipeline = CommunicationLabPipeline(cache_root=paths.root)
            artifacts = pipeline.compare(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "result": artifacts.result.model_dump(mode="json"),
            "environment": artifacts.environment_report.model_dump(mode="json"),
            "json_path": str(artifacts.json_path),
            "report_path": str(artifacts.report_path),
        }

    @app.get("/api/report")
    def report(path: str):
        report_path = Path(path).expanduser().resolve()
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        return FileResponse(report_path)

    @app.get("/api/result")
    def result(path: str):
        result_path = Path(path).expanduser().resolve()
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result not found")
        return json.loads(result_path.read_text(encoding="utf-8"))

    @app.get("/api/file")
    def file(path: str):
        target_path = Path(path).expanduser().resolve()
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target_path)

    return app


def _materialize_request(payload: CompareRequestPayload, cache_root: Path) -> CompareRequest:
    paths = get_app_paths(cache_root)
    variants: list[InputVariant] = []
    for raw_variant in payload.variants:
        if raw_variant.source_type == "text":
            temp_path = paths.uploads / f"{raw_variant.id}_{uuid.uuid4().hex}.txt"
            temp_path.write_text((raw_variant.text_content or "").strip(), encoding="utf-8")
            file_path = temp_path
        else:
            file_path = Path(raw_variant.file_path or "").expanduser().resolve()
        variants.append(
            InputVariant(
                id=raw_variant.id,
                label=raw_variant.label,
                modality=raw_variant.modality or Modality.text,
                file_path=file_path,
                domain_pack=raw_variant.domain_pack,
                notes=raw_variant.notes,
            )
        )
    export_dir = get_app_paths(cache_root).exports / payload.export_name
    return CompareRequest(
        variants=variants,
        primary_proxy_family=payload.primary_proxy_family,
        top_k_rois=payload.top_k_rois,
        max_duration_s=payload.max_duration_s,
        export_dir=export_dir,
    )


async def _materialize_single_variant(
    cache_root: Path,
    label: str,
    domain_pack: DomainPack,
    notes: str,
    source_type: Literal["text", "path", "upload"],
    modality: Modality | None,
    text_content: str | None,
    file_path: str | None,
    uploaded_file: UploadFile | None,
) -> InputVariant:
    paths = get_app_paths(cache_root)
    variant_id = "uploaded_input"
    resolved_modality = modality
    if source_type == "text":
        content = (text_content or "").strip()
        if not content:
            raise ValueError("text_content is required for text input")
        materialized_path = paths.uploads / f"{variant_id}_{uuid.uuid4().hex}.txt"
        materialized_path.write_text(content, encoding="utf-8")
        resolved_modality = Modality.text
    elif source_type == "path":
        raw_path = (file_path or "").strip()
        if not raw_path:
            raise ValueError("file_path is required for path input")
        materialized_path = Path(raw_path).expanduser().resolve()
        resolved_modality = resolved_modality or _infer_modality_from_path(materialized_path)
    else:
        if uploaded_file is None or not uploaded_file.filename:
            raise ValueError("uploaded_file is required for upload input")
        suffix = Path(uploaded_file.filename).suffix.lower()
        if not suffix:
            raise ValueError("Uploaded files must include an extension.")
        materialized_path = paths.uploads / f"{variant_id}_{uuid.uuid4().hex}{suffix}"
        materialized_path.write_bytes(await uploaded_file.read())
        resolved_modality = resolved_modality or _infer_modality_from_path(materialized_path)
    return InputVariant(
        id=variant_id,
        label=label,
        modality=resolved_modality or Modality.text,
        file_path=materialized_path,
        domain_pack=domain_pack,
        notes=notes,
    )


def _infer_modality_from_path(path: Path) -> Modality:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return Modality.text
    if suffix in {".wav", ".mp3", ".flac", ".ogg"}:
        return Modality.audio
    if suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        return Modality.video
    raise ValueError(f"Unsupported file type: {suffix}")


def run() -> None:
    uvicorn.run("communication_lab.api:create_app", host="127.0.0.1", port=8000, factory=True)
