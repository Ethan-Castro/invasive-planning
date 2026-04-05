from __future__ import annotations

import os

from communication_lab.domain_packs import get_domain_pack_spec
from communication_lab.schemas import CompareResult, InputVariant, ProxyFamily


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_natural_language_explanation(
    result: CompareResult,
    variant: InputVariant,
    primary_proxy_family: ProxyFamily,
    audience: str = "general",
) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set, so LLM explanations are unavailable.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Re-run `uv sync --extra runtime --extra test`."
        ) from exc

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    client = OpenAI(api_key=api_key)
    proxy_scores = result.proxy_scores[variant.id]
    top_rois = result.top_rois[variant.id]
    segment_times = result.segment_times_s[variant.id]
    spec = get_domain_pack_spec(variant.domain_pack)
    context = {
        "label": variant.label,
        "domain_pack": variant.domain_pack.value,
        "notes": variant.notes,
        "primary_proxy_family": primary_proxy_family.value,
        "proxy_scores": proxy_scores,
        "top_rois": top_rois,
        "segment_times_s": segment_times,
        "prediction_shape": result.prediction_shapes.get(variant.id),
        "warnings": result.warnings,
        "model_scope_note": result.model_scope_note,
        "domain_disclaimers": list(spec.disclaimers),
        "domain_interpretation_notes": list(spec.interpretation_notes),
        "hemodynamic_lag_s": result.hemodynamic_lag_s,
    }
    instructions = (
        "You explain TRIBE v2 inference outputs in careful plain language. "
        "Do not claim to predict behavior, outcomes, persuasion, diagnosis, or intent. "
        "Explain what the strongest proxy signals and ROIs suggest, what timing means, "
        "and what cannot be concluded. Keep the explanation natural and useful to a "
        f"{audience} audience."
    )
    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        instructions=instructions,
        input=str(context),
    )
    return response.output_text.strip(), model
