from __future__ import annotations

import json
from base64 import b64encode
from pathlib import Path

from jinja2 import Template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from communication_lab.config import REPORT_TEMPLATE_VERSION
from communication_lab.domain_packs import get_domain_pack_spec
from communication_lab.environment import EnvironmentReport
from communication_lab.schemas import CompareRequest, CompareResult
from communication_lab.analysis import VariantAnalysis

REPORT_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #152033; background: #f7f6f1; }
      h1, h2, h3 { color: #0f2d3d; }
      .card { background: white; padding: 20px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(15,45,61,0.08); }
      .warning { color: #8b3d1f; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { text-align: left; padding: 8px; border-bottom: 1px solid #d9e1e5; }
      img { max-width: 100%; border-radius: 12px; }
    </style>
  </head>
  <body>
    <h1>{{ title }}</h1>
    <div class="card">
      <p>{{ summary }}</p>
      <p><strong>Template version:</strong> {{ template_version }}</p>
      <p><strong>Scope note:</strong> {{ result.model_scope_note }}</p>
      <p><strong>Hemodynamic lag:</strong> {{ result.hemodynamic_lag_s }} seconds</p>
      <ul>
      {% for warning in result.warnings %}
        <li class="warning">{{ warning }}</li>
      {% endfor %}
      </ul>
    </div>
    {% if result.llm_explanation %}
    <div class="card">
      <h2>Natural-Language Explanation</h2>
      <p><strong>Model:</strong> {{ result.llm_explanation_model }}</p>
      <p>{{ result.llm_explanation }}</p>
    </div>
    {% endif %}
    <div class="grid">
      <div class="card">{{ proxy_bar|safe }}</div>
      <div class="card">{{ time_series|safe }}</div>
    </div>
    <div class="card">{{ roi_heatmap|safe }}</div>
    {% for variant in variants %}
    <div class="card">
      <h2>{{ variant.label }}</h2>
      <p><strong>Domain pack:</strong> {{ variant.domain_pack }}</p>
      <p><strong>Prediction shape:</strong> {{ result.prediction_shapes[variant.id] }}</p>
      <p><strong>Top ROIs:</strong> {{ result.top_rois[variant.id]|join(", ") }}</p>
      <p><strong>Raw predictions:</strong> {{ result.raw_vertex_preds_path[variant.id] }}</p>
      {% if cortical_images[variant.id] %}
      <img alt="Cortical map for {{ variant.label }}" src="data:image/png;base64,{{ cortical_images[variant.id] }}" />
      {% endif %}
      <div>{{ variant_tables[variant.id]|safe }}</div>
    </div>
    {% endfor %}
    <div class="card">
      <h2>Environment</h2>
      <table>
        <tr><th>Python</th><td>{{ environment.python_version }}</td></tr>
        <tr><th>ffmpeg</th><td>{{ environment.ffmpeg.summary }}</td></tr>
        <tr><th>HF token</th><td>{{ environment.huggingface_token.summary }}</td></tr>
        <tr><th>LLaMA access</th><td>{{ environment.llama_gate.summary }}</td></tr>
      </table>
    </div>
  </body>
</html>
"""
)


def export_json_report(result: CompareResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def export_html_report(
    request: CompareRequest,
    result: CompareResult,
    analyses: dict[str, VariantAnalysis],
    environment: EnvironmentReport,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_bar = build_proxy_bar_chart(result).to_html(full_html=False, include_plotlyjs="cdn")
    time_series = build_primary_proxy_chart(result, request, analyses).to_html(
        full_html=False,
        include_plotlyjs=False,
    )
    roi_heatmap = build_roi_heatmap(result, analyses).to_html(full_html=False, include_plotlyjs=False)
    variant_tables = {
        variant.id: pd.DataFrame(
            {
                "proxy": list(result.proxy_scores[variant.id].keys()),
                "score": list(result.proxy_scores[variant.id].values()),
            }
        ).to_html(index=False)
        for variant in request.variants
    }
    cortical_images = {
        variant.id: _base64_path(Path(result.cortical_image_paths[variant.id]))
        if result.cortical_image_paths.get(variant.id)
        else None
        for variant in request.variants
    }
    primary_spec = get_domain_pack_spec(request.variants[0].domain_pack)
    is_single_run = len(request.variants) == 1
    html = REPORT_TEMPLATE.render(
        title="TRIBE v2 Communication Lab Report",
        summary=primary_spec.description,
        template_version=REPORT_TEMPLATE_VERSION,
        is_single_run=is_single_run,
        request=request,
        result=result,
        variants=request.variants,
        proxy_bar=proxy_bar,
        time_series=time_series,
        roi_heatmap=roi_heatmap,
        variant_tables=variant_tables,
        cortical_images=cortical_images,
        environment=environment,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_cortical_image(signal, output_path: Path, title: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from tribev2.plotting.cortical import PlotBrainNilearn
    except ImportError:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    brain = PlotBrainNilearn(mesh="fsaverage5")
    fig, axes = brain.get_fig_axes(views=["left", "right", "posterior"])
    brain.plot_surf(signal, views=["left", "right", "posterior"], axes=axes, colorbar=False)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_proxy_bar_chart(result: CompareResult):
    rows = []
    for variant_id, proxy_scores in result.proxy_scores.items():
        for proxy_name, score in proxy_scores.items():
            rows.append({"variant": variant_id, "proxy": proxy_name, "score": score})
    frame = pd.DataFrame(rows)
    title = "Proxy Scores" if frame.variant.nunique() == 1 else "Proxy Score Comparison"
    return px.bar(frame, x="proxy", y="score", color="variant", barmode="group", title=title)


def build_primary_proxy_chart(
    result: CompareResult,
    request: CompareRequest,
    analyses: dict[str, VariantAnalysis],
):
    figure = go.Figure()
    primary = request.primary_proxy_family.value
    for variant in request.variants:
        series = analyses[variant.id].proxy_timeseries[primary]
        times = result.segment_times_s[variant.id][: len(series)]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=series,
                mode="lines+markers",
                name=variant.label,
            )
        )
    figure.update_layout(
        title=f"{primary} over time",
        xaxis_title="Predicted BOLD time (s)",
        yaxis_title="Proxy score",
    )
    return figure


def build_roi_heatmap(result: CompareResult, analyses: dict[str, VariantAnalysis]):
    roi_union: list[str] = []
    for top_rois in result.top_rois.values():
        for roi in top_rois:
            if roi not in roi_union:
                roi_union.append(roi)
    rows = []
    for variant_id, analysis in analyses.items():
        roi_means = analysis.roi_timeseries.abs().mean(axis=0)
        rows.append([float(roi_means.get(roi, 0.0)) for roi in roi_union])
    frame = pd.DataFrame(rows, index=list(analyses.keys()), columns=roi_union)
    return px.imshow(frame, aspect="auto", title="Mean absolute ROI response")


def _base64_path(path: Path) -> str | None:
    if not path.exists():
        return None
    return b64encode(path.read_bytes()).decode("ascii")
