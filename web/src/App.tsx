import { useEffect, useMemo, useState } from "react";
import type { DomainPack, EnvironmentReport, InferenceDraft, InferenceResponse, MetaPayload, Modality, ProxyFamily } from "./types";

const API_BASE = "http://127.0.0.1:8000";

const initialDraft: InferenceDraft = {
  label: "Teacher sample",
  domainPack: "teacher",
  notes: "",
  inputMode: "path",
  modality: "text",
  textContent: "",
  filePath: "",
  uploadedFile: null,
  primaryProxy: "language_processing_proxy",
  topK: 8,
  maxDuration: 120,
  exportName: "single_run",
  explainWithLlm: false,
  explanationAudience: "general",
};

export function App() {
  const [meta, setMeta] = useState<MetaPayload | null>(null);
  const [health, setHealth] = useState<EnvironmentReport | null>(null);
  const [draft, setDraft] = useState<InferenceDraft>(initialDraft);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    void Promise.all([
      fetch(`${API_BASE}/api/meta`).then((res) => res.json()),
      fetch(`${API_BASE}/api/health`).then((res) => res.json()),
    ])
      .then(([metaPayload, healthPayload]) => {
        const typedMeta = metaPayload as MetaPayload;
        setMeta(typedMeta);
        setHealth(healthPayload.environment as EnvironmentReport);
        const sample = typedMeta.sample_files.teacher?.[0] ?? "";
        setDraft((current) => ({ ...current, filePath: sample }));
      })
      .catch((fetchError: Error) => {
        setError(fetchError.message);
      });
  }, []);

  const activePack = draft.domainPack;
  const packSpec = meta?.domain_packs[activePack];
  const firstVariantId = useMemo(() => Object.keys(result?.result.proxy_scores ?? {})[0] ?? "uploaded_input", [result]);
  const proxyScoreEntries = useMemo(
    () => Object.entries(result?.result.proxy_scores?.[firstVariantId] ?? {}).sort((a, b) => b[1] - a[1]),
    [result, firstVariantId],
  );
  const roiEntries = useMemo(
    () => result?.result.top_rois?.[firstVariantId] ?? [],
    [result, firstVariantId],
  );

  const canRun =
    draft.label.trim().length > 0 &&
    (draft.inputMode === "text"
      ? draft.textContent.trim().length > 0
      : draft.inputMode === "path"
        ? draft.filePath.trim().length > 0
        : draft.uploadedFile !== null);

  const applySample = (pack: DomainPack, index = 0) => {
    const sample = meta?.sample_files[pack]?.[index] ?? "";
    setDraft((current) => ({
      ...current,
      domainPack: pack,
      inputMode: "path",
      modality: inferModality(sample),
      label: `${capitalize(pack)} sample ${index + 1}`,
      filePath: sample,
      textContent: "",
      uploadedFile: null,
      primaryProxy: defaultProxyForPack(pack),
    }));
  };

  const runInference = async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.set("label", draft.label);
      formData.set("domain_pack", draft.domainPack);
      formData.set("notes", draft.notes);
      formData.set("source_type", draft.inputMode);
      formData.set("modality", draft.modality);
      formData.set("primary_proxy_family", draft.primaryProxy);
      formData.set("top_k_rois", String(draft.topK));
      formData.set("max_duration_s", String(draft.maxDuration));
      formData.set("export_name", draft.exportName);
      formData.set("explain_with_llm", String(draft.explainWithLlm));
      formData.set("explanation_audience", draft.explanationAudience);
      if (draft.inputMode === "text") {
        formData.set("text_content", draft.textContent);
      } else if (draft.inputMode === "path") {
        formData.set("file_path", draft.filePath);
      } else if (draft.uploadedFile) {
        formData.set("uploaded_file", draft.uploadedFile);
      }
      const response = await fetch(`${API_BASE}/api/infer`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Inference failed");
      }
      setResult(payload as InferenceResponse);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Inference failed");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="left-rail">
        <div className="brand-card">
          <p className="eyebrow">TRIBE v2 Upload Runner</p>
          <h1>Upload one item, run the model, get the output.</h1>
          <p className="lede">
            This app runs TRIBE v2 on a single text, audio, or video input and returns neural-proxy summaries, raw output paths, and an optional natural-language explanation.
          </p>
        </div>
        <div className="status-card">
          <h2>Runtime</h2>
          {health ? (
            <>
              <StatusLine ok={health.ffmpeg.ok} label={health.ffmpeg.summary} />
              <StatusLine ok={health.huggingface_token.ok} label={health.huggingface_token.summary} />
              <StatusLine ok={health.llama_gate.ok} label={health.llama_gate.summary} />
              <StatusLine ok={health.openai_api_key.ok} label={health.openai_api_key.summary} />
            </>
          ) : (
            <p>Loading runtime checks…</p>
          )}
          <p className="helper">Hugging Face access is required for TRIBE inference. OpenAI access is only required for optional explanations.</p>
        </div>
        <div className="status-card">
          <h2>Quick Samples</h2>
          <div className="chip-grid">
            {(["teacher", "doctor", "marketing", "investor"] as DomainPack[]).map((pack) => (
              <button key={pack} className="chip" onClick={() => applySample(pack)}>
                {pack}
              </button>
            ))}
          </div>
          <p className="helper">Each chip loads one repo sample as a single-run input.</p>
        </div>
      </aside>

      <main className="main-panel">
        <section className="hero">
          <div className="panel">
            <p className="eyebrow">What this project does</p>
            <p className="hero-copy">
              It takes one uploaded or pasted communication asset, runs it through TRIBE v2, and returns predicted average-subject cortical responses plus ROI and proxy summaries. If requested, it also uses an LLM to explain the output in natural language without overclaiming what the model means.
            </p>
          </div>
          {packSpec ? (
            <div className="pack-card">
              <h2>{packSpec.title}</h2>
              <p>{packSpec.description}</p>
              <ul className="flat-list">
                {packSpec.disclaimers.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>

        <section className="panel">
          <div className="section-header">
            <h2>Input</h2>
          </div>
          <div className="settings-grid">
            <label>
              Label
              <input value={draft.label} onChange={(event) => setDraft((current) => ({ ...current, label: event.target.value }))} />
            </label>
            <label>
              Domain pack
              <select
                value={draft.domainPack}
                onChange={(event) => {
                  const domainPack = event.target.value as DomainPack;
                  setDraft((current) => ({
                    ...current,
                    domainPack,
                    primaryProxy: defaultProxyForPack(domainPack),
                  }));
                }}
              >
                <option value="teacher">teacher</option>
                <option value="doctor">doctor</option>
                <option value="marketing">marketing</option>
                <option value="investor">investor</option>
              </select>
            </label>
            <label>
              Primary proxy
              <select
                value={draft.primaryProxy}
                onChange={(event) => setDraft((current) => ({ ...current, primaryProxy: event.target.value as ProxyFamily }))}
              >
                {meta?.proxy_families.map((proxy) => (
                  <option key={proxy} value={proxy}>
                    {proxy}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Modality
              <select value={draft.modality} onChange={(event) => setDraft((current) => ({ ...current, modality: event.target.value as Modality }))}>
                <option value="text">text</option>
                <option value="audio">audio</option>
                <option value="video">video</option>
              </select>
            </label>
          </div>

          <div className="segmented">
            <button className={draft.inputMode === "path" ? "active" : ""} onClick={() => setDraft((current) => ({ ...current, inputMode: "path" }))}>
              Path
            </button>
            <button className={draft.inputMode === "upload" ? "active" : ""} onClick={() => setDraft((current) => ({ ...current, inputMode: "upload" }))}>
              Upload
            </button>
            <button className={draft.inputMode === "text" ? "active" : ""} onClick={() => setDraft((current) => ({ ...current, inputMode: "text", modality: "text" }))}>
              Paste Text
            </button>
          </div>

          {draft.inputMode === "path" ? (
            <label>
              Local file path
              <input
                value={draft.filePath}
                placeholder="/absolute/path/to/file"
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    filePath: event.target.value,
                    modality: inferModality(event.target.value),
                  }))
                }
              />
            </label>
          ) : null}

          {draft.inputMode === "upload" ? (
            <label>
              Upload file
              <input
                type="file"
                accept=".txt,.wav,.mp3,.flac,.ogg,.mp4,.avi,.mkv,.mov,.webm"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setDraft((current) => ({
                    ...current,
                    uploadedFile: file,
                    modality: inferModality(file?.name ?? current.filePath),
                  }));
                }}
              />
            </label>
          ) : null}

          {draft.inputMode === "text" ? (
            <label>
              Text content
              <textarea rows={10} value={draft.textContent} onChange={(event) => setDraft((current) => ({ ...current, textContent: event.target.value }))} />
            </label>
          ) : null}

          <div className="settings-grid">
            <label>
              Top ROIs
              <input type="number" min={1} max={25} value={draft.topK} onChange={(event) => setDraft((current) => ({ ...current, topK: Number(event.target.value) }))} />
            </label>
            <label>
              Max duration (s)
              <input
                type="number"
                min={15}
                max={600}
                value={draft.maxDuration}
                onChange={(event) => setDraft((current) => ({ ...current, maxDuration: Number(event.target.value) }))}
              />
            </label>
            <label>
              Export folder
              <input value={draft.exportName} onChange={(event) => setDraft((current) => ({ ...current, exportName: event.target.value }))} />
            </label>
            <label>
              Explanation audience
              <input
                value={draft.explanationAudience}
                onChange={(event) => setDraft((current) => ({ ...current, explanationAudience: event.target.value }))}
                placeholder="general"
              />
            </label>
          </div>

          <label>
            Notes
            <textarea rows={3} value={draft.notes} onChange={(event) => setDraft((current) => ({ ...current, notes: event.target.value }))} />
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={draft.explainWithLlm}
              onChange={(event) => setDraft((current) => ({ ...current, explainWithLlm: event.target.checked }))}
            />
            <span>Generate natural-language explanation with an LLM</span>
          </label>

          <div className="action-row">
            <button className="primary" disabled={!canRun || running} onClick={runInference}>
              {running ? "Running…" : "Run TRIBE v2"}
            </button>
            <p className="helper">This is a single-item inference flow, not a multi-draft comparison flow.</p>
          </div>
          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <section className="panel">
          <div className="section-header">
            <h2>Output</h2>
            {result ? (
              <a className="secondary-link" href={`${API_BASE}/api/report?path=${encodeURIComponent(result.report_path)}`} target="_blank" rel="noreferrer">
                Open HTML report
              </a>
            ) : null}
          </div>
          {!result ? (
            <p className="empty-state">Run one item through the model to see proxy scores, top ROIs, raw output paths, and an optional explanation.</p>
          ) : (
            <>
              <div className="warning-box">
                <p>{result.result.model_scope_note}</p>
                <ul className="flat-list">
                  {result.result.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
              <div className="summary-grid">
                <div className="metric-card">
                  <p className="eyebrow">Prediction shape</p>
                  <strong>{(result.result.prediction_shapes[firstVariantId] ?? []).join(" x ") || "n/a"}</strong>
                </div>
                <div className="metric-card">
                  <p className="eyebrow">Hemodynamic lag</p>
                  <strong>{result.result.hemodynamic_lag_s}s</strong>
                </div>
                <div className="metric-card">
                  <p className="eyebrow">Raw output</p>
                  <strong className="path-pill">{result.result.raw_vertex_preds_path[firstVariantId]}</strong>
                </div>
              </div>

              <div className="result-grid">
                <div className="result-card">
                  <h3>Proxy scores</h3>
                  <ul className="flat-list compact">
                    {proxyScoreEntries.map(([name, value]) => (
                      <li key={name}>
                        <span>{name}</span>
                        <strong>{value.toFixed(4)}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="result-card">
                  <h3>Top ROIs</h3>
                  <ul className="flat-list compact">
                    {roiEntries.map((roi) => (
                      <li key={roi}>
                        <span>{roi}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {result.result.llm_explanation ? (
                <div className="result-card explanation-card">
                  <h3>Natural-language explanation</h3>
                  <p className="helper">Model: {result.result.llm_explanation_model}</p>
                  <p>{result.result.llm_explanation}</p>
                </div>
              ) : null}

              {result.result.cortical_image_paths[firstVariantId] ? (
                <div className="result-card">
                  <h3>Cortical image</h3>
                  <img
                    className="brain-image"
                    src={`${API_BASE}/api/file?path=${encodeURIComponent(result.result.cortical_image_paths[firstVariantId] ?? "")}`}
                    alt="Mean cortical response"
                  />
                </div>
              ) : null}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function StatusLine({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="status-line">
      <span className={ok ? "status-dot ok" : "status-dot bad"} />
      <span>{label}</span>
    </div>
  );
}

function inferModality(fileName: string): Modality {
  const suffix = fileName.split(".").pop()?.toLowerCase() ?? "";
  if (suffix === "txt") return "text";
  if (["wav", "mp3", "flac", "ogg"].includes(suffix)) return "audio";
  return "video";
}

function defaultProxyForPack(pack: DomainPack): ProxyFamily {
  if (pack === "doctor") return "emotional_social_proxy";
  if (pack === "marketing") return "visual_salience_proxy";
  return "language_processing_proxy";
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
