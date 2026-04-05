export type DomainPack = "teacher" | "doctor" | "marketing" | "investor";
export type ProxyFamily =
  | "language_processing_proxy"
  | "emotional_social_proxy"
  | "auditory_salience_proxy"
  | "visual_salience_proxy"
  | "cross_region_spread_proxy";
export type Modality = "text" | "audio" | "video";
export type InputMode = "text" | "path" | "upload";

export type EnvironmentReport = {
  python_version: string;
  ffmpeg: { ok: boolean; summary: string };
  huggingface_token: { ok: boolean; summary: string };
  llama_gate: { ok: boolean; summary: string };
  openai_api_key: { ok: boolean; summary: string };
  warnings: string[];
};

export type MetaPayload = {
  domain_packs: Record<
    DomainPack,
    {
      title: string;
      description: string;
      focus_proxies: ProxyFamily[];
      disclaimers: string[];
      interpretation_notes: string[];
    }
  >;
  proxy_families: ProxyFamily[];
  input_modes: InputMode[];
  sample_files: Record<DomainPack, string[]>;
};

export type InferenceResponse = {
  result: {
    segment_times_s: Record<string, number[]>;
    prediction_shapes: Record<string, number[]>;
    raw_vertex_preds_path: Record<string, string>;
    roi_timeseries: Record<string, Record<string, number[]>>;
    top_rois: Record<string, string[]>;
    proxy_scores: Record<string, Record<string, number>>;
    pairwise_deltas: Array<{
      variant_a_id: string;
      variant_b_id: string;
      proxy_deltas: Record<string, number>;
      top_roi_deltas: Record<string, number>;
      divergence_windows: Array<{ start_time_s: number; end_time_s: number; peak_delta: number }>;
    }>;
    warnings: string[];
    report_path: string;
    cortical_image_paths: Record<string, string | null>;
    hemodynamic_lag_s: number;
    llm_explanation: string | null;
    llm_explanation_model: string | null;
    model_scope_note: string;
  };
  environment: EnvironmentReport;
  json_path: string;
  report_path: string;
};

export type InferenceDraft = {
  label: string;
  domainPack: DomainPack;
  notes: string;
  inputMode: InputMode;
  modality: Modality;
  textContent: string;
  filePath: string;
  uploadedFile: File | null;
  primaryProxy: ProxyFamily;
  topK: number;
  maxDuration: number;
  exportName: string;
  explainWithLlm: boolean;
  explanationAudience: string;
};
