// Domain types: architectures, components, benchmarks

export type ArchStatus = 'baseline' | 'experimental' | 'deprecated' | 'production';

export type ComponentTag =
  | 'memory_bank'
  | 'teacher_student'
  | 'reconstruction'
  | 'latent_compactness'
  | 'score_fusion_weighted'
  | 'score_fusion_learned'
  | 'synthetic_anomalies'
  | 'multi_scale'
  | 'topk_aggregation'
  | 'threshold_calibration'
  | 'frozen_backbone'
  | 'kcenter_coreset';

export interface PipelineStep {
  label: string;
  detail: string;
  type?: 'input' | 'process' | 'storage' | 'output';
}

export interface Architecture {
  id: string;
  name: string;
  shortName: string;
  date: string;
  family: 'OCGAN' | 'PatchCore' | 'Hybrid';
  status: ArchStatus;
  macroAUROC: number;
  inferenceMs: number;
  trainTimeSec: number;
  bankSizeMB: number;
  paramsMillions: number;
  architecture_type: string;
  core_idea: string;
  strengths: string[];
  weaknesses: string[];
  improvements: string[];
  components: ComponentTag[];
  hyperparameters: Record<string, string | number>;
  pipeline: PipelineStep[];
  files: string[];
  available_for_inference: boolean;
  notes?: string;
  delta_vs_previous?: number;
}

export interface ArchComponent {
  id: ComponentTag;
  name: string;
  short: string;
  description: string;
  formula?: string;
  intuition: string;
  tradeoffs: string[];
  used_in: string[];
  paper_ref?: string;
  code_snippet?: string;
}

export interface PerCategoryRow {
  category: string;
  auroc: number;
  auprc: number;
  best_f1: number;
  fpr95: number;
  elapsed_s: number;
  n_seeds: number;
  feature_level: string;
  aggregation: string;
  topk: number;
  coreset: number;
  approximated?: boolean;
}

export interface Benchmarks {
  per_category: Record<string, PerCategoryRow[]>;
  macros: Record<string, number>;
}
