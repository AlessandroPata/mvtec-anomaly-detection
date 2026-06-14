export interface VariantInfo {
  id: string; label: string; kind: 'production' | 'reconstructed' | 'gan';
  aggregation: string | null; topk: number | null; coreset: number | null;
  description: string; available: boolean; approximate: boolean;
}
export interface DefectTypeInfo { name: string; count: number; is_anomaly: boolean }
export interface CategoryMeta { name: string; test_total: number; defect_types: DefectTypeInfo[]; variants: VariantInfo[] }
export interface Meta { categories: CategoryMeta[]; device: string; dataset_available: boolean }

export interface ArenaImage { idx: number; defect_type: string; filename: string; ground_truth_anomaly: boolean }
export interface ArenaStartResponse { job_id: string; seed: number; n: number; category: string; variant: string; images: ArenaImage[] }
export type Verdict = 'tp' | 'tn' | 'fp' | 'fn' | 'error';
export interface ArenaResult {
  idx: number; defect_type: string; filename: string; ground_truth_anomaly: boolean;
  anomaly_score?: number; anomaly_probability?: number | null; is_anomaly?: boolean;
  threshold?: number; inference_ms?: number; verdict: Verdict; correct?: boolean; error?: string;
}
export interface ArenaSummary {
  n: number; errors: number; accuracy: number | null; precision: number | null;
  recall: number | null; f1: number | null; auroc: number | null;
  confusion: { tp: number; tn: number; fp: number; fn: number };
  by_defect?: Record<string, { n: number; correct: number; is_anomaly: boolean; accuracy: number | null }>;
  mean_ms: number | null; p95_ms: number | null;
}
export interface ArenaFinal { status: 'done' | 'cancelled' | 'error'; summary: ArenaSummary | null; error: string | null }
export interface ArenaPoll { job_id: string; status: string; results: ArenaResult[]; summary: ArenaSummary | null; error: string | null; total: number; done: number }

export interface PredictionResponse {
  anomaly_score: number; anomaly_probability: number | null; is_anomaly: boolean;
  calibrated_probability?: number | null;
  threshold: number; category: string; inference_ms: number;
  heatmap_base64: string; model_variant?: string;
  defect_type?: string; filename?: string; ground_truth_anomaly?: boolean;
}
export interface HealthResponse { status: string; models_loaded: string[]; backend: string }
export interface TestImagesResponse { category: string; defect_types: { defect_type: string; is_anomaly: boolean; count: number; images: string[] }[] }
