import rawBenchmarks from './benchmarks.json';
import { ARCHITECTURES } from './architectures';
import type { Architecture } from '../types/domain';

export interface BenchmarkRow {
  category: string; auroc: number; auroc_std?: number; auprc: number; best_f1: number;
  fpr95: number; elapsed_s: number | null; n_seeds: number;
  feature_level: string | null; aggregation: string | null; topk: number | null; coreset: number | null;
}
export interface Benchmarks { per_category: Record<string, BenchmarkRow[]>; macros: Record<string, number> }

export const BENCHMARKS = rawBenchmarks as unknown as Benchmarks;

export const CATEGORIES = [
  'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
] as const;

/** Chronological (curated date field). The narrative journey: 7 curated cards. */
export const MODELS: Architecture[] = [...ARCHITECTURES].sort((a, b) => a.date.localeCompare(b.date));

export const macroOf = (id: string): number | undefined => BENCHMARKS.macros[id];
export const rowsOf = (id: string): BenchmarkRow[] => BENCHMARKS.per_category[id] ?? [];
export const rowFor = (id: string, category: string): BenchmarkRow | undefined =>
  rowsOf(id).find((r) => r.category === category);
export const archOf = (id: string): Architecture | undefined => MODELS.find((m) => m.id === id);

/**
 * Evaluation Lab entries — driven by the regenerated benchmarks.json keys so
 * every number shown traces back to a real result CSV. PatchCore ids match the
 * curated cards 1:1; the two GAN aggregates have no exact card (the old draft
 * shipped invented per-category GAN values) so they appear benchmark-only.
 */
export interface EvalModel { id: string; name: string; family: string; archId?: string }
export const EVAL_MODELS: EvalModel[] = [
  { id: 'ocgan_final', name: 'OCGAN final (per-cat)', family: 'OCGAN' },
  { id: 'ocgan_optv2', name: 'OCGAN optv2', family: 'OCGAN' },
  { id: 'patchcore_v1', name: 'PatchCore v1', family: 'PatchCore', archId: 'patchcore_v1' },
  { id: 'patchcore_v2', name: 'PatchCore v2', family: 'PatchCore', archId: 'patchcore_v2' },
  { id: 'patchcore_v3', name: 'PatchCore v3', family: 'PatchCore', archId: 'patchcore_v3' },
  { id: 'patchcore_p1', name: 'PatchCore L1+L2+L3', family: 'PatchCore', archId: 'patchcore_p1' },
  { id: 'production_final', name: 'Production', family: 'PatchCore', archId: 'production_final' },
];

export type MetricKey = 'auroc' | 'auprc' | 'best_f1' | 'fpr95';
export const METRICS: { key: MetricKey; label: string; higherIsBetter: boolean }[] = [
  { key: 'auroc', label: 'AUROC', higherIsBetter: true },
  { key: 'auprc', label: 'AUPRC', higherIsBetter: true },
  { key: 'best_f1', label: 'Best F1', higherIsBetter: true },
  { key: 'fpr95', label: 'FPR@95TPR', higherIsBetter: false },
];

export function macroMetric(id: string, metric: MetricKey): number | undefined {
  const rows = rowsOf(id);
  if (!rows.length) return undefined;
  return rows.reduce((s, r) => s + r[metric], 0) / rows.length;
}
