import { EVAL_MODELS, macroMetric, METRICS, type MetricKey } from '../../data/models';

export function leaderboardRows(metric: MetricKey) {
  const meta = METRICS.find((m) => m.key === metric)!;
  return EVAL_MODELS
    .map((m) => ({ id: m.id, name: m.name, family: m.family, archId: m.archId, value: macroMetric(m.id, metric) }))
    .filter((r): r is typeof r & { value: number } => r.value !== undefined)
    .sort((a, b) => (meta.higherIsBetter ? b.value - a.value : a.value - b.value));
}
