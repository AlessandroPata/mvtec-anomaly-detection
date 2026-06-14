import { beforeEach, describe, expect, it } from 'vitest';
import { useArena } from '../arena';
import type { ArenaResult } from '../../types/api';

const result = (idx: number, verdict: ArenaResult['verdict'], correct = verdict === 'tp' || verdict === 'tn'): ArenaResult => ({
  idx, defect_type: 'crack', filename: `${idx}.png`, ground_truth_anomaly: true,
  anomaly_score: 0.9, is_anomaly: true, inference_ms: 12, verdict, correct,
});

describe('arena store', () => {
  beforeEach(() => useArena.getState().reset());

  it('applyResult fills the map and live counters', () => {
    useArena.getState().beginJob('j1', 7, [
      { idx: 0, defect_type: 'crack', filename: '0.png', ground_truth_anomaly: true },
      { idx: 1, defect_type: 'good', filename: '1.png', ground_truth_anomaly: false },
    ]);
    useArena.getState().applyResult(result(0, 'tp'));
    useArena.getState().applyResult(result(1, 'fp', false));
    const st = useArena.getState();
    expect(st.done).toBe(2);
    expect(st.correct).toBe(1);
    expect(st.results[0]?.verdict).toBe('tp');
    expect(st.liveAccuracy).toBeCloseTo(0.5);
  });

  it('duplicate results (SSE reconnect overlap) are idempotent', () => {
    useArena.getState().beginJob('j1', 7, [{ idx: 0, defect_type: 'crack', filename: '0.png', ground_truth_anomaly: true }]);
    useArena.getState().applyResult(result(0, 'tp'));
    useArena.getState().applyResult(result(0, 'tp'));
    expect(useArena.getState().done).toBe(1);
  });

  it('finishJob stores summary and phase', () => {
    useArena.getState().beginJob('j1', 7, []);
    useArena.getState().finishJob({
      status: 'done',
      summary: { n: 0, errors: 0, accuracy: null, precision: null, recall: null, f1: null, auroc: null, confusion: { tp: 0, tn: 0, fp: 0, fn: 0 }, mean_ms: null, p95_ms: null },
      error: null,
    });
    expect(useArena.getState().phase).toBe('done');
    expect(useArena.getState().summary?.n).toBe(0);
  });
});
