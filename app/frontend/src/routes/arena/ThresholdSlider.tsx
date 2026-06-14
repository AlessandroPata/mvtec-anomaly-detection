import { useMemo, useState } from 'react';
import type { ArenaResult } from '../../types/api';

/**
 * Interactive operating-point explorer: recomputes accuracy / precision / recall / F1
 * client-side from the streamed per-image scores as you drag the threshold, and can
 * jump to the F1-optimal cut. Makes the single arena "accuracy" number honest by
 * showing it's just one point on a curve.
 */
export function ThresholdSlider({ results }: { results: Record<number, ArenaResult> }) {
  const pts = useMemo(
    () =>
      Object.values(results)
        .filter((r) => r.anomaly_score != null && r.verdict !== 'error')
        .map((r) => ({ s: r.anomaly_score as number, y: r.ground_truth_anomaly ? 1 : 0 })),
    [results],
  );
  const modelThr = useMemo(
    () => Object.values(results).find((r) => r.threshold != null)?.threshold ?? 0,
    [results],
  );

  const metricsAt = (t: number) => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const p of pts) {
      const pred = p.s >= t ? 1 : 0;
      if (p.y === 1 && pred === 1) tp++;
      else if (p.y === 0 && pred === 1) fp++;
      else if (p.y === 0 && pred === 0) tn++;
      else fn++;
    }
    const n = tp + fp + tn + fn;
    const prec = tp + fp ? tp / (tp + fp) : 0;
    const rec = tp + fn ? tp / (tp + fn) : 0;
    return {
      acc: n ? (tp + tn) / n : 0,
      prec,
      rec,
      f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0,
    };
  };

  const bestF1 = useMemo(() => {
    const cands = Array.from(new Set(pts.map((p) => p.s))).sort((a, b) => a - b);
    let best = { t: modelThr, f1: -1 };
    for (const t of cands) {
      const f1 = metricsAt(t).f1;
      if (f1 > best.f1) best = { t, f1 };
    }
    return best;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts, modelThr]);

  const [thr, setThr] = useState(modelThr);
  if (pts.length < 2) return null;
  const lo = Math.min(...pts.map((p) => p.s));
  const hi = Math.max(...pts.map((p) => p.s));
  const m = metricsAt(thr);

  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-steel">Try a threshold</div>
        <button className="text-accent text-xs hover:underline" onClick={() => setThr(bestF1.t)}>
          jump to F1-optimal
        </button>
      </div>
      <input
        type="range"
        min={lo}
        max={hi}
        step={(hi - lo) / 200 || 0.001}
        value={thr}
        onChange={(e) => setThr(Number(e.target.value))}
        className="w-full accent-[var(--color-accent)]"
        aria-label="decision threshold"
      />
      <div className="grid grid-cols-4 gap-2 text-center">
        {([['accuracy', m.acc], ['precision', m.prec], ['recall', m.rec], ['F1', m.f1]] as const).map(
          ([k, v]) => (
            <div key={k}>
              <div className="text-[10px] uppercase tracking-widest text-steel">{k}</div>
              <div className="num text-lg">{v.toFixed(3)}</div>
            </div>
          ),
        )}
      </div>
      <div className="text-[10px] text-steel num">
        threshold {thr.toFixed(4)} · model {modelThr.toFixed(4)} · F1-optimal {bestF1.t.toFixed(4)}
      </div>
    </div>
  );
}
