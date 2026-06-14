export function ScoreGauge({ score, threshold, max }: { score: number; threshold: number; max?: number }) {
  const hi = max ?? Math.max(score, threshold) * 1.4;
  const pct = (v: number) => Math.max(0, Math.min(100, (v / hi) * 100));
  return (
    <div className="space-y-1">
      <div className="relative h-3 rounded-full bg-panel2 border border-line overflow-hidden">
        <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-ok/50 to-alert/60" style={{ width: `${pct(score)}%` }} />
        <div className="absolute inset-y-0 w-0.5 bg-warn" style={{ left: `${pct(threshold)}%` }} title="threshold" />
      </div>
      <div className="flex justify-between text-xs text-steel num">
        <span>score {score.toFixed(4)}</span>
        <span className="text-warn">thr {threshold.toFixed(4)}</span>
      </div>
    </div>
  );
}
