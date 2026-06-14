import { useEffect, useState } from 'react';
import { Ban } from 'lucide-react';
import { ConfusionMatrix } from '../../components/ConfusionMatrix';
import { useArena } from '../../stores/arena';
import { ThresholdSlider } from './ThresholdSlider';

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel px-3 py-2 text-center">
      <div className="text-[10px] uppercase tracking-widest text-steel">{label}</div>
      <div className="num text-lg">{value}</div>
    </div>
  );
}
const fmt = (v: number | null, d = 3) => (v == null ? '—' : v.toFixed(d));

export function StatusBar() {
  const { phase, done, images, liveAccuracy, seed, cancel, startedAt } = useArena();
  const [now, setNow] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  if (phase !== 'running' && phase !== 'starting') return null;
  const total = images.length || 1;
  const elapsed = startedAt && now > startedAt ? (now - startedAt) / 1000 : 0;
  return (
    <div className="panel p-4 flex items-center gap-4">
      <div className="flex-1">
        <div className="flex justify-between text-xs text-steel mb-1">
          <span className="num">{done}/{images.length} scored · seed {seed ?? '…'}</span>
          <span className="num">live accuracy {(liveAccuracy * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2 rounded-full bg-panel2 overflow-hidden">
          <div className="h-full bg-accent transition-all duration-300" style={{ width: `${(done / total) * 100}%` }} />
        </div>
      </div>
      {startedAt && <span className="num text-xs text-steel">{elapsed.toFixed(0)}s</span>}
      <button onClick={cancel} className="flex items-center gap-1 text-alert text-sm hover:underline"><Ban size={14} /> Cancel</button>
    </div>
  );
}

export function SummaryPanel() {
  const { phase, summary, results, seed } = useArena();
  if (!summary || (phase !== 'done' && phase !== 'cancelled')) return null;
  const scores = Object.values(results).filter((r) => r.anomaly_score != null);
  const max = Math.max(...scores.map((r) => r.anomaly_score!), 1e-9);
  const thr = scores[0]?.threshold ?? 0;
  return (
    <div className="space-y-4">
      {phase === 'cancelled' && <div className="text-warn text-sm">Run cancelled — partial results below.</div>}
      <div className="grid grid-cols-3 md:grid-cols-7 gap-2">
        <Chip label="accuracy" value={fmt(summary.accuracy)} />
        <Chip label="precision" value={fmt(summary.precision)} />
        <Chip label="recall" value={fmt(summary.recall)} />
        <Chip label="F1" value={fmt(summary.f1)} />
        <Chip label="AUROC" value={fmt(summary.auroc, 4)} />
        <Chip label="mean ms" value={fmt(summary.mean_ms, 0)} />
        <Chip label="seed" value={String(seed ?? '—')} />
      </div>
      <div className="grid md:grid-cols-2 gap-4 items-start">
        <div className="panel p-4"><ConfusionMatrix confusion={summary.confusion} /></div>
        <div className="panel p-4">
          <div className="text-xs uppercase tracking-widest text-steel mb-3">Score distribution</div>
          <div className="relative h-24 border-b border-line">
            {scores.map((r) => (
              <span key={r.idx} title={`${r.filename}: ${r.anomaly_score!.toFixed(4)}`}
                className={`absolute bottom-0 w-1 rounded-t ${r.ground_truth_anomaly ? 'bg-alert/70' : 'bg-ok/70'}`}
                style={{ left: `${(r.anomaly_score! / max) * 98}%`, height: `${20 + (r.idx % 5) * 14}%` }} />
            ))}
            <span className="absolute top-0 bottom-0 w-0.5 bg-warn" style={{ left: `${(thr / max) * 98}%` }} title="threshold" />
          </div>
          <div className="flex gap-4 mt-2 text-[10px] text-steel">
            <span><span className="inline-block w-2 h-2 bg-ok/70 rounded-sm mr-1" />ground-truth good</span>
            <span><span className="inline-block w-2 h-2 bg-alert/70 rounded-sm mr-1" />ground-truth defect</span>
            <span className="text-warn">| threshold</span>
          </div>
        </div>
      </div>

      <ThresholdSlider results={results} />

      {summary.by_defect && Object.keys(summary.by_defect).length > 0 && (
        <div className="panel p-4">
          <div className="text-xs uppercase tracking-widest text-steel mb-3">Per defect type — caught vs missed</div>
          <div className="space-y-1.5">
            {Object.entries(summary.by_defect)
              .sort((a, b) => (a[1].accuracy ?? 0) - (b[1].accuracy ?? 0))
              .map(([d, e]) => (
                <div key={d} className="flex items-center gap-2 text-xs">
                  <span className={`w-2 h-2 rounded-sm shrink-0 ${e.is_anomaly ? 'bg-alert/70' : 'bg-ok/70'}`} />
                  <span className="flex-1 truncate">{d.replace(/_/g, ' ')}</span>
                  <span className="num text-steel">{e.correct}/{e.n}</span>
                  <div className="w-20 h-1.5 bg-panel2 rounded-full overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${(e.accuracy ?? 0) * 100}%` }} />
                  </div>
                  <span className="num w-9 text-right">{((e.accuracy ?? 0) * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
