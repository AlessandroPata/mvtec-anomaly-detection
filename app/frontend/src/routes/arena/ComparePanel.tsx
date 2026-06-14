import { X } from 'lucide-react';
import { useArena, type ArenaSnapshot } from '../../stores/arena';

const METRICS: { key: 'accuracy' | 'precision' | 'recall' | 'f1' | 'auroc'; label: string; d: number }[] = [
  { key: 'accuracy', label: 'accuracy', d: 3 },
  { key: 'precision', label: 'precision', d: 3 },
  { key: 'recall', label: 'recall', d: 3 },
  { key: 'f1', label: 'F1', d: 3 },
  { key: 'auroc', label: 'AUROC', d: 4 },
];

function Header({ snap, slot }: { snap: ArenaSnapshot; slot: 'A' | 'B' }) {
  return (
    <div className="text-center">
      <div className="text-[10px] uppercase tracking-widest text-steel">slot {slot}</div>
      <div className="font-semibold text-sm">{snap.variant}</div>
      <div className="text-[10px] text-steel num">{snap.category} · seed {snap.seed ?? '—'} · n {snap.n}</div>
    </div>
  );
}

export function ComparePanel() {
  const { snapA, snapB, snapshot, summary, phase, clearSnapshots } = useArena();
  const canPin = !!summary && (phase === 'done' || phase === 'cancelled');
  if (!snapA && !snapB && !canPin) return null;

  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-widest text-steel">Compare two models</div>
        <div className="flex items-center gap-2 text-xs">
          {canPin && <button onClick={() => snapshot('A')} className="px-2 py-1 rounded border border-line hover:border-accent text-steel hover:text-accent">Pin current → A</button>}
          {canPin && <button onClick={() => snapshot('B')} className="px-2 py-1 rounded border border-line hover:border-accent text-steel hover:text-accent">Pin current → B</button>}
          {(snapA || snapB) && <button onClick={clearSnapshots} className="flex items-center gap-1 text-steel hover:text-alert"><X size={12} /> clear</button>}
        </div>
      </div>

      {!snapA || !snapB ? (
        <p className="text-xs text-steel">
          Run a model, then <strong>Pin current → A</strong>. Run another model (same category &amp; seed for a fair
          comparison), then <strong>Pin current → B</strong>. Both summaries appear side by side with the deltas.
          {snapA && <span className="text-accent"> Slot A is set ({snapA.variant}) — pin a second run to B.</span>}
          {snapB && !snapA && <span className="text-accent"> Slot B is set ({snapB.variant}) — pin a run to A.</span>}
        </p>
      ) : (
        <>
          <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-center">
            <Header snap={snapA} slot="A" />
            <div className="text-[10px] text-steel">vs</div>
            <Header snap={snapB} slot="B" />
          </div>
          <div className="space-y-1.5">
            {METRICS.map(({ key, label, d }) => {
              const a = snapA.summary[key] as number | null;
              const b = snapB.summary[key] as number | null;
              const delta = a != null && b != null ? b - a : null;
              return (
                <div key={key} className="grid grid-cols-[1fr_auto_1fr] gap-3 items-center text-xs">
                  <div className="text-right num" style={{ color: a != null && b != null && a > b ? '#1f9d55' : undefined }}>{a == null ? '—' : a.toFixed(d)}</div>
                  <div className="text-center text-steel w-24">
                    <div className="text-[10px] uppercase tracking-widest">{label}</div>
                    {delta != null && <div className="num" style={{ color: delta > 0 ? '#1f9d55' : delta < 0 ? '#c2553a' : '#6b747e' }}>{delta > 0 ? '+' : ''}{delta.toFixed(d)}</div>}
                  </div>
                  <div className="text-left num" style={{ color: a != null && b != null && b > a ? '#1f9d55' : undefined }}>{b == null ? '—' : b.toFixed(d)}</div>
                </div>
              );
            })}
          </div>
          {snapA.seed !== snapB.seed && (
            <p className="text-[10px] text-warn">⚠ different seeds ({snapA.seed} vs {snapB.seed}) — the two runs saw different images, so deltas mix model and sampling.</p>
          )}
        </>
      )}
    </div>
  );
}
