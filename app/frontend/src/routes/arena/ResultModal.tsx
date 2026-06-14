import { useEffect, useState } from 'react';
import { Modal } from '../../components/Modal';
import { ScoreGauge } from '../../components/ScoreGauge';
import { Badge } from '../../components/ui';
import { verdictClasses } from '../../components/heat';
import { predictFromDataset, sampleUrl } from '../../services/api';
import { useArena } from '../../stores/arena';

export function ResultModal({ idx, onClose }: { idx: number | null; onClose: () => void }) {
  const { images, results, config } = useArena();
  const img = idx == null ? undefined : images.find((i) => i.idx === idx);
  const res = idx == null ? undefined : results[idx];
  const [heat, setHeat] = useState<{ idx: number; b64: string | null } | null>(null);
  const [showHeat, setShowHeat] = useState(true);
  const [opacity, setOpacity] = useState(0.55);

  const wantHeat = idx != null && img != null && res != null && res.verdict !== 'error';
  useEffect(() => {
    if (idx == null || !img || !res || res.verdict === 'error') return;
    let cancelled = false;
    predictFromDataset(config.category, img.defect_type, img.filename, config.variant)
      .then((p) => { if (!cancelled) setHeat({ idx, b64: p.heatmap_base64 }); })
      .catch(() => { if (!cancelled) setHeat({ idx, b64: null }); });
    return () => { cancelled = true; };
  }, [idx]);  // eslint-disable-line react-hooks/exhaustive-deps
  const heatmap = wantHeat && heat?.idx === idx ? heat.b64 : null;
  const loading = wantHeat && heat?.idx !== idx;

  if (!img) return null;
  const v = res ? verdictClasses(res.verdict) : null;
  return (
    <Modal open={idx != null} onClose={onClose} wide>
      <div className="grid md:grid-cols-[1.2fr_1fr] gap-6">
        <div className="relative rounded-lg overflow-hidden border border-line">
          <img src={sampleUrl(config.category, img.defect_type, img.filename)} alt={img.filename} className="w-full" />
          {showHeat && heatmap && (
            <img src={`data:image/png;base64,${heatmap}`} alt="anomaly heatmap"
              className="absolute inset-0 w-full h-full mix-blend-screen pointer-events-none"
              style={{ opacity, filter: 'sepia(1) saturate(6) hue-rotate(-50deg)' }} />
          )}
          {loading && <div className="absolute inset-0 grid place-items-center bg-ink/40 text-xs text-steel">computing heatmap…</div>}
        </div>
        <div className="space-y-4">
          <div>
            <div className="text-xs text-steel num">{img.defect_type}/{img.filename}</div>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              {v && <span className={`px-2 py-1 rounded-md text-sm font-semibold ${v.chip}`}>{v.label}</span>}
              <Badge tone={img.ground_truth_anomaly ? 'alert' : 'ok'}>
                GT: {img.ground_truth_anomaly ? `defect (${img.defect_type})` : 'good'}
              </Badge>
            </div>
          </div>
          {res?.anomaly_score != null && res.threshold != null && (
            <ScoreGauge score={res.anomaly_score} threshold={res.threshold} />
          )}
          <dl className="text-sm space-y-1.5">
            {res?.anomaly_probability != null && (
              <div className="flex justify-between"><dt className="text-steel">anomaly probability</dt><dd className="num">{(res.anomaly_probability * 100).toFixed(1)}%</dd></div>
            )}
            {res?.inference_ms != null && (
              <div className="flex justify-between"><dt className="text-steel">inference</dt><dd className="num">{res.inference_ms.toFixed(0)} ms</dd></div>
            )}
            <div className="flex justify-between"><dt className="text-steel">model</dt><dd>{config.variant}</dd></div>
          </dl>
          <div className="flex items-center gap-3 text-xs text-steel">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} className="accent-cyan-400" />
              heatmap
            </label>
            <input type="range" min={0.1} max={1} step={0.05} value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))} className="flex-1 accent-cyan-400" />
          </div>
          {res?.verdict === 'error' && <p className="text-alert text-sm">{res.error}</p>}
        </div>
      </div>
    </Modal>
  );
}
