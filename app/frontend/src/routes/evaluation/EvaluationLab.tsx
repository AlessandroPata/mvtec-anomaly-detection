import { useState } from 'react';
import { Section } from '../../components/ui';
import { METRICS, type MetricKey } from '../../data/models';
import { Leaderboard } from './Leaderboard';
import { Heatmap } from './Heatmap';
import { Evolution } from './Evolution';
import { Insights } from './Insights';
import { InfoNote } from '../../components/InfoNote';
import { aurocColor } from '../../components/heat';
import { MACRO_PIXEL_AUROC, MACRO_AUPRO, PIXEL_METRICS } from '../../data/pixelMetrics';

export default function EvaluationLab() {
  const [metric, setMetric] = useState<MetricKey>('auroc');
  return (
    <div className="space-y-12">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Evaluation Lab</h1>
        <p className="text-steel mt-2">Every model, every category, every metric — side by side.</p>
      </header>

      <InfoNote title="How these numbers are computed" defaultOpen>
        <p>
          <strong>AUROC</strong> (the headline) measures how well a model <em>ranks</em> defective
          images above normal ones. It is <strong>threshold-free</strong>, which makes it the
          fairest cross-model comparison: 1.0 = perfect separation, 0.5 = chance.
        </p>
        <p>
          These figures are aggregated over the <strong>full test set and multiple seeds</strong>,
          precomputed from the training runs. <strong>best-F1</strong> and <strong>FPR@95</strong>
          summarise behaviour at a chosen operating point.
        </p>
        <p>
          AUROC says nothing about <em>which</em> threshold a model uses to decide. To watch a model
          actually call images normal vs. anomalous at its threshold, use the{' '}
          <strong>Test Arena</strong> — that's why a 0.93-AUROC model can still look weaker there if
          its cut-off is off.
        </p>
      </InfoNote>

      <Section title="Leaderboard" sub="Macro average across the 15 MVTec categories"
        right={
          <div className="flex rounded-lg border border-line overflow-hidden text-xs">
            {METRICS.map((m) => (
              <button key={m.key} onClick={() => setMetric(m.key)}
                className={`px-3 py-1.5 ${metric === m.key ? 'bg-accent/15 text-accent' : 'text-steel hover:text-fog'}`}>
                {m.label}
              </button>
            ))}
          </div>
        }>
        <Leaderboard metric={metric} />
      </Section>

      <Section title="Macro AUROC evolution" sub="Chronological — the PatchCore jump is the story">
        <Evolution />
      </Section>

      <Section title="Model × category heatmap" sub="Where each architecture wins and where it breaks down">
        <Heatmap />
      </Section>

      <Section title="Ablation insights" sub="The three ingredients behind +19.8 pp, from the tuning logs">
        <Insights />
      </Section>

      <Section title="Localization (pixel-level)"
        sub={`Macro pixel-AUROC ${MACRO_PIXEL_AUROC.toFixed(4)} · macro AUPRO@30% ${MACRO_AUPRO.toFixed(4)} — raw anomaly map vs. ground-truth masks, full test set`}>
        <InfoNote title="How localization is scored">
          MVTec AD ships pixel-accurate masks, so beyond the image-level verdict we score
          <em> where</em> each defect is. <strong>pixel-AUROC</strong> ranks every pixel (defect vs. normal);
          <strong> pixel-AP</strong> is the precision–recall area on the heavily imbalanced pixel labels
          (defect pixels are often &lt;3% of the image, so even a strong model scores far below its AUROC);
          <strong> AUPRO@30%</strong> is the official MVTec metric — it weights every connected defect region
          equally (a tiny scratch counts as much as a large stain) and integrates the per-region overlap up to a
          30% false-positive rate. All three come from the same raw anomaly map, no per-image normalization.
        </InfoNote>
        <div className="panel p-4 space-y-1.5">
          {[...PIXEL_METRICS].sort((a, b) => a.pixel_auroc - b.pixel_auroc).map((r) => (
            <div key={r.category} className="flex items-center gap-3 text-xs">
              <span className="w-20 truncate">{r.category.replace(/_/g, ' ')}</span>
              <div className="flex-1 h-2 bg-panel2 rounded-full overflow-hidden">
                <div className="h-full rounded-full"
                  style={{ width: `${Math.max(2, (r.pixel_auroc - 0.5) / 0.5 * 100)}%`, background: aurocColor(r.pixel_auroc) }} />
              </div>
              <span className="num w-16 text-right" style={{ color: aurocColor(r.pixel_auroc) }}>{r.pixel_auroc.toFixed(4)}</span>
              <span className="num w-20 text-right" style={{ color: aurocColor(r.aupro) }} title="AUPRO@30% — per-region overlap">PRO {r.aupro.toFixed(3)}</span>
              <span className="num w-16 text-right text-steel" title="pixel average precision">AP {r.pixel_ap.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
