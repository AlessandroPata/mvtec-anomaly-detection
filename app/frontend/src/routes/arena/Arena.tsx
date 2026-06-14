import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { OfflineCard } from '../../components/ui';
import { useHealth } from '../../stores/health';
import { useArena } from '../../stores/arena';
import { ConfigPanel } from './ConfigPanel';
import { LiveGrid } from './LiveGrid';
import { StatusBar, SummaryPanel } from './SummaryPanel';
import { ComparePanel } from './ComparePanel';
import { SingleTest } from './SingleTest';
import { InfoNote } from '../../components/InfoNote';

export default function Arena() {
  const { online } = useHealth();
  const { config, setConfig, seed, phase } = useArena();
  const [params, setParams] = useSearchParams();
  const [tab, setTab] = useState<'batch' | 'single'>('batch');

  useEffect(() => {   // hydrate config from URL once
    const cat = params.get('cat'); const variant = params.get('variant');
    const n = params.get('n'); const s = params.get('seed');
    setConfig({
      ...(cat ? { category: cat } : {}), ...(variant ? { variant } : {}),
      ...(n ? { n: Number(n) } : {}), ...(s ? { seed: Number(s) } : {}),
    });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {   // reflect the running config (incl. resolved seed) into the URL
    if (phase === 'running' || phase === 'done') {
      setParams({ cat: config.category, variant: config.variant, n: String(config.n), ...(seed != null ? { seed: String(seed) } : {}) }, { replace: true });
    }
  }, [phase, seed]);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-8">
      <header className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Test Arena</h1>
          <p className="text-steel mt-2">Sample random test images, pick a model, watch it classify live.</p>
        </div>
        <div className="flex rounded-lg border border-line overflow-hidden text-sm">
          <button onClick={() => setTab('batch')} className={`px-4 py-2 ${tab === 'batch' ? 'bg-accent/15 text-accent' : 'text-steel'}`}>Batch run</button>
          <button onClick={() => setTab('single')} className={`px-4 py-2 ${tab === 'single' ? 'bg-accent/15 text-accent' : 'text-steel'}`}>Single test</button>
        </div>
      </header>

      <InfoNote title="How the Arena scores" defaultOpen>
        <p>
          The Arena runs the selected model <strong>live</strong> on a random sample of real test
          images and reports <strong>accuracy at the model's decision threshold</strong> — the
          share of images it labels correctly as normal vs. anomalous.
        </p>
        <p>
          This is deliberately different from the <strong>Evaluation</strong> page. Evaluation
          reports <strong>AUROC</strong>, a threshold-free ranking score (how well a model
          separates defective from normal overall, averaged over seeds). The Arena fixes one
          operating point, so a model with a high AUROC can still miss defects here if its
          threshold is set too conservatively.
        </p>
        <p>
          Each category uses its own <strong>best-F1 threshold</strong>, because every product has
          a different score distribution — one global cut-off can't fit all 15 categories.
        </p>
      </InfoNote>

      {online === false ? (
        <OfflineCard what="The Test Arena" onRetry={() => window.location.reload()} />
      ) : tab === 'batch' ? (
        <div className="space-y-6">
          <ConfigPanel />
          <StatusBar />
          <SummaryPanel />
          <ComparePanel />
          <LiveGrid />
        </div>
      ) : (
        <SingleTest />
      )}
    </div>
  );
}
