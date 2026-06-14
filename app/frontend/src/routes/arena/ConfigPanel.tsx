import { Dices, Play } from 'lucide-react';
import { Badge } from '../../components/ui';
import { useHealth } from '../../stores/health';
import { useArena } from '../../stores/arena';
import { macroOf } from '../../data/models';
import { thumbUrl } from '../../services/api';

const N_OPTIONS = [25, 50, 100, 150];
const VARIANT_MACRO: Record<string, string> = {
  production: 'production_final', patchcore_v2: 'patchcore_v2', patchcore_v1: 'patchcore_v1',
};

export function ConfigPanel() {
  const { meta, online } = useHealth();
  const { config, setConfig, start, phase } = useArena();
  const cat = meta?.categories.find((c) => c.name === config.category);
  const busy = phase === 'starting' || phase === 'running';

  return (
    <div className="panel p-5 space-y-5">
      <div>
        <div className="text-xs uppercase tracking-widest text-steel mb-2">Category</div>
        <div className="grid grid-cols-5 md:grid-cols-8 xl:[grid-template-columns:repeat(15,minmax(0,1fr))] gap-1.5">
          {meta?.categories.map((c) => (
            <button key={c.name} onClick={() => setConfig({ category: c.name })} disabled={busy}
              className={`rounded-lg overflow-hidden border transition-colors ${config.category === c.name ? 'border-accent' : 'border-line hover:border-steel'}`}
              title={`${c.name} · ${c.test_total} test images`}>
              <img src={thumbUrl(c.name, 'good', '000.png', 64)} alt={c.name}
                className="w-full aspect-square object-cover" loading="lazy"
                onError={(e) => ((e.target as HTMLImageElement).style.visibility = 'hidden')} />
              <div className="text-[9px] py-0.5 truncate px-1 text-steel">{c.name.replace('_', ' ')}</div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-widest text-steel mb-2">Model</div>
        <div className="grid md:grid-cols-3 gap-1.5">
          {cat?.variants.map((v) => (
            <button key={v.id} onClick={() => setConfig({ variant: v.id })} disabled={busy || !v.available}
              className={`text-left rounded-lg border p-3 transition-colors ${
                config.variant === v.id ? 'border-accent bg-accent/5' : 'border-line hover:border-steel'} ${!v.available ? 'opacity-40' : ''}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">{v.label}</span>
                <span className="flex gap-1">
                  {v.approximate && <Badge tone="warn">approx</Badge>}
                  <Badge tone={v.kind === 'production' ? 'ok' : v.kind === 'gan' ? 'accent' : 'steel'}>{v.kind}</Badge>
                </span>
              </div>
              <div className="flex justify-between mt-1 text-xs text-steel gap-2">
                <span className="line-clamp-2">{v.description}</span>
                <span className="num shrink-0">{(macroOf(VARIANT_MACRO[v.id] ?? v.id) ?? 0).toFixed(4)}</span>
              </div>
            </button>
          ))}
        </div>
        <p className="text-[11px] text-steel mt-2">
          Every model from the Evaluation Lab that exists as a runnable checkpoint is here. PatchCore v3 is
          the Production configuration itself (only screw differed, by feature layers), and the p1 row is a
          layer-ablation series rather than a separate model — so neither gets its own card. The OCGAN
          variants load the original training checkpoints; the first run per category rebuilds their memory
          bank (~1 min).
        </p>
      </div>

      <div className="flex gap-4 items-end flex-wrap">
        <div>
          <div className="text-xs uppercase tracking-widest text-steel mb-2">Images</div>
          <div className="flex rounded-lg border border-line overflow-hidden">
            {N_OPTIONS.map((n) => (
              <button key={n} onClick={() => setConfig({ n })} disabled={busy}
                className={`px-3 py-1.5 text-sm num ${config.n === n ? 'bg-accent/15 text-accent' : 'text-steel hover:text-fog'}`}>{n}</button>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-widest text-steel mb-2">Seed</div>
          <div className="flex items-center gap-1">
            <input value={config.seed ?? ''} placeholder="random" disabled={busy}
              onChange={(e) => setConfig({ seed: e.target.value === '' ? null : Number(e.target.value) || 0 })}
              className="w-24 bg-panel2 border border-line rounded-lg px-2 py-1.5 text-sm num focus:border-accent outline-none" />
            <button onClick={() => setConfig({ seed: Math.floor(Math.random() * 1_000_000) })} disabled={busy}
              className="p-2 text-steel hover:text-accent" title="Roll a seed"><Dices size={16} /></button>
          </div>
        </div>
        <button onClick={() => void start()} disabled={busy || !online}
          className="ml-auto inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent text-ink font-semibold text-sm
                     hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
          <Play size={15} /> {phase === 'starting' ? 'Loading model…' : 'Run batch'}
        </button>
      </div>
    </div>
  );
}
