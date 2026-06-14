import insights from '../../data/insights.json';
import { Sparkline } from '../../components/Sparkline';

interface Delta { category: string; delta: number }
const topDeltas = (rows: Delta[], n = 3) => [...rows].sort((a, b) => b.delta - a.delta).slice(0, n);

export function Insights() {
  const coreset = insights.coreset_effect as Delta[];
  const agg = insights.aggregation_effect as Delta[];
  const layers = insights.layer_ablation as { category: string; configs: Record<string, number> }[];
  const screw = layers.find((l) => l.category === 'screw');

  return (
    <div className="grid md:grid-cols-3 gap-4">
      <article className="panel p-5 space-y-2">
        <h3 className="font-medium text-sm">Don't prune the bank</h3>
        <p className="text-xs text-steel">Full 70k-patch bank vs 10k coreset (same aggregation):</p>
        <ul className="space-y-1">
          {topDeltas(coreset).map((d) => (
            <li key={d.category} className="flex justify-between text-sm">
              <span>{d.category}</span><span className="num text-ok">+{d.delta.toFixed(4)}</span>
            </li>
          ))}
        </ul>
        <Sparkline values={coreset.map((d) => d.delta)} width={200} height={28} stroke="var(--color-ok)" />
      </article>

      <article className="panel p-5 space-y-2">
        <h3 className="font-medium text-sm">Reweighted top-k beats plain top-k</h3>
        <p className="text-xs text-steel">topk_reweighted k=9 vs topk_mean k=3, both on the 10k coreset:</p>
        <ul className="space-y-1">
          {topDeltas(agg).map((d) => (
            <li key={d.category} className="flex justify-between text-sm">
              <span>{d.category}</span><span className="num text-ok">+{d.delta.toFixed(4)}</span>
            </li>
          ))}
        </ul>
        <Sparkline values={agg.map((d) => d.delta)} width={200} height={28} stroke="var(--color-ok)" />
      </article>

      <article className="panel p-5 space-y-2">
        <h3 className="font-medium text-sm">Screw needs layer1 detail</h3>
        <p className="text-xs text-steel">Fine thread defects benefit from earlier features:</p>
        {screw && (
          <ul className="space-y-1">
            {Object.entries(screw.configs).map(([fl, v]) => (
              <li key={fl} className="flex justify-between text-sm">
                <span className="text-steel">{fl}</span><span className="num">{v.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-steel">The only per-category override that made production.</p>
      </article>
    </div>
  );
}
