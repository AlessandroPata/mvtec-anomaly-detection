import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { CountUp } from '../components/CountUp';
import { Sparkline } from '../components/Sparkline';
import { Badge, CTA, StatCard } from '../components/ui';
import { MODELS, macroOf, rowsOf } from '../data/models';
import { InfoNote } from '../components/InfoNote';

export default function Home() {
  const journey = MODELS.map((m) => ({ ...m, macro: macroOf(m.id) ?? m.macroAUROC }));
  const finalMacro = macroOf('production_final') ?? 0.9846;
  const perfect = rowsOf('production_final').filter((r) => r.auroc >= 0.9999).length;
  const gain = (finalMacro - 0.7866) * 100; // GAN baseline from the project README

  return (
    <div className="space-y-14">
      <header className="relative blueprint rounded-2xl border border-line p-10 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-ink via-ink/60 to-transparent pointer-events-none" />
        <div className="relative space-y-4 max-w-2xl">
          <Badge tone="accent">industrial anomaly detection</Badge>
          <h1 className="text-4xl font-bold tracking-tight leading-tight">
            From a one-class GAN to <span className="text-accent">frozen-feature PatchCore</span>
          </h1>
          <p className="text-steel">
            Seven model iterations on the MVTec AD benchmark — 15 industrial categories, image-level
            anomaly detection. The journey ends at <span className="num text-fog">0.9846</span> macro
            AUROC with no training at all: frozen ImageNet features, a full memory bank, and a smarter
            aggregation.
          </p>
          <div className="flex gap-3 pt-2">
            <CTA to="/arena">Run the live arena <ArrowRight size={14} /></CTA>
            <CTA to="/evaluation">Compare all models <ArrowRight size={14} /></CTA>
          </div>
        </div>
      </header>

      <InfoNote title="About this showcase">
        <p>
          This site walks through an industrial anomaly-detection project on the{' '}
          <strong>MVTec AD</strong> benchmark: seven model iterations, from a one-class GAN to a
          training-free PatchCore that reaches <strong>0.9846</strong> macro AUROC across 15 product
          categories.
        </p>
        <p>
          <strong>Models</strong> compares every iteration · <strong>Evaluation</strong> breaks down
          the metrics · <strong>Test Arena</strong> runs the models live on real images ·{' '}
          <strong>Dataset</strong> browses the categories · <strong>Methodology</strong> explains why
          frozen features beat the GAN.
        </p>
      </InfoNote>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Final macro AUROC" value={<CountUp value={finalMacro * 100} decimals={2} suffix="%" />} sub="production PatchCore, 15 categories" />
        <StatCard label="Gain vs GAN baseline" value={<CountUp value={gain} decimals={1} suffix=" pp" />} sub="0.7866 → 0.9846" />
        <StatCard label="Perfect categories" value={<CountUp value={perfect} />} sub="AUROC = 1.0000" />
        <StatCard label="Model iterations" value={<CountUp value={MODELS.length} />} sub="OCGAN v1 → Production" />
      </div>

      <section className="space-y-4">
        <div className="flex items-end justify-between">
          <h2 className="text-xl font-semibold">The journey</h2>
          <Sparkline values={journey.map((j) => j.macro)} width={160} height={40} />
        </div>
        <ol className="grid md:grid-cols-2 xl:grid-cols-4 gap-3">
          {journey.map((m, i) => (
            <li key={m.id}>
              <Link to={`/models/${m.id}`} className="panel block p-4 hover:border-accent/50 transition-colors h-full">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-steel num">{String(i + 1).padStart(2, '0')} · {m.date}</span>
                  <Badge tone={m.status === 'production' ? 'ok' : 'steel'}>{m.status}</Badge>
                </div>
                <div className="font-medium mt-2">{m.shortName}</div>
                <div className="num text-2xl mt-1" style={{ color: m.macro >= 0.9 ? 'var(--color-ok)' : 'var(--color-steel)' }}>
                  {m.macro.toFixed(4)}
                </div>
                <div className="text-xs text-steel mt-1 line-clamp-2">{m.architecture_type}</div>
              </Link>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
