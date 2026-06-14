import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Check, X } from 'lucide-react';
import { Badge, Section } from '../../components/ui';
import { aurocColor } from '../../components/heat';
import { archOf, macroOf, MODELS, rowsOf } from '../../data/models';

const TYPE_TONE: Record<string, string> = {
  input: 'border-accent/40 bg-accent/5', process: 'border-line bg-panel2',
  storage: 'border-warn/40 bg-warn/5', output: 'border-ok/40 bg-ok/5',
};

export default function ModelDetail() {
  const { id = '' } = useParams();
  const arch = archOf(id);
  if (!arch) return <div className="text-steel">Unknown model. <Link className="text-accent" to="/models">Back to gallery</Link></div>;

  const idx = MODELS.findIndex((m) => m.id === id);
  const prev = MODELS[idx - 1]; const next = MODELS[idx + 1];
  const macro = macroOf(id) ?? arch.macroAUROC;
  const rows = rowsOf(id);

  return (
    <div className="space-y-10">
      <header className="space-y-3">
        <div className="flex items-center justify-between">
          <Link to="/models" className="text-steel text-sm hover:text-fog flex items-center gap-1"><ArrowLeft size={14} /> Models</Link>
          <div className="flex gap-3 text-sm">
            {prev && <Link className="text-steel hover:text-accent flex items-center gap-1" to={`/models/${prev.id}`}><ArrowLeft size={12} />{prev.shortName}</Link>}
            {next && <Link className="text-steel hover:text-accent flex items-center gap-1" to={`/models/${next.id}`}>{next.shortName}<ArrowRight size={12} /></Link>}
          </div>
        </div>
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">{arch.name}</h1>
            <p className="text-steel mt-1 text-sm">{arch.architecture_type} · {arch.date}</p>
          </div>
          <div className="text-right">
            <div className="text-[10px] uppercase tracking-widest text-steel">macro auroc</div>
            <div className="num text-4xl font-semibold" style={{ color: aurocColor(macro) }}>{macro.toFixed(4)}</div>
            <Badge tone={arch.status === 'production' ? 'ok' : 'steel'}>{arch.status}</Badge>
          </div>
        </div>
        <p className="max-w-3xl text-sm text-fog/90">{arch.core_idea}</p>
      </header>

      <Section title="Pipeline">
        <ol className="flex flex-wrap items-stretch gap-2">
          {arch.pipeline.map((s, i) => (
            <li key={i} className="flex items-center gap-2">
              <div className={`rounded-lg border px-3 py-2 ${TYPE_TONE[s.type ?? 'process'] ?? TYPE_TONE.process}`}>
                <div className="text-xs font-medium">{s.label}</div>
                <div className="text-[10px] text-steel max-w-44">{s.detail}</div>
              </div>
              {i < arch.pipeline.length - 1 && <ArrowRight size={14} className="text-steel shrink-0" />}
            </li>
          ))}
        </ol>
      </Section>

      <div className="grid md:grid-cols-2 gap-6">
        <Section title="Strengths">
          <ul className="space-y-2 text-sm">
            {arch.strengths.map((s) => <li key={s} className="flex gap-2"><Check size={15} className="text-ok shrink-0 mt-0.5" />{s}</li>)}
          </ul>
        </Section>
        <Section title="Weaknesses">
          <ul className="space-y-2 text-sm">
            {arch.weaknesses.map((s) => <li key={s} className="flex gap-2"><X size={15} className="text-alert shrink-0 mt-0.5" />{s}</li>)}
          </ul>
        </Section>
      </div>

      <Section title="What changed vs the previous iteration">
        <div className="flex flex-wrap gap-2">
          {arch.improvements.map((s) => <Badge key={s} tone="accent">{s}</Badge>)}
        </div>
      </Section>

      <Section title="Hyperparameters">
        <div className="panel overflow-hidden">
          <table className="w-full text-sm">
            <tbody>
              {Object.entries(arch.hyperparameters).map(([k, v]) => (
                <tr key={k} className="border-b border-line/50 last:border-0">
                  <td className="px-4 py-2 text-steel w-56">{k}</td>
                  <td className="px-4 py-2 num">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {rows.length > 0 && (
        <Section title="Per-category results">
          <div className="panel overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-steel uppercase tracking-wider">
                <tr className="border-b border-line">
                  <th className="px-4 py-2">Category</th><th className="px-4 py-2">AUROC</th>
                  <th className="px-4 py-2">AUPRC</th><th className="px-4 py-2">Best F1</th><th className="px-4 py-2">FPR@95</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.category} className="border-b border-line/40 last:border-0">
                    <td className="px-4 py-2">{r.category}</td>
                    <td className="px-4 py-2 num" style={{ color: aurocColor(r.auroc) }}>{r.auroc.toFixed(4)}</td>
                    <td className="px-4 py-2 num text-steel">{r.auprc.toFixed(4)}</td>
                    <td className="px-4 py-2 num text-steel">{r.best_f1.toFixed(4)}</td>
                    <td className="px-4 py-2 num text-steel">{r.fpr95.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {arch.notes && <p className="text-xs text-steel border-l-2 border-warn/50 pl-3">{arch.notes}</p>}
    </div>
  );
}
