import { Link } from 'react-router-dom';
import { Badge } from '../../components/ui';
import { aurocColor } from '../../components/heat';
import { CATEGORIES, macroOf, MODELS, rowFor } from '../../data/models';
import { InfoNote } from '../../components/InfoNote';

export default function ModelGallery() {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Models</h1>
        <p className="text-steel mt-2">Seven iterations, two families. Click any card for the full anatomy.</p>
      </header>

      <InfoNote title="Reading the model families">
        <p>
          Two families are compared. <strong>OCGAN</strong> (ocgan_final, ocgan_optv2) is the original
          one-class GAN — it reconstructs a normal image and scores the reconstruction error.{' '}
          <strong>PatchCore</strong> (production, plus reconstructed v1/v2) is <strong>training-free</strong>:
          it stores frozen ImageNet patch features in a memory bank and scores each patch by its distance
          to the nearest bank entry.
        </p>
        <p>
          The bar strip on each card is per-category AUROC. <strong>production</strong> is the shipped
          model (full memory bank); <strong>v1/v2</strong> are smaller-coreset reconstructions of earlier
          eras, kept for comparison.
        </p>
      </InfoNote>
      <div className="grid md:grid-cols-2 gap-4">
        {MODELS.map((m, i) => {
          const macro = macroOf(m.id) ?? m.macroAUROC;
          return (
            <Link key={m.id} to={`/models/${m.id}`} className="panel p-5 hover:border-accent/50 transition-colors space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="num text-steel text-xs">{String(i + 1).padStart(2, '0')}</span>
                  <h2 className="font-semibold">{m.name}</h2>
                </div>
                <Badge tone={m.status === 'production' ? 'ok' : m.family === 'OCGAN' ? 'alert' : 'steel'}>{m.status}</Badge>
              </div>
              <p className="text-xs text-steel line-clamp-2">{m.core_idea}</p>
              <div className="flex items-end justify-between">
                <div>
                  <div className="text-[10px] uppercase tracking-widest text-steel">macro auroc</div>
                  <div className="num text-2xl" style={{ color: aurocColor(macro) }}>{macro.toFixed(4)}</div>
                </div>
                <div className="flex gap-0.5 items-end h-8">
                  {CATEGORIES.map((c) => {
                    const r = rowFor(m.id, c);
                    return <div key={c} title={`${c}: ${r?.auroc.toFixed(4) ?? 'n/a'}`}
                      className="w-1.5 rounded-t-sm"
                      style={{ height: `${r ? Math.max(8, (r.auroc - 0.4) / 0.6 * 100) : 6}%`,
                               background: r ? aurocColor(r.auroc) : 'var(--color-panel2)' }} />;
                  })}
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
