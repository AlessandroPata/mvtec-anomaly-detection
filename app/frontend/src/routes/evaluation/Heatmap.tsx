import { useState } from 'react';
import { aurocColor } from '../../components/heat';
import { CATEGORIES, EVAL_MODELS, rowFor } from '../../data/models';
import { CategoryBars } from './CategoryBars';

export function Heatmap() {
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <div className="space-y-6">
      <div className="panel p-4 overflow-x-auto">
        <div className="grid gap-px" style={{ gridTemplateColumns: `9rem repeat(${CATEGORIES.length}, minmax(2.4rem, 1fr))` }}>
          <div />
          {CATEGORIES.map((c) => (
            <button key={c} onClick={() => setSelected(c === selected ? null : c)}
              className={`text-[10px] py-1 truncate px-0.5 ${selected === c ? 'text-accent' : 'text-steel hover:text-fog'}`}
              title={c}>{c.replace('_', ' ')}</button>
          ))}
          {EVAL_MODELS.map((m) => <Row key={m.id} modelId={m.id} name={m.name} selected={selected} onSelect={setSelected} />)}
        </div>
        <div className="flex items-center gap-2 mt-3 text-[10px] text-steel">
          <span>AUROC</span>
          {[0.5, 0.7, 0.85, 0.95, 1].map((v) => (
            <span key={v} className="flex items-center gap-1">
              <span className="w-4 h-3 rounded-sm inline-block" style={{ background: aurocColor(v) }} />{v.toFixed(2)}
            </span>
          ))}
          <span className="ml-auto">click a cell or header to drill into a category</span>
        </div>
      </div>
      {selected && <CategoryBars category={selected} />}
    </div>
  );
}

function Row({ modelId, name, selected, onSelect }: {
  modelId: string; name: string; selected: string | null; onSelect: (c: string) => void;
}) {
  return (
    <>
      <div className="text-xs text-steel pr-2 py-1 truncate self-center" title={name}>{name}</div>
      {CATEGORIES.map((c) => {
        const row = rowFor(modelId, c);
        return (
          <button key={c} onClick={() => onSelect(c)}
            className={`h-8 rounded-sm transition-transform hover:scale-110 hover:z-10 ${selected === c ? 'ring-1 ring-accent' : ''}`}
            style={{ background: row ? aurocColor(row.auroc) : 'var(--color-panel2)' }}
            title={row ? `${name} · ${c}: ${row.auroc.toFixed(4)}` : `${name} · ${c}: n/a`}>
            <span className="sr-only">{row?.auroc.toFixed(4) ?? 'n/a'}</span>
          </button>
        );
      })}
    </>
  );
}
