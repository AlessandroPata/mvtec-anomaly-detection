import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { aurocColor } from '../../components/heat';
import { EVAL_MODELS, rowFor } from '../../data/models';

export function CategoryBars({ category }: { category: string }) {
  const data = EVAL_MODELS
    .map((m) => ({ name: m.name, auroc: rowFor(m.id, category)?.auroc }))
    .filter((d): d is { name: string; auroc: number } => d.auroc !== undefined);
  return (
    <div className="panel p-4">
      <h3 className="text-sm font-medium mb-3">All models on <span className="text-accent">{category}</span></h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="var(--color-line)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: 'var(--color-steel)', fontSize: 11 }} interval={0} angle={-18} textAnchor="end" height={52} />
          <YAxis domain={[0.4, 1]} tick={{ fill: 'var(--color-steel)', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: 'var(--color-panel)', border: '1px solid var(--color-line)', borderRadius: 8 }}
            formatter={(v) => Number(v).toFixed(4)} />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
            {data.map((d) => <Cell key={d.name} fill={aurocColor(d.auroc)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
