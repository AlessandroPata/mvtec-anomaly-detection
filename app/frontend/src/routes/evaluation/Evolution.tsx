import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { macroOf, MODELS } from '../../data/models';

export function Evolution() {
  const data = MODELS.map((m) => ({ name: m.shortName, date: m.date, macro: macroOf(m.id) ?? m.macroAUROC }));
  return (
    <div className="panel p-4">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="var(--color-line)" strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fill: 'var(--color-steel)', fontSize: 11 }} interval={0} angle={-18} textAnchor="end" height={52} />
          <YAxis domain={[0.7, 1]} tick={{ fill: 'var(--color-steel)', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: 'var(--color-panel)', border: '1px solid var(--color-line)', borderRadius: 8 }}
            formatter={(v) => Number(v).toFixed(4)} labelFormatter={(l, p) => `${l} · ${(p?.[0]?.payload as { date?: string })?.date ?? ''}`} />
          <ReferenceLine y={0.9846} stroke="var(--color-ok)" strokeDasharray="4 4"
            label={{ value: 'production 0.9846', fill: 'var(--color-ok)', fontSize: 11, position: 'insideTopRight' }} />
          <Line type="monotone" dataKey="macro" stroke="var(--color-accent)" strokeWidth={2}
            dot={{ fill: 'var(--color-accent)', r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
