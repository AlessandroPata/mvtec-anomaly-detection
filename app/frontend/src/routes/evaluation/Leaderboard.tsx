import { Link } from 'react-router-dom';
import { Badge } from '../../components/ui';
import { archOf, type MetricKey } from '../../data/models';
import { leaderboardRows } from './leaderboard-rows';

export function Leaderboard({ metric }: { metric: MetricKey }) {
  const rows = leaderboardRows(metric);
  const best = rows[0]?.value ?? 1;
  const worst = rows[rows.length - 1]?.value ?? 0;
  const span = Math.abs(best - worst) || 1;
  return (
    <div className="panel overflow-hidden">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-steel uppercase tracking-wider">
          <tr className="border-b border-line">
            <th className="px-4 py-3">#</th><th className="px-4 py-3">Model</th>
            <th className="px-4 py-3">Family</th><th className="px-4 py-3 w-1/3">Macro {metric.toUpperCase()}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id} className="border-b border-line/50 hover:bg-panel2 transition-colors">
              <td className="px-4 py-3 num text-steel">{i + 1}</td>
              <td className="px-4 py-3">
                {r.archId
                  ? <Link to={`/models/${r.archId}`} className="hover:text-accent">{r.name}</Link>
                  : <span>{r.name}</span>}{' '}
                {r.archId && archOf(r.archId)?.status === 'production' && <Badge tone="ok">prod</Badge>}
              </td>
              <td className="px-4 py-3 text-steel">{r.family}</td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className="flex-1 h-1.5 rounded bg-panel2">
                    <div className="h-full rounded bg-accent" style={{ width: `${10 + 90 * Math.abs(r.value - worst) / span}%` }} />
                  </div>
                  <span className="num w-16 text-right">{r.value.toFixed(4)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
