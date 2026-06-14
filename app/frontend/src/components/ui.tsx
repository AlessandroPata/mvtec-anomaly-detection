import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';

export function Spinner({ className = 'w-6 h-6' }: { className?: string }) {
  return <div className={`${className} border-2 border-accent border-t-transparent rounded-full animate-spin`} aria-label="loading" />;
}

export function Badge({ children, tone = 'steel' }: { children: ReactNode; tone?: 'ok' | 'alert' | 'warn' | 'accent' | 'steel' }) {
  const tones = {
    ok: 'bg-ok/15 text-ok', alert: 'bg-alert/15 text-alert', warn: 'bg-warn/15 text-warn',
    accent: 'bg-accent/15 text-accent', steel: 'bg-steel/15 text-steel',
  } as const;
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium tracking-wide uppercase ${tones[tone]}`}>{children}</span>;
}

export function Section({ title, sub, children, right }: { title: string; sub?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="space-y-4">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
          {sub && <p className="text-sm text-steel mt-1">{sub}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function StatCard({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="panel p-5">
      <div className="text-xs uppercase tracking-widest text-steel">{label}</div>
      <div className="num text-3xl font-semibold mt-2">{value}</div>
      {sub && <div className="text-xs text-steel mt-1">{sub}</div>}
    </div>
  );
}

export function OfflineCard({ what, onRetry }: { what: string; onRetry?: () => void }) {
  return (
    <div className="panel p-8 text-center space-y-3">
      <div className="text-warn text-sm font-medium">Backend offline</div>
      <p className="text-steel text-sm max-w-md mx-auto">
        {what} needs the inference server. Start it with{' '}
        <code className="num text-fog bg-panel2 px-1.5 py-0.5 rounded">python server.py --device auto</code>{' '}
        in <code className="num">ocgan-modernized/</code>.
      </p>
      {onRetry && <button onClick={onRetry} className="text-accent text-sm hover:underline">Retry</button>}
    </div>
  );
}

export function CTA({ to, children }: { to: string; children: ReactNode }) {
  return (
    <Link to={to} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 transition-colors text-sm font-medium">
      {children}
    </Link>
  );
}
