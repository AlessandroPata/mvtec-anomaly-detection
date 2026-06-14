import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Activity, BarChart3, BookOpenText, Boxes, Home, Layers, Swords } from 'lucide-react';
import { useHealth } from '../stores/health';

const NAV = [
  { to: '/', label: 'Overview', icon: Home, end: true },
  { to: '/models', label: 'Models', icon: Layers },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
  { to: '/arena', label: 'Test Arena', icon: Swords },
  { to: '/dataset', label: 'Dataset', icon: Boxes },
  { to: '/methodology', label: 'Methodology', icon: BookOpenText },
];

export function AppShell() {
  const { online, start } = useHealth();
  const location = useLocation();
  useEffect(() => start(), [start]);
  return (
    <div className="min-h-screen flex">
      <aside className="w-56 shrink-0 border-r border-line bg-panel/60 backdrop-blur sticky top-0 h-screen flex flex-col">
        <div className="p-5 border-b border-line">
          <div className="font-semibold tracking-widest text-sm">MVTEC·AD <span className="text-accent">LAB</span></div>
          <div className="text-[11px] text-steel mt-1">anomaly detection showcase</div>
        </div>
        <nav className="p-3 space-y-1 flex-1">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive ? 'bg-accent/10 text-accent' : 'text-steel hover:text-fog hover:bg-panel2'}`}>
              <Icon size={16} /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-line flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${online ? 'bg-ok animate-pulse' : online === false ? 'bg-alert' : 'bg-steel'}`} />
          <span className="text-steel flex items-center gap-1">
            <Activity size={12} /> {online ? 'inference online' : online === false ? 'inference offline' : 'checking…'}
          </span>
        </div>
      </aside>
      <main className="flex-1 min-w-0">
        <AnimatePresence mode="wait">
          <motion.div key={location.pathname} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.18 }} className="max-w-6xl mx-auto px-8 py-10 space-y-12">
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
