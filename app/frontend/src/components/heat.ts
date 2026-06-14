import type { Verdict } from '../types/api';

/** AUROC 0.5..1.0 → hex color, red→amber→emerald, clamped. */
export function aurocColor(v: number): string {
  const t = Math.max(0, Math.min(1, (v - 0.5) / 0.5));
  const stops: [number, [number, number, number]][] = [
    [0.0, [127, 29, 29]],
    [0.5, [180, 110, 30]],
    [0.8, [22, 101, 52]],
    [1.0, [16, 185, 129]],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break; }
  }
  const f = hi[0] === lo[0] ? 0 : (t - lo[0]) / (hi[0] - lo[0]);
  const rgb = lo[1].map((c, i) => Math.round(c + f * (hi[1][i] - c)));
  return `#${rgb.map((c) => c.toString(16).padStart(2, '0')).join('')}`;
}

export function verdictClasses(v: Verdict): { border: string; chip: string; label: string } {
  switch (v) {
    case 'tp': return { border: 'border-ok', chip: 'bg-ok/15 text-ok', label: 'TP — defect caught' };
    case 'tn': return { border: 'border-ok', chip: 'bg-ok/15 text-ok', label: 'TN — good confirmed' };
    case 'fp': return { border: 'border-alert', chip: 'bg-alert/15 text-alert', label: 'FP — false alarm' };
    case 'fn': return { border: 'border-alert', chip: 'bg-alert/15 text-alert', label: 'FN — defect missed' };
    default:   return { border: 'border-warn', chip: 'bg-warn/15 text-warn', label: 'Error' };
  }
}
