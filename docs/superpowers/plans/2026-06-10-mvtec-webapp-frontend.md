# MVTec Webapp — Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `frontend/src` into a polished 6-page showcase (Home, Models, Evaluation, Arena, Dataset, Methodology) on the existing stack, consuming the backend from the companion plan (`2026-06-10-mvtec-webapp-backend.md`).

**Architecture:** React 19 + Vite 8 + TypeScript + Tailwind 4 (dark industrial "QC instrument" theme), react-router 7, Recharts for charts, Framer Motion for motion, Zustand for the arena store. Static pages read bundled JSON (`src/data/*`); live pages call `/api` (Vite proxy → :8000, already configured). The arena consumes SSE with polling fallback.

**Tech Stack:** existing `frontend/package.json` deps (react 19, react-router-dom 7, recharts 3, framer-motion 12, zustand 5, lucide-react, react-dropzone, tailwindcss 4) + dev: vitest, jsdom, @testing-library/react, @testing-library/jest-dom.

**Working directory for all commands:** `D:\OCGAN\project\storage_project_outputs_datasets\project\frontend` (PowerShell).

**Prerequisite:** backend plan executed at least through Task 8 (benchmarks.json + insights.json regenerated; server endpoints live for manual checks).

**Conventions:**
- UI language: **English**. AUROC as `0.9846` (4 dp) in tables, `98.5%` only in hero stats. Monospace (`font-mono`) for metric digits.
- Verdict colors: TP/TN `text-ok` (emerald), FP/FN `text-alert` (red); threshold `text-warn` (amber); interactive accent cyan.
- Keep existing `src/data/architectures.ts` and `src/types/domain.ts` (curated model cards — reused as-is). Everything else in `src/` is rebuilt.

---

### Task 0: Repo init, deps, clean slate

- [ ] **Step 0.1: git init + baseline commit of the draft (preserves what existed)**

```powershell
git init
git add -A
git commit -m "chore: baseline — draft frontend before rebuild"
```
Expected: repo created, 1 commit. Check `.gitignore` covers `node_modules/` and `dist/` first (the Vite template ships one).

- [ ] **Step 0.2: Install deps + test tooling**

```powershell
npm install
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom
```
Expected: clean install against react 19 (testing-library ≥16 supports it).

- [ ] **Step 0.3: Wire vitest into `vite.config.ts`** (keep the existing proxy!)

```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    globals: true,
  },
})
```
Create `src/test/setup.ts`:
```ts
import '@testing-library/jest-dom/vitest';
```
Add scripts to `package.json`: `"test": "vitest run", "test:watch": "vitest"`.

- [ ] **Step 0.4: Remove the draft routes/components (keep data + domain types)**

```powershell
Remove-Item -Recurse -Force src\routes, src\components, src\services
Remove-Item -Force src\types\api.ts, src\App.tsx, src\index.css, src\main.tsx
```
Keep: `src/data/architectures.ts`, `src/data/benchmarks.json`, `src/data/insights.json` (from backend Task 8), `src/types/domain.ts`, `src/test/setup.ts`.

- [ ] **Step 0.5: Commit**

```powershell
git add -A
git commit -m "chore: test tooling + clean slate (kept curated data and domain types)"
```

---

### Task 1: Theme, fonts, app entry

**Files:**
- Create: `src/index.css`, `src/main.tsx`, `src/App.tsx` + stub pages
- Modify: `index.html`

- [ ] **Step 1.1: `index.html`** — title, fonts, favicon

```html
<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔍</text></svg>" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MVTec AD Lab — Anomaly Detection Showcase</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 1.2: `src/index.css`** — Tailwind 4 theme tokens + base

```css
@import "tailwindcss";

@theme {
  --color-ink: #0b0f14;
  --color-panel: #11161d;
  --color-panel2: #18202b;
  --color-line: #233041;
  --color-steel: #8b98a8;
  --color-fog: #e6edf3;
  --color-ok: #34d399;
  --color-alert: #f87171;
  --color-warn: #fbbf24;
  --color-accent: #22d3ee;
  --font-display: "Space Grotesk", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;
}

@layer base {
  html { background: var(--color-ink); color-scheme: dark; }
  body { @apply bg-ink text-fog font-display antialiased; }
  ::selection { background: color-mix(in srgb, var(--color-accent) 30%, transparent); }
}

@layer utilities {
  .blueprint {
    background-image:
      linear-gradient(var(--color-line) 1px, transparent 1px),
      linear-gradient(90deg, var(--color-line) 1px, transparent 1px);
    background-size: 32px 32px;
  }
  .num { @apply font-mono tabular-nums; }
  .panel { @apply bg-panel border border-line rounded-xl; }
}
```

- [ ] **Step 1.3: `src/main.tsx`**

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 1.4: `src/App.tsx`** — router with lazy pages; create one-line stub default exports for every referenced page plus stub `src/components/AppShell.tsx` / `src/components/ui.tsx` so the build is green from day one (real versions land in Tasks 2-4)

```tsx
import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import { Spinner } from './components/ui';

const Home = lazy(() => import('./routes/Home'));
const Models = lazy(() => import('./routes/models/ModelGallery'));
const ModelDetail = lazy(() => import('./routes/models/ModelDetail'));
const Evaluation = lazy(() => import('./routes/evaluation/EvaluationLab'));
const Arena = lazy(() => import('./routes/arena/Arena'));
const Dataset = lazy(() => import('./routes/dataset/DatasetExplorer'));
const DatasetCategory = lazy(() => import('./routes/dataset/DatasetCategory'));
const Methodology = lazy(() => import('./routes/Methodology'));

const page = (el: React.ReactNode) => (
  <Suspense fallback={<div className="flex justify-center py-24"><Spinner /></div>}>{el}</Suspense>
);

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: page(<Home />) },
      { path: 'models', element: page(<Models />) },
      { path: 'models/:id', element: page(<ModelDetail />) },
      { path: 'evaluation', element: page(<Evaluation />) },
      { path: 'arena', element: page(<Arena />) },
      { path: 'dataset', element: page(<Dataset />) },
      { path: 'dataset/:category', element: page(<DatasetCategory />) },
      { path: 'methodology', element: page(<Methodology />) },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
```

- [ ] **Step 1.5: Verify build + commit**

```powershell
npm run build
git add -A ; git commit -m "feat: theme tokens, fonts, router skeleton"
```

---

### Task 2: Types, API client, health store, data module

**Files:**
- Create: `src/types/api.ts`, `src/services/api.ts`, `src/stores/health.ts`, `src/data/models.ts`

- [ ] **Step 2.1: `src/types/api.ts`** (mirrors backend responses exactly)

```ts
export interface VariantInfo {
  id: string; label: string; kind: 'production' | 'reconstructed';
  aggregation: string | null; topk: number | null; coreset: number | null;
  description: string; available: boolean; approximate: boolean;
}
export interface DefectTypeInfo { name: string; count: number; is_anomaly: boolean }
export interface CategoryMeta { name: string; test_total: number; defect_types: DefectTypeInfo[]; variants: VariantInfo[] }
export interface Meta { categories: CategoryMeta[]; device: string; dataset_available: boolean }

export interface ArenaImage { idx: number; defect_type: string; filename: string; ground_truth_anomaly: boolean }
export interface ArenaStartResponse { job_id: string; seed: number; n: number; category: string; variant: string; images: ArenaImage[] }
export type Verdict = 'tp' | 'tn' | 'fp' | 'fn' | 'error';
export interface ArenaResult {
  idx: number; defect_type: string; filename: string; ground_truth_anomaly: boolean;
  anomaly_score?: number; anomaly_probability?: number | null; is_anomaly?: boolean;
  threshold?: number; inference_ms?: number; verdict: Verdict; correct?: boolean; error?: string;
}
export interface ArenaSummary {
  n: number; errors: number; accuracy: number | null; precision: number | null;
  recall: number | null; f1: number | null; auroc: number | null;
  confusion: { tp: number; tn: number; fp: number; fn: number };
  mean_ms: number | null; p95_ms: number | null;
}
export interface ArenaFinal { status: 'done' | 'cancelled' | 'error'; summary: ArenaSummary | null; error: string | null }
export interface ArenaPoll { job_id: string; status: string; results: ArenaResult[]; summary: ArenaSummary | null; error: string | null; total: number; done: number }

export interface PredictionResponse {
  anomaly_score: number; anomaly_probability: number | null; is_anomaly: boolean;
  threshold: number; category: string; inference_ms: number;
  heatmap_base64: string; model_variant?: string;
  defect_type?: string; filename?: string; ground_truth_anomaly?: boolean;
}
export interface HealthResponse { status: string; models_loaded: string[]; backend: string }
export interface TestImagesResponse { category: string; defect_types: { defect_type: string; is_anomaly: boolean; count: number; images: string[] }[] }
```

- [ ] **Step 2.2: `src/services/api.ts`**

```ts
import type {
  ArenaFinal, ArenaPoll, ArenaResult, ArenaStartResponse, HealthResponse,
  Meta, PredictionResponse, TestImagesResponse,
} from '../types/api';

const BASE = '/api';

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = (body as { detail?: unknown }).detail;
    throw new Error(typeof detail === 'string' ? detail : res.statusText);
  }
  return res.json() as Promise<T>;
}

export const fetchHealth = async () => jsonOrThrow<HealthResponse>(await fetch(`${BASE}/health`));
export const fetchMeta = async () => jsonOrThrow<Meta>(await fetch(`${BASE}/meta`));
export const fetchTestImages = async (cat: string) =>
  jsonOrThrow<TestImagesResponse>(await fetch(`${BASE}/dataset/test-images?cat=${encodeURIComponent(cat)}`));

export const thumbUrl = (cat: string, defect: string, filename: string, size = 128) =>
  `${BASE}/dataset/thumb?cat=${encodeURIComponent(cat)}&defect=${encodeURIComponent(defect)}&filename=${encodeURIComponent(filename)}&size=${size}`;
export const sampleUrl = (cat: string, defect: string, filename: string) =>
  `${BASE}/dataset/sample?cat=${encodeURIComponent(cat)}&defect=${encodeURIComponent(defect)}&filename=${encodeURIComponent(filename)}`;
export const maskUrl = (cat: string, defect: string, filename: string) =>
  `${BASE}/dataset/mask?cat=${encodeURIComponent(cat)}&defect=${encodeURIComponent(defect)}&filename=${encodeURIComponent(filename)}`;

export async function startArena(body: { category: string; variant: string; n_images: number; seed: number | null }) {
  return jsonOrThrow<ArenaStartResponse>(await fetch(`${BASE}/arena/start`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, seed: body.seed ?? undefined }),
  }));
}
export const cancelArena = async (jobId: string) =>
  fetch(`${BASE}/arena/jobs/${jobId}/cancel`, { method: 'POST' });
export const pollArena = async (jobId: string, since: number) =>
  jsonOrThrow<ArenaPoll>(await fetch(`${BASE}/arena/jobs/${jobId}?since=${since}`));

/** SSE subscription; returns an unsubscribe fn. Caller handles fallback on disconnect. */
export function streamArena(
  jobId: string, since: number,
  on: { result: (r: ArenaResult) => void; final: (f: ArenaFinal) => void; disconnect: () => void },
): () => void {
  const es = new EventSource(`${BASE}/arena/jobs/${jobId}/stream?since=${since}`);
  es.addEventListener('result', (e) => on.result(JSON.parse((e as MessageEvent).data)));
  es.addEventListener('summary', (e) => { on.final(JSON.parse((e as MessageEvent).data)); es.close(); });
  es.onerror = () => { es.close(); on.disconnect(); };
  return () => es.close();
}

export async function predictFromDataset(category: string, defect: string, filename: string, variant: string) {
  const fd = new FormData();
  fd.set('category', category); fd.set('defect', defect); fd.set('filename', filename);
  fd.set('model_variant', variant);
  return jsonOrThrow<PredictionResponse>(await fetch(`${BASE}/predict/from-dataset`, { method: 'POST', body: fd }));
}
export async function predictUpload(file: File, category: string, variant: string) {
  const fd = new FormData();
  fd.set('file', file, file.name); fd.set('category', category); fd.set('model_variant', variant);
  return jsonOrThrow<PredictionResponse>(await fetch(`${BASE}/predict`, { method: 'POST', body: fd }));
}
```

- [ ] **Step 2.3: `src/stores/health.ts`**

```ts
import { create } from 'zustand';
import { fetchHealth, fetchMeta } from '../services/api';
import type { Meta } from '../types/api';

interface HealthState {
  online: boolean | null;          // null = checking
  meta: Meta | null;
  start: () => void;
}

let timer: number | undefined;

export const useHealth = create<HealthState>((set, get) => ({
  online: null,
  meta: null,
  start: () => {
    if (timer !== undefined) return;
    const tick = async () => {
      try {
        await fetchHealth();
        if (!get().meta) set({ meta: await fetchMeta() });
        set({ online: true });
      } catch {
        set({ online: false });
      }
    };
    void tick();
    timer = window.setInterval(tick, 20_000);
  },
}));
```

- [ ] **Step 2.4: `src/data/models.ts`** — joins curated cards + regenerated benchmarks

```ts
import rawBenchmarks from './benchmarks.json';
import { ARCHITECTURES } from './architectures';
import type { Architecture } from '../types/domain';

export interface BenchmarkRow {
  category: string; auroc: number; auroc_std?: number; auprc: number; best_f1: number;
  fpr95: number; elapsed_s: number | null; n_seeds: number;
  feature_level: string | null; aggregation: string | null; topk: number | null; coreset: number | null;
}
export interface Benchmarks { per_category: Record<string, BenchmarkRow[]>; macro: Record<string, number> }

export const BENCHMARKS = rawBenchmarks as unknown as Benchmarks;

export const CATEGORIES = [
  'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
] as const;

/** Chronological (curated date field). */
export const MODELS: Architecture[] = [...ARCHITECTURES].sort((a, b) => a.date.localeCompare(b.date));

export const macroOf = (id: string): number | undefined => BENCHMARKS.macro[id];
export const rowsOf = (id: string): BenchmarkRow[] => BENCHMARKS.per_category[id] ?? [];
export const rowFor = (id: string, category: string): BenchmarkRow | undefined =>
  rowsOf(id).find((r) => r.category === category);
export const archOf = (id: string): Architecture | undefined => MODELS.find((m) => m.id === id);

export type MetricKey = 'auroc' | 'auprc' | 'best_f1' | 'fpr95';
export const METRICS: { key: MetricKey; label: string; higherIsBetter: boolean }[] = [
  { key: 'auroc', label: 'AUROC', higherIsBetter: true },
  { key: 'auprc', label: 'AUPRC', higherIsBetter: true },
  { key: 'best_f1', label: 'Best F1', higherIsBetter: true },
  { key: 'fpr95', label: 'FPR@95TPR', higherIsBetter: false },
];

export function macroMetric(id: string, metric: MetricKey): number | undefined {
  const rows = rowsOf(id);
  if (!rows.length) return undefined;
  return rows.reduce((s, r) => s + r[metric], 0) / rows.length;
}
```
Check `src/types/domain.ts` exports `Architecture` with the fields used here and in Tasks 4/6 (`id, date, name, shortName, family, status, macroAUROC, architecture_type, core_idea, strengths, weaknesses, improvements, hyperparameters, pipeline, notes`). If a name differs, adapt at the usage site — do not edit the curated data file.

- [ ] **Step 2.5: Verify + commit**

```powershell
npx tsc -b
git add -A ; git commit -m "feat: api client, health store, benchmark data layer"
```

---

### Task 3: Shared UI components (+ first unit tests)

**Files:**
- Create: `src/components/ui.tsx` (replace stub), `src/components/heat.ts`, `src/components/CountUp.tsx`, `src/components/Modal.tsx`, `src/components/ConfusionMatrix.tsx`, `src/components/Sparkline.tsx`, `src/components/ScoreGauge.tsx`
- Create: `src/components/__tests__/heat.test.ts`, `src/components/__tests__/ConfusionMatrix.test.tsx`

- [ ] **Step 3.1: Failing tests first**

`src/components/__tests__/heat.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { aurocColor, verdictClasses } from '../heat';

describe('aurocColor', () => {
  it('clamps endpoints', () => {
    expect(aurocColor(0.2)).toBe(aurocColor(0.5));
    expect(aurocColor(1.3)).toBe(aurocColor(1.0));
  });
  it('is greener for higher auroc', () => {
    const g = (c: string) => parseInt(c.slice(3, 5), 16);
    expect(g(aurocColor(0.95))).toBeGreaterThanOrEqual(g(aurocColor(0.6)));
  });
});

describe('verdictClasses', () => {
  it('marks correct verdicts ok and wrong ones alert', () => {
    expect(verdictClasses('tp').border).toContain('ok');
    expect(verdictClasses('tn').border).toContain('ok');
    expect(verdictClasses('fp').border).toContain('alert');
    expect(verdictClasses('fn').border).toContain('alert');
    expect(verdictClasses('error').border).toContain('warn');
  });
});
```
`src/components/__tests__/ConfusionMatrix.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ConfusionMatrix } from '../ConfusionMatrix';

describe('ConfusionMatrix', () => {
  it('renders the four cells with counts', () => {
    render(<ConfusionMatrix confusion={{ tp: 41, tn: 52, fp: 3, fn: 4 }} />);
    expect(screen.getByText('41')).toBeInTheDocument();
    expect(screen.getByText('52')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText(/true positive/i)).toBeInTheDocument();
  });
});
```
Run: `npm test` → Expected: FAIL (modules missing).

- [ ] **Step 3.2: `src/components/heat.ts`**

```ts
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
```

- [ ] **Step 3.3: `src/components/ui.tsx`** (atoms — replaces the Task 1 stub)

```tsx
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
```

- [ ] **Step 3.4: `src/components/CountUp.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react';

export function CountUp({ value, decimals = 0, suffix = '', duration = 900 }: {
  value: number; decimals?: number; suffix?: string; duration?: number;
}) {
  const [shown, setShown] = useState(0);
  const raf = useRef(0);
  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) { setShown(value); return; }
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / duration);
      setShown(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [value, duration]);
  return <span>{shown.toFixed(decimals)}{suffix}</span>;
}
```

- [ ] **Step 3.5: `src/components/Modal.tsx`**

```tsx
import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import type { ReactNode } from 'react';

export function Modal({ open, onClose, children, wide = false }: {
  open: boolean; onClose: () => void; children: ReactNode; wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = ''; };
  }, [open, onClose]);
  if (!open) return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className={`relative panel p-6 max-h-[90vh] overflow-y-auto w-full ${wide ? 'max-w-4xl' : 'max-w-xl'}`}>
        <button onClick={onClose} aria-label="Close" className="absolute top-3 right-3 text-steel hover:text-fog"><X size={18} /></button>
        {children}
      </div>
    </div>,
    document.body,
  );
}
```

- [ ] **Step 3.6: `src/components/ConfusionMatrix.tsx`**

```tsx
export function ConfusionMatrix({ confusion }: { confusion: { tp: number; tn: number; fp: number; fn: number } }) {
  const cell = 'rounded-lg p-4 text-center';
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center text-sm">
      <div />
      <div className="text-center text-xs text-steel">Predicted anomaly</div>
      <div className="text-center text-xs text-steel">Predicted good</div>
      <div className="text-xs text-steel [writing-mode:vertical-rl] rotate-180">Actual anomaly</div>
      <div className={`${cell} bg-ok/10 border border-ok/30`}>
        <div className="num text-2xl text-ok">{confusion.tp}</div>
        <div className="text-xs text-steel mt-1">True positive</div>
      </div>
      <div className={`${cell} bg-alert/10 border border-alert/30`}>
        <div className="num text-2xl text-alert">{confusion.fn}</div>
        <div className="text-xs text-steel mt-1">False negative</div>
      </div>
      <div className="text-xs text-steel [writing-mode:vertical-rl] rotate-180">Actual good</div>
      <div className={`${cell} bg-alert/10 border border-alert/30`}>
        <div className="num text-2xl text-alert">{confusion.fp}</div>
        <div className="text-xs text-steel mt-1">False positive</div>
      </div>
      <div className={`${cell} bg-ok/10 border border-ok/30`}>
        <div className="num text-2xl text-ok">{confusion.tn}</div>
        <div className="text-xs text-steel mt-1">True negative</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3.7: `src/components/Sparkline.tsx`** + **`src/components/ScoreGauge.tsx`**

```tsx
// Sparkline.tsx
export function Sparkline({ values, width = 120, height = 32, stroke = 'var(--color-accent)' }: {
  values: number[]; width?: number; height?: number; stroke?: string;
}) {
  if (values.length < 2) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * width},${height - 3 - ((v - min) / span) * (height - 6)}`).join(' ');
  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline points={pts} fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
```
```tsx
// ScoreGauge.tsx — score vs threshold on a horizontal band
export function ScoreGauge({ score, threshold, max }: { score: number; threshold: number; max?: number }) {
  const hi = max ?? Math.max(score, threshold) * 1.4;
  const pct = (v: number) => Math.max(0, Math.min(100, (v / hi) * 100));
  return (
    <div className="space-y-1">
      <div className="relative h-3 rounded-full bg-panel2 border border-line overflow-hidden">
        <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-ok/50 to-alert/60" style={{ width: `${pct(score)}%` }} />
        <div className="absolute inset-y-0 w-0.5 bg-warn" style={{ left: `${pct(threshold)}%` }} title="threshold" />
      </div>
      <div className="flex justify-between text-xs text-steel num">
        <span>score {score.toFixed(4)}</span>
        <span className="text-warn">thr {threshold.toFixed(4)}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 3.8: Run tests + build, commit**

```powershell
npm test
npm run build
git add -A ; git commit -m "feat: shared UI atoms, heat scale, confusion matrix, gauges (tested)"
```
Expected: vitest green, build green.

---

### Task 4: AppShell + Home

**Files:**
- Create: `src/components/AppShell.tsx`, `src/routes/Home.tsx` (replace stubs)

- [ ] **Step 4.1: `src/components/AppShell.tsx`**

```tsx
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
```

- [ ] **Step 4.2: `src/routes/Home.tsx`**

```tsx
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { CountUp } from '../components/CountUp';
import { Sparkline } from '../components/Sparkline';
import { Badge, CTA, StatCard } from '../components/ui';
import { MODELS, macroOf, rowsOf } from '../data/models';

export default function Home() {
  const journey = MODELS.map((m) => ({ ...m, macro: macroOf(m.id) ?? m.macroAUROC }));
  const finalMacro = macroOf('production_final') ?? 0.9846;
  const perfect = rowsOf('production_final').filter((r) => r.auroc >= 0.9999).length;
  const gain = (finalMacro - (macroOf('ocgan_v1') ?? 0.7866)) * 100;

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
```
If `Architecture.status` values differ from `'production'` (check `src/types/domain.ts`), adapt the badge condition.

- [ ] **Step 4.3: Visual check + commit**

```powershell
npm run dev   # sidebar + LED, hero, 4 count-up stats, 7 journey cards in date order
npm run build ; git add -A ; git commit -m "feat: app shell with health LED + home (hero, stats, journey)"
```

---

### Task 5: Evaluation Lab

**Files:**
- Create: `src/routes/evaluation/EvaluationLab.tsx` (replace stub), `src/routes/evaluation/Leaderboard.tsx`, `src/routes/evaluation/Heatmap.tsx`, `src/routes/evaluation/Evolution.tsx`, `src/routes/evaluation/CategoryBars.tsx`, `src/routes/evaluation/Insights.tsx`
- Create: `src/routes/evaluation/__tests__/leaderboard.test.ts`

- [ ] **Step 5.1: Failing test for the rank helper**

`src/routes/evaluation/__tests__/leaderboard.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { leaderboardRows } from '../Leaderboard';

describe('leaderboardRows', () => {
  it('ranks desc for higher-is-better metrics', () => {
    const rows = leaderboardRows('auroc');
    expect(rows.length).toBeGreaterThanOrEqual(6);
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i - 1].value).toBeGreaterThanOrEqual(rows[i].value);
    }
  });
  it('ranks asc for fpr95 (lower is better)', () => {
    const rows = leaderboardRows('fpr95');
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i - 1].value).toBeLessThanOrEqual(rows[i].value);
    }
  });
});
```
Run `npm test` → FAIL (module missing).

- [ ] **Step 5.2: `Leaderboard.tsx`**

```tsx
import { Link } from 'react-router-dom';
import { Badge } from '../../components/ui';
import { archOf, macroMetric, MODELS, METRICS, type MetricKey } from '../../data/models';

export function leaderboardRows(metric: MetricKey) {
  const meta = METRICS.find((m) => m.key === metric)!;
  return MODELS
    .map((m) => ({ id: m.id, name: m.shortName, family: m.family, value: macroMetric(m.id, metric) }))
    .filter((r): r is typeof r & { value: number } => r.value !== undefined)
    .sort((a, b) => (meta.higherIsBetter ? b.value - a.value : a.value - b.value));
}

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
                <Link to={`/models/${r.id}`} className="hover:text-accent">{r.name}</Link>{' '}
                {archOf(r.id)?.status === 'production' && <Badge tone="ok">prod</Badge>}
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
```
Run `npm test` → PASS.

- [ ] **Step 5.3: `Heatmap.tsx`**

```tsx
import { useState } from 'react';
import { aurocColor } from '../../components/heat';
import { CATEGORIES, MODELS, rowFor } from '../../data/models';
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
          {MODELS.map((m) => <Row key={m.id} modelId={m.id} name={m.shortName} selected={selected} onSelect={setSelected} />)}
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
```

- [ ] **Step 5.4: `CategoryBars.tsx`** + **`Evolution.tsx`** (Recharts)

```tsx
// CategoryBars.tsx
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { aurocColor } from '../../components/heat';
import { MODELS, rowFor } from '../../data/models';

export function CategoryBars({ category }: { category: string }) {
  const data = MODELS
    .map((m) => ({ name: m.shortName, auroc: rowFor(m.id, category)?.auroc }))
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
            formatter={(v: number) => v.toFixed(4)} />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
            {data.map((d) => <Cell key={d.name} fill={aurocColor(d.auroc)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```
```tsx
// Evolution.tsx
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
            formatter={(v: number) => v.toFixed(4)} labelFormatter={(l, p) => `${l} · ${(p?.[0]?.payload as { date?: string })?.date ?? ''}`} />
          <ReferenceLine y={0.9846} stroke="var(--color-ok)" strokeDasharray="4 4"
            label={{ value: 'production 0.9846', fill: 'var(--color-ok)', fontSize: 11, position: 'insideTopRight' }} />
          <Line type="monotone" dataKey="macro" stroke="var(--color-accent)" strokeWidth={2}
            dot={{ fill: 'var(--color-accent)', r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 5.5: `Insights.tsx`**

```tsx
import insights from '../../data/insights.json';
import { Sparkline } from '../../components/Sparkline';

interface Delta { category: string; delta: number }
const topDeltas = (rows: Delta[], n = 3) => [...rows].sort((a, b) => b.delta - a.delta).slice(0, n);

export function Insights() {
  const coreset = insights.coreset_effect as Delta[];
  const agg = insights.aggregation_effect as Delta[];
  const layers = insights.layer_ablation as { category: string; configs: Record<string, number> }[];
  const screw = layers.find((l) => l.category === 'screw');

  return (
    <div className="grid md:grid-cols-3 gap-4">
      <article className="panel p-5 space-y-2">
        <h3 className="font-medium text-sm">Don't prune the bank</h3>
        <p className="text-xs text-steel">Full 70k-patch bank vs 10k coreset (same aggregation):</p>
        <ul className="space-y-1">
          {topDeltas(coreset).map((d) => (
            <li key={d.category} className="flex justify-between text-sm">
              <span>{d.category}</span><span className="num text-ok">+{d.delta.toFixed(4)}</span>
            </li>
          ))}
        </ul>
        <Sparkline values={coreset.map((d) => d.delta)} width={200} height={28} stroke="var(--color-ok)" />
      </article>

      <article className="panel p-5 space-y-2">
        <h3 className="font-medium text-sm">Reweighted top-k beats plain top-k</h3>
        <p className="text-xs text-steel">topk_reweighted k=9 vs topk_mean k=3, both on the 10k coreset:</p>
        <ul className="space-y-1">
          {topDeltas(agg).map((d) => (
            <li key={d.category} className="flex justify-between text-sm">
              <span>{d.category}</span><span className="num text-ok">+{d.delta.toFixed(4)}</span>
            </li>
          ))}
        </ul>
        <Sparkline values={agg.map((d) => d.delta)} width={200} height={28} stroke="var(--color-ok)" />
      </article>

      <article className="panel p-5 space-y-2">
        <h3 className="font-medium text-sm">Screw needs layer1 detail</h3>
        <p className="text-xs text-steel">Fine thread defects benefit from earlier features:</p>
        {screw && (
          <ul className="space-y-1">
            {Object.entries(screw.configs).map(([fl, v]) => (
              <li key={fl} className="flex justify-between text-sm">
                <span className="text-steel">{fl}</span><span className="num">{v.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-steel">The only per-category override that made production.</p>
      </article>
    </div>
  );
}
```

- [ ] **Step 5.6: `EvaluationLab.tsx`**

```tsx
import { useState } from 'react';
import { Section } from '../../components/ui';
import { METRICS, type MetricKey } from '../../data/models';
import { Leaderboard } from './Leaderboard';
import { Heatmap } from './Heatmap';
import { Evolution } from './Evolution';
import { Insights } from './Insights';

export default function EvaluationLab() {
  const [metric, setMetric] = useState<MetricKey>('auroc');
  return (
    <div className="space-y-12">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Evaluation Lab</h1>
        <p className="text-steel mt-2">Every model, every category, every metric — side by side.</p>
      </header>

      <Section title="Leaderboard" sub="Macro average across the 15 MVTec categories"
        right={
          <div className="flex rounded-lg border border-line overflow-hidden text-xs">
            {METRICS.map((m) => (
              <button key={m.key} onClick={() => setMetric(m.key)}
                className={`px-3 py-1.5 ${metric === m.key ? 'bg-accent/15 text-accent' : 'text-steel hover:text-fog'}`}>
                {m.label}
              </button>
            ))}
          </div>
        }>
        <Leaderboard metric={metric} />
      </Section>

      <Section title="Macro AUROC evolution" sub="Chronological — the PatchCore jump is the story">
        <Evolution />
      </Section>

      <Section title="Model × category heatmap" sub="Where each architecture wins and where it breaks down">
        <Heatmap />
      </Section>

      <Section title="Ablation insights" sub="The three ingredients behind +19.8 pp, from the tuning logs">
        <Insights />
      </Section>
    </div>
  );
}
```

- [ ] **Step 5.7: Tests + visual check + commit**

```powershell
npm test
npm run dev   # /evaluation: switcher re-sorts; heatmap colored; drilldown bars; insights populated
npm run build ; git add -A ; git commit -m "feat: evaluation lab (leaderboard, evolution, heatmap, insights)"
```

---

### Task 6: Models gallery + detail

**Files:**
- Create: `src/routes/models/ModelGallery.tsx`, `src/routes/models/ModelDetail.tsx` (replace stubs)

- [ ] **Step 6.1: `ModelGallery.tsx`**

```tsx
import { Link } from 'react-router-dom';
import { Badge } from '../../components/ui';
import { aurocColor } from '../../components/heat';
import { CATEGORIES, macroOf, MODELS, rowFor } from '../../data/models';

export default function ModelGallery() {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Models</h1>
        <p className="text-steel mt-2">Seven iterations, two families. Click any card for the full anatomy.</p>
      </header>
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
```

- [ ] **Step 6.2: `ModelDetail.tsx`**

```tsx
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
              <div className={`rounded-lg border px-3 py-2 ${TYPE_TONE[s.type] ?? TYPE_TONE.process}`}>
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
```
Adapt field names to `src/types/domain.ts` if any differ.

- [ ] **Step 6.3: Visual check + commit**

```powershell
npm run dev   # /models: 7 cards with mini bars; details render pipeline, tables, prev/next
npm run build ; git add -A ; git commit -m "feat: model gallery + anatomy detail pages"
```

---

### Task 7: Test Arena (the core demo)

**Files:**
- Create: `src/stores/arena.ts`, `src/routes/arena/Arena.tsx` (replace stub), `src/routes/arena/ConfigPanel.tsx`, `src/routes/arena/LiveGrid.tsx`, `src/routes/arena/SummaryPanel.tsx`, `src/routes/arena/ResultModal.tsx`, `src/routes/arena/SingleTest.tsx`
- Create: `src/stores/__tests__/arena.test.ts`

- [ ] **Step 7.1: Failing store tests**

`src/stores/__tests__/arena.test.ts`:
```ts
import { beforeEach, describe, expect, it } from 'vitest';
import { useArena } from '../arena';
import type { ArenaResult } from '../../types/api';

const result = (idx: number, verdict: ArenaResult['verdict'], correct = verdict === 'tp' || verdict === 'tn'): ArenaResult => ({
  idx, defect_type: 'crack', filename: `${idx}.png`, ground_truth_anomaly: true,
  anomaly_score: 0.9, is_anomaly: true, inference_ms: 12, verdict, correct,
});

describe('arena store', () => {
  beforeEach(() => useArena.getState().reset());

  it('applyResult fills the map and live counters', () => {
    useArena.getState().beginJob('j1', 7, [
      { idx: 0, defect_type: 'crack', filename: '0.png', ground_truth_anomaly: true },
      { idx: 1, defect_type: 'good', filename: '1.png', ground_truth_anomaly: false },
    ]);
    useArena.getState().applyResult(result(0, 'tp'));
    useArena.getState().applyResult(result(1, 'fp', false));
    const st = useArena.getState();
    expect(st.done).toBe(2);
    expect(st.correct).toBe(1);
    expect(st.results[0]?.verdict).toBe('tp');
    expect(st.liveAccuracy).toBeCloseTo(0.5);
  });

  it('duplicate results (SSE reconnect overlap) are idempotent', () => {
    useArena.getState().beginJob('j1', 7, [{ idx: 0, defect_type: 'crack', filename: '0.png', ground_truth_anomaly: true }]);
    useArena.getState().applyResult(result(0, 'tp'));
    useArena.getState().applyResult(result(0, 'tp'));
    expect(useArena.getState().done).toBe(1);
  });

  it('finishJob stores summary and phase', () => {
    useArena.getState().beginJob('j1', 7, []);
    useArena.getState().finishJob({
      status: 'done',
      summary: { n: 0, errors: 0, accuracy: null, precision: null, recall: null, f1: null, auroc: null, confusion: { tp: 0, tn: 0, fp: 0, fn: 0 }, mean_ms: null, p95_ms: null },
      error: null,
    });
    expect(useArena.getState().phase).toBe('done');
    expect(useArena.getState().summary?.n).toBe(0);
  });
});
```
Run `npm test` → FAIL.

- [ ] **Step 7.2: `src/stores/arena.ts`**

```ts
import { create } from 'zustand';
import { cancelArena, pollArena, startArena, streamArena } from '../services/api';
import type { ArenaFinal, ArenaImage, ArenaResult, ArenaSummary } from '../types/api';

export type ArenaPhase = 'idle' | 'starting' | 'running' | 'done' | 'cancelled' | 'error';

interface ArenaConfig { category: string; variant: string; n: number; seed: number | null }

interface ArenaState {
  config: ArenaConfig;
  phase: ArenaPhase;
  jobId: string | null;
  seed: number | null;
  images: ArenaImage[];
  results: Record<number, ArenaResult>;
  done: number; correct: number; liveAccuracy: number;
  summary: ArenaSummary | null;
  error: string | null;
  startedAt: number | null;

  setConfig: (c: Partial<ArenaConfig>) => void;
  start: () => Promise<void>;
  beginJob: (jobId: string, seed: number, images: ArenaImage[]) => void;
  applyResult: (r: ArenaResult) => void;
  finishJob: (f: ArenaFinal) => void;
  cancel: () => void;
  reset: () => void;
}

let unsubscribe: (() => void) | null = null;

function subscribe(jobId: string, get: () => ArenaState) {
  unsubscribe?.();
  unsubscribe = streamArena(jobId, Object.keys(get().results).length, {
    result: (r) => get().applyResult(r),
    final: (f) => get().finishJob(f),
    disconnect: () => {
      const poll = async () => {
        try {
          const st = get();
          if (st.phase !== 'running' || st.jobId !== jobId) return;
          const p = await pollArena(jobId, st.done);
          p.results.forEach((r) => get().applyResult(r));
          if (p.status !== 'running') {
            get().finishJob({ status: p.status as ArenaFinal['status'], summary: p.summary, error: p.error });
            return;
          }
        } catch { /* keep trying until cancelled/unmounted */ }
        window.setTimeout(poll, 800);
      };
      void poll();
    },
  });
}

export const useArena = create<ArenaState>((set, get) => ({
  config: { category: 'bottle', variant: 'production', n: 100, seed: null },
  phase: 'idle', jobId: null, seed: null, images: [], results: {},
  done: 0, correct: 0, liveAccuracy: 0, summary: null, error: null, startedAt: null,

  setConfig: (c) => set((s) => ({ config: { ...s.config, ...c } })),

  start: async () => {
    const { config } = get();
    set({ phase: 'starting', error: null, summary: null, results: {}, done: 0, correct: 0, liveAccuracy: 0 });
    try {
      const res = await startArena({ category: config.category, variant: config.variant, n_images: config.n, seed: config.seed });
      get().beginJob(res.job_id, res.seed, res.images);
      subscribe(res.job_id, get);
    } catch (e) {
      set({ phase: 'error', error: e instanceof Error ? e.message : String(e) });
    }
  },

  beginJob: (jobId, seed, images) => set({
    phase: 'running', jobId, seed, images, results: {}, done: 0, correct: 0,
    liveAccuracy: 0, summary: null, error: null, startedAt: Date.now(),
  }),

  applyResult: (r) => set((s) => {
    if (s.results[r.idx]) return s;                       // idempotent on reconnect overlap
    const results = { ...s.results, [r.idx]: r };
    const done = s.done + 1;
    const correct = s.correct + (r.correct ? 1 : 0);
    const errors = Object.values(results).filter((x) => x.verdict === 'error').length;
    const scored = done - errors;
    return { results, done, correct, liveAccuracy: scored ? correct / scored : 0 };
  }),

  finishJob: (f) => {
    unsubscribe?.(); unsubscribe = null;
    set({ phase: f.status, summary: f.summary, error: f.error });
  },

  cancel: () => {
    const { jobId } = get();
    if (jobId) void cancelArena(jobId);
  },

  reset: () => {
    unsubscribe?.(); unsubscribe = null;
    set({ phase: 'idle', jobId: null, seed: null, images: [], results: {}, done: 0, correct: 0, liveAccuracy: 0, summary: null, error: null, startedAt: null });
  },
}));
```
Run `npm test` → PASS.

- [ ] **Step 7.3: `ConfigPanel.tsx`**

```tsx
import { Dices, Play } from 'lucide-react';
import { Badge } from '../../components/ui';
import { useHealth } from '../../stores/health';
import { useArena } from '../../stores/arena';
import { macroOf } from '../../data/models';
import { thumbUrl } from '../../services/api';

const N_OPTIONS = [25, 50, 100, 150];
const VARIANT_MACRO: Record<string, string> = {
  production: 'production_final', patchcore_v2: 'patchcore_v2', patchcore_v1: 'patchcore_v1',
};

export function ConfigPanel() {
  const { meta, online } = useHealth();
  const { config, setConfig, start, phase } = useArena();
  const cat = meta?.categories.find((c) => c.name === config.category);
  const busy = phase === 'starting' || phase === 'running';

  return (
    <div className="panel p-5 space-y-5">
      <div>
        <div className="text-xs uppercase tracking-widest text-steel mb-2">Category</div>
        <div className="grid grid-cols-5 md:grid-cols-8 xl:[grid-template-columns:repeat(15,minmax(0,1fr))] gap-1.5">
          {meta?.categories.map((c) => (
            <button key={c.name} onClick={() => setConfig({ category: c.name })} disabled={busy}
              className={`rounded-lg overflow-hidden border transition-colors ${config.category === c.name ? 'border-accent' : 'border-line hover:border-steel'}`}
              title={`${c.name} · ${c.test_total} test images`}>
              <img src={thumbUrl(c.name, 'good', '000.png', 64)} alt={c.name}
                className="w-full aspect-square object-cover" loading="lazy"
                onError={(e) => ((e.target as HTMLImageElement).style.visibility = 'hidden')} />
              <div className="text-[9px] py-0.5 truncate px-1 text-steel">{c.name.replace('_', ' ')}</div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-widest text-steel mb-2">Model</div>
        <div className="grid md:grid-cols-3 gap-1.5">
          {cat?.variants.map((v) => (
            <button key={v.id} onClick={() => setConfig({ variant: v.id })} disabled={busy || !v.available}
              className={`text-left rounded-lg border p-3 transition-colors ${
                config.variant === v.id ? 'border-accent bg-accent/5' : 'border-line hover:border-steel'} ${!v.available ? 'opacity-40' : ''}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">{v.label}</span>
                <span className="flex gap-1">
                  {v.approximate && <Badge tone="warn">approx</Badge>}
                  <Badge tone={v.kind === 'production' ? 'ok' : 'steel'}>{v.kind}</Badge>
                </span>
              </div>
              <div className="flex justify-between mt-1 text-xs text-steel gap-2">
                <span className="line-clamp-2">{v.description}</span>
                <span className="num shrink-0">{(macroOf(VARIANT_MACRO[v.id] ?? v.id) ?? 0).toFixed(4)}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-4 items-end flex-wrap">
        <div>
          <div className="text-xs uppercase tracking-widest text-steel mb-2">Images</div>
          <div className="flex rounded-lg border border-line overflow-hidden">
            {N_OPTIONS.map((n) => (
              <button key={n} onClick={() => setConfig({ n })} disabled={busy}
                className={`px-3 py-1.5 text-sm num ${config.n === n ? 'bg-accent/15 text-accent' : 'text-steel hover:text-fog'}`}>{n}</button>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-widest text-steel mb-2">Seed</div>
          <div className="flex items-center gap-1">
            <input value={config.seed ?? ''} placeholder="random" disabled={busy}
              onChange={(e) => setConfig({ seed: e.target.value === '' ? null : Number(e.target.value) || 0 })}
              className="w-24 bg-panel2 border border-line rounded-lg px-2 py-1.5 text-sm num focus:border-accent outline-none" />
            <button onClick={() => setConfig({ seed: Math.floor(Math.random() * 1_000_000) })} disabled={busy}
              className="p-2 text-steel hover:text-accent" title="Roll a seed"><Dices size={16} /></button>
          </div>
        </div>
        <button onClick={() => void start()} disabled={busy || !online}
          className="ml-auto inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent text-ink font-semibold text-sm
                     hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
          <Play size={15} /> {phase === 'starting' ? 'Loading model…' : 'Run batch'}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 7.4: `LiveGrid.tsx`**

```tsx
import { memo, useState } from 'react';
import { motion } from 'framer-motion';
import { verdictClasses } from '../../components/heat';
import { thumbUrl } from '../../services/api';
import { useArena } from '../../stores/arena';
import { ResultModal } from './ResultModal';
import type { ArenaImage, ArenaResult } from '../../types/api';

export function LiveGrid() {
  const { images, results, config } = useArena();
  const [open, setOpen] = useState<number | null>(null);
  if (!images.length) return null;
  return (
    <>
      <div className="grid gap-1.5" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(64px, 1fr))' }}>
        {images.map((img) => (
          <Cell key={img.idx} img={img} category={config.category} result={results[img.idx]} onClick={() => setOpen(img.idx)} />
        ))}
      </div>
      <ResultModal idx={open} onClose={() => setOpen(null)} />
    </>
  );
}

const Cell = memo(function Cell({ img, category, result, onClick }: {
  img: ArenaImage; category: string; result?: ArenaResult; onClick: () => void;
}) {
  const v = result ? verdictClasses(result.verdict) : null;
  return (
    <motion.button layout onClick={onClick} title={`${img.defect_type}/${img.filename}`}
      className={`relative aspect-square rounded-md overflow-hidden border-2 transition-colors ${
        v ? v.border : 'border-line animate-pulse'}`}>
      <img src={thumbUrl(category, img.defect_type, img.filename, 64)} alt=""
        className={`w-full h-full object-cover ${result ? '' : 'opacity-40 grayscale'}`} loading="lazy" />
      {result && (
        <span className={`absolute bottom-0 inset-x-0 text-[8px] font-semibold text-center py-0.5 ${v!.chip}`}>
          {result.verdict.toUpperCase()}
        </span>
      )}
    </motion.button>
  );
});
```

- [ ] **Step 7.5: `SummaryPanel.tsx`** (includes `StatusBar`)

```tsx
import { Ban } from 'lucide-react';
import { ConfusionMatrix } from '../../components/ConfusionMatrix';
import { useArena } from '../../stores/arena';

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel px-3 py-2 text-center">
      <div className="text-[10px] uppercase tracking-widest text-steel">{label}</div>
      <div className="num text-lg">{value}</div>
    </div>
  );
}
const fmt = (v: number | null, d = 3) => (v == null ? '—' : v.toFixed(d));

export function StatusBar() {
  const { phase, done, images, liveAccuracy, seed, cancel, startedAt } = useArena();
  if (phase !== 'running' && phase !== 'starting') return null;
  const total = images.length || 1;
  return (
    <div className="panel p-4 flex items-center gap-4">
      <div className="flex-1">
        <div className="flex justify-between text-xs text-steel mb-1">
          <span className="num">{done}/{images.length} scored · seed {seed ?? '…'}</span>
          <span className="num">live accuracy {(liveAccuracy * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2 rounded-full bg-panel2 overflow-hidden">
          <div className="h-full bg-accent transition-all duration-300" style={{ width: `${(done / total) * 100}%` }} />
        </div>
      </div>
      {startedAt && <span className="num text-xs text-steel">{((Date.now() - startedAt) / 1000).toFixed(0)}s</span>}
      <button onClick={cancel} className="flex items-center gap-1 text-alert text-sm hover:underline"><Ban size={14} /> Cancel</button>
    </div>
  );
}

export function SummaryPanel() {
  const { phase, summary, results, seed } = useArena();
  if (!summary || (phase !== 'done' && phase !== 'cancelled')) return null;
  const scores = Object.values(results).filter((r) => r.anomaly_score != null);
  const max = Math.max(...scores.map((r) => r.anomaly_score!), 1e-9);
  const thr = scores[0]?.threshold ?? 0;
  return (
    <div className="space-y-4">
      {phase === 'cancelled' && <div className="text-warn text-sm">Run cancelled — partial results below.</div>}
      <div className="grid grid-cols-3 md:grid-cols-7 gap-2">
        <Chip label="accuracy" value={fmt(summary.accuracy)} />
        <Chip label="precision" value={fmt(summary.precision)} />
        <Chip label="recall" value={fmt(summary.recall)} />
        <Chip label="F1" value={fmt(summary.f1)} />
        <Chip label="AUROC" value={fmt(summary.auroc, 4)} />
        <Chip label="mean ms" value={fmt(summary.mean_ms, 0)} />
        <Chip label="seed" value={String(seed ?? '—')} />
      </div>
      <div className="grid md:grid-cols-2 gap-4 items-start">
        <div className="panel p-4"><ConfusionMatrix confusion={summary.confusion} /></div>
        <div className="panel p-4">
          <div className="text-xs uppercase tracking-widest text-steel mb-3">Score distribution</div>
          <div className="relative h-24 border-b border-line">
            {scores.map((r) => (
              <span key={r.idx} title={`${r.filename}: ${r.anomaly_score!.toFixed(4)}`}
                className={`absolute bottom-0 w-1 rounded-t ${r.ground_truth_anomaly ? 'bg-alert/70' : 'bg-ok/70'}`}
                style={{ left: `${(r.anomaly_score! / max) * 98}%`, height: `${20 + (r.idx % 5) * 14}%` }} />
            ))}
            <span className="absolute top-0 bottom-0 w-0.5 bg-warn" style={{ left: `${(thr / max) * 98}%` }} title="threshold" />
          </div>
          <div className="flex gap-4 mt-2 text-[10px] text-steel">
            <span><span className="inline-block w-2 h-2 bg-ok/70 rounded-sm mr-1" />ground-truth good</span>
            <span><span className="inline-block w-2 h-2 bg-alert/70 rounded-sm mr-1" />ground-truth defect</span>
            <span className="text-warn">| threshold</span>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7.6: `ResultModal.tsx`**

```tsx
import { useEffect, useState } from 'react';
import { Modal } from '../../components/Modal';
import { ScoreGauge } from '../../components/ScoreGauge';
import { Badge } from '../../components/ui';
import { verdictClasses } from '../../components/heat';
import { predictFromDataset, sampleUrl } from '../../services/api';
import { useArena } from '../../stores/arena';

export function ResultModal({ idx, onClose }: { idx: number | null; onClose: () => void }) {
  const { images, results, config } = useArena();
  const img = idx == null ? undefined : images.find((i) => i.idx === idx);
  const res = idx == null ? undefined : results[idx];
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [showHeat, setShowHeat] = useState(true);
  const [opacity, setOpacity] = useState(0.55);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setHeatmap(null);
    if (!img || !res || res.verdict === 'error') return;
    setLoading(true);
    predictFromDataset(config.category, img.defect_type, img.filename, config.variant)
      .then((p) => setHeatmap(p.heatmap_base64))
      .catch(() => setHeatmap(null))
      .finally(() => setLoading(false));
  }, [idx]);  // eslint-disable-line react-hooks/exhaustive-deps

  if (!img) return null;
  const v = res ? verdictClasses(res.verdict) : null;
  return (
    <Modal open={idx != null} onClose={onClose} wide>
      <div className="grid md:grid-cols-[1.2fr_1fr] gap-6">
        <div className="relative rounded-lg overflow-hidden border border-line">
          <img src={sampleUrl(config.category, img.defect_type, img.filename)} alt={img.filename} className="w-full" />
          {showHeat && heatmap && (
            <img src={`data:image/png;base64,${heatmap}`} alt="anomaly heatmap"
              className="absolute inset-0 w-full h-full mix-blend-screen pointer-events-none"
              style={{ opacity, filter: 'sepia(1) saturate(6) hue-rotate(-50deg)' }} />
          )}
          {loading && <div className="absolute inset-0 grid place-items-center bg-ink/40 text-xs text-steel">computing heatmap…</div>}
        </div>
        <div className="space-y-4">
          <div>
            <div className="text-xs text-steel num">{img.defect_type}/{img.filename}</div>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              {v && <span className={`px-2 py-1 rounded-md text-sm font-semibold ${v.chip}`}>{v.label}</span>}
              <Badge tone={img.ground_truth_anomaly ? 'alert' : 'ok'}>
                GT: {img.ground_truth_anomaly ? `defect (${img.defect_type})` : 'good'}
              </Badge>
            </div>
          </div>
          {res?.anomaly_score != null && res.threshold != null && (
            <ScoreGauge score={res.anomaly_score} threshold={res.threshold} />
          )}
          <dl className="text-sm space-y-1.5">
            {res?.anomaly_probability != null && (
              <div className="flex justify-between"><dt className="text-steel">anomaly probability</dt><dd className="num">{(res.anomaly_probability * 100).toFixed(1)}%</dd></div>
            )}
            {res?.inference_ms != null && (
              <div className="flex justify-between"><dt className="text-steel">inference</dt><dd className="num">{res.inference_ms.toFixed(0)} ms</dd></div>
            )}
            <div className="flex justify-between"><dt className="text-steel">model</dt><dd>{config.variant}</dd></div>
          </dl>
          <div className="flex items-center gap-3 text-xs text-steel">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} className="accent-cyan-400" />
              heatmap
            </label>
            <input type="range" min={0.1} max={1} step={0.05} value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))} className="flex-1 accent-cyan-400" />
          </div>
          {res?.verdict === 'error' && <p className="text-alert text-sm">{res.error}</p>}
        </div>
      </div>
    </Modal>
  );
}
```

- [ ] **Step 7.7: `SingleTest.tsx`** + page **`Arena.tsx`**

```tsx
// SingleTest.tsx
import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud } from 'lucide-react';
import { Spinner } from '../../components/ui';
import { ScoreGauge } from '../../components/ScoreGauge';
import { predictUpload } from '../../services/api';
import { useArena } from '../../stores/arena';
import type { PredictionResponse } from '../../types/api';

export function SingleTest() {
  const { config } = useArena();
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<PredictionResponse | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = async (files: File[]) => {
    const f = files[0];
    if (!f) return;
    setBusy(true); setError(null); setRes(null);
    setPreview(URL.createObjectURL(f));
    try { setRes(await predictUpload(f, config.category, config.variant)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'image/*': [] }, maxFiles: 1 });

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div {...getRootProps()} className={`panel p-8 grid place-items-center text-center cursor-pointer border-dashed min-h-56 ${isDragActive ? 'border-accent' : ''}`}>
        <input {...getInputProps()} />
        {busy ? <Spinner /> : (
          <div className="space-y-2 text-steel text-sm">
            <UploadCloud className="mx-auto" />
            <p>Drop an image (category: <span className="text-fog">{config.category}</span>, model: <span className="text-fog">{config.variant}</span>)</p>
            <p className="text-xs">…or click to browse. Tip: pick category/model in the Batch tab first.</p>
          </div>
        )}
      </div>
      <div className="space-y-4">
        {preview && (
          <div className="relative rounded-lg overflow-hidden border border-line max-h-72">
            <img src={preview} alt="uploaded" className="w-full object-contain max-h-72" />
            {res && (
              <img src={`data:image/png;base64,${res.heatmap_base64}`} alt="heatmap"
                className="absolute inset-0 w-full h-full mix-blend-screen opacity-60 pointer-events-none"
                style={{ filter: 'sepia(1) saturate(6) hue-rotate(-50deg)' }} />
            )}
          </div>
        )}
        {res && (
          <div className="space-y-3">
            <div className={`text-lg font-semibold ${res.is_anomaly ? 'text-alert' : 'text-ok'}`}>
              {res.is_anomaly ? '⚠ Anomaly detected' : '✓ Looks normal'}
            </div>
            <ScoreGauge score={res.anomaly_score} threshold={res.threshold} />
            <div className="text-xs text-steel num">{res.inference_ms.toFixed(0)} ms · {res.model_variant}</div>
          </div>
        )}
        {error && <p className="text-alert text-sm">{error}</p>}
      </div>
    </div>
  );
}
```
```tsx
// Arena.tsx
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { OfflineCard } from '../../components/ui';
import { useHealth } from '../../stores/health';
import { useArena } from '../../stores/arena';
import { ConfigPanel } from './ConfigPanel';
import { LiveGrid } from './LiveGrid';
import { StatusBar, SummaryPanel } from './SummaryPanel';
import { SingleTest } from './SingleTest';

export default function Arena() {
  const { online } = useHealth();
  const { config, setConfig, seed, phase } = useArena();
  const [params, setParams] = useSearchParams();
  const [tab, setTab] = useState<'batch' | 'single'>('batch');

  useEffect(() => {   // hydrate config from URL once
    const cat = params.get('cat'); const variant = params.get('variant');
    const n = params.get('n'); const s = params.get('seed');
    setConfig({
      ...(cat ? { category: cat } : {}), ...(variant ? { variant } : {}),
      ...(n ? { n: Number(n) } : {}), ...(s ? { seed: Number(s) } : {}),
    });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {   // reflect the running config (incl. resolved seed) into the URL
    if (phase === 'running' || phase === 'done') {
      setParams({ cat: config.category, variant: config.variant, n: String(config.n), ...(seed != null ? { seed: String(seed) } : {}) }, { replace: true });
    }
  }, [phase, seed]);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-8">
      <header className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Test Arena</h1>
          <p className="text-steel mt-2">Sample random test images, pick a model, watch it classify live.</p>
        </div>
        <div className="flex rounded-lg border border-line overflow-hidden text-sm">
          <button onClick={() => setTab('batch')} className={`px-4 py-2 ${tab === 'batch' ? 'bg-accent/15 text-accent' : 'text-steel'}`}>Batch run</button>
          <button onClick={() => setTab('single')} className={`px-4 py-2 ${tab === 'single' ? 'bg-accent/15 text-accent' : 'text-steel'}`}>Single test</button>
        </div>
      </header>

      {online === false ? (
        <OfflineCard what="The Test Arena" onRetry={() => window.location.reload()} />
      ) : tab === 'batch' ? (
        <div className="space-y-6">
          <ConfigPanel />
          <StatusBar />
          <SummaryPanel />
          <LiveGrid />
        </div>
      ) : (
        <SingleTest />
      )}
    </div>
  );
}
```

- [ ] **Step 7.8: Full-loop manual test (server running with real banks)**

```powershell
npm test
npm run dev
```
At `/arena`: bottle + Production + 25 → Run. Expect: gray thumb grid immediately; cells fill with colored borders one by one; live accuracy ticks; summary with confusion matrix; cell click → image + red heatmap blob over the defect (computed on demand). Rerun with `patchcore_v1` on `zipper` or `capsule` — visibly worse accuracy (the coreset story). Refresh mid-run → reconnect/polling completes. `/arena?cat=screw&variant=production&n=25&seed=7` reproduces a sample; `approx` badge shows on screw's reconstructed variants.

- [ ] **Step 7.9: Commit**

```powershell
npm run build ; git add -A ; git commit -m "feat: test arena — streaming batch grid, summary, heatmap modal, single test"
```

---

### Task 8: Dataset explorer

**Files:**
- Create: `src/routes/dataset/DatasetExplorer.tsx`, `src/routes/dataset/DatasetCategory.tsx` (replace stubs)

- [ ] **Step 8.1: `DatasetExplorer.tsx`**

```tsx
import { Link } from 'react-router-dom';
import { OfflineCard } from '../../components/ui';
import { useHealth } from '../../stores/health';
import { thumbUrl } from '../../services/api';

export default function DatasetExplorer() {
  const { meta, online } = useHealth();
  if (online === false) return <OfflineCard what="The dataset explorer" />;
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">MVTec AD Dataset</h1>
        <p className="text-steel mt-2">15 industrial categories · defect-free training, mixed test set with pixel-accurate ground truth.</p>
      </header>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        {meta?.categories.map((c) => (
          <Link key={c.name} to={`/dataset/${c.name}`} className="panel overflow-hidden hover:border-accent/50 transition-colors">
            <img src={thumbUrl(c.name, 'good', '000.png', 256)} alt={c.name} className="w-full aspect-square object-cover" loading="lazy" />
            <div className="p-3">
              <div className="font-medium text-sm">{c.name.replace('_', ' ')}</div>
              <div className="text-xs text-steel num mt-0.5">
                {c.test_total} test · {c.defect_types.filter((d) => d.is_anomaly).length} defect types
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 8.2: `DatasetCategory.tsx`**

```tsx
import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Badge, OfflineCard, Spinner } from '../../components/ui';
import { Modal } from '../../components/Modal';
import { fetchTestImages, maskUrl, sampleUrl, thumbUrl } from '../../services/api';
import { useHealth } from '../../stores/health';
import type { TestImagesResponse } from '../../types/api';

export default function DatasetCategory() {
  const { category = '' } = useParams();
  const { online } = useHealth();
  const [data, setData] = useState<TestImagesResponse | null>(null);
  const [defect, setDefect] = useState<string>('');
  const [openImg, setOpenImg] = useState<string | null>(null);
  const [showMask, setShowMask] = useState(true);

  useEffect(() => {
    setData(null);
    fetchTestImages(category).then((d) => {
      setData(d);
      setDefect(d.defect_types[0]?.defect_type ?? '');
    }).catch(() => setData(null));
  }, [category]);

  if (online === false) return <OfflineCard what="The dataset explorer" />;
  if (!data) return <div className="flex justify-center py-24"><Spinner /></div>;
  const group = data.defect_types.find((d) => d.defect_type === defect);

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <Link to="/dataset" className="text-steel text-sm hover:text-fog flex items-center gap-1"><ArrowLeft size={14} /> Dataset</Link>
        <h1 className="text-3xl font-bold tracking-tight">{category.replace('_', ' ')}</h1>
      </header>
      <div className="flex flex-wrap gap-2 items-center">
        {data.defect_types.map((d) => (
          <button key={d.defect_type} onClick={() => setDefect(d.defect_type)}
            className={`px-3 py-1.5 rounded-lg border text-sm transition-colors ${
              defect === d.defect_type ? 'border-accent text-accent bg-accent/5' : 'border-line text-steel hover:text-fog'}`}>
            {d.defect_type} <span className="num text-xs opacity-70">({d.count})</span>
          </button>
        ))}
        {group?.is_anomaly && (
          <label className="ml-auto flex items-center gap-1.5 text-xs text-steel">
            <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} className="accent-cyan-400" />
            ground-truth mask overlay
          </label>
        )}
      </div>
      <div className="grid grid-cols-3 md:grid-cols-5 xl:grid-cols-7 gap-2">
        {group?.images.map((f) => (
          <button key={f} onClick={() => setOpenImg(f)} className="rounded-lg overflow-hidden border border-line hover:border-accent transition-colors">
            <img src={thumbUrl(category, defect, f, 128)} alt={f} className="w-full aspect-square object-cover" loading="lazy" />
          </button>
        ))}
      </div>
      <Modal open={openImg != null} onClose={() => setOpenImg(null)} wide>
        {openImg && group && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="num text-sm text-steel">{defect}/{openImg}</span>
              <Badge tone={group.is_anomaly ? 'alert' : 'ok'}>{group.is_anomaly ? 'defect' : 'good'}</Badge>
            </div>
            <div className="relative rounded-lg overflow-hidden border border-line">
              <img src={sampleUrl(category, defect, openImg)} alt={openImg} className="w-full" />
              {group.is_anomaly && showMask && (
                <img src={maskUrl(category, defect, openImg)} alt="ground-truth mask"
                  className="absolute inset-0 w-full h-full mix-blend-screen opacity-50 pointer-events-none"
                  style={{ filter: 'sepia(1) saturate(8) hue-rotate(-50deg)' }}
                  onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')} />
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
```

- [ ] **Step 8.3: Visual check + commit**

```powershell
npm run dev   # /dataset: 15 cards; category page: tabs, grid, mask overlay in modal
npm run build ; git add -A ; git commit -m "feat: dataset explorer with defect tabs and GT mask overlay"
```

---

### Task 9: Methodology + polish + docs + E2E

**Files:**
- Create: `src/routes/Methodology.tsx` (replace stub)
- Create: `../README.md` (project root)

- [ ] **Step 9.1: `Methodology.tsx`** (content from `ocgan-modernized/README.md`)

```tsx
import { Section } from '../components/ui';

function Code({ children }: { children: string }) {
  return <pre className="panel p-4 text-xs num overflow-x-auto whitespace-pre">{children}</pre>;
}

export default function Methodology() {
  return (
    <div className="space-y-12 max-w-3xl">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Methodology</h1>
        <p className="text-steel mt-2">Why the GAN lost, why frozen features won, and what keeps the numbers honest.</p>
      </header>

      <Section title="The arc">
        <p className="text-sm leading-relaxed text-fog/90">
          The project began as a one-class GAN: reconstruction plus seven fused scoring heads
          (perceptual, teacher-student, latent compactness, memory bank…). After Sprint 1 fixed three
          dead config flags, the honest macro AUROC stood at <span className="num">0.7866</span>.
          Sprint 4 deleted everything except the memory bank and scored frozen ImageNet features
          directly — no training, no fusion. First attempt: <span className="num">0.9051</span>.
          After tuning: <span className="num text-ok">0.9846</span>.
        </p>
      </Section>

      <Section title="The three ingredients (+19.8 pp)">
        <ol className="space-y-3 text-sm list-decimal pl-5">
          <li><span className="font-medium">No bank pruning when it fits.</span> Keeping all ≤70k patches instead of a 10k coreset took zipper from <span className="num">0.7184 → 0.9801</span> and capsule from <span className="num">0.7724 → 0.9824</span>.</li>
          <li><span className="font-medium">topk_reweighted aggregation.</span> A softmax-weighted top-k mean (k=9) that down-weights redundant top distances — beats plain top-k on every weak category.</li>
          <li><span className="font-medium">Multi-scale features.</span> layer2+layer3 concatenated; screw alone gains another <span className="num">+2.7 pp</span> from adding layer1 (fine thread detail).</li>
        </ol>
      </Section>

      <Section title="Threshold calibration" sub="Why you can't calibrate on the bank's own images">
        <p className="text-sm leading-relaxed text-fog/90">
          Every training patch is in the bank, so training images score ≈ 0 — calibrating there would
          flag everything as anomalous. Instead 15% of training images are held out (val_normal, seed
          43) and the threshold is their 99th-percentile score. The reconstructed v1/v2 variants in the
          Arena are recalibrated with exactly the same protocol.
        </p>
      </Section>

      <Section title="Honesty notes on the Arena">
        <ul className="text-sm space-y-2 list-disc pl-5 text-fog/90">
          <li><span className="text-fog font-medium">Production</span> is the shipped model: full bank, real thresholds.</li>
          <li><span className="text-fog font-medium">Reconstructed v1/v2</span> rebuild the historical configs (coreset 10k + their aggregation) from today's production bank — labeled, and marked <em>approx</em> for screw, whose production bank uses different feature layers than the originals did.</li>
          <li>The GAN iterations cannot run live (their inference pipeline was retired); they appear in the static evaluations only.</li>
          <li>Arena metrics are computed on the sampled subset — expect variance vs the full-test-set numbers in Evaluation.</li>
        </ul>
      </Section>

      <Section title="Reproduce">
        <Code>{`# evaluation (15 categories × 3 seeds, ~10 min on a single GPU)
bash scripts/run_patchcore_v3.sh

# rebuild production banks + thresholds
python scripts/export_patchcore_banks.py --device cuda

# variant thresholds + webapp data
python scripts/calibrate_variant_thresholds.py
python scripts/build_webapp_data.py`}</Code>
        <p className="text-xs text-steel">Project hardware: Quadro RTX 5000 (16 GB), PyTorch 2.1.1+cu121. Backbone: wide_resnet50_2, frozen.</p>
      </Section>
    </div>
  );
}
```

- [ ] **Step 9.2: Lint + tests + build**

```powershell
npm run lint
npm test
npm run build
```
Expected: all green (unused-import warnings are the usual fix).

- [ ] **Step 9.3: Project-root `README.md`** (`D:\OCGAN\project\storage_project_outputs_datasets\project\README.md`)

```markdown
# MVTec AD Lab — Model Showcase Webapp

Showcase of the project's anomaly-detection journey on MVTec AD (OCGAN → PatchCore,
0.7866 → **0.9846** macro AUROC) with a live Test Arena.

## Layout
- `ocgan-modernized/` — models, eval scripts, FastAPI inference server
- `frontend/` — React/Vite webapp
- `../datasets/mvtec_ad/` — dataset (15 categories)
- `docs/superpowers/` — design spec + implementation plans

## Run (development)
    # 1. backend
    cd ocgan-modernized
    .\.venv\Scripts\Activate.ps1
    python server.py --port 8000 --device auto

    # 2. frontend (second shell)
    cd frontend
    npm run dev          # http://localhost:5173 (proxies /api → :8000)

## Run (single process)
    cd frontend ; npm run build
    cd ..\ocgan-modernized ; python server.py --port 8000 --device auto
    # → http://localhost:8000 serves the built app

## First-time setup
See `ocgan-modernized/README.md` (venv, torch, calibrate_variant_thresholds.py,
build_webapp_data.py) and run `npm install` in `frontend/`.
```

- [ ] **Step 9.4: Manual E2E checklist (real server, real banks)**

- [ ] Home: stats count up; 7 journey cards in date order; CTAs navigate.
- [ ] Evaluation: metric switcher re-sorts; heatmap matches README values (bottle row green, OCGAN cable cell red); cell click opens drilldown; insights show zipper/capsule as top coreset wins.
- [ ] Models: 7 cards; `ocgan_v1` and `production_final` detail pages render every section.
- [ ] Arena: 100-image run on bottle/Production completes (record wall time); accuracy ≥ 0.9; cancel works; v1 vs Production difference visible on zipper/capsule; heatmap localizes a real defect; seeded URL reproduces the sample; `approx` badge on screw variants.
- [ ] Dataset: thumbs fast on second visit (cache); mask overlay aligns with defects.
- [ ] Offline: stop server → Arena/Dataset show OfflineCard, static pages fine, LED red.
- [ ] Production serve: `npm run build` then server at :8000 serves the app; deep link `/models/production_final` works (SPA fallback).

- [ ] **Step 9.5: Final commit**

```powershell
git add -A ; git commit -m "feat: methodology page, root README, E2E pass"
```

---

## Self-review notes

- Spec coverage: 6 pages ✔, streaming arena with verdict grid + on-demand heatmaps ✔, evaluation charts (leaderboard/evolution/heatmap/drilldown/insights) ✔, dataset explorer with GT masks ✔, offline degradation ✔, URL-shareable arena ✔, English UI ✔. The spec's radar + seed-stability charts were folded into leaderboard/insights to avoid chart sprawl — add later via the same `macroMetric`/`auroc_std` helpers if requested.
- Consistency watch-list: `Architecture` field names vs `src/types/domain.ts` (Tasks 2/4/6), `status` union values, benchmarks.json shape from backend Task 8 (`per_category` = model → **list**; `data/models.ts` assumes lists), thumb convention `good/000.png` (MVTec zero-padded; broken thumbs hidden gracefully), `beginJob` arity between store and tests.
- Deliberate cuts (YAGNI): no multi-model race on a single image (superseded by per-run comparison), no grid virtualization (≤150 cells), no i18n.
