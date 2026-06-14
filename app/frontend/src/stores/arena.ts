import { create } from 'zustand';
import { cancelArena, pollArena, startArena, streamArena } from '../services/api';
import type { ArenaFinal, ArenaImage, ArenaResult, ArenaSummary } from '../types/api';

export type ArenaPhase = 'idle' | 'starting' | 'running' | 'done' | 'cancelled' | 'error';

interface ArenaConfig { category: string; variant: string; n: number; seed: number | null }

export interface ArenaSnapshot { variant: string; category: string; seed: number | null; n: number; summary: ArenaSummary }

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
  snapA: ArenaSnapshot | null;
  snapB: ArenaSnapshot | null;

  setConfig: (c: Partial<ArenaConfig>) => void;
  start: () => Promise<void>;
  beginJob: (jobId: string, seed: number, images: ArenaImage[]) => void;
  applyResult: (r: ArenaResult) => void;
  finishJob: (f: ArenaFinal) => void;
  cancel: () => void;
  reset: () => void;
  snapshot: (slot: 'A' | 'B') => void;
  clearSnapshots: () => void;
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
  snapA: null, snapB: null,

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

  snapshot: (slot) => set((s) => {
    if (!s.summary) return s;
    const snap: ArenaSnapshot = { variant: s.config.variant, category: s.config.category, seed: s.seed, n: s.config.n, summary: s.summary };
    return slot === 'A' ? { snapA: snap } : { snapB: snap };
  }),
  clearSnapshots: () => set({ snapA: null, snapB: null }),
}));
