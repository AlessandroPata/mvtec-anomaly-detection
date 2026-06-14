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
