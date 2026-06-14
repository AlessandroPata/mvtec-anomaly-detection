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
