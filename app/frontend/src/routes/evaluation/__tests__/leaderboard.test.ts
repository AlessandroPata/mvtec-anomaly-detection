import { describe, expect, it } from 'vitest';
import { leaderboardRows } from '../leaderboard-rows';

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
