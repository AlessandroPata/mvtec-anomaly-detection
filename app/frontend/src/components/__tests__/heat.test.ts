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
