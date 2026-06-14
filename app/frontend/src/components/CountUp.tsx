import { useEffect, useRef, useState } from 'react';

export function CountUp({ value, decimals = 0, suffix = '', duration = 900 }: {
  value: number; decimals?: number; suffix?: string; duration?: number;
}) {
  const [shown, setShown] = useState(0);
  const raf = useRef(0);
  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = reduced ? 1 : Math.min(1, (t - t0) / duration);
      setShown(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [value, duration]);
  return <span>{shown.toFixed(decimals)}{suffix}</span>;
}
