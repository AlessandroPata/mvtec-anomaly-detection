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
