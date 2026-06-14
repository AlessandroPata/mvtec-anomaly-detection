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
