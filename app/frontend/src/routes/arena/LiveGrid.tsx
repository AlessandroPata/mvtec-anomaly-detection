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
