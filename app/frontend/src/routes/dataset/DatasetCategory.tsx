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
  const [loaded, setLoaded] = useState<{ category: string; data: TestImagesResponse } | null>(null);
  const [defect, setDefect] = useState<string>('');
  const [openImg, setOpenImg] = useState<string | null>(null);
  const [showMask, setShowMask] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchTestImages(category).then((d) => {
      if (cancelled) return;
      setLoaded({ category, data: d });
      setDefect(d.defect_types[0]?.defect_type ?? '');
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [category]);
  const data = loaded?.category === category ? loaded.data : null;

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
