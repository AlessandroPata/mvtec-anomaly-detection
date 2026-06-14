import { Link } from 'react-router-dom';
import { OfflineCard } from '../../components/ui';
import { useHealth } from '../../stores/health';
import { thumbUrl } from '../../services/api';
import { InfoNote } from '../../components/InfoNote';

export default function DatasetExplorer() {
  const { meta, online } = useHealth();
  if (online === false) return <OfflineCard what="The dataset explorer" />;
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">MVTec AD Dataset</h1>
        <p className="text-steel mt-2">15 industrial categories · defect-free training, mixed test set with pixel-accurate ground truth.</p>
      </header>

      <InfoNote title="About MVTec AD">
        <p>
          <strong>MVTec AD</strong> is the standard industrial anomaly-detection benchmark: 15 product
          categories, each with a <strong>defect-free training set</strong> and a{' '}
          <strong>mixed test set</strong> (normal + several defect types) with pixel-accurate
          ground-truth masks.
        </p>
        <p>
          The models never see a defect during training — they learn what “normal” looks like and flag
          anything that deviates. Open a category to browse its defect types and inspect ground-truth
          masks.
        </p>
      </InfoNote>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        {meta?.categories.map((c) => (
          <Link key={c.name} to={`/dataset/${c.name}`} className="panel overflow-hidden hover:border-accent/50 transition-colors">
            <img src={thumbUrl(c.name, 'good', '000.png', 256)} alt={c.name} className="w-full aspect-square object-cover" loading="lazy" />
            <div className="p-3">
              <div className="font-medium text-sm">{c.name.replace('_', ' ')}</div>
              <div className="text-xs text-steel num mt-0.5">
                {c.test_total} test · {c.defect_types.filter((d) => d.is_anomaly).length} defect types
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
