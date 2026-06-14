import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud } from 'lucide-react';
import { Spinner } from '../../components/ui';
import { ScoreGauge } from '../../components/ScoreGauge';
import { predictUpload } from '../../services/api';
import { useArena } from '../../stores/arena';
import type { PredictionResponse } from '../../types/api';

export function SingleTest() {
  const { config } = useArena();
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<PredictionResponse | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = async (files: File[]) => {
    const f = files[0];
    if (!f) return;
    setBusy(true); setError(null); setRes(null);
    setPreview(URL.createObjectURL(f));
    try { setRes(await predictUpload(f, config.category, config.variant)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'image/*': [] }, maxFiles: 1 });

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div {...getRootProps()} className={`panel p-8 grid place-items-center text-center cursor-pointer border-dashed min-h-56 ${isDragActive ? 'border-accent' : ''}`}>
        <input {...getInputProps()} />
        {busy ? <Spinner /> : (
          <div className="space-y-2 text-steel text-sm">
            <UploadCloud className="mx-auto" />
            <p>Drop an image (category: <span className="text-fog">{config.category}</span>, model: <span className="text-fog">{config.variant}</span>)</p>
            <p className="text-xs">…or click to browse. Tip: pick category/model in the Batch tab first.</p>
          </div>
        )}
      </div>
      <div className="space-y-4">
        {preview && (
          <div className="relative rounded-lg overflow-hidden border border-line max-h-72">
            <img src={preview} alt="uploaded" className="w-full object-contain max-h-72" />
            {res && (
              <img src={`data:image/png;base64,${res.heatmap_base64}`} alt="heatmap"
                className="absolute inset-0 w-full h-full mix-blend-screen opacity-60 pointer-events-none"
                style={{ filter: 'sepia(1) saturate(6) hue-rotate(-50deg)' }} />
            )}
          </div>
        )}
        {res && (
          <div className="space-y-3">
            <div className={`text-lg font-semibold ${res.is_anomaly ? 'text-alert' : 'text-ok'}`}>
              {res.is_anomaly ? '⚠ Anomaly detected' : '✓ Looks normal'}
            </div>
            <ScoreGauge score={res.anomaly_score} threshold={res.threshold} />
            <div className="text-xs text-steel num">{res.inference_ms.toFixed(0)} ms · {res.model_variant}</div>
          </div>
        )}
        {error && <p className="text-alert text-sm">{error}</p>}
      </div>
    </div>
  );
}
