import { Section } from '../components/ui';

function Code({ children }: { children: string }) {
  return <pre className="panel p-4 text-xs num overflow-x-auto whitespace-pre">{children}</pre>;
}

export default function Methodology() {
  return (
    <div className="space-y-12 max-w-3xl">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Methodology</h1>
        <p className="text-steel mt-2">Why the GAN lost, why frozen features won, and what keeps the numbers honest.</p>
      </header>

      <Section title="The arc">
        <p className="text-sm leading-relaxed text-fog/90">
          The project began as a one-class GAN: reconstruction plus seven fused scoring heads
          (perceptual, teacher-student, latent compactness, memory bank…). After Sprint 1 fixed three
          dead config flags, the honest macro AUROC stood at <span className="num">0.7866</span>.
          Sprint 4 deleted everything except the memory bank and scored frozen ImageNet features
          directly — no training, no fusion. First attempt: <span className="num">0.9051</span>.
          After tuning: <span className="num text-ok">0.9846</span>.
        </p>
      </Section>

      <Section title="The three ingredients (+19.8 pp)">
        <ol className="space-y-3 text-sm list-decimal pl-5">
          <li><span className="font-medium">No bank pruning when it fits.</span> Keeping all ≤70k patches instead of a 10k coreset took zipper from <span className="num">0.7184 → 0.9801</span> and capsule from <span className="num">0.7724 → 0.9824</span>.</li>
          <li><span className="font-medium">topk_reweighted aggregation.</span> A softmax-weighted top-k mean (k=9) that down-weights redundant top distances — beats plain top-k on every weak category.</li>
          <li><span className="font-medium">Multi-scale features.</span> layer2+layer3 concatenated; screw alone gains another <span className="num">+2.7 pp</span> from adding layer1 (fine thread detail).</li>
        </ol>
      </Section>

      <Section title="Threshold calibration" sub="Why you can't calibrate on the bank's own images">
        <p className="text-sm leading-relaxed text-fog/90">
          Every training patch is in the bank, so training images score ≈ 0 — calibrating there would
          flag everything as anomalous. Instead 15% of training images are held out (val_normal, seed
          43) and the threshold is their 99th-percentile score. The reconstructed v1/v2 variants in the
          Arena are recalibrated with exactly the same protocol.
        </p>
        <p className="text-sm leading-relaxed text-fog/90 mt-3">
          The p99 rule is intentionally conservative (few false positives), but for a category whose
          normal images vary a lot — <span className="num">screw</span> — it sits above most anomaly
          scores and the model under-detects. So for the live <span className="text-fog font-medium">Arena</span>
          each model uses a <span className="font-medium">per-category best-F1 operating point</span>,
          which makes the displayed accuracy reflect the model's best achievable trade-off. AUROC, shown
          in Evaluation, stays the threshold-free headline metric.
        </p>
      </Section>

      <Section title="Honesty notes on the Arena">
        <ul className="text-sm space-y-2 list-disc pl-5 text-fog/90">
          <li><span className="text-fog font-medium">Production</span> is the shipped model: full bank, real thresholds.</li>
          <li><span className="text-fog font-medium">Reconstructed v1/v2</span> rebuild the historical configs (coreset 10k + their aggregation) from today's production bank — labeled, and marked <em>approx</em> for screw, whose production bank uses different feature layers than the originals did.</li>
          <li><span className="text-fog font-medium">OCGAN final / optv2</span> run live from the original training checkpoints (one best seed per category, vs the multiseed means in Evaluation). <em>final</em> keeps its archived score calibration verbatim; <em>optv2</em> refits normalization, fusion and threshold on the validation splits at load, because its archived calibration came from a training-era code state and fp16 numerics that don't reproduce here.</li>
          <li>Arena metrics are computed on the sampled subset — expect variance vs the full-test-set numbers in Evaluation.</li>
        </ul>
      </Section>

      <Section title="Reproduce">
        <Code>{`# evaluation (15 categories × 3 seeds, ~10 min on a single GPU)
bash scripts/run_patchcore_v3.sh

# rebuild production banks + thresholds
python scripts/export_patchcore_banks.py --device cuda

# variant thresholds + webapp data
python scripts/calibrate_variant_thresholds.py
python scripts/build_webapp_data.py`}</Code>
        <p className="text-xs text-steel">Project hardware: Quadro RTX 5000 (16 GB), PyTorch 2.1.1+cu121. Backbone: wide_resnet50_2, frozen.</p>
      </Section>
    </div>
  );
}
