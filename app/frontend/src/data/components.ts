import type { ArchComponent } from '../types/domain';

export const COMPONENTS: ArchComponent[] = [
  {
    id: 'memory_bank',
    name: 'Memory Bank',
    short: 'Stored normal patch features for nearest-neighbour anomaly scoring',
    description:
      'Patch-level feature vectors extracted from normal training images, stored as a single large matrix. At inference time, every test patch finds its nearest neighbour in the bank — large distances indicate anomaly.',
    intuition:
      'Normal images live on a low-dimensional manifold in feature space. Anomalous patches fall outside this manifold and have large distance to any normal patch.',
    formula: 'score(x) = aggregate(min_{m ∈ M} ||f(x) - m||₂)',
    tradeoffs: [
      'More patches = better coverage but bigger bank',
      'Coreset prunes redundant patches but may drop rare normal patterns',
      'L2-normalised features → distances on unit sphere',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3', 'patchcore_v1', 'patchcore_v2', 'patchcore_v3', 'patchcore_p1', 'production_final'],
    paper_ref: 'Roth et al. "Towards Total Recall in Industrial Anomaly Detection" CVPR 2022',
  },
  {
    id: 'kcenter_coreset',
    name: 'K-Center Greedy Coreset',
    short: 'Diverse subsampling that maximises minimum pairwise distance',
    description:
      'Iterative greedy selection: pick the next patch that is farthest from all previously selected ones. Approximates the k-center problem (NP-hard) in O(k·n) time.',
    intuition:
      'Random sampling biases towards dense regions. K-center selects diverse exemplars covering the whole feature space.',
    formula: 'next ← argmax_{i ∉ S} min_{j ∈ S} ||x_i - x_j||₂',
    tradeoffs: [
      'O(k·n·d) compute',
      'Initialisation matters — init=mean gives deterministic results',
      'For very large n, candidate_pool sub-sampling helps',
    ],
    used_in: ['patchcore_v1', 'patchcore_v2', 'patchcore_v3'],
    paper_ref: 'Sener & Savarese "Active Learning for Convolutional Neural Networks: A Core-Set Approach" ICLR 2018',
  },
  {
    id: 'topk_aggregation',
    name: 'Top-K Aggregation',
    short: 'Image-level scoring from per-patch distances',
    description:
      'After computing per-patch min-distances, aggregate to a single image score. Three variants: mean (top-k), max (k=1), and reweighted (softmax-based).',
    formula:
      'reweighted: w_i = 1 - softmax_i(1 / d_i),  score = Σ w_i · d_i / Σ w_i',
    intuition:
      'Mean averages out, max is fragile, reweighted emphasises truly large distances while down-weighting redundant top-k entries.',
    tradeoffs: [
      'top-k=3 mean: simple, but smooths out true anomalies',
      'top-k=9 reweighted: most discriminative on hard categories',
      'max (k=1): fragile to single noisy patches',
    ],
    used_in: ['patchcore_v1', 'patchcore_v2', 'patchcore_v3', 'patchcore_p1', 'production_final', 'ocgan_v3'],
  },
  {
    id: 'multi_scale',
    name: 'Multi-Scale Features',
    short: 'Concatenate features from multiple backbone levels',
    description:
      'Pool deeper feature maps to a common spatial size and concatenate along the channel dimension. Combines low-level texture (layer1) with mid-level structure (layer2) and high-level semantics (layer3).',
    formula: 'F = cat(pool(L1, H₃×W₃), pool(L2, H₃×W₃), L3)',
    intuition:
      'Different defects appear at different scales. Fine textures need layer1, structural defects need layer3.',
    tradeoffs: [
      'layer2+layer3: best general-purpose default',
      'layer1+layer2+layer3: helps fine textures (screw threads) but adds dim',
      'More layers = bigger bank',
    ],
    used_in: ['patchcore_v1', 'patchcore_v2', 'patchcore_v3', 'patchcore_p1', 'production_final'],
  },
  {
    id: 'frozen_backbone',
    name: 'Frozen ImageNet Backbone',
    short: 'Pre-trained features without any fine-tuning',
    description:
      'Use ImageNet-pretrained wide_resnet50_2 with all parameters frozen. No task-specific training. Features come straight from ImageNet.',
    intuition:
      'For one-class anomaly detection, training on a small set of normal images often degrades the representation. ImageNet features are general and rich.',
    tradeoffs: [
      'No training time',
      'No risk of overfitting to normal class',
      'May miss domain-specific cues',
      'Best when normal class has visual diversity',
    ],
    used_in: ['patchcore_v1', 'patchcore_v2', 'patchcore_v3', 'patchcore_p1', 'production_final'],
    paper_ref: 'Bergmann et al. "Uninformed Students" CVPR 2020 (and PatchCore)',
  },
  {
    id: 'reconstruction',
    name: 'Reconstruction Head',
    short: 'Encoder-decoder that reconstructs the input',
    description:
      'A decoder (Residual or UNet with skip connections) trained to reconstruct normal images. Anomalous regions reconstruct poorly → large pixel/feature differences.',
    intuition:
      'The model learns the manifold of normal images. Anything outside (anomalies) cannot be reconstructed faithfully.',
    tradeoffs: [
      'Heatmaps come naturally from diff maps',
      'Trained encoder often weaker than frozen ImageNet',
      'Skip connections allow trivial pixel copying',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3'],
  },
  {
    id: 'teacher_student',
    name: 'Teacher-Student Distillation',
    short: 'Frozen teacher vs trainable student feature comparison',
    description:
      'Train a student network to mimic a frozen teacher on normal images only. At inference, large student-teacher feature differences indicate anomaly.',
    intuition:
      'On normal images, student matches teacher (training objective). On anomalies, student fails to mimic.',
    tradeoffs: [
      'Multi-level distillation richer than single-level',
      'Requires extra training time and memory',
      'Effective for subtle structural anomalies',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3'],
    paper_ref: 'Bergmann et al. "Uninformed Students" CVPR 2020',
  },
  {
    id: 'latent_compactness',
    name: 'Latent Compactness (DeepSVDD-style)',
    short: 'One-class loss pulling latents toward a single centre',
    description:
      'Encode each normal image to a latent vector. Penalise distance from a learned cluster centre. At inference, latents far from centre are anomalous.',
    formula: 'L = Σᵢ ||z_i - c||²',
    intuition:
      'Compresses the normal class into a tight ball in latent space. Anomalies are off-ball.',
    tradeoffs: [
      'Risk of trivial collapse (all latents → same vector)',
      'Sensitive to initialisation of c',
      'Often combined with reconstruction loss',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3'],
    paper_ref: 'Ruff et al. "Deep One-Class Classification" ICML 2018',
  },
  {
    id: 'score_fusion_weighted',
    name: 'Score Fusion (weighted)',
    short: 'Linear combination of multiple anomaly scores',
    description:
      'Each head produces a normalised score. The final anomaly score is a weighted sum with manually-tuned weights from config.',
    formula: 'score = Σ wᵢ · normalize(scoreᵢ)',
    intuition:
      'Different heads catch different anomaly types. Combining them increases coverage.',
    tradeoffs: [
      'Manual weight tuning is brittle',
      'Each component must be MAD-normalised against val_normal',
      'Often outperformed by learned fusion',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3'],
  },
  {
    id: 'score_fusion_learned',
    name: 'Score Fusion (learned)',
    short: 'Logistic regression on per-head normalised scores',
    description:
      'Fit a logistic regression on val_mixed (normal + synthetic-anomaly) where features are the per-head normalised scores. The probability output replaces the manual weighted sum.',
    intuition:
      'Lets the data choose the right weights instead of hand-tuning.',
    tradeoffs: [
      'Requires val_mixed with labelled positives',
      'High variance with <30 samples per class',
      'L2 regularisation prevents overfitting',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3'],
  },
  {
    id: 'synthetic_anomalies',
    name: 'Synthetic Anomaly Augmentation',
    short: 'Cutpaste / Perlin noise / DRAEM-style training-time anomalies',
    description:
      'Generate fake anomalies at training time to give the discriminator/classifier something to push against. Three flavours: cutpaste (random patch paste), Perlin noise (DRAEM-style), and uniform noise blobs.',
    intuition:
      'Without real anomalies in training, the model has no direct supervision for "what is wrong". Synthetic anomalies fill that gap.',
    tradeoffs: [
      'Cutpaste: simple, but anomalies look unnatural',
      'Perlin: more diverse, but expensive in pure-Python (vectorised in v3)',
      'Per-item deterministic seed required for reproducibility',
    ],
    used_in: ['ocgan_v1', 'ocgan_v3'],
    paper_ref: 'Zavrtanik et al. "DRAEM" ICCV 2021; Li et al. "CutPaste" CVPR 2021',
  },
  {
    id: 'threshold_calibration',
    name: 'Threshold Calibration',
    short: 'Set decision threshold on a held-out set of normal images',
    description:
      'Hold out 15% of training images as val_normal. Score them against the bank built from the other 85%. Set threshold = 99th percentile of these scores. Avoids the trap where bank patches score themselves at distance ≈ 0.',
    formula: 'τ = quantile(scores(val_normal), 0.99)',
    intuition:
      'Threshold should reflect the normal score distribution at inference time. Calibrating on training data alone gives near-zero distances since training patches ARE in the bank.',
    tradeoffs: [
      'Stricter percentile = fewer false positives, more false negatives',
      'Requires sufficient val_normal samples (~30+)',
      'For hard categories, anomaly_probability (sigmoid z-score) is more reliable than the binary flag',
    ],
    used_in: ['patchcore_v3', 'patchcore_p1', 'production_final'],
  },
];

export const COMPONENT_BY_ID: Record<string, ArchComponent> = Object.fromEntries(
  COMPONENTS.map((c) => [c.id, c])
);
