// Pixel-level localization metrics for the production PatchCore, computed by
// ocgan-modernized/pixel_metrics.py (raw anomaly map vs. ground-truth masks over
// the full test set). Image-level AUROC is the headline; these are the
// localization counterparts MVTec AD is actually built for.
//   - pixel_auroc : per-pixel ranking of defect vs. normal pixels
//   - pixel_ap    : average precision on the (heavily imbalanced) pixel labels
//   - aupro       : AUPRO@30% — the official MVTec metric; every connected
//                   defect region is weighted equally, integrated to FPR=0.30
export const MACRO_PIXEL_AUROC = 0.9714;
export const MACRO_AUPRO = 0.9127;

export interface PixelRow {
  category: string;
  pixel_auroc: number;
  pixel_ap: number;
  aupro: number;
}

export const PIXEL_METRICS: PixelRow[] = [
  { category: 'bottle', pixel_auroc: 0.9867, pixel_ap: 0.7875, aupro: 0.9545 },
  { category: 'cable', pixel_auroc: 0.9812, pixel_ap: 0.6242, aupro: 0.9331 },
  { category: 'capsule', pixel_auroc: 0.9783, pixel_ap: 0.3896, aupro: 0.9004 },
  { category: 'carpet', pixel_auroc: 0.9891, pixel_ap: 0.5537, aupro: 0.9438 },
  { category: 'grid', pixel_auroc: 0.9748, pixel_ap: 0.2938, aupro: 0.9174 },
  { category: 'hazelnut', pixel_auroc: 0.9837, pixel_ap: 0.5268, aupro: 0.8928 },
  { category: 'leather', pixel_auroc: 0.9925, pixel_ap: 0.4245, aupro: 0.9796 },
  { category: 'metal_nut', pixel_auroc: 0.9738, pixel_ap: 0.7975, aupro: 0.9335 },
  { category: 'pill', pixel_auroc: 0.9533, pixel_ap: 0.5970, aupro: 0.9107 },
  { category: 'screw', pixel_auroc: 0.9763, pixel_ap: 0.2268, aupro: 0.8927 },
  { category: 'tile', pixel_auroc: 0.9528, pixel_ap: 0.5139, aupro: 0.8649 },
  { category: 'toothbrush', pixel_auroc: 0.9872, pixel_ap: 0.5094, aupro: 0.8935 },
  { category: 'transistor', pixel_auroc: 0.9291, pixel_ap: 0.5355, aupro: 0.8586 },
  { category: 'wood', pixel_auroc: 0.9446, pixel_ap: 0.4534, aupro: 0.9023 },
  { category: 'zipper', pixel_auroc: 0.9679, pixel_ap: 0.4424, aupro: 0.9121 },
];
