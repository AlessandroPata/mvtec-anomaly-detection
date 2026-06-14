// Probability calibration for the production model, from
// ocgan-modernized/calibrate_probabilities.py (Platt vs. isotonic, picked per
// category by held-out CV Brier score). A raw anomaly score is monotonic with
// anomaly-ness but is not a probability: ECE 0.32 means a "0.8" score is far from
// 80% likely. Post-hoc calibration fixes that. Lower Brier / ECE is better.
export const MACRO_BRIER_RAW = 0.17592;
export const MACRO_BRIER_CAL = 0.03352;
export const MACRO_ECE_RAW = 0.31595;
export const MACRO_ECE_CAL = 0.03342;

export interface CalibrationRow {
  category: string;
  method: 'platt' | 'isotonic';
  brier_raw: number;
  brier_cal: number;
  ece_raw: number;
  ece_cal: number;
}

export const CALIBRATION: CalibrationRow[] = [
  { category: 'bottle', method: 'platt', brier_raw: 0.09774, brier_cal: 0.00017, ece_raw: 0.30217, ece_cal: 0.00156 },
  { category: 'cable', method: 'isotonic', brier_raw: 0.17032, brier_cal: 0.01794, ece_raw: 0.31013, ece_cal: 0.02498 },
  { category: 'capsule', method: 'isotonic', brier_raw: 0.25545, brier_cal: 0.03054, ece_raw: 0.38722, ece_cal: 0.02226 },
  { category: 'carpet', method: 'isotonic', brier_raw: 0.15062, brier_cal: 0.02452, ece_raw: 0.28609, ece_cal: 0.02474 },
  { category: 'grid', method: 'isotonic', brier_raw: 0.22076, brier_cal: 0.07682, ece_raw: 0.38589, ece_cal: 0.07536 },
  { category: 'hazelnut', method: 'platt', brier_raw: 0.19753, brier_cal: 0.00437, ece_raw: 0.37169, ece_cal: 0.00171 },
  { category: 'leather', method: 'platt', brier_raw: 0.11982, brier_cal: 0.00000, ece_raw: 0.33876, ece_cal: 0.00020 },
  { category: 'metal_nut', method: 'isotonic', brier_raw: 0.12864, brier_cal: 0.01866, ece_raw: 0.25766, ece_cal: 0.01604 },
  { category: 'pill', method: 'platt', brier_raw: 0.19757, brier_cal: 0.06056, ece_raw: 0.33476, ece_cal: 0.05664 },
  { category: 'screw', method: 'isotonic', brier_raw: 0.29679, brier_cal: 0.07751, ece_raw: 0.35509, ece_cal: 0.03904 },
  { category: 'tile', method: 'platt', brier_raw: 0.12417, brier_cal: 0.00013, ece_raw: 0.30630, ece_cal: 0.00186 },
  { category: 'toothbrush', method: 'platt', brier_raw: 0.15075, brier_cal: 0.08109, ece_raw: 0.19502, ece_cal: 0.10845 },
  { category: 'transistor', method: 'platt', brier_raw: 0.19481, brier_cal: 0.03034, ece_raw: 0.32239, ece_cal: 0.03522 },
  { category: 'wood', method: 'platt', brier_raw: 0.11338, brier_cal: 0.04260, ece_raw: 0.25859, ece_cal: 0.05903 },
  { category: 'zipper', method: 'platt', brier_raw: 0.22043, brier_cal: 0.03752, ece_raw: 0.32749, ece_cal: 0.03428 },
];
