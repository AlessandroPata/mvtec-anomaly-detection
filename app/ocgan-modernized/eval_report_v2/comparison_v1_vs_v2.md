# OCGAN2026 — v1 vs v2 Comparison

**v1** = final_per_category_multiseed (seed 43–47)
**v2** = optv2 retrain (seed 43–45, skip connections + cosine schedule + memory bank kcenter + learned fusion + EMA + AMP + augmentations)

## Per-category AUROC

| Category | seeds v1/v2 | v1 AUROC | v2 AUROC | Δ | v1 AUPRC | v2 AUPRC | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| screw | 3/3 | 0.9995 | 1.0000 | ⚪ +0.0005 | 0.9998 | 1.0000 | ⚪ +0.0002 |
| wood | 3/2 | 0.9889 | 1.0000 | 🟢 +0.0111 | 0.9972 | 1.0000 | ⚪ +0.0028 |
| hazelnut | 3/3 | 0.9891 | 0.9914 | ⚪ +0.0024 | 0.9947 | 0.9935 | ⚪ -0.0011 |
| grid | 3/3 | 0.9394 | 0.9687 | 🟢 +0.0293 | 0.9700 | 0.9863 | 🟢 +0.0163 |
| leather | 3/3 | 0.8976 | 0.9678 | 🟢 +0.0702 | 0.9648 | 0.9892 | 🟢 +0.0245 |
| tile | 3/3 | 0.9753 | 0.9407 | 🔴 -0.0346 | 0.9919 | 0.9814 | 🔴 -0.0105 |
| carpet | 3/3 | 0.9238 | 0.9069 | 🔴 -0.0169 | 0.9707 | 0.9654 | 🔴 -0.0053 |
| bottle | 3/3 | 0.8542 | 0.8958 | 🟢 +0.0417 | 0.9570 | 0.9695 | 🟢 +0.0126 |
| zipper | 3/2 | 0.8191 | 0.8958 | 🟢 +0.0767 | 0.9509 | 0.9722 | 🟢 +0.0214 |
| transistor | 5/3 | 0.7377 | 0.7878 | 🟢 +0.0501 | 0.7078 | 0.7309 | 🟢 +0.0231 |
| pill | 5/3 | 0.6707 | 0.7309 | 🟢 +0.0603 | 0.9057 | 0.9352 | 🟢 +0.0295 |
| toothbrush | 3/3 | 0.7037 | 0.6519 | 🔴 -0.0518 | 0.8811 | 0.8545 | 🔴 -0.0266 |
| capsule | 3/3 | 0.7839 | 0.6268 | 🔴 -0.1571 | 0.9340 | 0.8694 | 🔴 -0.0647 |
| metal_nut | 5/3 | 0.6085 | 0.6228 | 🟢 +0.0143 | 0.8967 | 0.8979 | ⚪ +0.0012 |
| cable | 5/3 | 0.5231 | 0.5797 | 🟢 +0.0566 | 0.6875 | 0.7397 | 🟢 +0.0521 |

## Macro averages (mean across 15 categories)

| Metric | v1 | v2 | Δ |
|---|---:|---:|---:|
| AUROC | 0.8276 | 0.8378 | 🟢 +0.0102 |
| AUPRC | 0.9207 | 0.9257 | 🟢 +0.0050 |
| Best F1 | 0.9027 | 0.9124 | 🟢 +0.0097 |
| F1@thr | 0.8664 | 0.8693 | ⚪ +0.0029 |
| FPR@95 | 0.5035 | 0.4791 | 🟢 -0.0244 |

## Improvements (Δ AUROC > +0.005)
- **zipper**: +0.0767
- **leather**: +0.0702
- **pill**: +0.0603
- **cable**: +0.0566
- **transistor**: +0.0501
- **bottle**: +0.0417
- **grid**: +0.0293
- **metal_nut**: +0.0143
- **wood**: +0.0111

## Regressions (Δ AUROC < -0.005)
- **capsule**: -0.1571
- **toothbrush**: -0.0518
- **tile**: -0.0346
- **carpet**: -0.0169

## Stable (|Δ| ≤ 0.005)
hazelnut, screw

## Tier transitions

| Category | v1 tier | v2 tier | note |
|---|:---:|:---:|---|
| grid | T2 | T1 | ⬆ upgrade |
| leather | T2 | T1 | ⬆ upgrade |
| tile | T1 | T2 | ⬇ downgrade |
