# Appunti per la relazione finale (working notes — NON il deliverable)

> Fonti lette PER INTERO: paper OCGAN (CVPR 2019, = fsfs.pdf arXiv 1903.08550),
> piano cronologico "binario B" (38 pp, docx/pdf utente).
> Testo estratto in `relazione/sources/{paper_ocgan,piano_cronologico}.txt`.
> ATTENZIONE: il docx contiene un token GitHub (ghp_VDmj…) — MAI citarlo nella relazione; utente avvisato di revocarlo.

## Richieste esplicite dell'utente per la relazione
- Ordine cronologico: da miglioramento soft di OCGAN → modelli potenti finali (ma posso proporre struttura migliore)
- Tutte le scelte fatte + differenze tra modelli e tra versioni dello stesso modello
- Spiegazione breve di tutte le tecniche (memory bank, learned fusion, teacher-student, ecc.)
- Confronto costante col paper originale OCGAN
- Molti grafici e dati (nostri + confronto paper)
- Per ogni modello: architettura, struttura, come funziona
- Paperspace: cos'è, perché vantaggioso
- Tutti i problemi/sfide del percorso, vantaggi/svantaggi di ogni scelta
- Libertà di aggiungere idee mie

## 1. Paper OCGAN (Perera, Nallapati, Xiang — CVPR 2019)
- Problema: one-class novelty detection (solo esempi in-class al training).
- Idea chiave: non basta rappresentare bene la classe; bisogna che il latent space rappresenti SOLO la classe (out-of-class mal rappresentati). AE su digit 8 ricostruisce bene anche 1,5,6,9 → falsi negativi.
- 4 componenti: denoising AE (noise gauss σ²=0.2, MSE), latent space bounded con tanh → (-1,1)^d;
  latent discriminator Dl (latent reali vs U(-1,1)^d → forza distribuzione uniforme);
  visual discriminator Dv (immagini generate da campioni casuali del latent vs reali);
  classifier C (debole, ricostruzioni=positivi, fake=negativi) che guida l'informative-negative mining:
  gradient ascent nel latent (5 step) per trovare punti che generano esempi out-of-class, usati per allenare il generatore.
- Training: 2 step alternati (classifier; poi AE+discriminatori adversarial), loss generatore = 10·MSE + l_latent + l_visual.
- Architettura: AE simmetrico 3 conv 5×5 stride2 + 3 deconv, BN+LeakyReLU(0.2), base 64 ch; Dv 12 ch, C 64 ch; Dl FC 128-64-32-16. Model selection su MSE di validation.
- Risultati: Protocol 1 (80/20 in-class, test bilanciato): MNIST 0.977, COIL100 0.995, fMNIST 0.924. Protocol 2 (split nativi): MNIST mean AUC 0.9750 (batte DSVDD 0.9480, AND 0.9671...), CIFAR10 0.6566 (vs DSVDD 0.6481).
- Ablation MNIST: AE solo 0.957 → +Dl 0.959 → +Dv 0.971 → +classifier/mining 0.975.
- Limiti dichiarati: funziona quando c'è un solo concetto centrato (MNIST/COIL/fMNIST); CIFAR10 difficile. Repo originale MXNet; esiste repo TF minimale (risultati misurati 1 sola volta per cifra → fragile).

## 2. Piano cronologico "Binario B — OCGAN modernizzato" (doc utente, 38 pp)
- Strategia: NON difendere OCGAN puro; tenere branch ricostruttiva + latent regularization (come idea) + informative-negative mining; innestarli in pipeline moderna: backbone pretrained (ResNet50/WideResNet50), score multipli, synthetic anomalies, branch discriminativa, memory signals, teacher-student discrepancy. Riferimenti moderni citati: PaDiM, PatchCore, DRAEM, RD4AD, EfficientAD, SimpleNet.
- Fase 1: rifondazione repo in PyTorch modulare (configs/datasets/models/losses/miners/scorers/metrics/trainers/callbacks/scripts), reproducibility hardening (seed, determinismo, config salvate, checkpoint top-k, resume, logging env, multi-run), training utilities (AMP, grad accum, clipping, EMA, profiler, smoke test, NaN test).
- Fase 2 protocollo: 4 split SEMPRE: train_normal / val_normal / val_mixed / test_blind. val_mixed per soglie/checkpoint/iperparametri; test MAI per tuning. Anti-leakage (dedup, near-dup, stat norm solo su train). Multi-seed obbligatorio (≥5 su MVTec). Metriche: AUROC, AUPRC, F1@val_thr, FPR@95TPR (+ pixel metrics per localization).
- Dataset order: MNIST/FMNIST/CIFAR10 (verifica pipeline) → MVTec AD (vero banco di prova: 15 categorie industriali, train defect-free, test misto con GT pixel-level) → MVTec AD 2 (futuro).
- Preprocessing: resize anti-aliased aspect-preserving, center crop, normalizzazione ImageNet se backbone pretrained; augmentations conservative (preservare la normalità): traslazioni/rotazioni piccole, jitter luminosità, noise lieve. NO elastic/cutout aggressivi.
- Architettura moderna: ricostruttore con residual blocks, anti-aliased down, upsample+conv, skip leggere/gated (skip troppo forti fanno passare le anomalie!); loss stack L1 + MS-SSIM + perceptual (MSE solo è troppo debole, oversmoothing).
- Latent: da prior U(-1,1)^d (storico) → center loss/compactness Deep-SVDD-like, one-class bottleneck (idea RD4AD).
- Mining modernizzato: multi-step PGD-style, warmup, replay buffer, guidato da score composito; anche feature-space negatives.
- Synthetic anomalies: pixel-space (CutPaste, Perlin masks, texture overlay) e feature-space (gaussian perturbation, mixing — linea SimpleNet). Usi: branch discriminativa, val_mixed, mappe, curriculum.
- Branch discriminativa DRAEM-like: input+ricostruzione → score + anomaly map densa.
- Memory scoring PatchCore-like: patch features dal backbone, NN distance vs reference set nominale, coreset; come score aggiuntivo.
- Teacher-student RD4AD-like leggero: teacher pretrained frozen, student/decoder, discrepancy multiscala come score ausiliario.
- Training: AdamW, warmup+cosine, EMA, clipping; se adversarial: hinge, spectral norm, R1, TTUR.
- Scoring finale: MAI un solo score. Normalizzazione robusta, weighted/rank fusion → logistic regression fusion su val_mixed. Thresholding da val_mixed / FPR target / EVT.

## 3. Risultati griglia di ablation (dal doc, log reali del progetto)
Fattori (nomi dei run: {cat}_{t}_{m}_{lf}_{oc}_s{seed}):
- teacher: t0 = nessun teacher-student; t1a/t1b = varianti TS; t1c/t1d = varianti TS successive (solo "new space")
- memory: m0 = no memory bank; m1a/m1b; m1c/m1d (new space)
- learned fusion: lf0 = no fusion (score semplice); lf1a/lf1b; lf1c/lf1d
- one-class: oc0 = senza one-class/latent score attivo; oc1 = con
(NB: mappare esattamente cosa sono a/b/c/d dai configs nel repo/outputs — TODO verifica)
- Run totali: old space 1380 + new space 1920 = 3300. Old: mean AUROC 0.7352; New: 0.8392.
- Effetto fattori (mean AUROC globale): lf è IL fattore dominante: lf0 0.6158 → lf1a 0.8250 / lf1b 0.8293 / lf1c 0.8397 / lf1d 0.8365 (+~0.21!).
  teacher: t0 0.7069 → t1a 0.7735 / t1b 0.7740 → t1c 0.8374 / t1d 0.8375.
  memory: m0 0.7474 → m1a 0.7526 / m1b 0.7545 → m1c 0.8356 / m1d 0.8415.
  one-class: oc0 0.8110 vs oc1 0.7489 → oc1 leggermente DANNOSO (config finale usa oc0).
- Shortlist finale (16 epoche, 15 cat × seed 43-46): vincitrice t1d_m1d_lf1c_oc0:
  macro AUROC 0.8510 (std seed 0.0240), AUPRC 0.9261, F1 0.9149, FPR95 0.4475, std cat 0.1249, worst cat 0.6301.
  Runner-up t1a_m1a_lf1b_oc0 (0.8456) — vince più categorie singole (6).
- Best per categoria (grid finale): bottle .9125, cable .6829, capsule .8121, carpet .9524, grid .8589,
  hazelnut 1.0, leather .8940, metal_nut .7021, pill .8028, screw 1.0, tile 1.0, toothbrush .8222,
  transistor .8183, wood 1.0, zipper .9125. Categorie difficili: cable, metal_nut, pill, transistor.
- Ablation memory bank (45 run, 15 cat × p1024/p2048/p4096, layer3, max_train_batches 4, agg=max):
  p1024 mean test AUROC 0.8753 > p2048 0.8611 > p4096 0.8551; ciascuna vince 5 categorie; scelta baseline: layer3/b4/p1024/max.

## 4. Storia successiva (dal lavoro webapp/repo — già nel contesto del progetto)
- Sprint 1 "v3 fixes": scoperti e corretti 3 flag config morti (use_skip_connections, unfreeze_from, scoring_topk) → macro AUROC onesto GAN = 0.7866.
- GAN finale: ocgan_final (multiseed per-categoria) macro 0.8276; ocgan_optv2 (retrain ottimizzato) macro 0.8378 (CSV: final_per_category_multiseed_aggregated.csv, optv2_multiseed_aggregated.csv).
- Sprint 4 svolta PatchCore puro (frozen ImageNet features + memory bank, NO training):
  v1 primo tentativo 0.9051 → v2 0.9397 (topk_mean k=3, coreset 10k) → v3 0.9828/0.9846 production.
- 3 ingredienti del +19.8pp finale: (1) niente coreset quando il bank ci sta (max_patches 70k; zipper 0.7184→0.9801, capsule 0.7724→0.9824); (2) topk_reweighted k=9 (softmax-weighted top-k); (3) multi-scale layer2+layer3 (screw: +layer1, +2.7pp).
- Production finale: wide_resnet50_2 frozen, macro AUROC 0.9846 (bottle/hazelnut/leather/tile/wood = 1.0000, screw 0.9419 peggiore).
- Calibrazione: threshold = p99 degli score su val_normal (15% held-out, seed 43) — NON si può calibrare sulle immagini del bank (distanza ≈ 0).
- Webapp finale: FastAPI server + frontend React/Vite (Evaluation Lab, Models, Test Arena live con SSE, Dataset explorer, Methodology). Arena: production + patchcore v1/v2 ricostruite + ocgan_final/optv2 LIVE dai checkpoint originali.

## 5. Problemi/sfide da raccontare (con pro/contro scelte)
- Repo storico TF/MXNet minimale e datato → porting PyTorch completo (costo alto, ma necessario per ablation).
- Risultati paper misurati 1 volta per cifra → protocollo multi-seed rigoroso (5 seed, mean±std).
- Flag config morti scoperti in Sprint 1 (il builder ignorava use_skip_connections ecc.) → i numeri precedenti erano di un'architettura diversa da quella dichiarata; fix → 0.7866 onesto.
- GAN pixel-centric debole su MVTec: anche modernizzato si ferma ~0.84-0.85; le feature ImageNet frozen lo battono senza training.
- oc1 (one-class score) in media dannoso → tolto nella config finale (esempio di ablation che boccia un componente "amato").
- Memory bank: più patch ≠ meglio con coreset piccolo (p1024>p2048>p4096 con quella config), MA in PatchCore puro niente pruning (70k) è la svolta — apparente contraddizione da spiegare (aggregation max vs topk_reweighted, layer3 solo vs layer2+3).
- fp16/AMP overflow su GPU consumer (Quadro T1000 locale): NaN da layer2 in poi sui backbone optv2 (87 param sbloccati) → fp32 per l'inferenza live.
- Drift del codice di training (Paperspace sync senza commit: base_trainer +1235 righe) → calibrazione archiviata optv2 non riproducibile → ricalibrazione onesta al load (AUROC test 0.8813 bottle live).
- Checkpoint optv2 con architettura diversa dal config (use_skip_connections=true ma pesi BaseReconstructor) → rilevazione architettura dai pesi.
- Paperspace: /notebooks/storage, gestione .Trash-0, df -h nel doc → ambiente cloud GPU (Quadro RTX 5000 16GB), PyTorch 2.1.1+cu121, numpy<2 pinned. Vantaggi: GPU potente a consumo, persistenza storage, Jupyter; svantaggi: sync senza git → drift del codice (lezione imparata!).
- Threshold calibration trap di PatchCore (bank images score ~0).

## 6. Struttura proposta relazione (bozza, da validare)
1. Introduzione e obiettivo (paper OCGAN → MVTec AD industriale)
2. Background: il paper OCGAN nel dettaglio (architettura, training, risultati, limiti)
3. Tecniche moderne usate (glossario ragionato: backbone pretrained, perceptual loss, MS-SSIM, latent compactness/Deep-SVDD, hard-negative mining, synthetic anomalies, branch discriminativa/DRAEM, memory bank/PatchCore, teacher-student/RD4AD, learned fusion, calibrazione/thresholding)
4. Setup sperimentale: MVTec AD, protocollo 4-split, metriche, multi-seed, Paperspace, hardware
5. Fase 1 — OCGAN modernizzato (binario B): architettura del nostro modello, fasi, ablation grid 3300 run, effetto fattori, config finale t1d_m1d_lf1c_oc0, risultati 0.8510 val-shortlist / final multiseed 0.8276-0.8378
6. Fase 2 — la svolta PatchCore: v1→v2→v3/production, i 3 ingredienti, 0.9846
7. Confronto globale e col paper (tabelle+grafici: OCGAN paper su MNIST vs nostro su MVTec; evoluzione macro AUROC 0.7866→0.8276→0.8378→0.9051→0.9397→0.9846)
8. Problemi e lezioni (sezione onesta)
9. Webapp dimostrativa (breve: arena live, evaluation lab)
10. Conclusioni e lavori futuri
Appendici: tabelle per-categoria complete, config, riproducibilità.

## 6bis. VERIFICATO SU DISCO (outputs + logs)
### Griglia su disco (outputs/ocgan-modernized, 11.738 dir)
- Naming: {cat}_{t}_{m}_{lf}_{mp}_{oc}_s{seed}_seed{seed}_{ts}. 192 combo × 15 cat × 4 seed (43-46) ≈ 11.5k run (date 2026-03-28).
- Mappatura fattori VERIFICATA (diff config): t = teacher_student.score_weight (t0x=0.05, t1a=0.1, t1b=0.2, t1d=0.4);
  m = memory_bank.score_weight (m0x=0.02, m1b=0.1, m1d=0.4, m1f=0.6); lf = C logistic regression (lf1b=1, lf1c=2, lf1d=4, lf1g=8);
  mp = max_patches (mp1a=1024, mp1b=2048, mp1c=4096); oc0 = one_class.score_weight 0 (loss_weight 0.1 resta attiva).
- QUINDI: questa è la griglia di TUNING DEI PESI (seconda campagna); la griglia on/off del doc (t0/m0/lf0, old space 1380 + new 1920) è una campagna precedente i cui run NON sono più su disco (pulizia Paperspace).
- Config base griglia: resnet50 frozen, base_reconstructor, L1+MS-SSIM (no MSE) + perceptual 0.1 (l2/3/4), compact_latent dim128 (compactness 0.1),
  TS layer3+4 (loss 0.05), memory bank layer3 l2 kcenter_greedy agg=max, cutpaste p=0.5 + feature synthetic gaussian layer4,
  mining 3 step PGD-like (warmup 1), fusion LR su val_mixed con MAD norm su val_normal, AdamW 1e-4 cosine warmup2, EMA .999, AMP, early stop patience 3 su composite val (auroc .5/auprc .3/f1 .2), 100 epoche max, 256px, anti-leakage sha1 dedup.
### Famiglie di run (diff config verificato)
- {cat}_final (53 dir, 2026-03-28, seed 43-45 +47; cable/metal_nut/pill/transistor con 5 seed): ts_w 0.1, mem_w 0.2, C=5.0, mp1024, cutpaste → final_per_category_multiseed_aggregated.csv (macro 0.8276).
- final_mvtec_t1d_m1d_lf1c_oc0 (15 dir, 2026-04-12 08:5x, seed42, epochs=1, length 8) = smoke/collection run veloci pre-production.
- {cat}_production (15 dir, 2026-04-12 09:0x, seed43, save_best=True): ts_w 0.4, mem_w 0.02, C=1.0, mp4096 → checkpoint deployati in production_models/{cat}/model.pt (= variante arena "ocgan_final").
- {cat}_optv2 (45 dir = 15×3 seed, 2026-04-13): RETRAIN OTTIMIZZATO: unfreeze_from layer3 (lr factor 0.1 → 87 param sbloccati), use_skip_connections=true (MA il builder dell'epoca lo ignorava → checkpoint base_reconstructor!), perlin invece di cutpaste, augmentation più forti (rot 5°, jitter .08, noise .008), scoring_topk=100, LR fusion cv_folds=5, model.scoring: memory_score=False, teacher_student_score=False (score solo recon/perceptual/feature), patience 5 → optv2_multiseed_aggregated.csv (macro 0.8378).
### Numeri per-categoria GAN (mean test AUROC multiseed)
- final: bottle .8542, cable .5231(!), capsule .7839, carpet .9238, grid .9394, hazelnut .9891, leather .8976, metal_nut .6085, pill .6707, screw .9995, tile .9753, toothbrush .7037, transistor .7377, wood .9889, zipper .8191
- optv2: bottle .8958, cable .5797, capsule .6268, carpet .9069, grid .9687, hazelnut .9914, leather .9678, metal_nut .6228, pill .7309, screw 1.0, tile .9407, toothbrush .6519, transistor .7878, wood 1.0, zipper .8958
- STORIA FORTE: GAN screw=1.0 (PatchCore lì è il peggiore .9419); GAN cable .52-.58 (PatchCore v2 .98!). Complementarità dei segnali.
### PatchCore (logs/*.csv, tutti 3 seed 43-45)
- patchcore_pure.csv = v1: layer3(?), topk_mean k=3, coreset 10k → capsule .769, pill .631, zipper deboli; ~35 s/cat.
- patchcore_tuning.csv: esplorazione (pill/grid/zipper/hazelnut/capsule/screw): topk_reweighted k9 > topk_mean; layer2+layer3 > layer3; pill .534 (topk_mean3 layer3) vs .872 (l2+l3 reweighted) — esempio drammatico.
- patchcore_v2.csv = v2: layer2+layer3, topk_reweighted k9, coreset 10k → bottle 1.0, cable .98, metal_nut .987; capsule ancora .81, grid .92.
- patchcore_lc.csv: coreset 50k (zipper .9801, capsule .9824, screw .9204 ma 468 s → kcenter su 50k costosissimo).
- patchcore_v3.csv = v3/production: max_patches 70k senza coreset → 5-7 s/cat (più veloce E migliore di v1/v2!), transistor .9933, metal_nut .9924, pill .958, screw .9147 (l2+l3).
- patchcore_p1/p1_ext.csv: ablation layer1+2+3: screw .9419 (+2.7pp), grid/toothbrush/pill nessun miglioramento → override solo screw.

### Hardware/ambiente (env_info.yaml verificati)
- Grid + final + production: Paperspace Gradient (Linux, cwd /notebooks/storage/...), NVIDIA **RTX A4000** 16GB, Python 3.11.7, torch 2.10.0+cu128, hostnames diversi per sessione (nzujijoayr, n3jmv9eyss, na1gz7w04r) → istanze effimere.
- optv2: **Quadro RTX 5000** 16GB, torch 2.1.1+cu121 (istanza diversa, 2026-04-13).
- Inferenza live locale webapp: Quadro T1000 (qui è emerso l'overflow fp16).
- Tutti i run committati allo stesso git_commit e4d4a3f (ma con drift locale non committato — lezione!).
### Architettura GAN modernizzato (codice verificato)
- ReconstructionModel: backbone resnet50 frozen → feats layer1-4 + global → projection MLP (Linear→ReLU→Linear) global_dim→latent 128 → reconstructor → immagine ricostruita → ri-encoding col backbone (recon_layer1-4 + recon_global per perceptual/feature score).
- BaseReconstructor: fc latent→256×8×8, poi 5× [Upsample bilineare ×2 → Conv3×3 → BN → ReLU] da 256ch a 3ch, Sigmoid (8→256 px). Varianti: residual_reconstructor, unet_reconstructor (skip features layer1-4).
- Score heads (7, fusione LR): norm_recon, norm_perceptual, norm_feature, norm_latent, norm_memory, norm_discriminative, norm_teacher_student; normalizzazione MAD su val_normal; soglia best-F1 su val_mixed.
### benchmarks.json (webapp, fonte per grafici)
macros: ocgan_final .8276, ocgan_optv2 .8378, patchcore_v1 .9051, patchcore_v2 .9397, patchcore_v3 .9828, patchcore_p1 .9562, production_final .9846. per_category: dict modello → lista per categoria.
### Materiale extra per la relazione
- Screenshot webapp già pronti: frontend/visual-shots/*.png (01-home, 06-arena-config, 07-arena-production-done, 09-arena-gan-done, 08-result-modal, 02-evaluation...).
- Paper OCGAN numeri per confronto: MNIST P2 0.9750, CIFAR10 P2 0.6566, ablation AE .957→+Dl .959→+Dv .971→full .975.

## 7. TODO operativi
- [ ] Esplorare D:\OCGAN\project\storage_project_outputs_datasets\outputs (run dirs GAN: csv aggregati multiseed, config)
- [ ] Esplorare ocgan-modernized/logs (patchcore_v3.csv, p1 ablation) e configs/ per mappare t1a..t1d, m1a..m1d, lf1a..lf1d, oc0/1 al significato esatto
- [ ] Verificare numeri: 0.7866 (post-fix), 0.8276 final, 0.8378 optv2, 0.9051 v1, 0.9397 v2, 0.9828 v3, 0.9846 production (benchmarks.json)
- [ ] Decidere formato output (proposta: Markdown master + grafici matplotlib PNG + conversione HTML→PDF via Edge headless; lingua: ITALIANO)
- [ ] Generare grafici: evoluzione macro AUROC; effetto fattori ablation; per-categoria production vs GAN; old vs new space; paper MNIST table vs nostro; heatmap modelli×categorie (riusare dati webapp benchmarks.json)
- [ ] Scrivere la relazione capitolo per capitolo
