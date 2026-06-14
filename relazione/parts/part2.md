---

# 4. Setup sperimentale: dataset, protocollo, infrastruttura

## 4.1 Il dataset: MVTec AD

Il banco di prova del progetto è **MVTec AD**, lo standard de facto per l'anomaly detection industriale: **15 categorie** (10 oggetti: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper; 5 texture: carpet, grid, leather, tile, wood), immagini ad alta risoluzione, training set **solo defect-free** e test set misto con difetti reali (graffi, contaminazioni, parti mancanti, deformazioni…) annotati anche a livello pixel.

La scelta di MVTec — e non MNIST/CIFAR come nel paper — è la decisione che dà senso a tutto il progetto: volevamo misurare le idee di OCGAN **sul tipo di problema per cui l'anomaly detection one-class esiste davvero**. Il prezzo: nessun confronto diretto numero-su-numero col paper (sezione 7.3); il guadagno: ogni conclusione è rilevante per un caso d'uso reale.

Pipeline di preprocessing: resize anti-aliased che preserva l'aspect ratio + center crop a **256×256**, normalizzazione ImageNet (necessaria con backbone pre-addestrati), augmentation conservative sul train (piccole traslazioni/rotazioni, jitter di luminosità, rumore lieve) — conservative perché un'augmentation aggressiva *cambia la definizione di normalità* che il modello deve imparare.

## 4.2 Il protocollo a 4 split

Tutti gli esperimenti usano **sempre** quattro split:

| Split | Contenuto | Uso |
|---|---|---|
| `train_normal` | solo immagini normali | addestramento / costruzione bank |
| `val_normal` | normali tenute fuori | normalizzazione MAD degli score, calibrazione p99 |
| `val_mixed` | normali + anomalie | scelta di soglie, checkpoint, iperparametri, fusione |
| `test_blind` | il test ufficiale | **solo valutazione finale, mai per tuning** |

Più anti-leakage: deduplicazione (hash SHA-1) e near-duplicate check tra split, statistiche di normalizzazione calcolate solo sul train. È la risposta diretta al limite metodologico del paper (misure singole, nessuna separazione esplicita tra selezione e valutazione): qui ogni numero "test" è davvero blind.

## 4.3 Le metriche

- **AUROC** (image-level) — metrica principale, threshold-free, confrontabile con la letteratura.
- **AUPRC** — più informativa quando le classi sono sbilanciate.
- **F1 alla soglia di validazione** — quanto è buono il sistema *operativo*, non solo il ranking.
- **FPR@95TPR** — quanti falsi allarmi per catturare il 95% dei difetti: la metrica che interessa a una linea di produzione.

Il "macro AUROC" citato in tutta la relazione è la media non pesata sulle 15 categorie.

## 4.4 Multi-seed obbligatorio

Ogni risultato finale è la media di **3–5 seed** (43–47) con deviazione standard riportata (tabella A2). Il motivo è già stato detto: i numeri pubblici di OCGAN sono misurati una volta sola, e la nostra esperienza ha mostrato deviazioni standard per-categoria fino a ±0.17 (toothbrush nel GAN finale!) — un singolo run può raccontare qualunque storia.

## 4.5 Paperspace: l'infrastruttura cloud del progetto

Tutti i training sono stati eseguiti su **Paperspace Gradient**.

**Cos'è.** Una piattaforma cloud di GPU computing orientata al machine learning: si lancia un *notebook* (un container Linux con Jupyter/terminale) su una macchina con GPU dedicata, pagando a consumo. Lo storage di progetto (`/notebooks/storage/...`) è **persistente** e sopravvive allo spegnimento dell'istanza, mentre la macchina vera e propria è **effimera**: a ogni sessione si riceve un host diverso (nei nostri `env_info.yaml` compaiono hostname sempre nuovi: `nzujijoayr`, `n3jmv9eyss`, `na1gz7w04r`…).

**Perché è così vantaggioso per un progetto come questo:**

1. **GPU da workstation a costo orario**: la griglia di ablation è girata su **NVIDIA RTX A4000 (16 GB)** e i retrain optv2 su **Quadro RTX 5000 (16 GB)** — hardware fuori portata per un laptop, affittato solo per le ore di training effettive.
2. **Zero amministrazione**: ambiente CUDA pronto (Python 3.11.7, PyTorch 2.10.0+cu128 sull'istanza della griglia; 2.1.1+cu121 su quella di optv2), niente driver da installare.
3. **Storage persistente**: dataset, checkpoint e log restano in `/notebooks/storage` tra una sessione e l'altra; le ~11.700 directory di output della griglia sono state scritte lì e poi sincronizzate in locale.
4. **Scalabilità del "fire-and-forget"**: lanci uno script di griglia la sera, spegni il laptop, l'istanza macina; il costo segue l'uso.

**I contro, vissuti sulla nostra pelle** (dettagli in sezione 8.7):

- **Istanze effimere + sync manuale senza git = drift del codice.** Il codice veniva sincronizzato tra locale e cloud copiando file; alcune modifiche fatte "al volo" su Paperspace (il `base_trainer.py` è arrivato a +1235 righe non committate) non sono mai rientrate nel repository → mesi dopo, la calibrazione archiviata di optv2 non era riproducibile con il codice committato.
- **La pulizia dello storage è a tuo carico**: i run cancellati finiscono in `.Trash-0` e continuano a occupare quota; la prima campagna di griglia (3.300 run) è stata eliminata dallo storage per far posto alla seconda — sopravvive solo nei CSV aggregati e nel diario di progetto.
- GPU disponibili a rotazione: il tipo di scheda può cambiare tra sessioni (A4000 ↔ RTX 5000), il che è innocuo per i risultati ma è un'altra variabile da registrare (per questo ogni run salva `env_info.yaml`).

**Il ruolo del laptop locale** (Quadro T1000, 4 GB): sviluppo, smoke test, e soprattutto **l'inferenza live della webapp** (sezione 9) — dove l'hardware consumer ci ha regalato il bug più istruttivo del progetto (overflow fp16, sezione 8.6).

## 4.6 Riproducibilità by design

Ogni run salva: configurazione YAML completa risolta, `env_info.yaml` (GPU, versioni, hostname, commit git), seed, checkpoint top-k, log per epoca. Il repository è stato rifondato all'inizio del progetto proprio attorno a questi requisiti (sezione 5.2).

---

# 5. Fase 1 — OCGAN modernizzato (il "binario B")

## 5.1 La strategia

All'inizio del progetto erano sul tavolo due binari:

- **Binario A** — riprodurre e difendere OCGAN così com'è, portandolo su MVTec con modifiche minime.
- **Binario B** — trattare OCGAN come **fonte di idee** (ricostruzione + controllo del latent + mining) e innestarle in una pipeline allo stato dell'arte: backbone pre-addestrato, score multipli, anomalie sintetiche, memory bank, teacher–student, fusione appresa.

Abbiamo scelto il binario B, per una ragione che il paper stesso suggerisce (sezione 2.8): l'architettura 2019 è progettata per immagini 28–32 px mono-concetto, e nessuna quantità di tuning l'avrebbe resa competitiva su immagini industriali a 256 px. I riferimenti moderni da cui abbiamo attinto: PaDiM, **PatchCore**, DRAEM, RD4AD, EfficientAD, SimpleNet.

*Pro della scelta*: ogni componente moderno è misurabile via ablation; il progetto produce un sistema rilevante, non un reperto. *Contro*: il "modello" diventa una pipeline complessa con molti iperparametri — è il motivo per cui la griglia di ablation (5.5–5.6) è stata così grande.

## 5.2 La rifondazione del repository

Prima di qualunque esperimento, il codice è stato riscritto da zero in **PyTorch modulare**: `configs/` (Hydra-style, ogni run salva il config risolto), `datasets/`, `models/` (backbone, reconstruction, score heads), `losses/`, `miners/`, `scorers/`, `metrics/`, `trainers/`, `callbacks/`, `scripts/`. Più l'hardening: seed e determinismo, resume, checkpoint top-k, logging d'ambiente, multi-run, smoke test e NaN-test automatici, AMP, gradient accumulation, EMA, profiler.

*Pro*: la griglia da 11.000+ run sarebbe stata impossibile senza run "lanciabili a lotti" e auto-documentanti. *Contro*: settimane di ingegneria prima del primo numero utile — un investimento che si è ripagato, ma che all'inizio è sembrato lentezza.

## 5.3 L'architettura del nostro modello, nel dettaglio

Il modello finale della fase 1 (`ReconstructionModel` nel codice) è organizzato così:

```
immagine 256×256
   │
   ▼
ResNet50 pre-addestrato, CONGELATO          ──► feature layer1…layer4 + global pool
   │                                              │
   ▼                                              │ (le stesse feature alimentano:
projection MLP (Linear → ReLU → Linear)           │  perceptual loss, memory bank,
   │        global_dim → latent 128               │  teacher–student, feature score)
   ▼                                              │
latent compatto z ∈ R^128  (compactness 0.1)      │
   │                                              │
   ▼                                              ▼
BaseReconstructor (decoder)                ri-encoding della ricostruzione
   fc: 128 → 256·8·8                       con lo stesso backbone congelato
   poi 5 × [Upsample ×2 bilineare           (recon_layer1…4 + recon_global)
            → Conv 3×3 → BN → ReLU]
   → Conv finale + Sigmoid
   8×8 → 256×256, canali 256→3
```

**Come funziona il forward.** L'immagine passa nel backbone congelato; il global pooling viene proiettato da un MLP in un latent di 128 dimensioni; il decoder (`BaseReconstructor`) risale da 8×8 a 256×256 con cinque blocchi upsample+conv (upsampling bilineare seguito da convoluzione: la scelta classica anti-checkerboard, al posto delle deconvoluzioni del paper 2019). La ricostruzione viene **ri-encodata dallo stesso backbone**, così da poter confrontare originale e ricostruzione anche in feature space.

**Le 7 teste di score.** Da questo flusso il modello estrae sette punteggi di anomalia, ognuno normalizzato con MAD su `val_normal` (sezione 3.11):

| Testa | Segnale |
|---|---|
| `norm_recon` | errore di ricostruzione in pixel (L1 + MS-SSIM) |
| `norm_perceptual` | distanza tra feature di originale e ricostruzione (layer 2–4) |
| `norm_feature` | discrepanza di feature aggiuntiva (global) |
| `norm_latent` | anomalia nello spazio latente |
| `norm_memory` | distanza kNN da un piccolo memory bank di patch normali (layer3) |
| `norm_discriminative` | branch discriminativa addestrata su anomalie sintetiche |
| `norm_teacher_student` | discrepanza teacher–student (layer 3–4) |

La **regressione logistica su `val_mixed`** fonde le sette teste; soglia a best-F1.

**Confronto diretto con OCGAN 2019** (architettura per architettura):

| | OCGAN (paper) | Nostro modello |
|---|---|---|
| Encoder | 3 conv da zero, 64 ch | ResNet50 ImageNet **congelato** |
| Latent | tanh-bounded (−1,1)^d, riempito uniformemente (D_l) | 128-d, **compattato** attorno a un centro (Deep-SVDD-like) |
| Decoder | 3 deconv | 5 blocchi upsample bilineare + conv 3×3 + BN + ReLU |
| Loss ricostruzione | 10·MSE | **L1 + MS-SSIM + perceptual 0.1** (niente MSE) |
| Negativi | generazioni da latent casuali | **anomalie sintetiche** (CutPaste / Perlin / gaussian feature) |
| Mining | 5 step gradient ascent guidati dal classifier | 3 step PGD-like guidati dallo score composito, warmup 1 epoca |
| Score | solo MSE | **7 teste fuse con regressione logistica** |
| Discriminatori adversariali | D_l + D_v | nessuno (sostituiti da compactness + branch discriminativa supervisionata) |

L'ultima riga merita una nota: abbiamo rinunciato al training adversariale vero e proprio (instabile, sensibile, costoso da bilanciare) e mantenuto la *funzione* dei due discriminatori con mezzi più stabili — la compattezza del latent al posto di D_l, la branch discriminativa su anomalie sintetiche al posto di D_v + classifier. *Pro*: training stabile e riproducibile su 15 categorie × decine di configurazioni. *Contro*: si perde l'eleganza della "pattuglia del latent space" del paper; il mining ne conserva lo spirito.

**Varianti del decoder** disponibili nel codice: `base_reconstructor` (sopra), `residual_reconstructor` (blocchi residui), `unet_reconstructor` (skip connection dalle feature layer1–4 dell'encoder). Le skip sono a doppio taglio in anomaly detection: troppo forti, e il decoder "copia" l'input **difetti compresi**, azzerando il segnale di ricostruzione. La configurazione finale usa il `base_reconstructor` puro — e proprio attorno al flag `use_skip_connections` è nato uno dei bug più istruttivi del progetto (sezioni 5.7 e 8.8).

## 5.4 Il training

Configurazione comune a tutta la fase sperimentale: AdamW lr 1e-4, warmup 2 epoche + cosine, EMA 0.999, AMP, gradient clipping, early stopping (pazienza 3) sulla metrica composita di `val_mixed`, massimo 100 epoche, batch a 256 px. Anomalie sintetiche: CutPaste con p = 0.5 in pixel space + perturbazione gaussiana al layer4 in feature space. Mining: 3 step, attivo dopo la prima epoca.

## 5.5 Campagna 1 — la griglia on/off (3.300 run): cosa serve davvero?

La prima campagna di ablation ha acceso e spento i quattro componenti principali — **t**eacher–student, **m**emory bank, **l**earned **f**usion, **o**ne-**c**lass score — su tutte le 15 categorie, in due "spazi" di configurazione:

- **old space** (1.380 run): iperparametri di contorno della prima implementazione → AUROC medio **0.7352**;
- **new space** (1.920 run): dopo la revisione di preprocessing/loss/schedule → **0.8392**.

![Old space vs new space](figures/fig3_old_new_space.png)
*Figura 3 — Stessa griglia di componenti, due spazi di iperparametri: il contesto vale +0.10 AUROC prima ancora di toccare i componenti.*

L'effetto dei singoli fattori (AUROC medio su tutti i run della campagna):

![Effetto dei fattori](figures/fig2_fattori_ablation.png)
*Figura 2 — Effetto marginale dei quattro fattori della griglia on/off.*

| Fattore | Spento | Acceso (migliore variante) | Δ |
|---|---|---|---|
| **Learned fusion** | 0.6158 | **0.8397** (lf1c) | **+0.224** |
| Teacher–student | 0.7069 | 0.8375 (t1d) | +0.131 |
| Memory bank | 0.7474 | 0.8415 (m1d) | +0.094 |
| One-class score | **0.8110** (oc0) | 0.7489 (oc1) | **−0.062** |

Tre conclusioni nette:

1. **La fusione appresa è IL fattore**: +0.22 da sola. Un singolo score, per quanto buono, non basta su 15 categorie eterogenee: ciò che separa normale da difettoso su `carpet` (texture) non è ciò che li separa su `transistor` (struttura). La regressione logistica impara, per categoria, *quale mix di segnali guardare*.
2. Teacher–student e memory bank aiutano sostanzialmente — sono i due segnali "a feature congelate", e qui c'è il presagio della fase 2: **i segnali che non dipendono dal training del GAN sono i più affidabili**.
3. **Lo score one-class fa danno** (−0.06 in media): la distanza dal centro del latent è ridondante rispetto alle altre teste e rumorosa. È stato disattivato (oc0) nella configurazione finale **nonostante fosse l'erede concettuale del cuore di OCGAN** — l'ablation non guarda in faccia l'affetto per le idee. (La regolarizzazione di compattezza in loss resta attiva: aiuta la geometria del latent anche senza usare quella distanza come score.)

> **Nota di archivio**: i 3.300 run di questa campagna sono documentati nei CSV aggregati e nel diario di progetto; le directory originali sono state eliminate dallo storage Paperspace per liberare quota (sezione 4.5). La campagna i cui run sono ancora integralmente su disco è la seconda, qui sotto.

## 5.6 Campagna 2 — il tuning dei pesi (11.738 run su disco)

Stabilito *cosa* tenere acceso, la seconda griglia ha cercato *quanto* pesare ogni componente. Sul disco del progetto ci sono **11.738 directory di run** (date 28/03/2026, RTX A4000), con naming `{categoria}_{t}_{m}_{lf}_{mp}_{oc}_s{seed}`: 192 combinazioni × 15 categorie × 4 seed (43–46). La mappatura dei codici — ricostruita e verificata facendo il diff dei config YAML salvati nei run — è:

| Codice | Parametro | Valori |
|---|---|---|
| `t0x/t1a/t1b/t1d` | `teacher_student.score_weight` | 0.05 / 0.1 / 0.2 / 0.4 |
| `m0x/m1b/m1d/m1f` | `memory_bank.score_weight` | 0.02 / 0.1 / 0.4 / 0.6 |
| `lf1b/lf1c/lf1d/lf1g` | C della regressione logistica | 1 / 2 / 4 / 8 |
| `mp1a/mp1b/mp1c` | `max_patches` del memory bank | 1024 / 2048 / 4096 |
| `oc0` | `one_class.score_weight` = 0 | (loss di compattezza 0.1 sempre attiva) |

Sotto-ablation interessante del memory bank ausiliario (45 run dedicati, layer3, aggregazione max): **più patch NON è meglio** — p1024 dà AUROC medio 0.8753 contro 0.8611 (p2048) e 0.8551 (p4096). Con aggregazione `max` e coreset k-center, un bank più grande aggiunge più rumore di quanto aggiunga copertura. Tenete a mente questo risultato: in PatchCore puro scopriremo l'esatto contrario (bank pieno = svolta), ed è un'apparente contraddizione che si scioglie solo guardando *come* il bank viene interrogato (sezioni 6.4 e 8.5, figura 9).

**La shortlist finale** (16 epoche, 15 categorie × seed 43–46) ha incoronato:

> **`t1d_m1d_lf1c_oc0`** — teacher 0.4, memory 0.4, C = 2, niente one-class score:
> macro AUROC **0.8510** (±0.0240 tra seed), AUPRC 0.9261, F1 0.9149, FPR@95 0.4475; peggior categoria 0.6301.

Runner-up `t1a_m1a_lf1b_oc0` (0.8456), che però vinceva più categorie singole (6): la vincitrice è stata scelta per la media e la stabilità, non per i picchi. *Pro di questa scelta*: robustezza cross-categoria, che è ciò che un sistema "general purpose" deve avere. *Contro*: rinuncia a ~6 vittorie per-categoria — il primo indizio che "un modello solo per tutte le categorie" è un vincolo costoso.

## 5.7 Lo Sprint 1: i flag morti e il reset dell'onestà

Durante il consolidamento del codice è emerso che **tre flag di configurazione erano "morti"**: `use_skip_connections`, `unfreeze_from` e `scoring_topk` venivano letti dal config ma **ignorati dal builder del modello**. Tutti i run che credevano di usare skip connection o backbone parzialmente sbloccato avevano in realtà addestrato il `base_reconstructor` puro su backbone interamente congelato.

Conseguenza: parte dei numeri storici descriveva un'architettura diversa da quella dichiarata. Dopo il fix e il re-test, la baseline onesta della famiglia GAN si è assestata a **macro AUROC 0.7866**. È il punto di partenza della curva di figura 1 — più basso dei numeri "pre-fix", ma vero. I dettagli e la morale in sezione 8.3.

## 5.8 Le tre famiglie finali del GAN: `final`, `production`, `optv2`

Dalla configurazione vincente sono derivate tre famiglie di run (tutte verificate sul disco con diff dei config):

**`{cat}_final`** — la misura ufficiale multi-seed (53 run, marzo 2026, seed 43–45, esteso a 5 seed sulle categorie difficili cable/metal_nut/pill/transistor): pesi di score teacher 0.1 / memory 0.2, C = 5, `max_patches` 1024, CutPaste. → **macro AUROC 0.8276** (tabella A2).

**`{cat}_production`** — il run per i checkpoint deployati nella webapp (15 run, aprile 2026, seed 43, `save_best=True`): teacher 0.4 / memory 0.02, C = 1, `max_patches` 4096. I suoi checkpoint sono i `production_models/{cat}/model.pt` serviti live dall'arena come variante "ocgan_final" (sezione 9).

**`{cat}_optv2`** — il **retrain ottimizzato** (45 run = 15 categorie × 3 seed, aprile 2026, Quadro RTX 5000), l'esperimento "spingiamo il GAN al massimo":

- backbone **parzialmente sbloccato** da layer3 in su (87 parametri, learning rate ridotto ×0.1);
- `use_skip_connections=true` nel config (ma il builder dell'epoca lo ignorava ancora: i checkpoint sono `base_reconstructor` — sezione 8.8);
- anomalie sintetiche **Perlin** al posto di CutPaste;
- augmentation più forti (rotazioni 5°, jitter 0.08, rumore 0.008);
- `scoring_topk` = 100; fusione con cross-validation a 5 fold; pazienza early-stop 5;
- **memory score e teacher–student score disattivati** (score solo da ricostruzione/perceptual/feature) — un esperimento di parsimonia.

→ **macro AUROC 0.8378** (+0.010 sul final, tabella A2): qualche categoria sale molto (leather +0.07, zipper +0.08, screw → 1.0000), altre crollano (capsule −0.16 con std ±0.16 — instabile tra seed). *Pro di optv2*: dimostra che margini ci sono ancora. *Contro*: +0.01 macro al prezzo di un retrain completo e di più instabilità — il rapporto costo/beneficio della famiglia GAN era ormai chiaro.

![GAN per categoria](figures/fig4_gan_per_categoria.png)
*Figura 4 — `final` vs `optv2` per categoria (test AUROC, media multi-seed). Le categorie "strutturali" (cable, metal_nut, pill, transistor, toothbrush) restano il tallone d'Achille di tutta la famiglia.*

## 5.9 Lettura critica della fase 1

Il quadro a fine fase 1:

- la modernizzazione **funziona**: dal 0.7866 onesto al 0.8276–0.8378 multi-seed, con punte per-categoria notevoli (screw **0.9995–1.0000**, hazelnut 0.989, grid 0.94–0.97, wood 0.99–1.0);
- ma la famiglia **satura attorno a 0.83–0.84**: le categorie dove il difetto è un dettaglio strutturale fine su un oggetto complesso (cable 0.52–0.58!, metal_nut 0.61–0.62, pill 0.67–0.73) non si spostano, qualunque sia il tuning;
- e i segnali che reggono meglio sono proprio quelli **che non si addestrano** (memory, teacher–student — entrambi su feature ImageNet congelate).

La conclusione operativa fu quasi obbligata: *se i segnali a feature congelate sono i migliori del nostro GAN, cosa succede se togliamo tutto il resto?* È la domanda da cui nasce la fase 2.
