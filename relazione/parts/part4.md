---

# 8. Problemi, sfide e lezioni imparate

Questa sezione raccoglie tutto ciò che è andato storto o ha sorpreso, in ordine più o meno cronologico. Per ogni episodio: cosa è successo, come l'abbiamo risolto, e il bilancio della scelta fatta.

## 8.1 Il repo di partenza inservibile

**Problema.** Il codice ufficiale di OCGAN è in MXNet, framework di fatto abbandonato; l'alternativa pubblica in TensorFlow è minimale e misura ogni cifra una sola volta.
**Scelta**: riscrivere tutto da zero in PyTorch modulare (sezione 5.2), invece di rattoppare il porting.
**Pro**: senza quella base (config salvati, multi-run, seed, smoke test) le campagne da migliaia di run sarebbero state ingestibili. **Contro**: settimane senza "numeri nuovi"; la tentazione di saltare l'ingegneria era forte ed è giusto ammetterlo.

## 8.2 Numeri di letteratura senza varianza

**Problema.** I risultati pubblici di OCGAN sono single-run; replicandoli si ottengono valori che ballano.
**Scelta**: multi-seed obbligatorio (3–5) con std dichiarata, ovunque.
**Pro**: scoperte come la std ±0.17 di toothbrush (GAN) cambiano le conclusioni. **Contro**: costo computazionale ×3–5 su ogni esperimento — è il prezzo dell'affidabilità, e Paperspace l'ha reso pagabile.

## 8.3 I flag di configurazione "morti" (il bug più importante del progetto)

**Problema.** Tre opzioni del config (`use_skip_connections`, `unfreeze_from`, `scoring_topk`) venivano accettate ma **ignorate dal builder**: esperimenti che "confrontavano" architetture diverse stavano in realtà rieseguendo la stessa. Scoperto nello Sprint 1 ispezionando i checkpoint (i pesi non contenevano le skip che il config dichiarava).
**Scelta**: fix del builder + re-test completo + **azzeramento dei numeri storici** non riproducibili → baseline onesta 0.7866.
**Pro**: da lì in avanti ogni numero della relazione è difendibile. **Contro**: giorni di lavoro e numeri "peggiori" sulla carta.
**Lezione (la più generalizzabile del progetto)**: *un config accettato non è un config applicato*. Da allora: test automatico che ogni chiave di config raggiunga un effetto misurabile sul modello (il numero di parametri o l'architettura serializzata devono cambiare).

## 8.4 L'ablation boccia il componente "affezionato"

**Problema.** Lo score one-class (distanza dal centro del latent) — l'erede più diretto dell'idea centrale di OCGAN — in media **toglie** 6 punti di AUROC (sezione 5.5).
**Scelta**: disattivarlo (oc0) nella configurazione finale, mantenendo la compactness in loss.
**Pro**: +0.06 di AUROC medio e un esempio perfetto di "i dati battono l'affetto". **Contro**: nessuno, tecnicamente; narrativamente, ammettere che il pezzo più "OCGAN" del modello non aiutava è stato il momento più umbratile del binario B.

## 8.5 Il paradosso del memory bank (due regimi)

**Problema.** Dentro il GAN, allargare il bank ausiliario peggiorava (p1024 0.8753 > p4096 0.8551); in PatchCore puro, riempire il bank è stata la svolta (+0.04 macro). Contraddizione?
**Spiegazione.** No: cambiano interrogazione e ruolo. Nel GAN il bank (coreset k-center piccolo) era letto con **aggregazione max** — più patch = più probabilità che una singola distanza rumorosa diventi lo score — e il suo output era *uno dei sette* segnali, riconciliato dalla fusione. In PatchCore il bank pieno è letto con **top-k reweighted**, robusto al rumore del singolo vicino, ed è *l'unico* segnale: lì la copertura è tutto.
**Lezione**: i componenti non hanno proprietà assolute; hanno proprietà *nel contesto della pipeline*. ("Più memoria è meglio?" non è una domanda ben posta; "più memoria, letta come?" sì.)

## 8.6 Overflow fp16 sull'hardware consumer

**Problema.** Durante l'integrazione live nella webapp, i checkpoint `optv2` (gli unici con 87 parametri di backbone sbloccati) producevano score **tutti zero** sulla GPU locale (Quadro T1000): attivazioni NaN/Inf da layer2 in poi in **fp16**. In training su A4000/RTX 5000 l'AMP non aveva mai dato problemi; in inferenza su una GPU senza tensor core moderni e con range numerici al limite, sì.
**Scelta**: inferenza live in **fp32** per i checkpoint GAN.
**Pro**: corretto e stabile. **Contro**: ~2× più lento — irrilevante per una demo (4 s per 10 immagini restano interattivi).
**Lezione**: l'AMP è una proprietà del *deployment*, non del modello: va ri-validata su ogni hardware di destinazione.

## 8.7 Il drift del codice su Paperspace

**Problema.** Il flusso di lavoro cloud sincronizzava i file **senza passare da git**: sul `base_trainer.py` dell'istanza si erano accumulate +1235 righe mai committate. Mesi dopo, la calibrazione archiviata di `optv2` (mediane/MAD/pesi di fusione salvati nei run) **non era riproducibile** con il codice del repository: gli score live uscivano su una scala diversa da quella attesa.
**Scelta**: invece di inseguire la versione perduta del codice, **ricalibrazione onesta al load**: al primo caricamento di un checkpoint optv2, la webapp ricalcola mediane/MAD su `val_normal` e ri-fitta la fusione su `val_mixed` con il codice attuale (verificato: AUROC test live 0.8813 su bottle, plausibile e stabile).
**Pro**: risultato verificabile oggi, con il codice di oggi. **Contro**: i numeri live non coincidono al decimale con i CSV d'archivio (e la relazione lo dichiara).
**Lezione**: *mai* sincronizzare codice verso il cloud fuori da git. Una riga di `git commit && git push` prima di ogni run costa 10 secondi; la sua assenza è costata giorni.

## 8.8 Checkpoint che mentono sul proprio config

**Problema.** I run `optv2` dichiarano `use_skip_connections=true` nel config salvato, ma i pesi nei checkpoint sono di un `base_reconstructor` senza skip (il flag era ancora morto al tempo del lancio — strascico del bug 8.3). Caricarli costruendo il modello "dal config" falliva.
**Scelta**: il loader della webapp **deduce l'architettura dai pesi** (dalle chiavi/forme dello state_dict), non dal config.
**Lezione**: in caso di conflitto, *i pesi sono la verità*; il config è una dichiarazione d'intenti.

## 8.9 La trappola della calibrazione di PatchCore

**Problema.** Calibrare la soglia di PatchCore sugli stessi dati del bank produce soglie assurde: quelle immagini hanno distanza ≈ 0 dal bank *per costruzione* (il loro vicino più prossimo sono loro stesse).
**Scelta**: 15% di `train_normal` (seed 43) tenuto **fuori** dal bank come `val_normal` di calibrazione; soglia = 99° percentile (sezione 6.5).
**Pro**: falsi positivi attesi ~1% per costruzione. **Contro**: il bank perde il 15% delle patch — trascurabile visto il margine.
**Lezione**: per i modelli a memoria, "training set" e "set di calibrazione" devono essere disgiunti *anche se nessuno dei due serve a un training nel senso classico*.

## 8.10 L'outlier hazelnut: 799 secondi

**Problema.** In v3/production tutte le categorie costruiscono il bank in 4–8 s, tranne hazelnut: **799 s, su tutti e tre i seed**.
**Diagnosi** (verificata nel codice: il coreset scatta solo `if bank > max_patches`): hazelnut ha il train set più grande di MVTec (391 immagini) ed è **l'unica categoria che sfora le 70.000 patch** → solo lì il k-center-greedy si attiva, e su un pool così grande è lento.
**Scelta**: lasciato così — 13 minuti una tantum per la categoria col bank più ricco (e AUROC 1.0000) non valgono un parametro in più.
**Lezione**: un outlier di *tempo* costante tra i seed non è rumore: è un ramo di codice diverso. I log dei tempi sono diagnostica, non contorno.

## 8.11 La prima campagna cancellata

**Problema.** Per liberare quota sullo storage Paperspace, le directory dei 3.300 run della campagna on/off sono state eliminate; sopravvivono i CSV aggregati e il diario di progetto.
**Bilancio**: accettabile (i numeri aggregati bastano per ogni conclusione tratta), ma i config esatti di quella campagna non sono più ispezionabili — infatti in questa relazione la mappatura fine dei fattori è dichiarata solo per la campagna 2, verificata sui config su disco.
**Lezione**: prima di cancellare run, archiviare *almeno* un config per combinazione.

## 8.12 Riepilogo scelte / pro / contro

| Scelta | Vantaggio | Svantaggio |
|---|---|---|
| Binario B (modernizzare, non difendere) | sistema rilevante, ablation possibili | complessità, tanti iperparametri |
| MVTec invece di MNIST | conclusioni industrialmente sensate | nessun confronto diretto col paper |
| Riscrittura PyTorch completa | 11k run gestibili, riproducibilità | settimane di setup |
| Protocollo 4-split + multi-seed | numeri difendibili | costo ×3–5 |
| Paperspace | GPU 16 GB a consumo, storage persistente | drift di codice se si lavora fuori git |
| Backbone congelato | generalità preservata, zero overfit | cieco a domini lontani da ImageNet |
| Fusione appresa (LR) | +0.22, adattiva per categoria | richiede `val_mixed` con anomalie (sintetiche) |
| oc0 (no one-class score) | +0.06 | si rinuncia a un'eredità OCGAN |
| PatchCore bank pieno | +0.04 macro e 6× più veloce | memoria O(n); hazelnut 799 s |
| Config per-categoria (layer1 solo screw) | +1.8pp finale senza costi altrove | il sistema non è più "one config fits all" |
| fp32 in inferenza live | corretto su ogni GPU | ~2× più lento della fp16 |
| Ricalibrazione optv2 al load | onestà e verificabilità oggi | divergenza dichiarata dai numeri d'archivio |

---

# 9. La webapp dimostrativa

Il progetto si chiude con una webapp che rende tutti i risultati **ispezionabili e rieseguibili dal vivo** — il contrario di una tabella statica.

**Architettura.** Backend **FastAPI** (Python) che serve i modelli reali: per PatchCore carica i bank di produzione (con le varianti v1/v2 ricostruite dagli stessi bank per confronto storico); per i GAN carica **i checkpoint originali** `production` e `optv2` ed esegue l'inferenza live su GPU locale (in fp32, sezione 8.6; per optv2 con ricalibrazione al load, sezione 8.7). Frontend **React + Vite + Tailwind**, stato con Zustand, streaming dei risultati via **SSE** con fallback a polling.

**Le pagine:**

- **Home** — il riassunto del percorso, con i numeri chiave animati;
- **Evaluation Lab** — leaderboard, heatmap 7 modelli × 15 categorie, curva di evoluzione (gli stessi dati di questa relazione: `benchmarks.json` è generato dai CSV reali);
- **Models** — una scheda per modello con architettura e storia;
- **Test Arena** — la parte viva: si sceglie una categoria, un set di immagini di test e una o più varianti (production / PatchCore v1 / v2 / OCGAN final / optv2) e si guarda il modello classificare in streaming, immagine per immagine, con score, soglia e verdetto;
- **Dataset Explorer** — navigazione di MVTec AD con ground truth;
- **Methodology** — il protocollo sperimentale.

**Verifica.** 60 test pytest sul backend; smoke E2E su CUDA (arena: 10 immagini in ~4 s, accuracy 1.0 su bottle/production); QA visivo automatizzato con Playwright + Edge (12 pagine/stati, 0 errori console).

![Home](figures/webapp_home.png)
*Figura 12 — Home della webapp.*

![Arena config](figures/webapp_arena_config.png)
*Figura 13 — Test Arena: configurazione di un run live (categoria, varianti, immagini).*

![Arena GAN](figures/webapp_arena_gan.png)
*Figura 14 — Arena al termine di un run live della variante GAN: per ogni immagine score, soglia e verdetto.*

![Modal risultato](figures/webapp_modal.png)
*Figura 15 — Dettaglio di una singola predizione.*

---

# 10. Conclusioni e sviluppi futuri

Il progetto è partito da un paper del 2019 con un'idea elegante — *costringere lo spazio latente a rappresentare solo la normalità, cercandone attivamente i punti deboli* — e l'ha portato fino a un sistema di anomaly detection industriale con **macro AUROC 0.9846 su MVTec AD**, interrogabile dal vivo da una webapp.

Che cosa resta, riguardando tutto il percorso:

1. **Le idee sopravvivono ai modelli.** Di OCGAN-2019 nel sistema finale non c'è una riga di architettura; ci sono il mining (nel GAN modernizzato), la disciplina del latent (come compactness) e soprattutto l'atteggiamento: cercare il punto in cui il modello sbaglia, invece di aspettarlo.
2. **La fusione batte il segnale.** Il singolo risultato più grande della fase generativa (+0.22) non viene da un'architettura migliore ma dal *combinare* segnali eterogenei con una regressione logistica da poche decine di parametri.
3. **Le feature pre-addestrate sono il moltiplicatore.** Ogni segnale costruito su ImageNet congelato (perceptual, memory, teacher–student) ha sovraperformato ogni segnale addestrato da zero; portata all'estremo, questa osservazione *è* PatchCore.
4. **Il rigore è una feature del risultato.** Senza 4-split, multi-seed e ablation, questo progetto avrebbe "trovato" numeri migliori e conclusioni peggiori: i flag morti (8.3), lo score dannoso (8.4) e la trappola di calibrazione (8.9) sarebbero passati inosservati.
5. **Misurare i tempi è misurare il sistema.** Le due scoperte più redditizie della fase 2 (coreset inutile sotto le 70k patch; hazelnut su un ramo di codice diverso) sono uscite dai log dei *secondi*, non da quelli dell'AUROC.

**Sviluppi futuri.**

- **Localizzazione pixel-level**: le mappe di distanza patch di PatchCore sono già calcolate; mancano valutazione (PRO-score, pixel-AUROC) e visualizzazione in arena.
- **Ensemble per-categoria GAN + PatchCore**: sulla carta ~0.99 di macro (sezione 7.2); richiederebbe un criterio di routing onesto scelto su validation.
- **MVTec AD 2** e dataset più grandi: lì il bank pieno smette di stare in memoria e il coreset torna necessario — il prossimo paradosso da gestire.
- **Distillazione in stile EfficientAD** per l'edge: portare il sistema sotto i 10 ms/immagine su hardware industriale.
- **CI per la riproducibilità**: il test "ogni chiave di config ha un effetto misurabile" (lezione 8.3) come guardia permanente del repository.

---

# Appendice A — Tabelle complete

## A.1 — AUROC per categoria, tutti i modelli (test blind; in grassetto il migliore per riga)

| Categoria | OCGAN final | OCGAN optv2 | PC v1 | PC v2 | PC v3 | Production |
|---|---|---|---|---|---|---|
| bottle | 0.8542 | 0.8958 | 0.9783 | **1.0000** | **1.0000** | **1.0000** |
| cable | 0.5231 | 0.5797 | 0.9703 | 0.9824 | **0.9960** | **0.9960** |
| capsule | 0.7839 | 0.6268 | 0.7724 | 0.8173 | **0.9824** | **0.9824** |
| carpet | 0.9238 | 0.9069 | 0.9801 | 0.9924 | **0.9943** | **0.9943** |
| grid | 0.9394 | **0.9687** | 0.9404 | 0.9254 | 0.9680 | 0.9680 |
| hazelnut | 0.9891 | 0.9914 | 0.9434 | 0.9987 | **1.0000** | **1.0000** |
| leather | 0.8976 | 0.9678 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| metal_nut | 0.6085 | 0.6228 | 0.9496 | 0.9874 | **0.9924** | **0.9924** |
| pill | 0.6707 | 0.7309 | 0.6081 | 0.8643 | **0.9580** | **0.9580** |
| screw | 0.9995 | **1.0000** | 0.8751 | 0.8749 | 0.9147 | 0.9419 |
| tile | 0.9753 | 0.9407 | 0.9971 | **1.0000** | **1.0000** | **1.0000** |
| toothbrush | 0.7037 | 0.6519 | 0.9179 | **0.9710** | **0.9710** | **0.9710** |
| transistor | 0.7377 | 0.7878 | 0.9390 | 0.9250 | **0.9933** | **0.9933** |
| wood | 0.9889 | **1.0000** | 0.9862 | 0.9906 | 0.9911 | 0.9911 |
| zipper | 0.8191 | 0.8958 | 0.7184 | 0.7658 | **0.9801** | **0.9801** |
| **Macro** | **0.8276** | **0.8378** | **0.9051** | **0.9397** | **0.9828** | **0.9846** |

*Nota: i valori GAN sono medie multi-seed (A.2); i valori PatchCore sono deterministici dato il bank (verificati identici su seed 43–45).*

## A.2 — Famiglia GAN: multi-seed con deviazione standard

| Categoria | final AUROC (±std) | seed | optv2 AUROC (±std) | seed | Δ optv2−final |
|---|---|---|---|---|---|
| bottle | 0.8542 (±0.0611) | 3 | 0.8958 (±0.0226) | 3 | +0.0417 |
| cable | 0.5231 (±0.0283) | 5 | 0.5797 (±0.0274) | 3 | +0.0566 |
| capsule | 0.7839 (±0.0411) | 3 | 0.6268 (±0.1589) | 3 | −0.1571 |
| carpet | 0.9238 (±0.0141) | 3 | 0.9069 (±0.0315) | 3 | −0.0169 |
| grid | 0.9394 (±0.0267) | 3 | 0.9687 (±0.0249) | 3 | +0.0293 |
| hazelnut | 0.9891 (±0.0084) | 3 | 0.9914 (±0.0148) | 3 | +0.0024 |
| leather | 0.8976 (±0.0170) | 3 | 0.9678 (±0.0090) | 3 | +0.0702 |
| metal_nut | 0.6085 (±0.0140) | 5 | 0.6228 (±0.0687) | 3 | +0.0143 |
| pill | 0.6707 (±0.0412) | 5 | 0.7309 (±0.0599) | 3 | +0.0603 |
| screw | 0.9995 (±0.0009) | 3 | 1.0000 (±0.0000) | 3 | +0.0005 |
| tile | 0.9753 (±0.0150) | 3 | 0.9407 (±0.0095) | 3 | −0.0346 |
| toothbrush | 0.7037 (±0.1668) | 3 | 0.6519 (±0.0559) | 3 | −0.0518 |
| transistor | 0.7377 (±0.0513) | 5 | 0.7878 (±0.0356) | 3 | +0.0501 |
| wood | 0.9889 (±0.0192) | 3 | 1.0000 (±0.0000) | 2 | +0.0111 |
| zipper | 0.8191 (±0.0910) | 3 | 0.8958 (±0.0044) | 2 | +0.0767 |
| **Macro** | **0.8276** | | **0.8378** | | **+0.0102** |

## A.3 — Sistema di produzione (PatchCore): metriche complete e tempi

| Categoria | AUROC | AUPRC | best F1 | FPR@95TPR | feature levels | t (s/cat) |
|---|---|---|---|---|---|---|
| bottle | 1.0000 | 1.0000 | 1.0000 | 0.0000 | layer2+layer3 | 5.5 |
| cable | 0.9960 | 0.9978 | 0.9853 | 0.0000 | layer2+layer3 | 7.7 |
| capsule | 0.9824 | 0.9958 | 0.9762 | 0.0556 | layer2+layer3 | 7.2 |
| carpet | 0.9943 | 0.9984 | 0.9925 | 0.0000 | layer2+layer3 | 7.6 |
| grid | 0.9680 | 0.9903 | 0.9647 | 0.0625 | layer2+layer3 | 5.5 |
| hazelnut | 1.0000 | 1.0000 | 1.0000 | 0.0000 | layer2+layer3 | 799.0 |
| leather | 1.0000 | 1.0000 | 1.0000 | 0.0000 | layer2+layer3 | 7.1 |
| metal_nut | 0.9924 | 0.9981 | 0.9859 | 0.0588 | layer2+layer3 | 5.1 |
| pill | 0.9580 | 0.9925 | 0.9665 | 0.1000 | layer2+layer3 | 6.4 |
| screw | 0.9419 | 0.9776 | 0.9405 | 0.2581 | layer1+layer2+layer3 | 7.3 |
| tile | 1.0000 | 1.0000 | 1.0000 | 0.0000 | layer2+layer3 | 5.9 |
| toothbrush | 0.9710 | 0.9890 | 0.9565 | 0.1111 | layer2+layer3 | 4.1 |
| transistor | 0.9933 | 0.9910 | 0.9492 | 0.0222 | layer2+layer3 | 7.0 |
| wood | 0.9911 | 0.9972 | 0.9670 | 0.0000 | layer2+layer3 | 6.9 |
| zipper | 0.9801 | 0.9947 | 0.9670 | 0.1250 | layer2+layer3 | 5.9 |

*Configurazione comune: WideResNet50-2 congelata, `topk_reweighted` k = 9, `max_patches` 70.000 (coreset attivo solo per hazelnut, sezione 8.10), soglia = p99 su `val_normal` (15% held-out, seed 43). Il tempo è build del bank + valutazione completa della categoria.*

---

# Appendice B — Riproducibilità

**Ambienti di esecuzione** (da `env_info.yaml` salvato in ogni run):

| Campagna | GPU | Software |
|---|---|---|
| Griglia 11.738 run + final + production (mar–apr 2026) | NVIDIA RTX A4000 16 GB (Paperspace) | Python 3.11.7, PyTorch 2.10.0+cu128, CUDA 12.8 |
| Retrain optv2 (apr 2026) | Quadro RTX 5000 16 GB (Paperspace) | PyTorch 2.1.1+cu121 |
| Inferenza live webapp | Quadro T1000 4 GB (locale, fp32) | PyTorch locale, FastAPI |

**Convenzione di naming dei run**: `{categoria}_{t}_{m}_{lf}_{mp}_{oc}_s{seed}_seed{seed}_{timestamp}` con la mappatura fattori→parametri della sezione 5.6. Ogni directory contiene `config.yaml` risolto, `env_info.yaml`, checkpoint top-k e log per epoca.

**Dati aggregati** (fonti di ogni numero della relazione):

- `final_per_category_multiseed_aggregated.csv`, `optv2_multiseed_aggregated.csv` — famiglia GAN;
- `logs/patchcore_pure.csv` (v1), `patchcore_tuning.csv`, `patchcore_v2.csv`, `patchcore_lc.csv` (coreset 50k), `patchcore_v3.csv`, `patchcore_p1*.csv` (ablation layer1) — famiglia PatchCore;
- `frontend/src/data/benchmarks.json` — aggregato unico generato dai CSV reali (`scripts/build_webapp_data.py`); macro verificati: final 0.8276, optv2 0.8378, v1 0.9051, v2 0.9397, v3 0.9828, production 0.9846.

**Figure e tabelle di questa relazione**: generate da `relazione/gen_figures.py` e `relazione/gen_tables.py` esclusivamente a partire dalle fonti sopra (nessun numero inserito a mano se non quelli del paper OCGAN e della campagna 1, citati dal diario di progetto).

**Nota di fedeltà**: tutti i run cloud puntano allo stesso commit git (`e4d4a3f`), ma con drift locale non committato sull'istanza (sezione 8.7): è la ragione per cui la riproducibilità *bit-exact* della calibrazione optv2 è stata sostituita da una ricalibrazione dichiarata al load.

---

*Fine della relazione.*
