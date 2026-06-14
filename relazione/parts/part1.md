# Da OCGAN a PatchCore

## Un percorso completo di anomaly detection one-class: dalla riproduzione di un paper CVPR 2019 a un sistema industriale su MVTec AD

**Autore:** Alessandro Pata
**Data:** giugno 2026
**Codice:** repository `ocgan-modernized` (PyTorch) + webapp dimostrativa (FastAPI + React)

---

## Indice

1. [Introduzione e obiettivo](#1-introduzione-e-obiettivo)
2. [Il punto di partenza: il paper OCGAN (CVPR 2019)](#2-il-punto-di-partenza-il-paper-ocgan-cvpr-2019)
3. [Le tecniche utilizzate, spiegate una per una](#3-le-tecniche-utilizzate-spiegate-una-per-una)
4. [Setup sperimentale: dataset, protocollo, infrastruttura](#4-setup-sperimentale-dataset-protocollo-infrastruttura)
5. [Fase 1 — OCGAN modernizzato (il "binario B")](#5-fase-1--ocgan-modernizzato-il-binario-b)
6. [Fase 2 — La svolta PatchCore](#6-fase-2--la-svolta-patchcore)
7. [Confronto globale e confronto con il paper originale](#7-confronto-globale-e-confronto-con-il-paper-originale)
8. [Problemi, sfide e lezioni imparate](#8-problemi-sfide-e-lezioni-imparate)
9. [La webapp dimostrativa](#9-la-webapp-dimostrativa)
10. [Conclusioni e sviluppi futuri](#10-conclusioni-e-sviluppi-futuri)

Appendici: [A — Tabelle complete](#appendice-a--tabelle-complete) · [B — Riproducibilità](#appendice-b--riproducibilità)

---

# 1. Introduzione e obiettivo

Il rilevamento di anomalie **one-class** è il problema di riconoscere esempi "anomali" disponendo, in fase di addestramento, **soltanto di esempi normali**. È lo scenario tipico del controllo qualità industriale: di un pezzo meccanico esistono migliaia di foto di esemplari corretti, ma i difetti sono rari, costosi da raccogliere e — soprattutto — imprevedibili: il modello deve segnalare anche tipologie di difetto mai viste prima.

Il progetto è partito da un paper preciso: **OCGAN — One-Class Novelty Detection Using GANs with Constrained Latent Representations** (Perera, Nallapati, Xiang — CVPR 2019), che affronta il problema con un autoencoder adversariale dal design molto elegante (sezione 2). Da lì il percorso si è sviluppato in due fasi:

1. **Fase 1 — "binario B": modernizzare OCGAN.** Non una difesa filologica del modello del 2019, ma un innesto delle sue idee (ricostruzione, regolarizzazione del latent, mining di negativi informativi) in una pipeline moderna: backbone pre-addestrato, loss percettive, anomalie sintetiche, score multipli fusi con una regressione logistica. Il tutto validato con una campagna di ablation di migliaia di run.
2. **Fase 2 — la svolta PatchCore.** Constatato il plateau della famiglia generativa (~0.84 macro AUROC), abbiamo adottato e ottimizzato un approccio a memoria di feature congelate (PatchCore), portandolo con tre interventi mirati a **0.9846 macro AUROC** su MVTec AD — senza alcun addestramento.

La figura 1 riassume l'intero arco del progetto in un'unica curva: dal **0.7866** della baseline GAN "onesta" (dopo la correzione di alcuni bug di configurazione, sezione 8.3) fino al **0.9846** del sistema di produzione.

![Evoluzione del progetto](figures/fig1_evoluzione.png)
*Figura 1 — L'evoluzione del macro AUROC su MVTec AD (15 categorie) lungo tutto il progetto: la famiglia generativa (rosso) migliora ma satura; la famiglia a memory bank (blu/verde) cambia regime.*

Tre principi hanno guidato tutto il lavoro, e questa relazione li riflette:

- **Onestà sperimentale**: protocollo a 4 split con test set mai usato per il tuning, multi-seed obbligatorio, numeri riportati con deviazione standard, bug raccontati invece che nascosti.
- **Ablation prima delle opinioni**: ogni componente è stato acceso/spento e pesato in griglie sistematiche (in totale oltre 11.000 run su disco); alcune componenti "affezionate" sono state eliminate perché i numeri le bocciavano (sezione 5.5).
- **Confronto costante con il paper di partenza**: ogni scelta è messa in relazione con la corrispondente scelta di OCGAN (sezione 7.3).

### Come leggere la relazione

La sezione 2 descrive in dettaglio il paper di partenza. La sezione 3 è un glossario ragionato di tutte le tecniche usate (memory bank, teacher–student, learned fusion, ecc.): chi le conosce già può saltarla e tornarci dai rimandi. La sezione 4 descrive dataset, protocollo e l'infrastruttura cloud (Paperspace). Le sezioni 5 e 6 sono il cuore sperimentale (architettura e funzionamento di ogni modello, versioni, scelte con pro e contro). La sezione 7 confronta tutto, anche col paper. La sezione 8 raccoglie problemi e lezioni. Le appendici riportano le tabelle complete per categoria.

---

# 2. Il punto di partenza: il paper OCGAN (CVPR 2019)

## 2.1 Il problema

Nella **one-class novelty detection** il training set contiene una sola classe ("in-class") e al test bisogna distinguere in-class da out-of-class. L'approccio classico è l'autoencoder: si addestra a ricostruire la classe normale e si usa l'errore di ricostruzione come punteggio di anomalia, assumendo che ciò che non sa ricostruire sia anomalo.

Il paper parte da un'osservazione sperimentale che demolisce questa assunzione: un autoencoder addestrato **solo sulla cifra 8** di MNIST ricostruisce sorprendentemente bene anche 1, 5, 6 e 9. Le feature apprese (curve, tratti) sono abbastanza generiche da rappresentare anche ciò che non si è mai visto → errore di ricostruzione basso sugli outlier → **falsi negativi**.

## 2.2 L'intuizione chiave

> Non basta chiedere che la classe normale sia ben rappresentata nello spazio latente: bisogna chiedere che **l'intero spazio latente rappresenti solo la classe normale**.

OCGAN attacca il problema "in negativo": invece di limitarsi a minimizzare l'errore sugli esempi normali, **esplora attivamente lo spazio latente a caccia di regioni che producono immagini fuori classe**, e costringe il generatore a "normalizzarle". Se ogni punto del latent decodifica in qualcosa che sembra in-class, allora un esempio out-of-class non può che essere ricostruito male — ed è esattamente ciò che vogliamo.

## 2.3 L'architettura: quattro componenti

OCGAN è composto da quattro reti che si allenano insieme:

| Componente | Struttura | Ruolo |
|---|---|---|
| **Denoising autoencoder** (En + De) | encoder: 3 conv 5×5 stride 2, BatchNorm, LeakyReLU(0.2), 64 canali base; latent limitato con **tanh** in (−1,1)^d; decoder simmetrico con 3 deconvoluzioni | ricostruire l'input (a cui viene aggiunto rumore gaussiano con σ² = 0.2); l'errore di ricostruzione è lo score finale |
| **Latent discriminator** D_l | MLP fully-connected 128→64→32→16 | distingue i latent "veri" (encodings di immagini normali) da campioni **uniformi** U(−1,1)^d; in equilibrio costringe l'encoder a distribuire i latent su tutto il cubo |
| **Visual discriminator** D_v | CNN leggera (12 canali base) | distingue le immagini **generate da latent casuali** dalle immagini reali: ogni punto del cubo latente deve decodificare in qualcosa di plausibilmente in-class |
| **Classifier** C | CNN (64 canali base) | classificatore "debole" che giudica quanto un'immagine generata sembra in-class (positivi = ricostruzioni, negativi = generazioni da latent casuali); non serve allo score, serve a **guidare il mining** |

Due dettagli di design sono particolarmente importanti:

- Il **tanh sul latent** rende lo spazio dei codici un cubo chiuso e limitato: questo rende sensato campionarci dentro in modo uniforme e "pattugliarlo" tutto.
- Il **denoising** (input rumoroso) evita la soluzione banale dell'identità e regolarizza.

## 2.4 L'informative-negative mining

È il contributo più originale del paper. A ogni iterazione:

1. si campionano punti casuali nel cubo latente;
2. per ogni punto, si fa **gradient ascent nel latent per 5 passi**, massimizzando il giudizio "out-of-class" del classifier C — cioè ci si sposta verso le regioni dove il generatore produce immagini che *non* sembrano la classe normale;
3. i punti trovati ("negativi informativi") vengono usati per addestrare il generatore a produrre, anche lì, immagini in-class.

In pratica è una ricerca adversariale interna al modello, parente stretta degli attacchi PGD: invece di aspettare che il caso esponga le zone difettose del latent, **le si cerca attivamente col gradiente**. Questa idea sopravvive, modernizzata, anche nel nostro modello (sezioni 3.5 e 5.4).

## 2.5 Training e loss

Il training alterna due step: prima si aggiorna il classifier C, poi (con C congelato) autoencoder e discriminatori in modo adversariale. La loss del generatore è:

```
L = 10 · MSE(ricostruzione) + l_latent (adversarial vs D_l) + l_visual (adversarial vs D_v)
```

Il peso 10 sulla MSE dice già molto: la ricostruzione resta il segnale dominante, l'apparato adversariale è una **regolarizzazione dello spazio**, non il fine. La model selection è fatta sull'MSE di validazione.

## 2.6 I risultati del paper

Il paper valuta su dataset di immagini piccole e mono-oggetto, con due protocolli:

| Protocollo | Dataset | Risultato (mean AUROC) |
|---|---|---|
| P1 (80/20 in-class, test bilanciato) | MNIST | 0.977 |
| P1 | COIL-100 | 0.995 |
| P1 | fashion-MNIST | 0.924 |
| P2 (split nativi del dataset) | MNIST | **0.9750** (vs Deep-SVDD 0.9480, AND 0.9671) |
| P2 | CIFAR-10 | **0.6566** (vs Deep-SVDD 0.6481) |

Su MNIST, OCGAN era lo stato dell'arte del momento. Su CIFAR-10 il margine è minimo e il valore assoluto è basso: il metodo fatica quando la "classe normale" è visivamente eterogenea.

## 2.7 L'ablation del paper

Il paper smonta il proprio modello pezzo per pezzo (MNIST, P2):

| Configurazione | mean AUROC |
|---|---|
| Solo autoencoder | 0.957 |
| + latent discriminator | 0.959 |
| + visual discriminator | 0.971 |
| + classifier / mining (modello completo) | **0.975** |

Vale la pena notare la forma di questa curva: la base ricostruttiva fa quasi tutto, e ogni componente adversariale aggiunge poco. Ritroveremo una struttura analoga — ma con pesi diversissimi — nella nostra ablation (sezione 5.5 e figura 11).

## 2.8 I limiti (dichiarati e non)

- **Dichiarati dagli autori**: il metodo funziona quando la classe normale è un concetto singolo e centrato (MNIST, COIL, fMNIST); su CIFAR-10 — più classi di sfondi e pose — è molto più debole.
- **Pratici, scoperti da noi**: il repo ufficiale è in **MXNet** (framework ormai abbandonato), e la reimplementazione TensorFlow disponibile è minimale; i risultati pubblici sono misurati **una sola volta per cifra**, senza varianza — fragile dal punto di vista statistico (questo ha motivato il nostro obbligo di multi-seed, sezione 4.4).
- **Strutturale**: tutto il design è **pixel-centrico** (l'immagine è il segnale). Su immagini industriali ad alta risoluzione con difetti piccoli e texture complesse, vedremo che questo è il vero tetto (sezioni 5.9 e 7).

---

# 3. Le tecniche utilizzate, spiegate una per una

Questa sezione è il glossario ragionato di tutte le tecniche comparse nel progetto. Per ciascuna: che cos'è, come funziona, e dove la usiamo.

## 3.1 Backbone pre-addestrato e congelato

**Cos'è.** Una CNN (per noi ResNet50 nel modello GAN, WideResNet50-2 in PatchCore) addestrata su ImageNet e usata come estrattore di feature **senza aggiornarne i pesi**.

**Come funziona.** I layer intermedi di una rete addestrata su milioni di immagini naturali producono descrizioni locali (texture, bordi, parti) estremamente generali. Per l'anomaly detection sono preziose per due motivi: (1) non serve impararle da zero da poche centinaia di immagini; (2) **congelarle impedisce il "catastrophic forgetting" della generalità** — se le si fine-tunasse solo su immagini normali, potrebbero specializzarsi al punto da mappare normale e difettoso sugli stessi codici.

**Dove la usiamo.** Ovunque: encoder del GAN modernizzato (sezione 5.3), feature per perceptual loss, memory bank, teacher–student, e come unico componente di PatchCore. È la differenza più profonda rispetto a OCGAN 2019, che addestrava l'encoder da zero.

## 3.2 Perceptual loss (feature loss)

**Cos'è.** Una loss di ricostruzione calcolata **nello spazio delle feature** di una rete pre-addestrata, invece che sui pixel.

**Come funziona.** Ricostruzione e originale vengono ri-encodate dal backbone; si confrontano le feature (per noi: layer2, layer3, layer4 con peso 0.1). Due immagini possono essere vicine pixel per pixel ma semanticamente diverse — e viceversa: la perceptual loss premia la somiglianza percettiva, penalizza l'oversmoothing tipico della sola MSE.

**Dove la usiamo.** Nello stack di ricostruzione del GAN modernizzato (L1 + MS-SSIM + perceptual), e come una delle teste di score (`norm_perceptual`). Il paper OCGAN usava solo MSE.

## 3.3 MS-SSIM

**Cos'è.** Multi-Scale Structural Similarity: una misura di somiglianza strutturale (luminanza, contrasto, struttura locale) calcolata a più scale.

**Come funziona.** Invece di confrontare i singoli pixel, confronta statistiche locali di patch a risoluzioni diverse. È molto più sensibile della MSE alle alterazioni di struttura (esattamente ciò che è un difetto) e più tollerante a piccole differenze di intensità globale.

**Dove la usiamo.** Nella loss di ricostruzione (insieme a L1; abbiamo eliminato del tutto la MSE) e nel calcolo dello score di ricostruzione.

## 3.4 Latent compactness (stile Deep-SVDD)

**Cos'è.** Una regolarizzazione che spinge i codici latenti delle immagini normali a concentrarsi **attorno a un centro**.

**Come funziona.** Deep-SVDD impara una sfera minima che racchiude le rappresentazioni dei dati normali: la distanza dal centro è il punteggio di anomalia. Nel nostro modello usiamo una versione leggera: un termine di compattezza (peso 0.1) sulla testa latente (`compact_latent`, 128 dimensioni), più uno score opzionale "one-class" basato sulla distanza dal centro.

**Dove la usiamo.** È il nostro **erede diretto del latent discriminator di OCGAN**: il paper costringeva i latent a riempire uniformemente un cubo; noi li costringiamo a stringersi attorno a un centro. Stessa filosofia (controllare la geometria del latent), meccanismo opposto. Nota onesta: nella configurazione finale lo *score* one-class è disattivato perché l'ablation lo ha bocciato (sezione 5.5); resta attiva la regolarizzazione in loss.

## 3.5 Informative-negative mining moderno (PGD-style)

**Cos'è.** La versione modernizzata del mining di OCGAN (sezione 2.4): ricerca attiva, col gradiente, di punti latenti che producono ricostruzioni "difficili".

**Come funziona.** Multi-step (3 passi di tipo PGD nel nostro setup, contro i 5 del paper), con warmup (parte dopo la prima epoca), guidato dallo score composito invece che da un classificatore dedicato. I negativi trovati alimentano il training della branch discriminativa.

**Dove la usiamo.** Nel GAN modernizzato. È l'idea di OCGAN che abbiamo conservato più volentieri, perché è indipendente dall'architettura.

## 3.6 Anomalie sintetiche (CutPaste, Perlin, perturbazioni in feature space)

**Cos'è.** Difetti artificiali generati al volo sulle immagini normali, per dare al modello esempi "negativi" senza possedere veri difetti.

**Come funziona.**
- **CutPaste**: si ritaglia una patch dall'immagine e la si incolla altrove (p = 0.5 nel nostro training) → discontinuità locali simili a graffi/contaminazioni.
- **Maschere di Perlin**: rumore coerente a bassa frequenza usato come maschera per fondere texture estranee → difetti dalle forme organiche (usate nella variante optv2).
- **Perturbazioni gaussiane in feature space** (linea SimpleNet): si aggiunge rumore alle feature del backbone (per noi al layer4) → "difetti semantici" a costo zero in pixel.

**Dove le usiamo.** Per addestrare la branch discriminativa del GAN modernizzato e per popolare il segnale di validazione. Il rischio noto (e il motivo per cui le usiamo con prudenza): il modello può imparare a riconoscere *l'artefatto sintetico* invece del concetto di difetto.

## 3.7 Branch discriminativa (stile DRAEM)

**Cos'è.** Una testa di rete addestrata in modo **supervisionato** a distinguere normale da anomalo, usando come anomali le anomalie sintetiche.

**Come funziona.** DRAEM concatena immagine e sua ricostruzione e addestra una rete a segmentare il difetto. La nostra versione produce uno score discriminativo a livello immagine (`norm_discriminative`) a partire da input + ricostruzione.

**Dove la usiamo.** Come una delle 7 teste di score del GAN modernizzato. È il "discendente funzionale" del classifier C di OCGAN — ma addestrato su difetti sintetici realistici invece che su generazioni da latent casuali.

## 3.8 Memory bank + nearest neighbour (PatchCore) e coreset

**Cos'è.** L'idea di **non addestrare nulla**: si memorizzano le feature locali (patch) di tutte le immagini normali di training in una "banca", e al test si misura la distanza di ogni patch dell'immagine dalla patch normale più vicina.

**Come funziona.**
1. Ogni immagine di training passa nel backbone congelato; le mappe di feature (per noi layer2 + layer3 di WideResNet50-2, concatenate dopo riallineamento spaziale) diventano un insieme di vettori-patch.
2. Tutti i vettori finiscono nel **memory bank**.
3. Al test, per ogni patch dell'immagine si calcola la distanza dal vicino più prossimo nel bank; l'aggregazione delle distanze patch-level dà lo score immagine (sezione 3.9).
4. Se il bank è troppo grande si applica il **coreset subsampling k-center-greedy**: si seleziona iterativamente il punto più lontano dall'insieme già scelto, ottenendo un sottoinsieme che "copre" bene lo spazio con una frazione dei punti.

**Dove la usiamo.** In due regimi molto diversi (ed è una delle lezioni più interessanti del progetto, sezione 8.5): come **score ausiliario** dentro il GAN modernizzato (bank piccolo, 1024–4096 patch, aggregazione max) e come **modello a sé** in PatchCore (bank pieno fino a 70.000 patch, top-k reweighted) — dove è la svolta dell'intero progetto.

## 3.9 Aggregazione top-k reweighted

**Cos'è.** Il modo di passare dalle distanze delle singole patch allo score dell'immagine.

**Come funziona.** Le alternative provate: massimo (fragile: un solo outlier di feature decide), media dei top-k (`topk_mean`, k = 3 in PatchCore v1), e **top-k reweighted** (k = 9): si prendono le k patch più anomale e le si pesa con un softmax sulle distanze — un compromesso che valorizza il picco senza dipendere da una sola patch. È la generalizzazione del reweighting proposto nel paper PatchCore originale.

**Dove la usiamo.** `topk_reweighted` con k = 9 è uno dei tre ingredienti della svolta di PatchCore v2/v3 (sezione 6.3).

## 3.10 Teacher–Student (stile RD4AD)

**Cos'è.** Due reti: un **teacher** pre-addestrato e congelato e uno **student** addestrato a imitarne le feature *solo su immagini normali*.

**Come funziona.** Sui dati normali lo student impara a riprodurre il teacher quasi perfettamente. Su un'anomalia — mai vista — lo student generalizza male e la **discrepanza teacher-student** (per noi sui layer 3 e 4, multiscala) esplode: quella discrepanza è lo score.

**Dove la usiamo.** Come testa di score `norm_teacher_student` del GAN modernizzato (peso in loss 0.05, peso di score regolato dalla griglia di tuning — fattore `t`).

## 3.11 Learned fusion (regressione logistica) + normalizzazione MAD

**Cos'è.** Il meccanismo che fonde le 7 teste di score in un unico punteggio.

**Come funziona.**
1. Ogni score grezzo ha scala e distribuzione proprie → si normalizza in modo **robusto** con mediana e MAD (Median Absolute Deviation) calcolate su `val_normal`: a differenza di media/deviazione standard, mediana e MAD non vengono trascinate dagli outlier.
2. Una **regressione logistica** addestrata su `val_mixed` (normali + anomalie) impara i pesi ottimali della combinazione; il parametro di regolarizzazione C è uno dei fattori della griglia (fattore `lf`).
3. La soglia operativa si sceglie massimizzando F1 su `val_mixed`.

**Dove la usiamo.** Nel GAN modernizzato. Anticipazione del risultato più netto di tutta l'ablation: la fusione appresa è **il singolo fattore più importante del modello generativo**, +0.21 di AUROC medio rispetto allo score singolo (sezione 5.5). Il paper OCGAN usava un solo score (MSE); questa è la differenza metodologica più redditizia che abbiamo introdotto.

## 3.12 Calibrazione della soglia

**Cos'è.** La scelta del valore di score oltre il quale un'immagine viene dichiarata anomala. L'AUROC non ne ha bisogno (è threshold-free), ma un sistema reale sì.

**Come funziona (nostre due varianti).**
- GAN modernizzato: soglia a **best-F1 su `val_mixed`**.
- PatchCore production: **99° percentile degli score su `val_normal`** (15% di immagini normali tenute fuori dal bank, seed 43) → per costruzione ~1% di falsi positivi attesi sulle normali. Il motivo per cui le immagini del bank non si possono usare per calibrare è una trappola classica raccontata in sezione 8.9.

## 3.13 Strumenti di training moderni

Usati nel GAN modernizzato (tutti assenti nel paper 2019):

- **AdamW** (lr 1e-4) — Adam con weight decay disaccoppiato.
- **Warmup + cosine schedule** (2 epoche di warmup) — il learning rate sale dolcemente e poi decade a coseno: stabilizza le prime epoche adversariali.
- **EMA dei pesi** (decay 0.999) — si mantiene una media mobile esponenziale dei parametri e la si usa per la valutazione: riduce il rumore degli ultimi step.
- **AMP / mixed precision** — calcoli in fp16 dove sicuro, fp32 dove serve: ~2× di velocità in training su GPU con tensor core (ma vedremo un caso in cui l'fp16 in *inferenza* ci ha morso, sezione 8.6).
- **Gradient clipping** — limite alla norma del gradiente, paracadute contro i picchi adversariali.
- **Early stopping multi-metrica** (pazienza 3) su una validazione composita: 0.5·AUROC + 0.3·AUPRC + 0.2·F1 su `val_mixed`.
