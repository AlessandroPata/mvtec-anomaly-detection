---

# 6. Fase 2 — La svolta PatchCore

## 6.1 Architettura e funzionamento

PatchCore (Roth et al., 2022) è l'antitesi metodologica di tutto ciò che abbiamo costruito nella fase 1: **niente training, niente loss, niente seed**. Il "modello" è una banca di ricordi.

```
                       COSTRUZIONE (una volta, pochi secondi)
train_normal ──► WideResNet50-2 congelata ──► feature layer2 + layer3
                                              (riallineate e concatenate)
                                                  │
                                                  ▼
                                       memory bank di vettori-patch
                                       (se > max_patches: coreset k-center-greedy)

                       INFERENZA (per ogni immagine di test)
immagine ──► stesse feature ──► per ogni patch: distanza dal vicino
                                più prossimo nel bank
                                       │
                                       ▼
                       score immagine = top-k reweighted (k = 9)
                       soglia = 99° percentile su val_normal
```

**Come funziona, in due righe.** Un difetto è, per definizione, *qualcosa che non assomiglia a nessuna porzione di nessuna immagine normale mai vista*. PatchCore prende questa definizione alla lettera: descrive ogni zona dell'immagine con feature ImageNet di media profondità (abbastanza astratte da ignorare il rumore, abbastanza locali da localizzare il difetto) e misura letteralmente la distanza dal normale più vicino. Tutta l'"intelligenza" è nel backbone pre-addestrato; tutta la "conoscenza del dominio" è nel bank.

Perché layer2+layer3 e non layer4? I layer profondi sono troppo semantici e a risoluzione troppo bassa (un difetto di 30 px scompare); i layer bassi troppo rumorosi. I layer intermedi sono il punto dolce — e l'eccezione di `screw` (sezione 6.5) conferma la regola.

Il nostro percorso su PatchCore è passato per tre versioni, ognuna delle quali corregge un'ipotesi sbagliata della precedente.

![PatchCore v1 v2 v3](figures/fig7_patchcore_v1v2v3.png)
*Figura 7 — Le tre versioni di PatchCore per categoria: ogni ingrediente recupera categorie diverse.*

## 6.2 v1 — il primo tentativo: 0.9051

Configurazione: WideResNet50-2, **una sola scala** (layer3), aggregazione `topk_mean` con k = 3, **coreset 10.000 patch** (k-center-greedy), ~35 s per categoria. Risultato: **macro 0.9051** — già sopra qualunque GAN della fase 1, al primo colpo, senza addestrare nulla.

Ma con buchi vistosi: pill **0.6081**, zipper **0.7184**, capsule **0.7724**. Tre categorie sotto il GAN. La media nasconde; il per-categoria accusa.

## 6.3 v2 — top-k reweighted e multi-scala: 0.9397

Una campagna di tuning mirata sulle categorie deboli (log `patchcore_tuning.csv`) ha isolato due colpevoli:

1. **L'aggregazione**: `topk_mean` su k = 3 patch è fragile; `topk_reweighted` con k = 9 (softmax sulle distanze, sezione 3.9) è sistematicamente migliore.
2. **La scala singola**: layer3 da solo perde i difetti fini; **layer2 + layer3** li cattura.

L'esempio più drammatico: **pill passa da 0.534 (layer3, topk_mean 3) a 0.872 (layer2+3, reweighted)** — +0.34 di AUROC *cambiando solo come si interroga lo stesso bank*.

v2 = layer2+layer3 + `topk_reweighted` k = 9 (coreset ancora 10k): **macro 0.9397**. Bottle sale a 1.0000, cable a 0.9824, metal_nut a 0.9874 — le categorie "strutturali" che il GAN non ha mai saputo trattare sono risolte. Restano sotto: capsule 0.8173, zipper 0.7658, grid 0.9254.

## 6.4 v3 — il paradosso del coreset: 0.9828

L'ipotesi successiva: le categorie rimaste indietro soffrono perché **10.000 patch non bastano a coprire la variabilità del normale**. Test intermedio con coreset a 50.000 (log `patchcore_lc.csv`): zipper 0.7658 → **0.9801**, capsule 0.8173 → **0.9824**. Ipotesi confermata — ma il k-center-greedy su pool così grandi costa fino a **468 s per categoria**: insostenibile.

E qui il ribaltamento concettuale: *se vogliamo tutte le patch, perché stiamo ancora pagando un algoritmo per sceglierne un sottoinsieme?* Il coreset esiste per comprimere bank enormi; ma su MVTec una categoria ha 200–400 immagini di training → le patch totali stanno (quasi) tutte sotto le 70.000.

**v3**: `max_patches = 70.000` e **nessun coreset quando il bank ci sta** (cioè per 14 categorie su 15). Risultato: **macro 0.9828**, e — paradosso completo — **~6× più veloce di v1/v2**: 6.3 s di media per categoria contro 35 s, perché il tempo dominante non era mai stata la ricerca kNN, era la *selezione* del coreset.

![Tempi](figures/fig8_tempi.png)
*Figura 8 — Il paradosso del coreset: il bank pieno è insieme il più accurato e il più veloce. (\*media su 14 categorie: hazelnut, l'unica oltre le 70k patch, attiva il coreset e impiega 799 s — sezione 8.10.)*

La figura 9 mette fianco a fianco i **due regimi del memory bank** incontrati nel progetto — quello ausiliario dentro il GAN (dove più patch peggiorava!) e quello di PatchCore (dove il bank pieno è la svolta). La spiegazione dell'apparente contraddizione è in sezione 8.5.

![Memory bank: due regimi](figures/fig9_memorybank.png)
*Figura 9 — La stessa leva (dimensione del bank), due regimi opposti: score ausiliario con aggregazione max (sinistra) vs modello completo con top-k reweighted (destra).*

## 6.5 Production — l'ultimo punto: screw e il layer1: 0.9846

A 0.9828 restava una sola categoria sotto 0.95: **screw** (0.9147). Le viti di MVTec sono piccole, metalliche, su sfondo uniforme, con difetti sottilissimi (filettature rovinate, punte smussate): difetti *ad altissima frequenza spaziale*, proprio ciò che layer2+layer3 vedono peggio.

Ablation dedicata (log `patchcore_p1*.csv`): aggiungere **layer1** alla piramide porta screw a **0.9419 (+2.7 punti)**; sulle altre categorie provate (grid, toothbrush, pill) non dà alcun miglioramento — e infatti la configurazione `p1` applicata a tappeto su tutte le categorie *peggiora* la macro (0.9562). La risposta giusta non è "layer1 ovunque" ma **una configurazione per-categoria**: layer2+layer3 per 14 categorie, layer1+layer2+layer3 solo per screw.

> **Production finale: macro AUROC 0.9846.** Quattro categorie perfette (bottle, hazelnut, leather, tile = 1.0000), undici sopra 0.96, peggiore screw 0.9419. Configurazione completa e metriche per categoria in tabella A3 (appendice).

La calibrazione della soglia merita una riga: il **99° percentile degli score su `val_normal`** — un 15% di immagini normali (seed 43) tenute **fuori dal bank**. Non si può calibrare sulle immagini che stanno nel bank: la loro distanza dal vicino più prossimo è ≈ 0 per costruzione (il vicino sono loro stesse), e la soglia risulterebbe assurdamente bassa. Trappola classica, dettagli in sezione 8.9.

## 6.6 I tre ingredienti, riassunti

Il +19.8 punti da GAN finale (0.8276 multi-seed) a production (0.9846) — e il +7.95 da PatchCore v1 a production — si scompone in tre mosse, tutte e tre *a costo computazionale negativo o nullo*:

| Ingrediente | Cosa cambia | Chi recupera |
|---|---|---|
| 1. Bank pieno (70k, niente coreset) | la memoria copre tutta la variabilità del normale | zipper +0.21, capsule +0.17 |
| 2. `topk_reweighted` k = 9 | lo score non dipende più da 1–3 patch | pill +0.34 (con il n.3) |
| 3. Multi-scala layer2+layer3 (+layer1 per screw) | difetti fini visibili | cable, metal_nut, transistor ~+0.99; screw +2.7pp |

*Pro dell'approccio PatchCore*: accuratezza, velocità (secondi per categoria), zero training, zero seed-variance (è deterministico dato il bank), semplicità operativa. *Contro*: il bank cresce col training set (memoria O(n)); nessuna rappresentazione "compressa" del concetto di normalità; un backbone ImageNet può essere cieco a domini molto lontani dalle immagini naturali; e non impara nulla dal dominio — è esattamente il punto di forza e il limite filosofico.

---

# 7. Confronto globale e confronto con il paper originale

## 7.1 Tutti i modelli, fianco a fianco

La tabella A1 (appendice) riporta l'AUROC di tutti i 6 modelli su tutte le 15 categorie; la heatmap la riassume:

![Heatmap modelli per categorie](figures/fig6_heatmap.png)
*Figura 6 — AUROC per modello × categoria. Si vede a colpo d'occhio dove ogni famiglia soffre: i GAN sulle categorie strutturali (cable, metal_nut, pill), i PatchCore di prima generazione su capsule/zipper/pill, production quasi uniformemente sopra 0.94.*

![Production vs GAN](figures/fig5_prod_vs_gan.png)
*Figura 5 — Production vs miglior GAN per categoria: il distacco è massimo proprio dove il GAN era più debole.*

La progressione macro completa:

| Modello | Macro AUROC | Note |
|---|---|---|
| GAN baseline onesta (post-fix) | 0.7866 | sezione 5.7 |
| GAN `final` multi-seed | 0.8276 | 3–5 seed |
| GAN `optv2` retrain | 0.8378 | 3 seed |
| PatchCore v1 | 0.9051 | zero training |
| PatchCore v2 | 0.9397 | +reweighted, +multi-scala |
| PatchCore v3 | 0.9828 | +bank pieno |
| **Production** | **0.9846** | +layer1 su screw |

## 7.2 La complementarità GAN ↔ PatchCore

Il dato più interessante del confronto non è la vittoria di PatchCore — è **dove non vince**:

- **screw**: GAN `optv2` **1.0000** (e `final` 0.9995) contro 0.9419 di production. La categoria *peggiore* di PatchCore è la *migliore* del GAN. Le viti hanno una normalità geometrica precisa che la ricostruzione cattura perfettamente, mentre i difetti ad alta frequenza si mimetizzano nelle feature di media profondità.
- **cable**: GAN 0.52–0.58 (poco più del caso) contro **0.9960** di production. I cavi in sezione hanno combinazioni di fili colorati troppo variabili per essere ricostruite, ma ogni patch normale assomiglia a qualche patch già vista.
- grid e wood: il GAN resta competitivo (0.9687 e 1.0000 in optv2, alla pari o sopra production).

I due paradigmi sbagliano su categorie **diverse** perché misurano cose diverse: *che cosa so rigenerare* (GAN) contro *che cosa ho già visto* (memoria). Un ensemble per-categoria dei due (production + GAN solo su screw/grid/wood) varrebbe sulla carta ~0.987–0.99 di macro; non l'abbiamo deployato — production resta un sistema solo, più onesto da dichiarare — ma la complementarità è il motivo per cui la webapp serve *entrambe* le famiglie live (sezione 9).

## 7.3 Il confronto con il paper OCGAN

Un confronto numero-contro-numero con il paper è impossibile per costruzione: OCGAN 2019 è valutato su MNIST/COIL/fMNIST/CIFAR-10 (immagini 28–128 px, one-class su classi semantiche), noi su MVTec AD (256 px, difetti industriali). La figura 10 mette i due mondi sulla stessa pagina **senza** pretendere che la differenza di altezza sia merito nostro: misura quanto strada c'è tra il benchmark accademico del 2019 e il problema industriale.

![Paper vs progetto](figures/fig10_paper_vs_progetto.png)
*Figura 10 — I risultati del paper sui suoi dataset (sinistra) e i nostri su MVTec AD (destra). I numeri non sono direttamente confrontabili: dataset, protocolli ed epoche di letteratura diverse.*

Il confronto **metodologicamente onesto** è tra le due *ablation*, cioè tra le due risposte alla domanda "da dove viene la performance?":

![Ablation a confronto](figures/fig11_ablation_paper.png)
*Figura 11 — L'ablation del paper (MNIST) e la nostra (MVTec). Nel 2019 la base ricostruttiva faceva il 98% del lavoro; nel nostro problema la singola componente più importante è la fusione appresa dei segnali.*

| | OCGAN (MNIST) | Nostro GAN (MVTec) |
|---|---|---|
| Base | AE: 0.957 | score singolo: 0.6158 |
| Contributo componenti | +0.018 totale (D_l +0.002, D_v +0.012, mining +0.004) | fusion **+0.224**, teacher +0.13, memory +0.09, one-class **−0.06** |
| Messaggio | l'AE basta quasi; l'adversarial rifinisce | nessun segnale basta; vince chi li combina |

Tre lezioni dal confronto:

1. **Il problema cambia la gerarchia delle idee.** Su MNIST lo spazio latente "pattugliato" rifinisce un AE già fortissimo. Su MVTec la ricostruzione da sola è debole, e il valore si sposta su segnali e fusione. Le idee di OCGAN non sono sbagliate: sono idee *da ultimo punto percentuale*, e noi avevamo bisogno di idee da venti punti.
2. **L'eredità viva di OCGAN nel progetto**: il mining PGD-style (usato fino all'ultima configurazione GAN), la regolarizzazione del latent (sopravvissuta come compactness in loss), e soprattutto l'impostazione mentale — *cercare attivamente dove il modello sbaglia invece di aspettare che succeda* — che è la stessa filosofia delle anomalie sintetiche.
3. **Il rigore statistico ripaga.** I numeri del paper sono single-run; i nostri multi-seed con std. Su toothbrush la std del GAN è ±0.17: con un seed fortunato avremmo potuto "pubblicare" 0.87, con uno sfortunato 0.53. Qualunque confronto serio richiede la distribuzione, non il punto.

## 7.4 Posizionamento rispetto alla letteratura

Per contesto: il paper PatchCore originale riporta ~0.991 di image-AUROC su MVTec AD nella sua configurazione di punta. Il nostro 0.9846 è nello stesso ordine, ottenuto con una reimplementazione indipendente, un protocollo 4-split con test rigorosamente blind e calibrazione dichiarata — senza tuning sul test set. Non rivendichiamo lo stato dell'arte: rivendichiamo un numero *riproducibile e onesto* a distanza di un punto da esso, partendo da un paper del 2019 e attraversando ogni scelta con un'ablation.
