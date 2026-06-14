# -*- coding: utf-8 -*-
"""Genera tutte le figure della relazione in relazione/figures/.

Fonti dati:
- frontend/src/data/benchmarks.json (per-categoria, generato dai CSV reali)
- ocgan-modernized/{final_per_category_multiseed_aggregated,optv2_multiseed_aggregated}.csv
- ocgan-modernized/logs/patchcore_*.csv (tempi)
- tabelle aggregate del documento di progetto (griglia on/off: hard-coded, i run non sono piu su disco)
- paper OCGAN CVPR 2019 (tabelle 1, 2, 4)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RELAZIONE = Path(__file__).resolve().parent
APP = RELAZIONE.parent / "app"
FIG = RELAZIONE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

bench = json.loads((APP / "frontend" / "src" / "data" / "benchmarks.json").read_text(encoding="utf-8"))
PC = bench["per_category"]
MACROS = bench["macros"]
CATS = [r["category"] for r in PC["production_final"]]

C_GAN = "#c2553a"      # famiglia GAN
C_PC = "#2d6a8f"       # famiglia PatchCore
C_PROD = "#1f9d55"     # production
C_GREY = "#8a8f98"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("ok:", name)


def auroc_of(model):
    return {r["category"]: r["auroc"] for r in PC[model]}


def std_of(model):
    return {r["category"]: r.get("auroc_std") or 0.0 for r in PC[model]}


# ---------------------------------------------------------------- fig 1
# Evoluzione del macro AUROC lungo il progetto
steps = [
    ("OCGAN mod.\nbaseline onesta\n(post-fix flag)", 0.7866, C_GAN),
    ("OCGAN mod.\nfinal\n(multiseed)", MACROS["ocgan_final"], C_GAN),
    ("OCGAN mod.\noptv2\n(fine-tuning)", MACROS["ocgan_optv2"], C_GAN),
    ("PatchCore v1\n(coreset 10k,\ntop-3 mean)", MACROS["patchcore_v1"], C_PC),
    ("PatchCore v2\n(layer2+3,\ntopk-rew. k9)", MACROS["patchcore_v2"], C_PC),
    ("PatchCore v3\n(bank pieno\n70k)", MACROS["patchcore_v3"], C_PC),
    ("Production\n(+layer1\nsu screw)", MACROS["production_final"], C_PROD),
]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
xs = np.arange(len(steps))
vals = [s[1] for s in steps]
ax.plot(xs, vals, "-", color=C_GREY, lw=1.4, zorder=1)
ax.scatter(xs, vals, c=[s[2] for s in steps], s=90, zorder=2)
for x, v in zip(xs, vals):
    ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(xs, [s[0] for s in steps], fontsize=8)
ax.set_ylim(0.74, 1.01)
ax.set_ylabel("Macro AUROC (15 categorie MVTec AD)")
ax.set_title("Evoluzione del progetto: dal GAN modernizzato a PatchCore production (+19.8 pp)")
ax.axvspan(-0.5, 2.5, color=C_GAN, alpha=0.05)
ax.axvspan(2.5, 6.5, color=C_PC, alpha=0.05)
ax.text(1.0, 0.748, "famiglia GAN", color=C_GAN, fontsize=9, ha="center")
ax.text(4.5, 0.748, "famiglia memory-bank", color=C_PC, fontsize=9, ha="center")
save(fig, "fig1_evoluzione.png")

# ---------------------------------------------------------------- fig 2
# Effetto medio dei fattori nella griglia on/off (dal documento di progetto)
factors = {
    "Teacher-student": [("off (t0)", 0.7069), ("t1a", 0.7735), ("t1b", 0.7740), ("t1c", 0.8374), ("t1d", 0.8375)],
    "Memory bank": [("off (m0)", 0.7474), ("m1a", 0.7526), ("m1b", 0.7545), ("m1c", 0.8356), ("m1d", 0.8415)],
    "Learned fusion": [("off (lf0)", 0.6158), ("lf1a", 0.8250), ("lf1b", 0.8293), ("lf1c", 0.8397), ("lf1d", 0.8365)],
    "One-class score": [("off (oc0)", 0.8110), ("on (oc1)", 0.7489)],
}
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
for ax, (name, rows) in zip(axes, factors.items()):
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [C_GREY if l.startswith("off") else C_PC for l in labels]
    if name == "One-class score":
        colors = [C_PC, C_GREY]
    ax.bar(labels, vals, color=colors, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_title(name, fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
axes[0].set_ylabel("AUROC medio (tutte le run)")
axes[0].set_ylim(0.5, 0.9)
fig.suptitle("Griglia di ablation (3 300 run): effetto medio di ogni componente sullo score finale", y=1.04)
save(fig, "fig2_fattori_ablation.png")

# ---------------------------------------------------------------- fig 3
# Old space vs new space
fig, ax = plt.subplots(figsize=(5.2, 3.4))
labels = ["Spazio 'old'\n(1 380 run)", "Spazio 'new'\n(1 920 run)"]
auroc = [0.7352, 0.8392]
bars = ax.bar(labels, auroc, color=[C_GREY, C_PC], width=0.5)
for b, v in zip(bars, auroc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0.6, 0.9)
ax.set_ylabel("AUROC medio su tutte le run")
ax.set_title("Prima vs seconda generazione dello spazio di ricerca")
save(fig, "fig3_old_new_space.png")

# ---------------------------------------------------------------- fig 4
# GAN final vs optv2 per categoria, con std multiseed
def read_agg(name):
    # il CSV "final" contiene la tabella duplicata (header incluso): pulizia + cast
    df = pd.read_csv(APP / "ocgan-modernized" / name)
    df = df[df.category != "category"].drop_duplicates("category")
    for col in df.columns:
        if col not in ("category", "best_run_name"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("category").loc[CATS]

fin = read_agg("final_per_category_multiseed_aggregated.csv")
opt = read_agg("optv2_multiseed_aggregated.csv")
x = np.arange(len(CATS))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(x - w / 2, fin["mean_test_auroc"], w, yerr=fin["std_test_auroc"], capsize=2,
       label=f"OCGAN final (macro {MACROS['ocgan_final']:.4f})", color=C_GAN, alpha=0.85)
ax.bar(x + w / 2, opt["mean_test_auroc"], w, yerr=opt["std_test_auroc"], capsize=2,
       label=f"OCGAN optv2 (macro {MACROS['ocgan_optv2']:.4f})", color="#e0936f", alpha=0.9)
ax.set_xticks(x, CATS, rotation=40, ha="right", fontsize=8.5)
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("Test AUROC (media multiseed ± std)")
ax.set_title("Le due versioni finali del GAN modernizzato per categoria")
ax.legend(loc="lower right", fontsize=9)
save(fig, "fig4_gan_per_categoria.png")

# ---------------------------------------------------------------- fig 5
# Production vs miglior GAN per categoria (complementarita')
prod = auroc_of("production_final")
gan_best = {c: max(fin.loc[c, "mean_test_auroc"], opt.loc[c, "mean_test_auroc"]) for c in CATS}
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(x - w / 2, [prod[c] for c in CATS], w, label=f"PatchCore production (macro {MACROS['production_final']:.4f})", color=C_PROD, alpha=0.9)
ax.bar(x + w / 2, [gan_best[c] for c in CATS], w, label="miglior OCGAN modernizzato (final/optv2)", color=C_GAN, alpha=0.85)
for i, c in enumerate(CATS):
    if c in ("screw", "cable"):
        ax.annotate(c, (i, max(prod[c], gan_best[c]) + 0.02), ha="center", fontsize=8.5,
                    fontweight="bold", color="#333")
ax.set_xticks(x, CATS, rotation=40, ha="right", fontsize=8.5)
ax.set_ylim(0.4, 1.08)
ax.set_ylabel("Test AUROC")
ax.set_title("Production vs GAN: su screw il GAN vince (1.000 vs 0.942), su cable crolla (0.58 vs 1.00)")
ax.legend(loc="lower right", fontsize=9)
save(fig, "fig5_prod_vs_gan.png")

# ---------------------------------------------------------------- fig 6
# Heatmap modelli x categorie
models = ["ocgan_final", "ocgan_optv2", "patchcore_v1", "patchcore_v2", "patchcore_v3", "patchcore_p1", "production_final"]
labels_m = ["OCGAN final", "OCGAN optv2", "PatchCore v1", "PatchCore v2", "PatchCore v3", "PatchCore L1+L2+L3", "Production"]
mat = np.array([[auroc_of(m).get(c, np.nan) for c in CATS] for m in models])
fig, ax = plt.subplots(figsize=(11, 4.6))
im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(CATS)), CATS, rotation=40, ha="right", fontsize=8.5)
ax.set_yticks(range(len(models)), labels_m, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if mat[i, j] > 0.62 else "white")
ax.set_title("Test AUROC per modello e categoria (heatmap completa)")
fig.colorbar(im, ax=ax, shrink=0.85, label="AUROC")
ax.grid(False)
save(fig, "fig6_heatmap.png")

# ---------------------------------------------------------------- fig 7
# I tre ingredienti PatchCore: v1 -> v2 -> v3 per categoria
v1, v2, v3 = auroc_of("patchcore_v1"), auroc_of("patchcore_v2"), auroc_of("patchcore_v3")
fig, ax = plt.subplots(figsize=(11, 4.2))
w3 = 0.27
ax.bar(x - w3, [v1[c] for c in CATS], w3, label=f"v1: layer3, top-3 mean, coreset 10k ({MACROS['patchcore_v1']:.4f})", color="#9fb8c9")
ax.bar(x, [v2[c] for c in CATS], w3, label=f"v2: +layer2, topk-reweighted k9 ({MACROS['patchcore_v2']:.4f})", color="#5a8fb0")
ax.bar(x + w3, [v3[c] for c in CATS], w3, label=f"v3: bank pieno 70k, no coreset ({MACROS['patchcore_v3']:.4f})", color=C_PC)
ax.set_xticks(x, CATS, rotation=40, ha="right", fontsize=8.5)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Test AUROC")
ax.set_title("PatchCore: effetto incrementale dei tre ingredienti")
ax.legend(loc="lower right", fontsize=8.5)
save(fig, "fig7_patchcore_v1v2v3.png")

# ---------------------------------------------------------------- fig 8
# Costo computazionale per categoria (build+eval)
logs = APP / "ocgan-modernized" / "logs"
t_v1 = pd.read_csv(logs / "patchcore_pure.csv")["elapsed_s"].mean()
t_v2 = pd.read_csv(logs / "patchcore_v2.csv")["elapsed_s"].mean()
df_v3 = pd.read_csv(logs / "patchcore_v3.csv")
# hazelnut esclusa dalla media: unica categoria che sfora le 70k patch
# (391 immagini di train) e attiva il k-center-greedy -> 799 s, outlier
t_v3 = df_v3[df_v3.category != "hazelnut"]["elapsed_s"].mean()
lc = pd.read_csv(logs / "patchcore_lc.csv")
t_lc_screw = lc[lc.category == "screw"]["elapsed_s"].mean()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = ["v1\n(coreset 10k)", "v2\n(coreset 10k)", "coreset 50k\n(solo screw)", "v3*\n(bank pieno,\nno coreset)"]
times = [t_v1, t_v2, t_lc_screw, t_v3]
colors = ["#9fb8c9", "#5a8fb0", "#c2553a", C_PC]
bars = ax.bar(names, times, color=colors, width=0.6)
for b, v in zip(bars, times):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.0f} s", ha="center", fontweight="bold", fontsize=9)
ax.set_yscale("log")
ax.set_ylabel("Tempo medio per categoria (s, scala log)")
ax.set_title("Il paradosso del coreset: il bank pieno e' anche ~6x piu' veloce")
ax.tick_params(axis="x", labelsize=8)
fig.text(0.02, -0.04,
         "* media su 14 categorie: hazelnut (391 img, unica oltre le 70k patch -> coreset attivo) impiega 799 s",
         fontsize=7.5, color="#555555")
save(fig, "fig8_tempi.png")

# ---------------------------------------------------------------- fig 9
# Memory bank: due regimi della stessa leva
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
sizes = ["1 024", "2 048", "4 096"]
vals9 = [0.8753, 0.8611, 0.8551]
ax.bar(sizes, vals9, color=C_GAN, width=0.55, alpha=0.85)
for i, v in enumerate(vals9):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylim(0.80, 0.90)
ax.set_title("GAN: memory score ausiliario\n(coreset k-center, aggregazione max)")
ax.set_xlabel("max_patches")
ax.set_ylabel("Test AUROC medio (15 cat)")
ax = axes[1]
sizes2 = ["10 000\n(v2)", "50 000\n(esperimento)", "70 000 = pieno\n(v3)"]
zipper = [0.7644 if False else None, None, None]
vals_zip = [0.7184, 0.9801, 0.9801]
vals_cap = [0.7724, 0.9824, 0.9824]
xx = np.arange(3)
ax.plot(xx, vals_zip, "o-", label="zipper", color=C_PC)
ax.plot(xx, vals_cap, "s-", label="capsule", color="#5a8fb0")
ax.set_xticks(xx, sizes2, fontsize=8.5)
ax.set_ylim(0.65, 1.02)
ax.set_title("PatchCore: il bank E' il modello\n(layer2+3, topk-reweighted)")
ax.set_xlabel("patch nel bank")
ax.legend(fontsize=9)
fig.suptitle("La stessa leva (dimensione memory bank) in due regimi diversi", y=1.04)
save(fig, "fig9_memorybank.png")

# ---------------------------------------------------------------- fig 10
# Confronto col paper: due mondi
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
paper = [("OCSVM", 0.9513), ("DAE", 0.8766), ("AnoGAN", 0.9127), ("DSVDD", 0.9480), ("AND", 0.9671), ("OCGAN\n(paper)", 0.9750)]
colors_p = [C_GREY] * 5 + [C_GAN]
ax.bar([p[0] for p in paper], [p[1] for p in paper], color=colors_p, width=0.6)
for i, (_, v) in enumerate(paper):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(0.82, 1.0)
ax.set_title("Paper (2019): MNIST protocol 2, mean AUC")
ax.tick_params(axis="x", labelsize=8)
ax = axes[1]
ours = [("OCGAN mod.\nbaseline", 0.7866, C_GAN), ("OCGAN mod.\noptv2", MACROS["ocgan_optv2"], C_GAN),
        ("PatchCore v1", MACROS["patchcore_v1"], C_PC), ("Production", MACROS["production_final"], C_PROD)]
ax.bar([o[0] for o in ours], [o[1] for o in ours], color=[o[2] for o in ours], width=0.6)
for i, (_, v, _c) in enumerate(ours):
    ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
ax.set_ylim(0.7, 1.02)
ax.set_title("Questo progetto: MVTec AD (15 categorie), macro AUROC")
ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Dal benchmark del paper al benchmark industriale", y=1.04)
save(fig, "fig10_paper_vs_progetto.png")

# ---------------------------------------------------------------- fig 11
# Ablation del paper vs nostra ablation (componenti)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
rows = [("AE solo", 0.957), ("+ latent discr.", 0.959), ("+ visual discr.", 0.971), ("+ classifier\n(mining)", 0.975)]
ax.bar([r[0] for r in rows], [r[1] for r in rows], color=["#d9b8ae", "#d09a85", "#ca7c5e", C_GAN], width=0.55)
for i, (_, v) in enumerate(rows):
    ax.text(i, v + 0.0006, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.95, 0.98)
ax.set_ylabel("Mean AUC (MNIST)")
ax.set_title("Ablation del paper OCGAN: ogni componente aggiunge poco ma stabile")
ax.tick_params(axis="x", labelsize=9)
save(fig, "fig11_ablation_paper.png")

print("\nMacro check: final", fin["mean_test_auroc"].mean().round(4), "| optv2", opt["mean_test_auroc"].mean().round(4))
print("Figures dir:", FIG)
