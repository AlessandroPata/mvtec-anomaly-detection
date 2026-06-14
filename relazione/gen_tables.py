# -*- coding: utf-8 -*-
"""Genera le tabelle markdown della relazione in relazione/tables/."""
import json
from pathlib import Path

import pandas as pd

RELAZIONE = Path(__file__).resolve().parent
APP = RELAZIONE.parent / "app"
TAB = RELAZIONE / "tables"
TAB.mkdir(parents=True, exist_ok=True)

bench = json.loads((APP / "frontend" / "src" / "data" / "benchmarks.json").read_text(encoding="utf-8"))
PC = bench["per_category"]
MACROS = bench["macros"]
CATS = [r["category"] for r in PC["production_final"]]

MODELS = [
    ("ocgan_final", "OCGAN final"),
    ("ocgan_optv2", "OCGAN optv2"),
    ("patchcore_v1", "PC v1"),
    ("patchcore_v2", "PC v2"),
    ("patchcore_v3", "PC v3"),
    ("production_final", "Production"),
]


def auroc_of(model):
    return {r["category"]: r["auroc"] for r in PC[model]}


# --- Tabella A1: AUROC per categoria, tutti i modelli ---
lines = ["| Categoria | " + " | ".join(lbl for _, lbl in MODELS) + " |",
         "|---|" + "---|" * len(MODELS)]
data = {m: auroc_of(m) for m, _ in MODELS}
for c in CATS:
    row = [c]
    best = max(data[m].get(c, -1) for m, _ in MODELS)
    for m, _ in MODELS:
        v = data[m].get(c)
        cell = "—" if v is None else (f"**{v:.4f}**" if abs(v - best) < 1e-9 else f"{v:.4f}")
        row.append(cell)
    lines.append("| " + " | ".join(row) + " |")
macro_row = ["**Macro**"]
for m, _ in MODELS:
    macro_row.append(f"**{MACROS[m]:.4f}**")
lines.append("| " + " | ".join(macro_row) + " |")
(TAB / "tab_a1_auroc_tutti.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- Tabella A2: GAN final e optv2 con std e seed ---
def read_agg(name):
    df = pd.read_csv(APP / "ocgan-modernized" / name)
    df = df[df.category != "category"].drop_duplicates("category")
    for col in df.columns:
        if col not in ("category", "best_run_name"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("category").loc[CATS]

fin = read_agg("final_per_category_multiseed_aggregated.csv")
opt = read_agg("optv2_multiseed_aggregated.csv")
lines = ["| Categoria | final AUROC (±std) | seed | optv2 AUROC (±std) | seed | Δ optv2−final |",
         "|---|---|---|---|---|---|"]
for c in CATS:
    f_m, f_s, f_n = fin.loc[c, "mean_test_auroc"], fin.loc[c, "std_test_auroc"], int(fin.loc[c, "num_seeds"])
    o_m, o_s, o_n = opt.loc[c, "mean_test_auroc"], opt.loc[c, "std_test_auroc"], int(opt.loc[c, "num_seeds"])
    d = o_m - f_m
    lines.append(f"| {c} | {f_m:.4f} (±{f_s:.4f}) | {f_n} | {o_m:.4f} (±{o_s:.4f}) | {o_n} | {d:+.4f} |")
lines.append(f"| **Macro** | **{fin['mean_test_auroc'].mean():.4f}** |  | **{opt['mean_test_auroc'].mean():.4f}** |  | **{opt['mean_test_auroc'].mean()-fin['mean_test_auroc'].mean():+.4f}** |")
(TAB / "tab_a2_gan_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- Tabella A3: production con config e tempi ---
lines = ["| Categoria | AUROC | AUPRC | best F1 | FPR@95TPR | feature levels | t (s/cat) |",
         "|---|---|---|---|---|---|---|"]
for r in PC["production_final"]:
    lines.append(f"| {r['category']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['best_f1']:.4f} | "
                 f"{r['fpr95']:.4f} | {r['feature_level']} | {r['elapsed_s']:.1f} |")
(TAB / "tab_a3_production.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- Tabella confronto architetture OCGAN paper vs nostro ---
print("ok tables ->", TAB)
for f in TAB.glob("*.md"):
    print(" ", f.name, f.stat().st_size, "bytes")
