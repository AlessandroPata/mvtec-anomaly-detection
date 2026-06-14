import csv
import math
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

path = Path("grid_runs/rich_crosscat_metrics_only_allcats_e8_s43_44/results.csv")
rows = list(csv.DictReader(path.open()))
rows = [r for r in rows if r.get("status") == "done"]

def f(x, default=float("nan")):
    try:
        return float(x)
    except:
        return default

old_pat = re.compile(
    r'^(?P<category>.+)_(?P<teacher>t0|t1a|t1b)_(?P<memory>m0|m1a|m1b)_(?P<lf>lf0|lf1a|lf1b)_(?P<oc>oc0|oc1)_s(?P<seed>\d+)$'
)

new_pat = re.compile(
    r'^(?P<category>.+)_(?P<teacher>t1a|t1b|t1c|t1d)_(?P<memory>m1a|m1b|m1c|m1d)_(?P<lf>lf1a|lf1b|lf1c|lf1d)_(?P<oc>oc0)_s(?P<seed>\d+)$'
)

parsed = []
for r in rows:
    m = new_pat.match(r["name"])
    space = "new"
    if not m:
        m = old_pat.match(r["name"])
        space = "old"
    if not m:
        continue

    rr = dict(r)
    rr.update(m.groupdict())
    rr["space"] = space
    rr["seed"] = int(rr["seed"])
    rr["best_val_selection"] = f(rr["best_val_selection"])
    rr["test_auroc"] = f(rr["test_auroc"])
    rr["test_auprc"] = f(rr["test_auprc"])
    rr["test_best_f1"] = f(rr["test_best_f1"])
    rr["f1_at_val_threshold"] = f(rr["f1_at_val_threshold"])
    rr["fpr_at_95_tpr"] = f(rr["fpr_at_95_tpr"])
    parsed.append(rr)

def fmt(x):
    if x != x:
        return "nan"
    return f"{x:.4f}"

print("=== COUNTS ===")
print("parsed rows:", len(parsed))
print("old space:", sum(r["space"] == "old" for r in parsed))
print("new space:", sum(r["space"] == "new" for r in parsed))
print()

# =========================
# TOP GLOBALI
# =========================
top_global = sorted(
    parsed,
    key=lambda r: (
        r["best_val_selection"],
        r["test_auroc"],
        r["test_auprc"],
        r["test_best_f1"],
        -r["fpr_at_95_tpr"] if r["fpr_at_95_tpr"] == r["fpr_at_95_tpr"] else -1e9,
    ),
    reverse=True,
)

print("=== TOP 40 GLOBALI ===")
for r in top_global[:40]:
    print(
        f'{r["name"]} | space={r["space"]} | cat={r["category"]} | seed={r["seed"]} | '
        f'val={fmt(r["best_val_selection"])} auroc={fmt(r["test_auroc"])} '
        f'auprc={fmt(r["test_auprc"])} f1={fmt(r["test_best_f1"])} '
        f'f1thr={fmt(r["f1_at_val_threshold"])} fpr95={fmt(r["fpr_at_95_tpr"])}'
    )
print()

# =========================
# BEST PER CATEGORIA
# =========================
best_cat = {}
for r in parsed:
    c = r["category"]
    if c not in best_cat or (
        r["best_val_selection"],
        r["test_auroc"],
        r["test_auprc"],
        r["test_best_f1"],
    ) > (
        best_cat[c]["best_val_selection"],
        best_cat[c]["test_auroc"],
        best_cat[c]["test_auprc"],
        best_cat[c]["test_best_f1"],
    ):
        best_cat[c] = r

print("=== BEST PER CATEGORIA ===")
for c in sorted(best_cat):
    r = best_cat[c]
    print(
        f'{c}: {r["name"]} | space={r["space"]} | '
        f'val={fmt(r["best_val_selection"])} auroc={fmt(r["test_auroc"])} '
        f'auprc={fmt(r["test_auprc"])} f1={fmt(r["test_best_f1"])} '
        f'f1thr={fmt(r["f1_at_val_threshold"])} fpr95={fmt(r["fpr_at_95_tpr"])}'
    )
print()

# =========================
# MEDIE PER COMBINAZIONE
# =========================
groups = defaultdict(list)
for r in parsed:
    key = (r["space"], r["teacher"], r["memory"], r["lf"], r["oc"])
    groups[key].append(r)

summary = []
for key, rs in groups.items():
    aurocs = [x["test_auroc"] for x in rs]
    auprcs = [x["test_auprc"] for x in rs]
    f1s = [x["test_best_f1"] for x in rs]
    vals = [x["best_val_selection"] for x in rs]
    f1ths = [x["f1_at_val_threshold"] for x in rs]
    fprs = [x["fpr_at_95_tpr"] for x in rs]
    summary.append({
        "space": key[0],
        "teacher": key[1],
        "memory": key[2],
        "lf": key[3],
        "oc": key[4],
        "n": len(rs),
        "mean_val": stats.mean(vals),
        "mean_auroc": stats.mean(aurocs),
        "std_auroc": stats.pstdev(aurocs) if len(aurocs) > 1 else 0.0,
        "mean_auprc": stats.mean(auprcs),
        "mean_f1": stats.mean(f1s),
        "mean_f1thr": stats.mean(f1ths),
        "mean_fpr95": stats.mean(fprs),
    })

summary = sorted(summary, key=lambda x: (x["mean_auroc"], x["mean_auprc"], x["mean_f1"]), reverse=True)

print("=== MEDIE PER COMBINAZIONE ===")
for s in summary:
    print(
        f'space={s["space"]} t={s["teacher"]} m={s["memory"]} lf={s["lf"]} oc={s["oc"]} '
        f'n={s["n"]} val={fmt(s["mean_val"])} auroc={fmt(s["mean_auroc"])} '
        f'auprc={fmt(s["mean_auprc"])} f1={fmt(s["mean_f1"])} '
        f'f1thr={fmt(s["mean_f1thr"])} fpr95={fmt(s["mean_fpr95"])} '
        f'std_auroc={fmt(s["std_auroc"])}'
    )
print()

# =========================
# EFFETTO MEDIO DELLE VARIABILI
# =========================
def print_effect(title, key_name, subset):
    print(f"=== EFFETTO MEDIO: {title} ===")
    buckets = defaultdict(list)
    for r in subset:
        buckets[r[key_name]].append(r)
    for k in sorted(buckets):
        rs = buckets[k]
        print(
            f'{k} n={len(rs)} '
            f'mean_val={fmt(stats.mean([x["best_val_selection"] for x in rs]))} '
            f'mean_auroc={fmt(stats.mean([x["test_auroc"] for x in rs]))} '
            f'std_auroc={fmt(stats.pstdev([x["test_auroc"] for x in rs]) if len(rs) > 1 else 0.0)} '
            f'mean_auprc={fmt(stats.mean([x["test_auprc"] for x in rs]))} '
            f'mean_f1={fmt(stats.mean([x["test_best_f1"] for x in rs]))} '
            f'mean_fpr95={fmt(stats.mean([x["fpr_at_95_tpr"] for x in rs]))}'
        )
    print()

print_effect("teacher | all", "teacher", parsed)
print_effect("memory | all", "memory", parsed)
print_effect("lf | all", "lf", parsed)
print_effect("oc | all", "oc", parsed)

print_effect("teacher | new only", "teacher", [r for r in parsed if r["space"] == "new"])
print_effect("memory | new only", "memory", [r for r in parsed if r["space"] == "new"])
print_effect("lf | new only", "lf", [r for r in parsed if r["space"] == "new"])

# =========================
# RANKING ROBUSTO MACRO
# =========================
combo_cat = defaultdict(list)
for r in parsed:
    key = (r["space"], r["teacher"], r["memory"], r["lf"], r["oc"], r["category"])
    combo_cat[key].append(r)

robust = []
combo_keys = set((r["space"], r["teacher"], r["memory"], r["lf"], r["oc"]) for r in parsed)
for combo in combo_keys:
    cat_scores = []
    for cat in sorted(set(r["category"] for r in parsed)):
        rs = combo_cat.get((*combo, cat), [])
        if not rs:
            continue
        cat_scores.append({
            "category": cat,
            "auroc": stats.mean([x["test_auroc"] for x in rs]),
            "auprc": stats.mean([x["test_auprc"] for x in rs]),
            "f1": stats.mean([x["test_best_f1"] for x in rs]),
            "val": stats.mean([x["best_val_selection"] for x in rs]),
            "fpr95": stats.mean([x["fpr_at_95_tpr"] for x in rs]),
        })
    if not cat_scores:
        continue
    robust.append({
        "space": combo[0],
        "teacher": combo[1],
        "memory": combo[2],
        "lf": combo[3],
        "oc": combo[4],
        "cats": len(cat_scores),
        "macro_val": stats.mean([x["val"] for x in cat_scores]),
        "macro_auroc": stats.mean([x["auroc"] for x in cat_scores]),
        "macro_auprc": stats.mean([x["auprc"] for x in cat_scores]),
        "macro_f1": stats.mean([x["f1"] for x in cat_scores]),
        "macro_fpr95": stats.mean([x["fpr95"] for x in cat_scores]),
        "std_cat_auroc": stats.pstdev([x["auroc"] for x in cat_scores]) if len(cat_scores) > 1 else 0.0,
        "worst_cat_auroc": min(x["auroc"] for x in cat_scores),
    })

robust = sorted(
    robust,
    key=lambda r: (
        r["macro_auroc"],
        r["macro_auprc"],
        r["macro_f1"],
        -r["std_cat_auroc"],
        r["worst_cat_auroc"],
    ),
    reverse=True,
)

print("=== RANKING ROBUSTO MACRO ===")
for r in robust:
    print(
        f'space={r["space"]} t={r["teacher"]} m={r["memory"]} lf={r["lf"]} oc={r["oc"]} '
        f'cats={r["cats"]} macro_val={fmt(r["macro_val"])} macro_auroc={fmt(r["macro_auroc"])} '
        f'macro_auprc={fmt(r["macro_auprc"])} macro_f1={fmt(r["macro_f1"])} '
        f'macro_fpr95={fmt(r["macro_fpr95"])} std_cat_auroc={fmt(r["std_cat_auroc"])} '
        f'worst_cat_auroc={fmt(r["worst_cat_auroc"])}'
    )
print()

print("=== TOP 20 ROBUSTI | NEW ONLY ===")
for r in [x for x in robust if x["space"] == "new"][:20]:
    print(
        f't={r["teacher"]} m={r["memory"]} lf={r["lf"]} oc={r["oc"]} '
        f'macro_auroc={fmt(r["macro_auroc"])} macro_auprc={fmt(r["macro_auprc"])} '
        f'macro_f1={fmt(r["macro_f1"])} std_cat_auroc={fmt(r["std_cat_auroc"])} '
        f'worst_cat_auroc={fmt(r["worst_cat_auroc"])}'
    )
print()

# =========================
# CONFRONTO SPAZI
# =========================
for space in ["old", "new"]:
    ss = [r for r in parsed if r["space"] == space]
    print(f"=== SPACE SUMMARY: {space} ===")
    print("n =", len(ss))
    print("mean_val =", fmt(stats.mean([x["best_val_selection"] for x in ss])))
    print("mean_auroc =", fmt(stats.mean([x["test_auroc"] for x in ss])))
    print("mean_auprc =", fmt(stats.mean([x["test_auprc"] for x in ss])))
    print("mean_f1 =", fmt(stats.mean([x["test_best_f1"] for x in ss])))
    print("mean_fpr95 =", fmt(stats.mean([x["fpr_at_95_tpr"] for x in ss])))
    print()

# =========================
# INTERSEZIONE DIRETTA FRA SPAZI
# =========================
old_compat = {}
new_compat = {}

for r in parsed:
    key = (r["category"], r["seed"], r["teacher"], r["memory"], r["lf"], r["oc"])
    if r["space"] == "old" and r["teacher"] in {"t1a","t1b"} and r["memory"] in {"m1a","m1b"} and r["lf"] in {"lf1a","lf1b"} and r["oc"] == "oc0":
        old_compat[key] = r
    if r["space"] == "new" and r["teacher"] in {"t1a","t1b"} and r["memory"] in {"m1a","m1b"} and r["lf"] in {"lf1a","lf1b"} and r["oc"] == "oc0":
        new_compat[key] = r

shared = sorted(set(old_compat) & set(new_compat))
print("=== SHARED DIRECT COMPARISON (old vs new on common configs) ===")
print("shared configs =", len(shared))

if shared:
    d_auroc = []
    d_auprc = []
    d_f1 = []
    d_val = []
    for k in shared:
        o = old_compat[k]
        n = new_compat[k]
        d_auroc.append(n["test_auroc"] - o["test_auroc"])
        d_auprc.append(n["test_auprc"] - o["test_auprc"])
        d_f1.append(n["test_best_f1"] - o["test_best_f1"])
        d_val.append(n["best_val_selection"] - o["best_val_selection"])
    print("mean_delta_val =", fmt(stats.mean(d_val)))
    print("mean_delta_auroc =", fmt(stats.mean(d_auroc)))
    print("mean_delta_auprc =", fmt(stats.mean(d_auprc)))
    print("mean_delta_f1 =", fmt(stats.mean(d_f1)))
    print()

    improved = sum(x > 0 for x in d_auroc)
    worsened = sum(x < 0 for x in d_auroc)
    equal = sum(abs(x) < 1e-12 for x in d_auroc)
    print("auroc improved =", improved)
    print("auroc worsened =", worsened)
    print("auroc equal =", equal)
