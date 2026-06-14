#!/usr/bin/env python3
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path
import re


RESULTS_PATH = Path("grid_runs/shortlist_final16/results.csv")
OUT_MD = Path("grid_runs/shortlist_final16/summary_report.md")


PAT = re.compile(
    r"^(?P<category>.+)_(?P<tag>t1[a-d]_m1[a-d]_lf1[a-d]_oc0)_s(?P<seed>\d+)$"
)


def f(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def fmt(x):
    if x != x:
        return "nan"
    return f"{x:.4f}"


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def parse_results(path: Path):
    rows = list(csv.DictReader(path.open()))
    rows = [r for r in rows if r.get("status") == "done"]

    parsed = []
    for r in rows:
        m = PAT.match(r["name"])
        if not m:
            continue
        rr = dict(r)
        rr.update(m.groupdict())
        rr["seed"] = int(rr["seed"])

        tag = rr["tag"]
        parts = tag.split("_")
        rr["teacher"] = parts[0]
        rr["memory"] = parts[1]
        rr["lf"] = parts[2]
        rr["oc"] = parts[3]

        rr["best_val_selection"] = f(rr["best_val_selection"])
        rr["test_auroc"] = f(rr["test_auroc"])
        rr["test_auprc"] = f(rr["test_auprc"])
        rr["test_best_f1"] = f(rr["test_best_f1"])
        rr["f1_at_val_threshold"] = f(rr["f1_at_val_threshold"])
        rr["fpr_at_95_tpr"] = f(rr["fpr_at_95_tpr"])
        parsed.append(rr)

    return parsed


def top_global(parsed, k=30):
    return sorted(
        parsed,
        key=lambda r: (
            r["best_val_selection"],
            r["test_auroc"],
            r["test_auprc"],
            r["test_best_f1"],
            -r["fpr_at_95_tpr"] if r["fpr_at_95_tpr"] == r["fpr_at_95_tpr"] else -1e9,
        ),
        reverse=True,
    )[:k]


def best_per_category(parsed):
    best = {}
    for r in parsed:
        c = r["category"]
        if c not in best or (
            r["best_val_selection"],
            r["test_auroc"],
            r["test_auprc"],
            r["test_best_f1"],
        ) > (
            best[c]["best_val_selection"],
            best[c]["test_auroc"],
            best[c]["test_auprc"],
            best[c]["test_best_f1"],
        ):
            best[c] = r
    return [best[c] for c in sorted(best)]


def combo_summary(parsed):
    groups = defaultdict(list)
    for r in parsed:
        key = (r["tag"], r["teacher"], r["memory"], r["lf"], r["oc"])
        groups[key].append(r)

    out = []
    for key, rs in groups.items():
        aurocs = [x["test_auroc"] for x in rs]
        auprcs = [x["test_auprc"] for x in rs]
        f1s = [x["test_best_f1"] for x in rs]
        vals = [x["best_val_selection"] for x in rs]
        f1ths = [x["f1_at_val_threshold"] for x in rs]
        fprs = [x["fpr_at_95_tpr"] for x in rs]
        out.append({
            "tag": key[0],
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

    out.sort(key=lambda x: (x["mean_auroc"], x["mean_auprc"], x["mean_f1"]), reverse=True)
    return out


def effect_table(parsed, key_name):
    buckets = defaultdict(list)
    for r in parsed:
        buckets[r[key_name]].append(r)

    out = []
    for k in sorted(buckets):
        rs = buckets[k]
        out.append({
            key_name: k,
            "n": len(rs),
            "mean_val": stats.mean([x["best_val_selection"] for x in rs]),
            "mean_auroc": stats.mean([x["test_auroc"] for x in rs]),
            "std_auroc": stats.pstdev([x["test_auroc"] for x in rs]) if len(rs) > 1 else 0.0,
            "mean_auprc": stats.mean([x["test_auprc"] for x in rs]),
            "mean_f1": stats.mean([x["test_best_f1"] for x in rs]),
            "mean_fpr95": stats.mean([x["fpr_at_95_tpr"] for x in rs]),
        })
    return out


def robust_macro(parsed):
    combo_cat = defaultdict(list)
    combos = set()

    for r in parsed:
        combo = (r["tag"], r["teacher"], r["memory"], r["lf"], r["oc"])
        combos.add(combo)
        combo_cat[(combo, r["category"])].append(r)

    out = []
    categories = sorted(set(r["category"] for r in parsed))

    for combo in combos:
        cat_scores = []
        for cat in categories:
            rs = combo_cat.get((combo, cat), [])
            if not rs:
                continue
            cat_scores.append({
                "category": cat,
                "val": stats.mean([x["best_val_selection"] for x in rs]),
                "auroc": stats.mean([x["test_auroc"] for x in rs]),
                "auprc": stats.mean([x["test_auprc"] for x in rs]),
                "f1": stats.mean([x["test_best_f1"] for x in rs]),
                "fpr95": stats.mean([x["fpr_at_95_tpr"] for x in rs]),
            })

        out.append({
            "tag": combo[0],
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

    out.sort(
        key=lambda r: (
            r["macro_auroc"],
            r["macro_auprc"],
            r["macro_f1"],
            -r["std_cat_auroc"],
            r["worst_cat_auroc"],
        ),
        reverse=True,
    )
    return out


def seed_stability(parsed):
    groups = defaultdict(list)
    for r in parsed:
        groups[(r["tag"], r["seed"])].append(r)

    seed_means = defaultdict(list)
    for (tag, seed), rs in groups.items():
        seed_means[tag].append({
            "seed": seed,
            "auroc": stats.mean([x["test_auroc"] for x in rs]),
            "auprc": stats.mean([x["test_auprc"] for x in rs]),
            "f1": stats.mean([x["test_best_f1"] for x in rs]),
            "val": stats.mean([x["best_val_selection"] for x in rs]),
        })

    out = []
    for tag, rs in seed_means.items():
        out.append({
            "tag": tag,
            "n_seeds": len(rs),
            "mean_seed_auroc": stats.mean([x["auroc"] for x in rs]),
            "std_seed_auroc": stats.pstdev([x["auroc"] for x in rs]) if len(rs) > 1 else 0.0,
            "mean_seed_auprc": stats.mean([x["auprc"] for x in rs]),
            "std_seed_auprc": stats.pstdev([x["auprc"] for x in rs]) if len(rs) > 1 else 0.0,
            "mean_seed_f1": stats.mean([x["f1"] for x in rs]),
            "std_seed_f1": stats.pstdev([x["f1"] for x in rs]) if len(rs) > 1 else 0.0,
            "mean_seed_val": stats.mean([x["val"] for x in rs]),
            "std_seed_val": stats.pstdev([x["val"] for x in rs]) if len(rs) > 1 else 0.0,
        })

    out.sort(key=lambda x: (x["mean_seed_auroc"], -x["std_seed_auroc"]), reverse=True)
    return out


def main():
    parsed = parse_results(RESULTS_PATH)
    if not parsed:
        raise SystemExit(f"Nessuna riga valida trovata in {RESULTS_PATH}")

    report = []

    report.append("# Shortlist final16 summary")
    report.append("")
    report.append(f"- rows parsed: {len(parsed)}")
    report.append(f"- categories: {len(set(r['category'] for r in parsed))}")
    report.append(f"- seeds: {sorted(set(r['seed'] for r in parsed))}")
    report.append(f"- configs: {len(set(r['tag'] for r in parsed))}")
    report.append("")

    tg = top_global(parsed, 30)
    tg_rows = [[
        r["name"], r["category"], r["seed"], fmt(r["best_val_selection"]),
        fmt(r["test_auroc"]), fmt(r["test_auprc"]), fmt(r["test_best_f1"]),
        fmt(r["f1_at_val_threshold"]), fmt(r["fpr_at_95_tpr"])
    ] for r in tg]
    report.append("## Top 30 globali")
    report.append(md_table(
        ["name", "cat", "seed", "val", "auroc", "auprc", "f1", "f1thr", "fpr95"],
        tg_rows
    ))
    report.append("")

    bpc = best_per_category(parsed)
    bpc_rows = [[
        r["category"], r["name"], fmt(r["best_val_selection"]), fmt(r["test_auroc"]),
        fmt(r["test_auprc"]), fmt(r["test_best_f1"]), fmt(r["f1_at_val_threshold"]),
        fmt(r["fpr_at_95_tpr"])
    ] for r in bpc]
    report.append("## Best per categoria")
    report.append(md_table(
        ["category", "name", "val", "auroc", "auprc", "f1", "f1thr", "fpr95"],
        bpc_rows
    ))
    report.append("")

    cs = combo_summary(parsed)
    cs_rows = [[
        r["tag"], r["n"], fmt(r["mean_val"]), fmt(r["mean_auroc"]), fmt(r["mean_auprc"]),
        fmt(r["mean_f1"]), fmt(r["mean_f1thr"]), fmt(r["mean_fpr95"]), fmt(r["std_auroc"])
    ] for r in cs]
    report.append("## Medie per combinazione")
    report.append(md_table(
        ["tag", "n", "mean_val", "mean_auroc", "mean_auprc", "mean_f1", "mean_f1thr", "mean_fpr95", "std_auroc"],
        cs_rows
    ))
    report.append("")

    for key in ["teacher", "memory", "lf", "oc"]:
        eff = effect_table(parsed, key)
        rows = [[
            r[key], r["n"], fmt(r["mean_val"]), fmt(r["mean_auroc"]),
            fmt(r["std_auroc"]), fmt(r["mean_auprc"]), fmt(r["mean_f1"]), fmt(r["mean_fpr95"])
        ] for r in eff]
        report.append(f"## Effetto medio: {key}")
        report.append(md_table(
            [key, "n", "mean_val", "mean_auroc", "std_auroc", "mean_auprc", "mean_f1", "mean_fpr95"],
            rows
        ))
        report.append("")

    rm = robust_macro(parsed)
    rm_rows = [[
        r["tag"], r["cats"], fmt(r["macro_val"]), fmt(r["macro_auroc"]), fmt(r["macro_auprc"]),
        fmt(r["macro_f1"]), fmt(r["macro_fpr95"]), fmt(r["std_cat_auroc"]), fmt(r["worst_cat_auroc"])
    ] for r in rm]
    report.append("## Ranking robusto macro")
    report.append(md_table(
        ["tag", "cats", "macro_val", "macro_auroc", "macro_auprc", "macro_f1", "macro_fpr95", "std_cat_auroc", "worst_cat_auroc"],
        rm_rows
    ))
    report.append("")

    ss = seed_stability(parsed)
    ss_rows = [[
        r["tag"], r["n_seeds"], fmt(r["mean_seed_val"]), fmt(r["std_seed_val"]),
        fmt(r["mean_seed_auroc"]), fmt(r["std_seed_auroc"]),
        fmt(r["mean_seed_auprc"]), fmt(r["std_seed_auprc"]),
        fmt(r["mean_seed_f1"]), fmt(r["std_seed_f1"])
    ] for r in ss]
    report.append("## Stabilità sui seed")
    report.append(md_table(
        ["tag", "n_seeds", "mean_seed_val", "std_seed_val", "mean_seed_auroc", "std_seed_auroc", "mean_seed_auprc", "std_seed_auprc", "mean_seed_f1", "std_seed_f1"],
        ss_rows
    ))
    report.append("")

    text = "\n".join(report)
    OUT_MD.write_text(text)

    print(text)
    print()
    print(f"[saved] {OUT_MD}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
