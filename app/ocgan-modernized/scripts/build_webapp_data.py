"""Regenerate frontend benchmark + insight JSON from the repo's result CSVs.

Run:  python scripts/build_webapp_data.py
Writes: ../frontend/src/data/benchmarks.json and insights.json (relative to the
repo root's parent). The frontend treats these files as the single source of
truth for metrics — every number traces back to a result CSV in this repo.

GAN sources (real multiseed aggregates; the old draft used invented values):
  ocgan_final  = final_per_category_multiseed_aggregated.csv  (runs *_final_s4x)
  ocgan_optv2  = optv2_multiseed_aggregated.csv               (runs *_optv2_s4x)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DATA = ROOT.parent / "frontend" / "src" / "data"

EXPECTED_MACRO = {  # from README / project history, tolerance ±0.005
    "patchcore_v1": 0.9051, "patchcore_v2": 0.9397, "patchcore_v3": 0.9828,
    "production_final": 0.9846,
}


def read_csv(path: Path) -> list[dict]:
    """DictReader + drop embedded header rows (some CSVs were concatenated)."""
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("category") not in (None, "", "category")]


def seed_runs_to_per_category(rows: list[dict]) -> dict[str, dict]:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        aurocs = [float(r["auroc"]) for r in rs]
        out[cat] = {
            "category": cat,
            "auroc": round(mean(aurocs), 4),
            "auroc_std": round(stdev(aurocs), 4) if len(aurocs) > 1 else 0.0,
            "auprc": round(mean(float(r["auprc"]) for r in rs), 4),
            "best_f1": round(mean(float(r["best_f1"]) for r in rs), 4),
            "fpr95": round(mean(float(r["fpr95"]) for r in rs), 4),
            "elapsed_s": round(mean(float(r["elapsed_s"]) for r in rs), 1) if rs[0].get("elapsed_s") else None,
            "n_seeds": len(rs),
            "feature_level": rs[0].get("feature_level", "layer2+layer3"),
            "aggregation": rs[0].get("aggregation"),
            "topk": int(rs[0]["topk"]) if rs[0].get("topk") else None,
            "coreset": int(rs[0]["coreset"]) if rs[0].get("coreset") else None,
        }
    return out


def ocgan_aggregated_to_per_category(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in sorted(rows, key=lambda r: r["category"]):
        out[r["category"]] = {  # duplicated rows are identical — last one wins
            "category": r["category"],
            "auroc": round(float(r["mean_test_auroc"]), 4),
            "auroc_std": round(float(r["std_test_auroc"]), 4),
            "auprc": round(float(r["mean_test_auprc"]), 4),
            "best_f1": round(float(r["mean_test_best_f1"]), 4),
            "fpr95": round(float(r["mean_test_fpr_at_95_tpr"]), 4),
            "elapsed_s": None,
            "n_seeds": int(r["num_seeds"]),
            "feature_level": None, "aggregation": None, "topk": None, "coreset": None,
        }
    return out


def macro_of(per_cat: dict[str, dict]) -> float:
    return round(mean(v["auroc"] for v in per_cat.values()), 4)


def main() -> None:
    per_category: dict[str, dict] = {}
    per_category["ocgan_final"] = ocgan_aggregated_to_per_category(
        read_csv(ROOT / "final_per_category_multiseed_aggregated.csv"))
    per_category["ocgan_optv2"] = ocgan_aggregated_to_per_category(
        read_csv(ROOT / "optv2_multiseed_aggregated.csv"))

    per_category["patchcore_v1"] = seed_runs_to_per_category(read_csv(ROOT / "logs" / "patchcore_pure.csv"))
    per_category["patchcore_v2"] = seed_runs_to_per_category(read_csv(ROOT / "logs" / "patchcore_v2.csv"))
    per_category["patchcore_v3"] = seed_runs_to_per_category(read_csv(ROOT / "logs" / "patchcore_v3.csv"))

    p1_rows = read_csv(ROOT / "logs" / "patchcore_p1.csv")
    p1_ext = ROOT / "logs" / "patchcore_p1_ext.csv"
    if p1_ext.exists():
        p1_rows += read_csv(p1_ext)
    per_category["patchcore_p1"] = seed_runs_to_per_category(p1_rows)

    production = dict(per_category["patchcore_v3"])
    screw_l123 = [r for r in p1_rows
                  if r["category"] == "screw" and r.get("feature_level") == "layer1+layer2+layer3"]
    if screw_l123:
        production["screw"] = seed_runs_to_per_category(screw_l123)["screw"]
    per_category["production_final"] = production

    macros = {model: macro_of(pc) for model, pc in per_category.items()}

    for model, expected in EXPECTED_MACRO.items():
        got = macros.get(model)
        ok = got is not None and abs(got - expected) <= 0.005
        print(f"[build] macro {model}: {got} (expected ~{expected}) {'OK' if ok else 'MISMATCH — investigate'}")
    for model in ("ocgan_final", "ocgan_optv2"):
        print(f"[build] macro {model}: {macros[model]} (real multiseed aggregate)")

    benchmarks = {
        "per_category": {m: list(pc.values()) for m, pc in per_category.items()},
        "macros": macros,
    }
    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
    print(f"[build] wrote benchmarks.json ({len(per_category)} models)")

    insights = {
        "coreset_effect": [
            {"category": c,
             "coreset_10k": per_category["patchcore_v2"][c]["auroc"],
             "full_bank": per_category["patchcore_v3"][c]["auroc"],
             "delta": round(per_category["patchcore_v3"][c]["auroc"]
                            - per_category["patchcore_v2"][c]["auroc"], 4)}
            for c in sorted(per_category["patchcore_v3"])
            if c in per_category["patchcore_v2"]
        ],
        "aggregation_effect": [
            {"category": c,
             "topk_mean": per_category["patchcore_v1"][c]["auroc"],
             "topk_reweighted": per_category["patchcore_v2"][c]["auroc"],
             "delta": round(per_category["patchcore_v2"][c]["auroc"]
                            - per_category["patchcore_v1"][c]["auroc"], 4)}
            for c in sorted(per_category["patchcore_v2"])
            if c in per_category["patchcore_v1"]
        ],
        "layer_ablation": [
            {"category": cat,
             "configs": {
                 fl: round(mean(float(r["auroc"]) for r in p1_rows
                                if r["category"] == cat and r["feature_level"] == fl), 4)
                 for fl in sorted({r["feature_level"] for r in p1_rows if r["category"] == cat})
             }}
            for cat in sorted({r["category"] for r in p1_rows})
        ],
        "seed_stability": [
            {"category": c, "auroc_std": per_category["patchcore_v3"][c]["auroc_std"]}
            for c in sorted(per_category["patchcore_v3"])
        ],
    }
    (FRONTEND_DATA / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print("[build] wrote insights.json")


if __name__ == "__main__":
    main()
