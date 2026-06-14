"""Map grid factor levels (t/m/lf/mp) to concrete config differences.

For each factor, pick runs identical in every other factor and print the
flattened config keys whose values differ across levels.
"""
import glob
import os

import yaml

OUT = r"D:\OCGAN\outputs\ocgan-modernized"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def load_first(pattern):
    hits = sorted(glob.glob(os.path.join(OUT, pattern)))
    if not hits:
        return None
    with open(os.path.join(hits[0], "config.yaml"), encoding="utf-8") as f:
        return flatten(yaml.safe_load(f))


IGNORE_PREFIXES = ("project.experiment_name", "project.seed", "dataset.category")

FACTORS = {
    "teacher": ["bottle_%s_m1d_lf1c_mp1c_oc0_s43_*" % t for t in ("t0x", "t1a", "t1b", "t1d")],
    "memory": ["bottle_t1d_%s_lf1c_mp1c_oc0_s43_*" % m for m in ("m0x", "m1b", "m1d", "m1f")],
    "fusion": ["bottle_t1d_m1d_%s_mp1c_oc0_s43_*" % lf for lf in ("lf1b", "lf1c", "lf1d", "lf1g")],
    "patches": ["bottle_t1d_m1d_lf1c_%s_oc0_s43_*" % mp for mp in ("mp1a", "mp1b", "mp1c")],
}

for factor, patterns in FACTORS.items():
    configs = {}
    for pat in patterns:
        level = pat.split("_")[1 if factor == "teacher" else 2 if factor == "memory" else 3 if factor == "fusion" else 4]
        cfg = load_first(pat)
        if cfg is None:
            print(f"[{factor}] MISSING: {pat}")
        else:
            configs[level] = cfg
    if len(configs) < 2:
        continue
    keys = set().union(*[set(c) for c in configs.values()])
    print(f"\n=== {factor}: levels {sorted(configs)} ===")
    for key in sorted(keys):
        if key.startswith(IGNORE_PREFIXES):
            continue
        values = {lvl: c.get(key, "<absent>") for lvl, c in configs.items()}
        if len({repr(v) for v in values.values()}) > 1:
            print(f"  {key}: " + " | ".join(f"{lvl}={values[lvl]}" for lvl in sorted(values)))
