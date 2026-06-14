"""Diff configs across run families: grid winner vs final vs optv2 vs production."""
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
    path = os.path.join(hits[0], "config.yaml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # some archived configs are dumped twice; keep the first copy
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip() == "project:"]
    if len(starts) > 1:
        text = "".join(lines[: starts[1]])
    return flatten(yaml.safe_load(text)), hits[0]


IGNORE = ("project.experiment_name", "project.seed", "dataset.category", "dataset.root",
          "project.output_dir", "dataset.seed")

families = {
    "grid_winner": "bottle_t1d_m1d_lf1c_mp1a_oc0_s43_*",
    "final_mvtec": "final_mvtec_t1d_m1d_lf1c_oc0_seed42_20260412_085009",
    "final": "bottle_final_s43_*",
    "production": "bottle_production_s43_*",
    "optv2": "bottle_optv2_s43_seed43_20260413_102015",
}

configs = {}
for name, pat in families.items():
    res = load_first(pat)
    if res is None:
        print(f"MISSING {name}: {pat}")
        continue
    cfg, path = res
    configs[name] = cfg
    print(f"{name}: {os.path.basename(path)}  ({len(cfg)} keys)")

keys = set().union(*[set(c) for c in configs.values()])
print("\n=== differing keys ===")
for key in sorted(keys):
    if key.startswith(IGNORE):
        continue
    values = {n: c.get(key, "<absent>") for n, c in configs.items()}
    if len({repr(v) for v in values.values()}) > 1:
        print(f"{key}:")
        for n in families:
            if n in values:
                print(f"    {n:12s} = {values[n]}")
