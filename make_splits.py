"""Frozen, leakage-free split grouped by study (seed 42, 80/20).

Studies whose image_id ranges interleave (2,3,4 / 7,8 / 11,12,13) are merged into one group each as a precaution.
Usage: python make_splits.py --manifest data_manifest_full.csv --out split_manifest.csv
"""
import argparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

GROUPS = {2: "g1", 3: "g1", 4: "g1", 7: "g2", 8: "g2", 11: "g3", 12: "g3", 13: "g3"}
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", default="split_manifest.csv")
a = ap.parse_args()

m = pd.read_csv(a.manifest)
studies = pd.DataFrame({"study": sorted(m["study"].unique())})
studies["group"] = studies.study.map(lambda s: GROUPS.get(s, f"s{s}"))
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(studies, groups=studies["group"]))
split = {int(studies.loc[i, "study"]): "train" for i in tr}
split.update({int(studies.loc[i, "study"]): "val" for i in va})
sm = m[["study", "image_id"]].drop_duplicates().copy()
sm["split"] = sm.study.map(split)
sm["group"] = sm.study.map(lambda s: GROUPS.get(s, f"s{s}"))
sm.to_csv(a.out, index=False)
print("groups crossing the split:", set(sm[sm.split == "train"].group) & set(sm[sm.split == "val"].group))
print(sm.groupby("split").agg(studies=("study", "nunique"), groups=("group", "nunique"), images=("image_id", "nunique")))
