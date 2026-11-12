"""Find duplicated target channels (for example Actin == Nucleus) and build a pairs file that excludes them.

Usage (Colab, from the repo folder):
  python check_duplicates.py --out OUT [--prepared-dir /content/prepared] [--exclude 3:Actin] [--w3-root OUTPUTS_W3]

1. Every usable image (train and validation) with at least two observed targets is checked. Two target channels of the
   same image are flagged when their normalized arrays are identical, or when their Pearson correlation on a 4x
   subsampled grid is at least 0.98. Normalized arrays come from --prepared-dir when given (fast), otherwise from the
   raw TIFFs. Every flagged pair is then confirmed on the raw files: raw_identical is True only if the raw pixel
   arrays are exactly equal.
2. All Study 3 images are listed with their split, observed channels and duplicate flags (study3_images.csv).
3. --exclude STUDY:ORGANELLE (repeatable) writes pairs_excluded.csv, a copy of pairs.csv where that target is marked
   as not observed. Images left without any observed target drop out of training and scoring automatically
   (dataset.load_pairs). The number of observed targets per organelle and split is printed before and after.
4. --w3-root rescores the existing Week 3 per_image_scores.csv files (best.pt) without the excluded targets, so the
   effect of the exclusion on the reported scores is visible without retraining.
"""
import argparse
import dataclasses
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from config import ORGANELLES, ChallengeConfig
from dataset import load_pairs, normalize
from evaluate import summarize

CORR_MIN = 0.98


def targets(cfg, r, prepared):
    """Normalized observed targets of one image as {organelle: array}."""
    obs = [o for o in ORGANELLES if bool(r[f"{o}_obs"])]
    if prepared is not None:
        arr = np.load(prepared / f"{int(r.study):02d}_{int(r.image_id):05d}.npy", mmap_mode="r")
        return {o: np.asarray(arr[1 + ORGANELLES.index(o)]) for o in obs}
    base = cfg.data_root / f"Study_{int(r.study):02d}"
    return {o: normalize(tifffile.imread(base / r[f"{o}_file"]), cfg.p_low, cfg.p_high).astype(np.float16) for o in obs}


def corr(a, b):
    a = a[::4, ::4].astype(np.float64).ravel()
    b = b[::4, ::4].astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def raw_identical(cfg, r, o1, o2):
    base = cfg.data_root / f"Study_{int(r.study):02d}"
    a = np.squeeze(tifffile.imread(base / r[f"{o1}_file"]))
    b = np.squeeze(tifffile.imread(base / r[f"{o2}_file"]))
    return bool(a.shape == b.shape and np.array_equal(a, b))


def counts(pairs):
    rows = []
    for split in ("train", "val"):
        d = pairs[(pairs.split == split) & ~pairs.excluded]
        keep = d[[f"{o}_obs" for o in ORGANELLES]].to_numpy().any(axis=1)
        d = d[keep]
        rows.append({"split": split, "images": len(d), **{o: int(d[f"{o}_obs"].sum()) for o in ORGANELLES}})
    return pd.DataFrame(rows)


def rescore(w3_root, excl, out):
    rows = []
    for run in sorted(Path(w3_root).glob("*_s*")):
        f = run / "per_image_scores.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        drop = np.zeros(len(df), dtype=bool)
        for study, org in excl:
            drop |= (df.study == study).to_numpy() & (df.organelle == org).to_numpy()
        before, after = summarize(df), summarize(df[~drop])
        cfg_name, seed = run.name.rsplit("_s", 1)
        row = {"config": cfg_name, "seed": int(seed), "dropped_rows": int(drop.sum()),
               "macro_ssim": before.attrs["macro_ssim"], "macro_pcc": before.attrs["macro_pcc"],
               "macro_ssim_excl": after.attrs["macro_ssim"], "macro_pcc_excl": after.attrs["macro_pcc"]}
        for o in ORGANELLES:
            a = after[after.organelle == o].iloc[0]
            row[f"{o}_n_excl"] = int(a.n_observed)
            row[f"{o}_ssim_excl"] = a.ssim_mean
            row[f"{o}_pcc_excl"] = a.pcc_mean
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "w3_rescored_excluded.csv", index=False)
    cols = ["macro_ssim", "macro_pcc", "macro_ssim_excl", "macro_pcc_excl", "Actin_ssim_excl", "Actin_pcc_excl"]
    agg = t.groupby("config")[cols].agg(["mean", "std"])
    print("\nWeek 3 best.pt scores, before and after the exclusion (mean and std over seeds)")
    print(agg.round(4).to_string())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--exclude", action="append", default=[], help="STUDY:ORGANELLE, for example 3:Actin")
    ap.add_argument("--w3-root", default=None)
    a = ap.parse_args()
    over = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(a.prepared_dir) if a.prepared_dir else None
    excl = []
    for e in a.exclude:
        s, o = e.split(":")
        assert o in ORGANELLES, f"unknown organelle {o}"
        excl.append((int(s), o))

    df = pd.concat([load_pairs(cfg, "train"), load_pairs(cfg, "val")], ignore_index=True)
    n_obs = df[[f"{o}_obs" for o in ORGANELLES]].sum(axis=1)
    todo = df[n_obs >= 2]
    print(f"{len(df)} usable images, {len(todo)} with two or more observed targets, source:",
          "prepared arrays" if prepared else "raw TIFFs")
    flags = []
    for i, (_, r) in enumerate(todo.iterrows()):
        t = targets(cfg, r, prepared)
        h = {o: hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest() for o, v in t.items()}
        for o1, o2 in combinations(t, 2):
            same = h[o1] == h[o2]
            c = 1.0 if same else corr(t[o1], t[o2])
            if same or (c == c and c >= CORR_MIN):
                flags.append({"study": int(r.study), "image_id": int(r.image_id), "split": r.split, "pair": f"{o1}={o2}",
                              "normalized_identical": same, "corr": round(c, 5), "raw_identical": raw_identical(cfg, r, o1, o2),
                              "file_1": r[f"{o1}_file"], "file_2": r[f"{o2}_file"]})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(todo)} checked, {len(flags)} flagged")
    fl = pd.DataFrame(flags, columns=["study", "image_id", "split", "pair", "normalized_identical", "corr",
                                      "raw_identical", "file_1", "file_2"])
    fl.to_csv(out / "duplicate_targets.csv", index=False)
    print(f"\nFlagged channel pairs: {len(fl)} (raw identical: {int(fl.raw_identical.sum())})")
    if len(fl):
        print(fl.drop(columns=["file_1", "file_2"]).to_string(index=False))
        print("\nBy study, split and pair:")
        print(fl.groupby(["study", "split", "pair"]).agg(images=("image_id", "size"), raw_identical=("raw_identical", "sum")).to_string())

    s3 = df[df.study == 3].copy()
    s3["observed"] = s3.apply(lambda r: ",".join(o for o in ORGANELLES if bool(r[f"{o}_obs"])), axis=1)
    s3f = fl[fl.study == 3].groupby("image_id").pair.apply(lambda p: ";".join(p)) if len(fl) else pd.Series(dtype=str)
    s3["flagged_pairs"] = s3.image_id.map(s3f).fillna("")
    s3 = s3[["image_id", "split", "modality", "observed", "flagged_pairs"]].sort_values(["split", "image_id"])
    s3.to_csv(out / "study3_images.csv", index=False)
    print("\nStudy 3 images (usable, with at least one observed target):")
    print(s3.to_string(index=False))

    if excl:
        pairs = pd.read_csv(cfg.pairs_csv)
        before = counts(pairs)
        for study, o in excl:
            m = pairs.study == study
            print(f"\nExcluding {o} of Study {study}: {int(pairs.loc[m, f'{o}_obs'].sum())} targets")
            pairs.loc[m, f"{o}_obs"] = False
        pairs.to_csv(out / "pairs_excluded.csv", index=False)
        after = counts(pairs)
        print("\nObserved targets per organelle (usable images), before:")
        print(before.to_string(index=False))
        print("after:")
        print(after.to_string(index=False))
        print(f"\nwrote {out / 'pairs_excluded.csv'} (pass it with --pairs-csv)")
        if a.w3_root:
            rescore(a.w3_root, excl, out)


if __name__ == "__main__":
    main()
