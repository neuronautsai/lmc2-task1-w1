"""Leakage check of the local train and validation split.

1. Split integrity: no study and no image id in both splits.
2. Duplicates across splits, on the prepared arrays (per-image normalized, so copies that differ only by a linear
   intensity change are caught too):
   a. exact: md5 of every channel (input and observed targets),
   b. near: input downsampled to 64 x 64 and z-scored, highest correlation of each validation input with any training
      input over the 8 flips and 90-degree rotations.
3. Selection subset: macro official-style SSIM and PCC of every per_image_official_scores.csv under --outputs, with and
   without the 100 validation images that train.py uses to pick best.pt (numpy default_rng(42)).
--view VAL_STUDY:VAL_ID,TRAIN_STUDY:TRAIN_ID (repeatable) reads the raw inputs of a candidate pair, prints the highest
correlation at 256 x 256 over the 8 orientations and saves the pairs side by side to --fig.

Usage:
  python leakage_check.py --pairs-csv PAIRS_EXCLUDED --prepared-dir /content/prepared --outputs OUT
  python leakage_check.py --pairs-csv PAIRS_EXCLUDED --data-root DATA --view 39:2074,35:2023 --fig pairs.png
"""
import argparse
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.transform import resize

from config import ChallengeConfig, ORGANELLES
from dataset import load_pairs


def orientations(a):
    for k in range(4):
        for flip in (False, True):
            v = np.rot90(a, k)
            yield v[:, ::-1] if flip else v


def split_integrity(tr, va):
    print("1) train", len(tr), "val", len(va))
    print("   study in both:", set(tr.study) & set(va.study))
    print("   image_id in both:", set(tr.image_id) & set(va.image_id))


def duplicates(tr, va, pdir, thr):
    hashes, thumbs = [], {}
    for split, df in (("train", tr), ("val", va)):
        for _, r in df.iterrows():
            a = np.load(pdir / f"{int(r.study):02d}_{int(r.image_id):05d}.npy", mmap_mode="r")
            obs = [0] + [i + 1 for i, o in enumerate(ORGANELLES) if r[f"{o}_obs"]]
            for c in obs:
                md5 = hashlib.md5(np.ascontiguousarray(a[c]).tobytes()).hexdigest()
                hashes.append((split, int(r.study), int(r.image_id), c, md5))
            t = resize(np.asarray(a[0], dtype=np.float32), (64, 64), anti_aliasing=True).ravel()
            thumbs[(split, int(r.study), int(r.image_id))] = (t - t.mean()) / (t.std() + 1e-8)
    h = pd.DataFrame(hashes, columns=["split", "study", "image_id", "ch", "md5"])
    cross = h.groupby("md5").filter(lambda g: g.split.nunique() > 1)
    print("2a) exact duplicates across splits (rows):", len(cross))
    if len(cross):
        print(cross.sort_values("md5").to_string(index=False))
    kt = [k for k in thumbs if k[0] == "train"]
    kv = [k for k in thumbs if k[0] == "val"]
    T = np.stack([thumbs[k] for k in kt]) / 64
    best = np.full(len(kv), -1.0)
    arg = np.zeros(len(kv), int)
    stacks = [np.stack(list(orientations(thumbs[x].reshape(64, 64)))) for x in kv]  # (8, 64, 64) per val image
    for o in range(8):
        V = np.stack([s[o] for s in stacks]).reshape(len(kv), -1) / 64
        S = V @ T.T
        m = S.max(1)
        upd = m > best
        best[upd] = m[upd]
        arg[upd] = S.argmax(1)[upd]
    near = pd.DataFrame({"val": kv, "closest_train": [kt[i] for i in arg], "corr": best})
    near = near.sort_values("corr", ascending=False)
    print(f"2b) val inputs with a train input correlation > {thr} (64x64 thumbnails):", int((near["corr"] > thr).sum()))
    print(near.head(10).to_string(index=False))


def selection_subset(va, outputs, n_subset):
    idx = sorted(np.random.default_rng(42).choice(len(va), size=min(n_subset, len(va)), replace=False).tolist())
    sub = set(zip(va.study.iloc[idx], va.image_id.iloc[idx]))
    print("3) subset images", len(sub))

    def macro(x):
        return x.groupby("organelle")[["off_ssim_dr1", "off_pcc"]].mean().mean()

    for f in sorted(glob.glob(f"{outputs}/**/per_image_official_scores.csv", recursive=True)):
        d = pd.read_csv(f)
        keep = [(s, i) not in sub for s, i in zip(d.study, d.image_id)]
        a, b = macro(d), macro(d[keep])
        print(f"   {Path(f).relative_to(outputs)}  all SSIM {a.off_ssim_dr1:.4f} PCC {a.off_pcc:.4f} | "
              f"without subset SSIM {b.off_ssim_dr1:.4f} PCC {b.off_pcc:.4f}")


def view_pairs(pairs_df, data_root, views, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    def load(study, image_id):
        r = pairs_df[(pairs_df.study == study) & (pairs_df.image_id == image_id)].iloc[0]
        x = tifffile.imread(Path(data_root) / f"Study_{study:02d}" / r.in_file).astype(np.float32)
        lo, hi = np.percentile(x, (1, 99.5))
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1), r.modality

    fig, ax = plt.subplots(len(views), 2, figsize=(8, 4 * len(views)), squeeze=False)
    for i, ((sv, iv), (st, it)) in enumerate(views):
        a, ma = load(sv, iv)
        b, mb = load(st, it)
        rb = resize(b, (256, 256))
        c = max(np.corrcoef(v.ravel(), rb.ravel())[0, 1] for v in orientations(resize(a, (256, 256))))
        print(f"val {sv}/{iv} {ma} {a.shape} | train {st}/{it} {mb} {b.shape} | best corr 256x256 {c:.3f}")
        ax[i, 0].imshow(a, cmap="gray")
        ax[i, 0].set_title(f"val {sv}/{iv}")
        ax[i, 1].imshow(b, cmap="gray")
        ax[i, 1].set_title(f"train {st}/{it}")
        for x in ax[i]:
            x.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)
    print("figure:", fig_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--prepared-dir", default=None, help="prepared folder (needed for step 2)")
    ap.add_argument("--outputs", default=None, help="folder searched for per_image_official_scores.csv (step 3)")
    ap.add_argument("--data-root", default=None, help="raw data folder (needed for --view)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--val-subset", type=int, default=100)
    ap.add_argument("--view", action="append", default=[], help="VAL_STUDY:VAL_ID,TRAIN_STUDY:TRAIN_ID")
    ap.add_argument("--fig", default="leakage_pairs.png")
    a = ap.parse_args()

    cfg = ChallengeConfig(pairs_csv=Path(a.pairs_csv))
    tr, va = load_pairs(cfg, "train"), load_pairs(cfg, "val")
    split_integrity(tr, va)
    if a.prepared_dir:
        duplicates(tr, va, Path(a.prepared_dir), a.threshold)
    if a.outputs:
        selection_subset(va, a.outputs, a.val_subset)
    if a.view:
        views = [tuple(tuple(int(x) for x in p.split(":")) for p in v.split(",")) for v in a.view]
        view_pairs(pd.read_csv(a.pairs_csv), a.data_root, views, a.fig)


if __name__ == "__main__":
    main()
