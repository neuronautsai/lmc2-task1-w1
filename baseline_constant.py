"""Baseline A: constant training-set mean normalized intensity per organelle (control).

Usage: python baseline_constant.py --out OUTPUT_DIR [--data-root DIR --pairs-csv CSV]
Mean = pooled over all pixels of all observed training targets, after per-image normalization.
PCC of a constant prediction is undefined and counted as such.
"""
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config import ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset, load_pairs
from evaluate import breakdown, score_image, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(ChallengeConfig.data_root))
    ap.add_argument("--pairs-csv", default=str(ChallengeConfig.pairs_csv))
    a = ap.parse_args()
    cfg = ChallengeConfig(data_root=Path(a.data_root), pairs_csv=Path(a.pairs_csv))
    out = a.out
    os.makedirs(out, exist_ok=True)

    train = LMC2Dataset(cfg, "train", train=False)
    sums, counts = np.zeros(4), np.zeros(4)

    def acc(i):
        r = train.df.iloc[i]
        s, c = np.zeros(4), np.zeros(4)
        for k, o in enumerate(ORGANELLES):
            if bool(r[f"{o}_obs"]):
                t = train._read(int(r.study), r[f"{o}_file"])
                s[k], c[k] = t.sum(dtype=np.float64), t.size
        return s, c

    with ThreadPoolExecutor(16) as ex:
        for s, c in ex.map(acc, range(len(train))):
            sums += s
            counts += c
    means = sums / counts
    json.dump(dict(zip(ORGANELLES, means.tolist())), open(f"{out}/constant_means.json", "w"))
    print("training means:", dict(zip(ORGANELLES, np.round(means, 4))))

    val = LMC2Dataset(cfg, "val", train=False)

    def ev(i):
        item = val[i]
        target = item["target"].numpy()
        pred = np.broadcast_to(means[:, None, None], target.shape)
        rows = score_image(pred, target, item["observed"].numpy())
        for r in rows:
            r.update(study=item["study"], image_id=item["image_id"], modality=item["modality"])
        return rows

    with ThreadPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(ev, range(len(val))) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/constant_per_image.csv", index=False)
    t = summarize(df)
    t.to_csv(f"{out}/constant_summary.csv", index=False)
    breakdown(df, "modality").to_csv(f"{out}/constant_by_modality.csv", index=False)
    breakdown(df, "study").to_csv(f"{out}/constant_by_study.csv", index=False)
    print(t.to_string(index=False))
    print("macro SSIM:", round(t.attrs["macro_ssim"], 4), "macro PCC:", round(t.attrs["macro_pcc"], 4))


if __name__ == "__main__":
    main()
