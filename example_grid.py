"""Fixed example set of #323 for any list of runs (#333 Week 2: inspect the same examples for every run).

Cases, picked from a reference per-image score file and never from the runs being compared (the #323 rule): for each
modality the validation image whose reference macro SSIM is the median of that modality, and for each organelle the
image with the lowest reference SSIM for that organelle. The reference is the per_image_scores.csv of base seed 42
(train.py output, best.pt), as in #323. Each figure shows the input, the target and every run's prediction per observed
organelle on the same [0, 1] display range (normalized intensities), with the official-style SSIM and PCC of each panel.

Usage:
  python example_grid.py --ref W3/base_s42/per_image_scores.csv --out OUT \
      --run base=W3/base_s42/best.pt smp_r34=W5/smp_r34_s42_v2/best.pt [--window hann --overlap 128]
A run may be an ensemble: NAME=a.pt,b.pt,c.pt.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch

from config import MODALITIES, ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from official_metrics import official_metrics
from predict import WINDOWS, predict_tta
from unet import unet_from_checkpoint


def pick_cases(ref):
    macro = ref.groupby(["study", "image_id", "modality"]).ssim.mean().reset_index()
    cases = []
    for mod in MODALITIES:
        m = macro[macro.modality == mod].sort_values("ssim").reset_index(drop=True)
        if len(m):
            r = m.iloc[len(m) // 2]
            cases.append((f"median_{mod}", int(r.study), int(r.image_id)))
    for org in ORGANELLES:
        o = ref[ref.organelle == org].sort_values("ssim")
        if len(o):
            r = o.iloc[0]
            cases.append((f"lowest_{org}", int(r.study), int(r.image_id)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="per_image_scores.csv of the reference run (base seed 42 in #323)")
    ap.add_argument("--run", nargs="+", required=True, help="NAME=checkpoint[,checkpoint...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--window", choices=WINDOWS, default="hann")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    kw = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = ChallengeConfig(**kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {}
    for spec in a.run:
        name, paths = spec.split("=", 1)
        runs[name] = [unet_from_checkpoint(torch.load(p, map_location=device, weights_only=False), device,
                                           len(MODALITIES) + 1) for p in paths.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = LMC2Dataset(cfg, "val", train=False)
    cases = pick_cases(pd.read_csv(a.ref))
    json.dump({"cases": cases, "ref": a.ref, "runs": a.run, "window": a.window, "overlap": a.overlap, "tta": a.tta},
              open(out / "cases.json", "w"), indent=1)
    for tag, study, image_id in cases:
        hit = np.flatnonzero((ds.df.study == study) & (ds.df.image_id == image_id))
        if not len(hit):
            print(f"{tag}: study {study} image {image_id} not in the validation pairs, skipped")
            continue
        r = ds.df.iloc[int(hit[0])]
        item = ds[int(hit[0])]
        x = item["input"][None].to(device)
        preds = {}
        for name, models in runs.items():
            ps = [predict_tta(m, x, cfg.tile_size, a.overlap, mod=item["modality_id"] if getattr(m, "n_mod", 0) else None,
                              window=a.window, n_tta=8 if a.tta else 1) for m in models]
            preds[name] = np.mean(ps, axis=0)
        obs = [k for k in range(len(ORGANELLES)) if item["observed"][k]]
        fig, ax = plt.subplots(len(obs), 2 + len(runs), figsize=(3 * (2 + len(runs)), 3.1 * len(obs)), squeeze=False)
        for row, k in enumerate(obs):
            org = ORGANELLES[k]
            raw = tifffile.imread(cfg.data_root / f"Study_{study:02d}" / r[f"{org}_file"]).astype(np.float64)
            panels = [("input " + item["modality"], item["input"][0].numpy()), ("target " + org, item["target"][k].numpy())]
            for name in runs:
                s = official_metrics(raw, preds[name][k])
                panels.append((f"{name} SSIM {s['off_ssim_dr1']:.2f} PCC {s['off_pcc']:.2f}", preds[name][k]))
            for col, (title, img) in enumerate(panels):
                ax[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
                ax[row, col].set_title(title, fontsize=9)
                ax[row, col].axis("off")
        fig.suptitle(f"{tag}: study {study}, image {image_id}, display range [0, 1], official-style scores", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"case_{tag}_s{study:02d}_{image_id:05d}.png", dpi=110)
        plt.close(fig)
        print("wrote", tag, study, image_id, flush=True)


if __name__ == "__main__":
    main()
