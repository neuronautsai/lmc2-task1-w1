"""Week 3 inspection: scores by modality and by study for every configuration, and the same examples for every model.

Usage: python w3_inspect.py --root OUTPUTS_W3 --out OUT_DIR [--seed 42]

1. Tables (from the per_image_scores.csv files written by train.py, best.pt): mean SSIM and PCC per configuration,
   modality (or study) and organelle, first averaged over images within each seed, then mean and std over seeds.
   Written as by_modality_w3.csv and by_study_w3.csv.
2. Figures: for one seed, the best.pt of base, cos and cosfilm (true modality label) predict the same validation
   images. The images are picked by a fixed rule from the base run of that seed, not by looking at the predictions of
   the other models: for each modality (BF, DIC, PC) the image whose base macro SSIM is the median of that modality,
   and for each organelle the image with the lowest base SSIM for that organelle (failure cases). Each figure shows the
   input, the target and the three predictions per observed organelle, all on the same [0, 1] display range
   (normalized intensities as in training and scoring).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import MODALITIES, ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from predict import predict_full
from unet import UNet

CFGS = ("base", "cos", "cosfilm")


def tables(root, out):
    frames = []
    for c in CFGS:
        for run in sorted(root.glob(f"{c}_s*")):
            f = run / "per_image_scores.csv"
            if f.exists() and run.name.split("_s")[0] == c:
                d = pd.read_csv(f)
                d["config"], d["seed"] = c, int(run.name.split("_s")[1])
                frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for by in ("modality", "study"):
        per_seed = df.groupby(["config", "seed", by, "organelle"]).agg(
            n=("ssim", "size"), ssim=("ssim", "mean"), pcc=("pcc", "mean")).reset_index()
        t = per_seed.groupby(["config", by, "organelle"]).agg(
            n_images=("n", "first"), seeds=("seed", "nunique"), ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
            pcc_mean=("pcc", "mean"), pcc_std=("pcc", "std")).reset_index()
        t.to_csv(out / f"by_{by}_w3.csv", index=False)
        piv = t.pivot_table(index=[by, "organelle"], columns="config", values=["ssim_mean", "pcc_mean"])
        print(f"\nBy {by} (mean over seeds of the per-seed image means)")
        print(piv.round(3).to_string())
    return df


def load(run, device):
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    n_mod = (len(MODALITIES) + 1) if ck["cfg"].get("modality_mode") == "film" else 0
    m = UNet(base=int(ck["cfg"]["base_channels"]), n_mod=n_mod).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, n_mod


def pick_cases(df, seed):
    b = df[(df.config == "base") & (df.seed == seed)]
    macro = b.groupby(["study", "image_id", "modality"]).ssim.mean().reset_index()
    cases = []
    for mod in MODALITIES:
        m = macro[macro.modality == mod].sort_values("ssim").reset_index(drop=True)
        if len(m):
            r = m.iloc[len(m) // 2]
            cases.append((f"median_{mod}", int(r.study), int(r.image_id)))
    for org in ORGANELLES:
        o = b[b.organelle == org].sort_values("ssim")
        if len(o):
            r = o.iloc[0]
            cases.append((f"lowest_{org}", int(r.study), int(r.image_id)))
    return cases


def figures(root, out, df, seed, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {c: load(root / f"{c}_s{seed}", device) for c in CFGS}
    ds = LMC2Dataset(cfg, "val", train=False)
    cases = pick_cases(df, seed)
    json.dump(cases, open(out / f"cases_seed{seed}.json", "w"), indent=1)
    for tag, study, image_id in cases:
        idx = int(np.flatnonzero((ds.df.study == study) & (ds.df.image_id == image_id))[0])
        item = ds[idx]
        x = item["input"][None].to(device)
        preds = {}
        for c, (m, n_mod) in models.items():
            preds[c] = predict_full(m, x, cfg.tile_size, cfg.tile_overlap, mod=item["modality_id"] if n_mod else None)
        obs = [k for k in range(len(ORGANELLES)) if item["observed"][k]]
        fig, ax = plt.subplots(len(obs), 5, figsize=(15, 3.1 * len(obs)), squeeze=False)
        sc = df[(df.seed == seed) & (df.study == study) & (df.image_id == image_id)]
        for row, k in enumerate(obs):
            org = ORGANELLES[k]
            panels = [("input " + item["modality"], item["input"][0].numpy()), ("target " + org, item["target"][k].numpy())]
            for c in CFGS:
                s = sc[(sc.config == c) & (sc.organelle == org)]
                lab = f"{c} SSIM {s.ssim.iloc[0]:.2f} PCC {s.pcc.iloc[0]:.2f}" if len(s) else c
                panels.append((lab, preds[c][k]))
            for col, (title, img) in enumerate(panels):
                ax[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
                ax[row, col].set_title(title, fontsize=9)
                ax[row, col].axis("off")
        fig.suptitle(f"{tag}: study {study}, image {image_id}, seed {seed} best.pt, display range [0, 1]", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"case_{tag}_s{study:02d}_{image_id:05d}.png", dpi=110)
        plt.close(fig)
        print("wrote", tag, study, image_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import dataclasses
    over = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    df = tables(root, out)
    figures(root, out, df, a.seed, cfg)


if __name__ == "__main__":
    main()
