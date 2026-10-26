"""Training curves and input / ground truth / prediction / error grids for a finished run.

Usage: python make_figures.py --run RUN_DIR
Writes curves.png and grid_<organelle>.png into RUN_DIR. Grid cases per organelle are the lowest, median and highest
SSIM images of the local validation set (taken from per_image_scores.csv).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from predict import predict_full
from unet import UNet


def curves(run):
    h = pd.read_csv(run / "history.csv")
    best = int(pd.read_json(run / "run_info.json", typ="series")["best_epoch"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a, col, title in zip(ax, ["train_loss", "val_macro_ssim", "val_macro_pcc"],
                             ["Training loss (observed-channel L1)", "Validation subset macro SSIM", "Validation subset macro PCC"]):
        a.plot(h.epoch, h[col])
        a.axvline(best, color="red", ls="--", lw=1, label=f"best epoch {best}")
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=130)
    plt.close(fig)


def grids(run, device):
    cfg = ChallengeConfig()
    ck = torch.load(run / "best.pt", map_location=device)
    model = UNet(base=int(ck["cfg"]["base_channels"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ds = LMC2Dataset(cfg, "val", train=False)
    pos = {iid: i for i, iid in enumerate(ds.df.image_id.tolist())}
    scores = pd.read_csv(run / "per_image_scores.csv")
    for k, org in enumerate(ORGANELLES):
        d = scores[scores.organelle == org].sort_values("ssim").reset_index(drop=True)
        picks = [("lowest SSIM", d.iloc[0]), ("median SSIM", d.iloc[(d.ssim - d.ssim.median()).abs().argmin()]),
                 ("highest SSIM", d.iloc[-1])]
        fig, ax = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
        for r, (label, row) in enumerate(picks):
            item = ds[pos[int(row.image_id)]]
            pred = predict_full(model, item["input"][None].to(device), cfg.tile_size, cfg.tile_overlap)[k]
            gt = item["target"][k].numpy()
            err = np.abs(pred - gt)
            panels = [(item["input"][0].numpy(), None), (gt, f"Ground truth {org}"),
                      (np.clip(pred, 0, 1), "Prediction"), (err, "Absolute error")]
            for c, (img, title) in enumerate(panels):
                ax[r, c].imshow(img, cmap="gray" if c < 3 else "magma", vmin=0, vmax=1)
                ax[r, c].axis("off")
                if title:
                    ax[r, c].set_title(title, fontsize=9)
            ax[r, 0].set_ylabel("")
            ax[r, 0].set_title(f"{label} | study {row.study} | {row.modality} | image {int(row.image_id)}\n"
                               f"SSIM {row.ssim:.3f} | PCC {row.pcc:.3f}\nInput (transmitted)", fontsize=8)
        fig.savefig(run / f"grid_{org}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curves(run)
    grids(run, device)
    print("written:", sorted(p.name for p in run.glob("*.png")))


if __name__ == "__main__":
    main()
