"""Compare tile blending settings of predict_full on the local validation split (review item 3).

Usage (Colab, from the repo folder):
  python compare_blending.py --checkpoint RUN/best.pt --pairs-csv PAIRS_EXCLUDED --out OUT [--prepared-dir DIR]
                             [--configs linear:64 linear:128 hann:128 gaussian:128] [--figure 4:75] [--resume]

Every validation image is read once and predicted with each setting WINDOW:OVERLAP (512 px tiles). Scores are the
local proxy (evaluate.score_image) per image and observed organelle, summarized per setting (macro SSIM and PCC).

Seam index: tile seams show up as steps between neighbouring pixels on the rows and columns where a tile starts or
ends. For each prediction and organelle, the mean absolute difference between neighbouring pixels across those tile
boundary lines is divided by the same mean over all lines of the image. 1 means the boundaries look like any other
line, values above 1 mean visible steps. It is reported as the mean over images and organelles, and separately for
the near-empty images (the 10 % of rows with the lowest target mean), where the Week 3 bands were seen.

--figure STUDY:IMAGE_ID saves the input, target and the prediction of every setting for that image (display range
[0, 1], the same for all panels), for example Study 4 image 75 from the Week 3 inspection.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import score_image, summarize
from predict import predict_full
from unet import unet_from_checkpoint

FACTOR = 16


def tile_lines(n, tile, overlap):
    """Positions (1 .. n-1) of the lines where a tile starts or ends, for the tiling used by predict_full."""
    N = n + ((-n) % FACTOR)
    step = max(tile - overlap, FACTOR)
    starts = list(range(0, max(N - tile, 0) + 1, step))
    if starts[-1] + tile < N:
        starts.append(N - tile)
    lines = set()
    for s in starts:
        lines.update((s, s + tile))
    return sorted(p for p in lines if 0 < p < n)


def seam_index(img, tile, overlap):
    """Mean |step| across tile boundary lines divided by the mean |step| across all lines (rows and columns)."""
    h, w = img.shape
    dy = np.abs(np.diff(img, axis=0)).mean(axis=1)  # dy[r - 1] is the step between rows r - 1 and r
    dx = np.abs(np.diff(img, axis=1)).mean(axis=0)
    rows, cols = tile_lines(h, tile, overlap), tile_lines(w, tile, overlap)
    if not rows and not cols:
        return np.nan
    edge = np.concatenate([dy[np.array(rows, dtype=int) - 1], dx[np.array(cols, dtype=int) - 1]])
    base = np.concatenate([dy, dx]).mean()
    return float(edge.mean() / base) if base > 0 else np.nan


def parse_configs(items):
    out = []
    for it in items:
        window, overlap = it.split(":")
        out.append((f"{window}_{int(overlap)}", window, int(overlap)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--configs", nargs="+", default=["linear:64", "linear:128", "hann:128", "gaussian:128"])
    ap.add_argument("--figure", nargs="*", default=["4:75"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    over = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv), ("prepared_dir", a.prepared_dir)) if v}
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    configs = parse_configs(a.configs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    assert ck["cfg"].get("modality_mode", "none") in (None, "none"), "use an unconditioned checkpoint"
    model = unet_from_checkpoint(ck, device)
    ds = LMC2Dataset(cfg, "val", train=False)
    figure_ids = {tuple(int(v) for v in f.split(":")) for f in a.figure}

    partial = out / "partial_rows.csv"
    rows, done = [], set()
    if a.resume and partial.exists():
        prev = pd.read_csv(partial)
        rows, done = prev.to_dict("records"), set(zip(prev.study, prev.image_id))
        print(f"resuming with {len(done)} images already scored", flush=True)
    secs = {name: 0.0 for name, _, _ in configs}
    n_done = 0
    for idx in range(len(ds)):
        r = ds.df.iloc[idx]
        key = (int(r.study), int(r.image_id))
        if key in done:
            continue
        item = ds[idx]
        x = item["input"][None].to(device)
        target, observed = item["target"].numpy(), item["observed"].numpy()
        preds = {}
        for name, window, overlap in configs:
            t0 = time.time()
            preds[name] = predict_full(model, x, cfg.tile_size, overlap, mod=None, window=window)
            if device.type == "cuda":
                torch.cuda.synchronize()
            secs[name] += time.time() - t0
            for s in score_image(preds[name], target, observed):
                k = ORGANELLES.index(s["organelle"])
                s.update(study=key[0], image_id=key[1], modality=item["modality"], setting=name,
                         target_mean=float(target[k].mean()),
                         seam=seam_index(preds[name][k], cfg.tile_size, overlap))
                rows.append(s)
        if key in figure_ids:
            obs = [k for k in range(len(ORGANELLES)) if observed[k]]
            fig, ax = plt.subplots(len(obs), 2 + len(configs), figsize=(3 * (2 + len(configs)), 3.1 * len(obs)), squeeze=False)
            for i, k in enumerate(obs):
                panels = [("input " + item["modality"], item["input"][0].numpy()), ("target " + ORGANELLES[k], target[k])]
                panels += [(f"{name} seam {seam_index(preds[name][k], cfg.tile_size, ov):.2f}", preds[name][k]) for name, _, ov in configs]
                for j, (title, img) in enumerate(panels):
                    ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
                    ax[i, j].set_title(title, fontsize=9)
                    ax[i, j].axis("off")
            fig.suptitle(f"Study {key[0]}, image {key[1]}: tile blending settings, display range [0, 1]", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / f"blending_s{key[0]:02d}_{key[1]:05d}.png", dpi=110)
            plt.close(fig)
        n_done += 1
        if n_done % 50 == 0:
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"{n_done} images", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_blending.csv", index=False)
    lo = df.target_mean <= df.target_mean.quantile(0.10)
    table = []
    for name, window, overlap in configs:
        d = df[df.setting == name]
        sm = summarize(d)
        row = {"setting": name, "macro_ssim": sm.attrs["macro_ssim"], "macro_pcc": sm.attrs["macro_pcc"],
               "seam_mean": d.seam.mean(), "seam_near_empty": d[lo[d.index]].seam.mean(),
               "s_per_image": secs[name] / max(n_done, 1)}
        for o in ORGANELLES:
            q = sm[sm.organelle == o].iloc[0]
            row[f"{o}_ssim"], row[f"{o}_pcc"] = q.ssim_mean, q.pcc_mean
        s4 = d[(d.study == 4) & (d.organelle == "Tubulin")]
        row["study4_tubulin_ssim"], row["study4_tubulin_n"] = s4.ssim.mean(), len(s4)
        table.append(row)
    t = pd.DataFrame(table)
    t.to_csv(out / "blending_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))
    print(f"\nnear-empty rows: target mean <= {df.target_mean.quantile(0.10):.4f} (10 % of image and organelle rows)")
    print(f"s_per_image counts only images predicted in this run ({n_done}); resumed images are not timed")
    json.dump({"checkpoint": a.checkpoint, "epoch": ck.get("epoch"), "pairs_csv": str(cfg.pairs_csv), "tile": cfg.tile_size,
               "configs": a.configs, "n_images": int(df[["study", "image_id"]].drop_duplicates().shape[0])},
              open(out / "blending_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
