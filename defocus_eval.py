"""Sensitivity of the model to the focal plane of the transmitted-light input (#333 item 1).

Training uses the best-focus transmitted-light slice as input. This script feeds the same validation images with the
slice at best focus + d for every offset d, keeps the best-focus fluorescence targets, and scores the prediction with
the official-style metrics. If the challenge inputs can be off focus, a steep drop away from d = 0 means the model is
trained on the wrong input distribution.

The slice at offset d is the transmitted-light file of the same study, image and channel with z = z_in + d, where z_in
is the z of the training input file (pairs csv column in_file), both read from the data manifest (inspect_data.py
manifest, columns study, image_id, channel, z, filename, kind). Images without a slice at that offset are skipped for
that offset, so the counts per offset are reported.

Usage:
  python defocus_eval.py --checkpoint W3/base_s42/best.pt [more checkpoints = ensemble] --manifest MANIFEST
      --pairs-csv PAIRS_EXCLUDED --out OUT --offsets -8 -4 -2 -1 0 1 2 4 8 --n-images 120
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from config import MODALITIES, ORGANELLES, ChallengeConfig
from dataset import load_pairs, normalize
from official_metrics import official_metrics
from predict import predict_tta
from unet import unet_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", type=int, nargs="+", default=[-8, -4, -2, -1, 0, 1, 2, 4, 8])
    ap.add_argument("--n-images", type=int, default=120, help="validation images, drawn with seed 0 (0 = all)")
    ap.add_argument("--window", default="hann")
    ap.add_argument("--overlap", type=int, default=128)
    a = ap.parse_args()
    kw = {"pairs_csv": Path(a.pairs_csv)}
    if a.data_root:
        kw["data_root"] = Path(a.data_root)
    cfg = ChallengeConfig(**kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [unet_from_checkpoint(torch.load(c, map_location=device), device, len(MODALITIES) + 1)
              for c in a.checkpoint]
    for m in models:
        if m.n_mod:
            raise ValueError("defocus_eval.py serves unconditioned models only")

    man = pd.read_csv(a.manifest)
    tl = man[man.kind == "transmitted"][["study", "image_id", "channel", "z", "filename"]]
    by_file = tl.set_index("filename")
    va = load_pairs(cfg, "val")
    if a.n_images and a.n_images < len(va):
        va = va.iloc[sorted(np.random.default_rng(0).choice(len(va), a.n_images, replace=False))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, (_, r) in enumerate(va.iterrows()):
        base = cfg.data_root / f"Study_{int(r.study):02d}"
        if r.in_file not in by_file.index:
            print(f"input {r.in_file} not in the manifest, skipped", flush=True)
            continue
        ref = by_file.loc[r.in_file]
        stack = tl[(tl.study == r.study) & (tl.image_id == r.image_id) & (tl.channel == ref.channel)].set_index("z")
        targets = {k: tifffile.imread(base / r[f"{o}_file"]).astype(np.float64)
                   for k, o in enumerate(ORGANELLES) if bool(r[f"{o}_obs"])}
        for d in a.offsets:
            z = int(ref.z) + d
            if z not in stack.index:
                continue
            raw = np.squeeze(tifffile.imread(base / stack.loc[z, "filename"]))
            x = torch.from_numpy(normalize(raw, cfg.p_low, cfg.p_high))[None, None].to(device)
            pred = np.mean([predict_tta(m, x, cfg.tile_size, a.overlap, window=a.window) for m in models], axis=0)
            for k, gt in targets.items():
                row = {"study": int(r.study), "image_id": int(r.image_id), "modality": r.modality,
                       "organelle": ORGANELLES[k], "offset": d, "z": z, "z_best": int(ref.z), "nz": len(stack)}
                row.update(official_metrics(gt, pred[k]))
                rows.append(row)
        if (n + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(out / "partial_rows.csv", index=False)
            print(f"{n + 1}/{len(va)} images", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_defocus.csv", index=False)
    cols = ["off_ssim_dr1", "off_pcc", "off_mae", "off_cos"]
    # macro over organelles per offset, on the images that have that offset
    per_org = df.groupby(["offset", "organelle"])[cols].mean()
    macro = per_org.groupby("offset").mean()
    macro["n_images"] = df.groupby("offset").apply(lambda g: g[["study", "image_id"]].drop_duplicates().shape[0])
    # same, restricted to images that have every offset, so the curve compares the same images
    full = df.groupby(["study", "image_id"]).offset.nunique()
    keep = set(full[full == len(a.offsets)].index)
    sub = df[[(s, i) in keep for s, i in zip(df.study, df.image_id)]]
    macro_same = sub.groupby(["offset", "organelle"])[cols].mean().groupby("offset").mean()
    macro_same["n_images"] = len(keep)
    pd.set_option("display.width", 200)
    print("macro over organelles, all images with that offset")
    print(macro.round(4).to_string())
    print("macro over organelles, only images that have every offset")
    print(macro_same.round(4).to_string())
    print("per modality, only images that have every offset, SSIM and PCC")
    print(sub.groupby(["modality", "offset"])[["off_ssim_dr1", "off_pcc"]].mean().unstack(0).round(4).to_string())
    macro.to_csv(out / "macro_by_offset.csv")
    macro_same.to_csv(out / "macro_by_offset_same_images.csv")
    json.dump({"checkpoints": a.checkpoint, "offsets": a.offsets, "n_images_drawn": len(va), "window": a.window,
               "overlap": a.overlap, "manifest": a.manifest}, open(out / "defocus_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
