"""Run inference.py on real validation images and compare with the training-time evaluation.

Usage (Colab, from the repo folder):
  python check_real_inputs.py --checkpoint CKPT --work /content/realcheck [--n 20]

Copies the first N validation transmitted-light files (raw OME-TIFFs) into the container input layout, runs
inference.py on them with CKPT, reads the four uint8 outputs back and scores them with the local proxy against the
targets, next to the same model's float predictions (predict_full, as in train.py). Checks: every output exists with the
input's shape, channel order is right (each output folder is compared with its own target), and the uint8 rounding
changes the scores by a negligible amount.
"""
import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from config import ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset, normalize
from evaluate import score_image
from predict import predict_full
from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    a = ap.parse_args()
    over = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    in_dir = work / "input" / "images" / "organelles-transmitted-light-ome-tiff"
    in_dir.mkdir(parents=True)
    (work / "model").mkdir()
    shutil.copy(a.checkpoint, work / "model" / "weights.pt")
    ds = LMC2Dataset(cfg, "val", train=False)
    rows = ds.df.iloc[: a.n]
    names = {}
    for _, r in rows.iterrows():
        src = cfg.data_root / f"Study_{int(r.study):02d}" / r.in_file
        name = f"s{int(r.study):02d}_{int(r.image_id):05d}.ome.tiff"
        shutil.copy(src, in_dir / name)
        names[name] = r
    with tifffile.TiffFile(next(in_dir.iterdir())) as t:
        print("first input:", t.series[0].shape, t.series[0].dtype, "| OME-XML present:", t.ome_metadata is not None)
    env = {**os.environ, "LMC2_INPUT": str(work / "input"), "LMC2_OUTPUT": str(work / "output"),
           "LMC2_MODEL_DIR": str(work / "model")}
    subprocess.run([sys.executable, "inference.py"], check=True, env=env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = UNet(base=int(ck["cfg"]["base_channels"]), n_mod=0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    out_rows, problems = [], []
    for name, r in names.items():
        idx = int(np.flatnonzero((ds.df.study == r.study) & (ds.df.image_id == r.image_id))[0])
        item = ds[idx]
        target, observed = item["target"].numpy(), item["observed"].numpy()
        raw = np.squeeze(tifffile.imread(in_dir / name)).astype(np.float32)
        x = torch.from_numpy(normalize(raw, cfg.p_low, cfg.p_high))[None, None].to(device)
        pred_float = predict_full(model, x, cfg.tile_size, cfg.tile_overlap)
        pred_file = np.zeros_like(pred_float)
        for k, org in enumerate(ORGANELLES):
            f = work / "output" / "images" / f"{org.lower()}-fluorescence-ome-tiff" / name
            if not f.exists():
                problems.append(f"missing {f}")
                continue
            arr = np.squeeze(tifffile.imread(f))
            if arr.shape != raw.shape or arr.dtype != np.uint8:
                problems.append(f"{f.name} {org}: {arr.shape} {arr.dtype}")
            pred_file[k] = arr.astype(np.float32) / 255.0
        for tag, p in (("float", pred_float), ("uint8_file", pred_file)):
            for s in score_image(p, target, observed):
                s.update(image=name, version=tag)
                out_rows.append(s)
    df = pd.DataFrame(out_rows)
    t = df.pivot_table(index="organelle", columns="version", values=["ssim", "pcc"], aggfunc="mean").reindex(ORGANELLES)
    pd.set_option("display.width", 200)
    print(t.round(5).to_string())
    m = df.pivot_table(index=["image", "organelle"], columns="version", values=["ssim", "pcc"])
    for met in ("ssim", "pcc"):
        d = (m[(met, "uint8_file")] - m[(met, "float")]).abs()
        print(f"max |uint8 file - float| {met}: {d.max():.5f}")
    print("PROBLEMS:\n" + "\n".join(problems) if problems else "no file problems")


if __name__ == "__main__":
    main()
