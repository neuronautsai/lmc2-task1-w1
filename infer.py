"""End-to-end inference: one transmitted-light plane in, four predicted fluorescence channels out.

Usage: python infer.py --checkpoint best.pt --input IMAGE.tif --out PRED.ome.tiff

The input is scaled by its own 1st and 99.5th percentile and clipped to [0, 1] (the declared rule), run through the
tiled U-Net and written as a float32 OME-TIFF with axes CYX in the order Nucleus, Mitochondria, Tubulin, Actin.
Predictions are in the normalized [0, 1] scale, not in original fluorescence intensity units.
"""
import argparse
import time

import numpy as np
import tifffile
import torch

from config import ChallengeConfig
from dataset import normalize
from predict import predict_full
from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device)
    cfg = ChallengeConfig()
    model = UNet(base=int(ck["cfg"]["base_channels"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    raw = tifffile.imread(a.input).astype(np.float32)
    assert raw.ndim == 2, f"expected one 2D plane, got shape {raw.shape}"
    x = torch.from_numpy(normalize(raw, cfg.p_low, cfg.p_high))[None, None].to(device)
    t0 = time.time()
    pred = predict_full(model, x, cfg.tile_size, cfg.tile_overlap)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    assert pred.shape == (4,) + raw.shape
    tifffile.imwrite(a.out, pred.astype(np.float32), ome=True, metadata={"axes": "CYX"})
    print(f"input {raw.shape} -> output {pred.shape} | {sec:.2f} s | epoch {ck['epoch']} | written {a.out}")


if __name__ == "__main__":
    main()
