"""Tiled full-image inference and local validation scoring.

Large images are processed in overlapping tiles with linear border blending, padded to a multiple of the network factor,
then restored to the original size before scoring. Output order is Nucleus, Mitochondria, Tubulin, Actin.
"""
import time

import numpy as np
import pandas as pd
import torch

from config import ORGANELLES
from evaluate import score_image


def _window(n, overlap):
    w = np.ones(n, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


@torch.no_grad()
def predict_full(model, x, tile=512, overlap=64, factor=16):
    """x: (1, 1, H, W) tensor on the model device. Returns (4, H, W) float32 numpy array in [0, 1]."""
    _, _, h, w = x.shape
    ph, pw = (-h) % factor, (-w) % factor
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    H, W = x.shape[-2:]
    out = torch.zeros((4, H, W), device=x.device)
    acc = torch.zeros((1, H, W), device=x.device)
    step = max(tile - overlap, factor)
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    for i in ys:
        for j in xs:
            patch = x[:, :, i:i + tile, j:j + tile]
            th, tw = patch.shape[-2:]
            wy = torch.from_numpy(_window(th, min(overlap, th // 2))).to(x.device)
            wx = torch.from_numpy(_window(tw, min(overlap, tw // 2))).to(x.device)
            weight = (wy[:, None] * wx[None, :])[None]
            pred = model(patch)[0].float()
            out[:, i:i + th, j:j + tw] += pred * weight
            acc[:, i:i + th, j:j + tw] += weight
    return (out / acc)[:, :h, :w].cpu().numpy()


def evaluate_model(model, dataset, cfg, device, indices=None):
    """Score full validation images. Returns (per image-channel DataFrame, seconds per image)."""
    model.eval()
    rows, t_inf, n = [], 0.0, 0
    for idx in (range(len(dataset)) if indices is None else indices):
        item = dataset[idx]
        x = item["input"][None].to(device)
        t0 = time.time()
        pred = predict_full(model, x, cfg.tile_size, cfg.tile_overlap)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.time() - t0
        n += 1
        for r in score_image(pred, item["target"].numpy(), item["observed"].numpy()):
            r.update(study=item["study"], image_id=item["image_id"], modality=item["modality"])
            rows.append(r)
    return pd.DataFrame(rows), t_inf / max(n, 1)
