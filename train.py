"""Train the four-output U-Net with observed-channel L1 loss, AdamW, AMP and checkpointing.

Checkpoint rule (fixed before comparing methods): maximum macro-average SSIM over the four organelles on a fixed
validation subset (cfg.val_subset images, drawn once with seed 42). The best checkpoint is then scored on the entire
local validation split. Local selection rule only, not a reconstruction of the official rank aggregation.

Usage: python train.py --out OUTPUT_DIR [--seed 42] [--epochs 50] [--overfit-n 8]
"""
import argparse
import dataclasses
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ChallengeConfig
from dataset import LMC2Dataset
from evaluate import breakdown, summarize
from predict import evaluate_model
from unet import UNet


def observed_channel_l1(prediction, target, observed):
    channel_errors = (prediction.float() - target.float()).abs().mean((-2, -1))
    valid = observed.bool()
    if not valid.any():
        raise ValueError("The batch has no observed fluorescence targets")
    return channel_errors[valid].mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro(df):
    t = summarize(df)
    return t.attrs["macro_ssim"], t.attrs["macro_pcc"], t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit-n", type=int, default=0, help="train and score on N training images only")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()

    over = {k: v for k, v in dict(
        seed=a.seed, epochs=a.epochs, patch_size=a.patch_size, base_channels=a.base_channels,
        num_workers=a.workers, val_subset=a.val_subset, learning_rate=a.lr,
        data_root=Path(a.data_root) if a.data_root else None,
        pairs_csv=Path(a.pairs_csv) if a.pairs_csv else None).items() if v is not None}
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    out = Path(a.out)
    cfg_dict = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_ds = LMC2Dataset(cfg, "train", train=True, seed=cfg.seed)
    if a.overfit_n:
        keep = np.linspace(0, len(train_ds.df) - 1, a.overfit_n).astype(int)
        train_ds.df = train_ds.df.iloc[keep].reset_index(drop=True)
        val_ds = LMC2Dataset(cfg, "train", train=False)
        val_ds.df = train_ds.df.copy()
        val_idx = list(range(len(val_ds)))
    else:
        val_ds = LMC2Dataset(cfg, "val", train=False)
        rng = np.random.default_rng(42)
        val_idx = sorted(rng.choice(len(val_ds), size=min(cfg.val_subset, len(val_ds)), replace=False).tolist())
    print(f"device {device} | train images {len(train_ds)} | val images {len(val_ds)} | val subset {len(val_idx)}")

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        drop_last=len(train_ds) >= cfg.batch_size,
                        generator=torch.Generator().manual_seed(cfg.seed))
    model = UNet(base=cfg.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    K = cfg.crops_per_image
    history, best, best_epoch = [], -1.0, -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            x = batch["input"].flatten(0, 1).to(device, non_blocking=True)
            y = batch["target"].flatten(0, 1).to(device, non_blocking=True)
            obs = batch["observed"].repeat_interleave(K, dim=0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(x)
            loss = observed_channel_l1(pred, y, obs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        df, _ = evaluate_model(model, val_ds, cfg, device, val_idx)
        m_ssim, m_pcc, _ = macro(df)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_ssim": m_ssim,
               "val_macro_pcc": m_pcc, "elapsed_s": time.time() - t_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print({k: round(v, 4) for k, v in row.items()})
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict}, out / "last.pt")
        if m_ssim > best:
            best, best_epoch = m_ssim, epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict}, out / "best.pt")
    train_time = time.time() - t_start

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    full_idx = val_idx if a.overfit_n else None
    df, sec_per_image = evaluate_model(model, val_ds, cfg, device, full_idx)
    df.to_csv(out / "per_image_scores.csv", index=False)
    m_ssim, m_pcc, table = macro(df)
    table.to_csv(out / "summary.csv", index=False)
    breakdown(df, "modality").to_csv(out / "by_modality.csv", index=False)
    breakdown(df, "study").to_csv(out / "by_study.csv", index=False)
    info = {"best_epoch": best_epoch, "train_seconds": train_time, "inference_seconds_per_image": sec_per_image,
            "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
            "macro_ssim": m_ssim, "macro_pcc": m_pcc, "config": cfg_dict,
            "overfit_n": a.overfit_n}
    json.dump(info, open(out / "run_info.json", "w"), indent=2, default=str)
    print(table.to_string(index=False))
    print({k: info[k] for k in ("best_epoch", "train_seconds", "inference_seconds_per_image", "peak_gpu_memory_gb",
                                "macro_ssim", "macro_pcc")})


if __name__ == "__main__":
    main()
