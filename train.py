"""Train the four-output U-Net with observed-channel L1 loss, AdamW, AMP and checkpointing.

Checkpoint rule (fixed before comparing methods): maximum macro-average SSIM over the four organelles on a fixed
validation subset (cfg.val_subset images, drawn once with seed 42). The best checkpoint is then scored on the entire
local validation split. Local selection rule only, not a reconstruction of the official rank aggregation.

Usage: python train.py --out OUTPUT_DIR [--seed 42] [--epochs 50] [--overfit-n 8]
       python train.py --out OUTPUT_DIR --config-from RUN/best.pt [other options override the copied config]

--config-from copies every setting of an earlier run from its checkpoint (paths included), so a controlled run changes
only what is passed on the command line. run_info.json records the package and CUDA versions of the run.
"""
import argparse
import dataclasses
import math
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import UNKNOWN_MODALITY, MODALITIES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import breakdown, summarize
from predict import evaluate_model
from unet import UNet

PATH_FIELDS = ("data_root", "pairs_csv", "output_dir", "prepared_dir", "train_prepared_dir")


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


def versions():
    import scipy
    import skimage
    v = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
         "scikit-image": skimage.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return v


def config_from(path):
    """ChallengeConfig with the settings stored in an earlier checkpoint. Unknown keys are refused, new fields keep their default."""
    stored = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
    names = {f.name for f in dataclasses.fields(ChallengeConfig)}
    extra = set(stored) - names
    if extra:
        raise SystemExit(f"{path} has config keys this code does not know: {sorted(extra)}")
    kw = {k: (Path(v) if k in PATH_FIELDS and v not in (None, "None") else (None if v == "None" else v))
          for k, v in stored.items()}
    return dataclasses.replace(ChallengeConfig(), **kw)


def macro(df):
    t = summarize(df)
    return t.attrs["macro_ssim"], t.attrs["macro_pcc"], t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="default 42, or the seed of --config-from")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit-n", type=int, default=0, help="train and score on N training images only")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", choices=["const", "cosine"], default=None)
    ap.add_argument("--modality", choices=["none", "film"], default=None, help="modality-aware conditioning")
    ap.add_argument("--modality-dropout", type=float, default=None)
    ap.add_argument("--prepared-dir", default=None, help="folder written by prepare_dataset.py (default: raw TIFFs, \"none\" forces raw TIFFs)")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/last.pt if it exists")
    ap.add_argument("--config-from", default=None, help="checkpoint whose config is copied (then overridden by the options)")
    ap.add_argument("--target-p-low", type=float, default=None, help="training targets only")
    ap.add_argument("--target-p-high", type=float, default=None, help="training targets only")
    ap.add_argument("--no-target-clip", action="store_true", help="training targets are not clipped to [0, 1]")
    ap.add_argument("--train-prepared-dir", default=None, help="prepared folder built with the training target rule")
    ap.add_argument("--output-activation", choices=["sigmoid", "linear"], default=None)
    a = ap.parse_args()

    over = {k: v for k, v in dict(
        seed=a.seed, epochs=a.epochs, patch_size=a.patch_size, base_channels=a.base_channels,
        num_workers=a.workers, val_subset=a.val_subset, learning_rate=a.lr,
        lr_schedule=a.lr_schedule, modality_mode=a.modality, modality_dropout=a.modality_dropout,
        data_root=Path(a.data_root) if a.data_root else None,
        prepared_dir=Path(a.prepared_dir) if a.prepared_dir and a.prepared_dir.lower() != "none" else None,
        pairs_csv=Path(a.pairs_csv) if a.pairs_csv else None,
        train_target_p_low=a.target_p_low, train_target_p_high=a.target_p_high,
        train_target_clip=False if a.no_target_clip else None,
        train_prepared_dir=Path(a.train_prepared_dir) if a.train_prepared_dir else None,
        output_activation=a.output_activation).items() if v is not None}
    base_cfg = config_from(a.config_from) if a.config_from else ChallengeConfig()
    if a.prepared_dir and a.prepared_dir.lower() == "none":
        over["prepared_dir"] = None  # read the raw TIFFs even if the copied config names a prepared folder
    cfg = dataclasses.replace(base_cfg, **over)
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
    n_mod = (len(MODALITIES) + 1) if cfg.modality_mode == "film" else 0
    model = UNet(base=cfg.base_channels, n_mod=n_mod, out_act=cfg.output_activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    K = cfg.crops_per_image
    history, best, best_epoch = [], -1.0, -1
    start_epoch, elapsed_before = 0, 0.0
    last_path = out / "last.pt"
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, history = ck["epoch"], ck["history"]
        best, best_epoch, elapsed_before = ck["best"], ck["best_epoch"], ck["elapsed"]
        print(f"resumed from epoch {start_epoch}, best epoch {best_epoch} ({best:.4f})")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time() - elapsed_before

    mod_rng = np.random.default_rng(cfg.seed + 7)
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.lr_schedule == "cosine":
            factor = cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * epoch / cfg.epochs))
            for g in opt.param_groups:
                g["lr"] = cfg.learning_rate * factor
        train_ds.set_epoch(epoch)
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            x = batch["input"].flatten(0, 1).to(device, non_blocking=True)
            y = batch["target"].flatten(0, 1).to(device, non_blocking=True)
            obs = batch["observed"].repeat_interleave(K, dim=0).to(device)
            mod = None
            if n_mod:
                ids = batch["modality_id"].numpy().copy()
                ids[mod_rng.random(len(ids)) < cfg.modality_dropout] = UNKNOWN_MODALITY
                mod = torch.from_numpy(ids).repeat_interleave(K).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(x) if mod is None else model(x, mod)
            loss = observed_channel_l1(pred, y, obs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        df, _ = evaluate_model(model, val_ds, cfg, device, val_idx)
        m_ssim, m_pcc, _ = macro(df)
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_loss": float(np.mean(losses)), "val_macro_ssim": m_ssim,
               "val_macro_pcc": m_pcc, "elapsed_s": time.time() - t_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print({k: round(v, 4) for k, v in row.items()})
        if m_ssim > best:
            best, best_epoch = m_ssim, epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict,
                    "history": json.loads(json.dumps(history, default=float)), "best": float(best), "best_epoch": best_epoch, "elapsed": row["elapsed_s"]},
                   last_path)
    train_time = time.time() - t_start

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    full_idx = val_idx if a.overfit_n else None
    df, sec_per_image = evaluate_model(model, val_ds, cfg, device, full_idx)
    df.to_csv(out / "per_image_scores.csv", index=False)
    m_ssim, m_pcc, table = macro(df)
    unk = {}
    if n_mod:  # same checkpoint with the modality label withheld, to measure how much the label is used
        df_u, _ = evaluate_model(model, val_ds, cfg, device, full_idx, label="unknown")
        df_u.to_csv(out / "per_image_scores_unknown_modality.csv", index=False)
        u_ssim, u_pcc, _ = macro(df_u)
        unk = {"macro_ssim_unknown_modality": u_ssim, "macro_pcc_unknown_modality": u_pcc}
    table.to_csv(out / "summary.csv", index=False)
    breakdown(df, "modality").to_csv(out / "by_modality.csv", index=False)
    breakdown(df, "study").to_csv(out / "by_study.csv", index=False)
    info = {"best_epoch": best_epoch, "train_seconds": train_time, "inference_seconds_per_image": sec_per_image,
            "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
            "macro_ssim": m_ssim, "macro_pcc": m_pcc, "config": cfg_dict,
            "overfit_n": a.overfit_n, "config_from": a.config_from, "versions": versions(), **unk}
    json.dump(info, open(out / "run_info.json", "w"), indent=2, default=str)
    print(table.to_string(index=False))
    print({k: info[k] for k in ("best_epoch", "train_seconds", "inference_seconds_per_image", "peak_gpu_memory_gb",
                                "macro_ssim", "macro_pcc")})


if __name__ == "__main__":
    main()
