"""Train the four-output U-Net with observed-channel L1 loss (optionally mixed with an SSIM term), AdamW, AMP and checkpointing.

Checkpoint rule (fixed before comparing methods): maximum macro-average SSIM over the four organelles on a fixed
validation subset (cfg.val_subset images, drawn once with seed 42). The best checkpoint is then scored on the entire
local validation split. Local selection rule only, not a reconstruction of the official rank aggregation.

Usage: python train.py --out OUTPUT_DIR [--seed 42] [--epochs 50] [--overfit-n 8]
       python train.py --out OUTPUT_DIR --config-from RUN/best.pt [other options override the copied config]

--config-from copies every setting of an earlier run from its checkpoint (paths included), so a controlled run changes
only what is passed on the command line. run_info.json records the package and CUDA versions of the run.

#333 options (defaults keep the earlier behaviour):
  --arch smp --encoder resnet34      U-Net decoder on an ImageNet-pretrained encoder, encoder lr x --encoder-lr-mult
  --select-on heldout                hold out --heldout-frac of the training groups (GroupShuffleSplit, seed 42), train on
                                     the rest and keep best.pt by the official-style mean of macro SSIM and PCC on them.
                                     The local validation split is then used for the final report only.
  --ema-decay 0.999                  exponential moving average of the weights, used for selection and saved checkpoints
  --pcc-weight 0.1 --ssim-norm       loss terms aligned with the official metric, optionally only for
  --loss-term-organelles Tubulin,Actin
  --ssl-checkpoint SSL/ssl_encoder.pt  encoder weights from ssl_pretrain.py loaded before training (Run B, --arch smp)
  --extra-input bm3d --extra-input-dir DIR   BM3D-denoised input as a second channel (item 11, bm3d_input.py build)
"""
import argparse
import copy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ORGANELLES, UNKNOWN_MODALITY, MODALITIES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import breakdown, summarize
from official_metrics import percentile_normalization
from predict import evaluate_model, predict_full
from unet import build_model

SPLIT_MANIFEST = Path(os.environ.get("LMC2_SPLIT_MANIFEST", Path(__file__).resolve().parent / "results" / "split_manifest.csv"))

PATH_FIELDS = ("data_root", "pairs_csv", "output_dir", "prepared_dir", "train_prepared_dir", "extra_input_dir")


def observed_channel_l1(prediction, target, observed):
    channel_errors = (prediction.float() - target.float()).abs().mean((-2, -1))
    valid = observed.bool()
    if not valid.any():
        raise ValueError("The batch has no observed fluorescence targets")
    return channel_errors[valid].mean()


def observed_channel_ssim(prediction, target, observed, win=7, k1=0.01, k2=0.03):
    """Mean SSIM over the observed (sample, channel) pairs, computed like skimage structural_similarity with its
    defaults and data_range 1 (the local proxy): uniform win x win window, sample covariance, valid region only."""
    p, t = prediction.float(), target.float()
    pool = lambda z: F.avg_pool2d(z, win, stride=1)
    ux, uy = pool(p), pool(t)
    norm = win * win / (win * win - 1.0)
    vx = norm * (pool(p * p) - ux * ux)
    vy = norm * (pool(t * t) - uy * uy)
    vxy = norm * (pool(p * t) - ux * uy)
    c1, c2 = k1 ** 2, k2 ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    valid = observed.bool()
    if not valid.any():
        raise ValueError("The batch has no observed fluorescence targets")
    return s.mean((-2, -1))[valid].mean()


def _norm_crop(z, p_low=2.0, p_high=99.8):
    """Per (sample, channel) percentile normalization without clipping, as the official scorer does on full images."""
    flat = z.float().flatten(-2)
    q = torch.quantile(flat, torch.tensor([p_low / 100, p_high / 100], device=z.device), dim=-1)
    lo, hi = q[0][..., None, None], q[1][..., None, None]
    return (z.float() - lo) / (hi - lo).clamp_min(1e-6)


def observed_channel_pcc(prediction, target, observed):
    """Mean Pearson correlation over the observed (sample, channel) pairs of the crops."""
    p = prediction.float().flatten(-2)
    t = target.float().flatten(-2)
    p = p - p.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    r = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-6)
    return r[observed.bool()].mean()


def training_loss(prediction, target, observed, ssim_weight, pcc_weight=0.0, ssim_norm=False, term_mask=None):
    """(1 - w) L1 + w (1 - SSIM) + beta (1 - PCC) on the observed channels. term_mask (4 booleans) limits the SSIM and
    PCC terms to some organelles. ssim_norm computes SSIM on prediction and target normalized per crop (p2, p99.8)."""
    l1 = observed_channel_l1(prediction, target, observed)
    obs_terms = observed.bool() if term_mask is None else observed.bool() & term_mask[None, :]
    has_terms = bool(obs_terms.any())
    loss = (1.0 - ssim_weight) * l1 if ssim_weight > 0 else l1
    if ssim_weight > 0 and has_terms:
        p, t = (_norm_crop(prediction), _norm_crop(target)) if ssim_norm else (prediction, target)
        loss = loss + ssim_weight * (1.0 - observed_channel_ssim(p, t, obs_terms))
    if pcc_weight > 0 and has_terms:
        loss = loss + pcc_weight * (1.0 - observed_channel_pcc(prediction, target, obs_terms))
    return loss


def heldout_groups(cfg, train_df):
    """Training groups held out for checkpoint selection: GroupShuffleSplit on the groups of the frozen split, seed 42."""
    from sklearn.model_selection import GroupShuffleSplit
    sm = pd.read_csv(SPLIT_MANIFEST)
    sm = sm[sm.split == "train"][["study", "group"]].drop_duplicates()
    study_group = dict(zip(sm.study, sm.group))
    groups = train_df.study.map(study_group)
    assert groups.notna().all(), "training study missing from the split manifest"
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.heldout_frac, random_state=42)
    _, hold = next(gss.split(train_df, groups=groups))
    return sorted(set(groups.iloc[hold]))


def official_select_scores(model, ds, idxs, cfg, device, cache):
    """Official-style SSIM (data range 1) and PCC per observed organelle on full images, Hann tiling with 128 px overlap.
    Raw targets are read once and kept in cache (dict)."""
    from skimage.metrics import structural_similarity
    import tifffile
    model.eval()
    rows = []
    for idx in idxs:
        item = ds[idx]
        r = ds.df.iloc[idx]
        x = item["input"][None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = predict_full(model, x, cfg.tile_size, 128, window="hann")
        for k, o in enumerate(ORGANELLES):
            if not bool(r[f"{o}_obs"]):
                continue
            key = (int(r.study), int(r.image_id), o)
            if key not in cache:
                raw = tifffile.imread(cfg.data_root / f"Study_{int(r.study):02d}" / r[f"{o}_file"])
                cache[key] = percentile_normalization(np.squeeze(raw).astype(np.float64)).astype(np.float32)
            g = cache[key]
            p = percentile_normalization(pred[k].astype(np.float64)).astype(np.float32)
            ssim = structural_similarity(g, p, data_range=1.0)
            pcc = float(np.corrcoef(g.ravel(), p.ravel())[0, 1]) if g.std() > 0 and p.std() > 0 else np.nan
            rows.append({"organelle": o, "ssim": ssim, "pcc": pcc})
    df = pd.DataFrame(rows)
    per_org = df.groupby("organelle")[["ssim", "pcc"]].mean()
    return float(per_org.ssim.mean()), float(per_org.pcc.mean())


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
    ap.add_argument("--target-clip", action=argparse.BooleanOptionalAction, default=None,
                    help="--no-target-clip: training targets are not clipped to [0, 1]")
    ap.add_argument("--train-prepared-dir", default=None, help="prepared folder built with the training target rule")
    ap.add_argument("--output-activation", choices=["sigmoid", "linear"], default=None)
    ap.add_argument("--ssim-weight", type=float, default=None, help="w in (1 - w) L1 + w (1 - SSIM), 0 = plain L1")
    ap.add_argument("--arch", choices=["unet", "smp"], default=None)
    ap.add_argument("--encoder", default=None, help="smp encoder name, e.g. resnet34 or tu-convnext_tiny")
    ap.add_argument("--encoder-weights", default=None, help="imagenet, or none for random initialization")
    ap.add_argument("--encoder-lr-mult", type=float, default=None)
    ap.add_argument("--ssim-norm", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--pcc-weight", type=float, default=None)
    ap.add_argument("--loss-term-organelles", default=None, help='e.g. "Tubulin,Actin", "" = all')
    ap.add_argument("--select-on", choices=["val_subset", "heldout"], default=None)
    ap.add_argument("--heldout-frac", type=float, default=None)
    ap.add_argument("--ema-decay", type=float, default=None)
    ap.add_argument("--ssl-checkpoint", default=None, help="ssl_encoder.pt from ssl_pretrain.py (--arch smp)")
    ap.add_argument("--extra-input", choices=["none", "bm3d"], default=None)
    ap.add_argument("--extra-input-dir", default=None, help="folder written by bm3d_input.py build")
    a = ap.parse_args()

    over = {k: v for k, v in dict(
        seed=a.seed, epochs=a.epochs, patch_size=a.patch_size, base_channels=a.base_channels,
        num_workers=a.workers, val_subset=a.val_subset, learning_rate=a.lr,
        lr_schedule=a.lr_schedule, modality_mode=a.modality, modality_dropout=a.modality_dropout,
        data_root=Path(a.data_root) if a.data_root else None,
        prepared_dir=Path(a.prepared_dir) if a.prepared_dir and a.prepared_dir.lower() != "none" else None,
        pairs_csv=Path(a.pairs_csv) if a.pairs_csv else None,
        train_target_p_low=a.target_p_low, train_target_p_high=a.target_p_high,
        train_target_clip=a.target_clip,
        train_prepared_dir=Path(a.train_prepared_dir) if a.train_prepared_dir and a.train_prepared_dir.lower() != "none" else None,
        output_activation=a.output_activation, ssim_weight=a.ssim_weight, arch=a.arch, encoder_name=a.encoder,
        encoder_weights=a.encoder_weights, encoder_lr_mult=a.encoder_lr_mult, ssim_norm=a.ssim_norm,
        pcc_weight=a.pcc_weight, loss_term_organelles=a.loss_term_organelles, select_on=a.select_on,
        heldout_frac=a.heldout_frac, ema_decay=a.ema_decay, ssl_checkpoint=a.ssl_checkpoint,
        extra_input=None if a.extra_input is None else ("" if a.extra_input == "none" else a.extra_input),
        extra_input_dir=Path(a.extra_input_dir) if a.extra_input_dir else None).items() if v is not None}
    base_cfg = config_from(a.config_from) if a.config_from else ChallengeConfig()
    if a.prepared_dir and a.prepared_dir.lower() == "none":
        over["prepared_dir"] = None  # read the raw TIFFs even if the copied config names a prepared folder
    if a.train_prepared_dir and a.train_prepared_dir.lower() == "none":
        over["train_prepared_dir"] = None
    cfg = dataclasses.replace(base_cfg, **over)
    out = Path(a.out)
    cfg_dict = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_ds = LMC2Dataset(cfg, "train", train=True, seed=cfg.seed)
    held, sel_ds, sel_idx, sel_cache = [], None, [], {}
    if cfg.select_on == "heldout" and not a.overfit_n:
        held = heldout_groups(cfg, train_ds.df)
        sm = pd.read_csv(SPLIT_MANIFEST)
        held_studies = sorted(set(sm[sm.group.isin(held)].study))
        in_held = train_ds.df.study.isin(held_studies)
        sel_ds = LMC2Dataset(cfg, "train", train=False)
        sel_ds.df = train_ds.df[in_held].reset_index(drop=True)
        train_ds.df = train_ds.df[~in_held].reset_index(drop=True)
        rng_h = np.random.default_rng(42)
        sel_idx = sorted(rng_h.choice(len(sel_ds), size=min(cfg.val_subset, len(sel_ds)), replace=False).tolist())
        json.dump({"groups": held, "studies": held_studies, "images": len(sel_ds), "scored_per_epoch": len(sel_idx)},
                  open(out / "heldout_studies.json", "w"), indent=2)
        print(f"held-out groups {held} | studies {held_studies} | images {len(sel_ds)} | scored per epoch {len(sel_idx)}")
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
    if cfg.arch == "smp" and n_mod:
        raise SystemExit("modality conditioning is only implemented for --arch unet")
    model = build_model(cfg, len(MODALITIES) + 1, pretrained=True).to(device)
    if cfg.ssl_checkpoint:
        if cfg.arch != "smp":
            raise SystemExit("--ssl-checkpoint needs --arch smp")
        sck = torch.load(cfg.ssl_checkpoint, map_location=device, weights_only=False)
        if sck["info"]["encoder"] != cfg.encoder_name:
            raise SystemExit(f"SSL encoder {sck['info']['encoder']} differs from --encoder {cfg.encoder_name}")
        model.net.encoder.load_state_dict(sck["encoder"], strict=True)
        print(f"encoder weights from {cfg.ssl_checkpoint} ({sck['info']['steps']} SSL steps)", flush=True)
    if cfg.arch == "smp":
        enc = {id(p) for p in model.encoder_parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) in enc], "base_lr": cfg.learning_rate * cfg.encoder_lr_mult},
                  {"params": [p for p in model.parameters() if id(p) not in enc], "base_lr": cfg.learning_rate}]
    else:
        groups = [{"params": list(model.parameters()), "base_lr": cfg.learning_rate}]
    for g in groups:
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    ema = copy.deepcopy(model).eval() if cfg.ema_decay > 0 else None
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    term_mask = None
    if cfg.loss_term_organelles:
        names = [o.strip() for o in cfg.loss_term_organelles.split(",") if o.strip()]
        assert all(o in ORGANELLES for o in names), f"unknown organelle in {cfg.loss_term_organelles!r}"
        term_mask = torch.tensor([o in names for o in ORGANELLES], device=device)
    print("parameters:", sum(p.numel() for p in model.parameters()), "| arch", cfg.arch,
          f"({cfg.encoder_name}, {cfg.encoder_weights})" if cfg.arch == "smp" else "")
    K = cfg.crops_per_image
    history, best, best_epoch = [], -1.0, -1
    best_mean, best_mean_epoch = -1.0, -1  # second rule: mean of macro SSIM and PCC (best_mean.pt)
    start_epoch, elapsed_before = 0, 0.0
    last_path = out / "last.pt"
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        stored = {k: v for k, v in ck["cfg"].items() if k not in PATH_FIELDS + ("num_workers",)}
        current = {k: v for k, v in cfg_dict.items() if k not in PATH_FIELDS + ("num_workers",)}
        changed = {k: (stored.get(k), current.get(k)) for k in set(stored) | set(current) if stored.get(k) != current.get(k)
                   and not (k not in stored and current.get(k) == getattr(ChallengeConfig(), k, None))}
        if changed:
            raise SystemExit(f"--resume with settings that differ from {last_path} (stored, now): {changed}. "
                             "Pass the same options as the interrupted run, or use another --out.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        start_epoch, history = ck["epoch"], ck["history"]
        best, best_epoch, elapsed_before = ck["best"], ck["best_epoch"], ck["elapsed"]
        if "best_mean" in ck:
            best_mean, best_mean_epoch = ck["best_mean"], ck["best_mean_epoch"]
        else:  # last.pt from before best_mean existed: the rule is rebuilt from the history, best_mean.pt is not
            means = [0.5 * (h["val_macro_ssim"] + h["val_macro_pcc"]) for h in history]
            if means:
                i = int(np.nanargmax(means))
                best_mean, best_mean_epoch = float(means[i]), i + 1
            print(f"note: {last_path} has no best_mean.pt record, best_mean.pt only covers epochs after {start_epoch}")
        print(f"resumed from epoch {start_epoch}, best epoch {best_epoch} ({best:.4f})")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time() - elapsed_before

    mod_rng = np.random.default_rng(cfg.seed + 7)
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.lr_schedule == "cosine":
            factor = cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * epoch / cfg.epochs))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * factor
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
            loss = training_loss(pred, y, obs, cfg.ssim_weight, cfg.pcc_weight, cfg.ssim_norm, term_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(cfg.ema_decay).add_(pm.detach(), alpha=1 - cfg.ema_decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            losses.append(loss.item())
        net = ema if ema is not None else model  # the weights that are evaluated and saved as best
        if sel_ds is not None:
            m_ssim, m_pcc = official_select_scores(net, sel_ds, sel_idx, cfg, device, sel_cache)
        else:
            df, _ = evaluate_model(net, val_ds, cfg, device, val_idx)
            m_ssim, m_pcc, _ = macro(df)
        row = {"epoch": epoch + 1, "lr": opt.param_groups[-1]["lr"], "train_loss": float(np.mean(losses)), "val_macro_ssim": m_ssim,
               "val_macro_pcc": m_pcc, "elapsed_s": time.time() - t_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print({k: round(v, 4) for k, v in row.items()})
        m_mean = 0.5 * (m_ssim + m_pcc)
        # proxy rule: best.pt on macro SSIM. Held-out rule: best.pt on the official-style mean of macro SSIM and PCC.
        crit = m_mean if sel_ds is not None else m_ssim
        if crit > best:
            best, best_epoch = crit, epoch + 1
            torch.save({"model": net.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict}, out / "best.pt")
        if m_mean > best_mean:
            best_mean, best_mean_epoch = m_mean, epoch + 1
            torch.save({"model": net.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict}, out / "best_mean.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1, "cfg": cfg_dict,
                    **({"ema": ema.state_dict()} if ema is not None else {}),
                    "history": json.loads(json.dumps(history, default=float)), "best": float(best), "best_epoch": best_epoch, "best_mean": float(best_mean),
                    "best_mean_epoch": best_mean_epoch, "elapsed": row["elapsed_s"]},
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
    info = {"best_epoch": best_epoch, "best_mean_epoch": best_mean_epoch, "train_seconds": train_time, "inference_seconds_per_image": sec_per_image,
            "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
            "macro_ssim": m_ssim, "macro_pcc": m_pcc, "config": cfg_dict,
            "overfit_n": a.overfit_n, "config_from": a.config_from, "versions": versions(),
            "heldout_groups": held, "parameters": sum(p.numel() for p in model.parameters()), **unk}
    json.dump(info, open(out / "run_info.json", "w"), indent=2, default=str)
    print(table.to_string(index=False))
    print({k: info[k] for k in ("best_epoch", "train_seconds", "inference_seconds_per_image", "peak_gpu_memory_gb",
                                "macro_ssim", "macro_pcc")})


if __name__ == "__main__":
    main()
