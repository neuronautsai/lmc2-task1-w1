"""Re-score a trained checkpoint on the local validation split with the current proxy and the official-style proxy.

Usage: python score_official.py --checkpoint OUT_RUN/best.pt --out OUT_SCORE [--subset N]

Inputs are read from the raw TIFFs with the declared rule (as in the Week 2 run), predictions come from the tiled
model, and the ground truth for the official-style metrics is the RAW target image. The current proxy columns
(proxy_ssim, proxy_pcc) are computed in the same pass on the normalized target, so they can be compared with the
Week 2 summary.csv. Official-style columns are described in official_metrics.py and are not official scores.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from config import ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import score_image
from official_metrics import METRIC_COLUMNS, official_metrics
from predict import predict_full
from unet import UNet


def retry(fn, tries=6, wait=5.0):
    """Drive mounts sometimes raise OSError (Errno 5) on a read. Retry with a growing pause before giving up."""
    for k in range(tries):
        try:
            return fn()
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"read failed ({e}), retry {k + 1}/{tries - 1}", flush=True)
            time.sleep(wait * (k + 1))


def invariance_check(raw, pred, n_images):
    """Affine rescaling of a prediction must not change the official-style scores (prediction is normalized by the evaluator)."""
    diffs = []
    for gt, p in zip(raw[:n_images], pred[:n_images]):
        a, b = official_metrics(gt, p), official_metrics(gt, p * 37.0 + 11.0)
        diffs.append(max(abs(a[k] - b[k]) for k in METRIC_COLUMNS if np.isfinite(a[k]) and np.isfinite(b[k])))
    return float(max(diffs)) if diffs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=0, help="score only the first N validation images (0 = all)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from partial_rows.csv in --out")
    a = ap.parse_args()
    kw = {}
    if a.data_root:
        kw["data_root"] = Path(a.data_root)
    if a.pairs_csv:
        kw["pairs_csv"] = Path(a.pairs_csv)
    cfg = ChallengeConfig(**kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device)
    model = UNet(base=int(ck["cfg"]["base_channels"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ds = LMC2Dataset(cfg, "val", train=False)
    idxs = range(len(ds)) if not a.subset else range(min(a.subset, len(ds)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, raw_keep, pred_keep = [], [], []
    partial = out / "partial_rows.csv"
    done = set()
    if a.resume and partial.exists():
        prev = pd.read_csv(partial)
        rows = prev.to_dict("records")
        done = set(zip(prev.study, prev.image_id))
        print(f"resuming with {len(done)} images already scored", flush=True)
    for n, idx in enumerate(idxs):
        r = ds.df.iloc[idx]
        if (int(r.study), int(r.image_id)) in done:
            continue
        item = retry(lambda: ds[idx])
        pred = predict_full(model, item["input"][None].to(device), cfg.tile_size, cfg.tile_overlap)
        obs = item["observed"].numpy()
        proxy = {x["organelle"]: x for x in score_image(pred, item["target"].numpy(), obs)}
        for k, organelle in enumerate(ORGANELLES):
            if not obs[k]:
                continue
            path = cfg.data_root / f"Study_{int(r.study):02d}" / r[f"{organelle}_file"]
            raw = retry(lambda: tifffile.imread(path)).astype(np.float64)
            assert raw.shape == pred[k].shape, f"shape mismatch {r.study} {r.image_id} {organelle}"
            row = {"study": int(r.study), "image_id": int(r.image_id), "modality": r.modality, "organelle": organelle,
                   "proxy_ssim": proxy[organelle]["ssim"], "proxy_pcc": proxy[organelle]["pcc"]}
            row.update(official_metrics(raw, pred[k]))
            rows.append(row)
            if len(raw_keep) < 5:
                raw_keep.append(raw)
                pred_keep.append(pred[k].astype(np.float64))
        if (n + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"{n + 1}/{len(idxs)} images", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_official_scores.csv", index=False)
    cols = ["proxy_ssim", "proxy_pcc", *METRIC_COLUMNS]
    summ = df.groupby("organelle")[cols].agg(["mean", "std"])
    summ.insert(0, ("n", "observed"), df.groupby("organelle").size())
    summ = summ.reindex(ORGANELLES)
    summ.to_csv(out / "summary_official.csv")
    by_mod = df.groupby(["modality", "organelle"])[cols].mean().assign(n=df.groupby(["modality", "organelle"]).size())
    by_mod.reset_index().to_csv(out / "by_modality_official.csv", index=False)
    by_study = df.groupby(["study", "organelle"])[cols].mean().assign(n=df.groupby(["study", "organelle"]).size())
    by_study.reset_index().to_csv(out / "by_study_official.csv", index=False)
    macro = {c: float(df.groupby("organelle")[c].mean().reindex(ORGANELLES).mean()) for c in cols}
    undefined = {c: int(df[c].isna().sum()) for c in cols}
    inv = invariance_check(raw_keep, pred_keep, 5)
    info = {"checkpoint": a.checkpoint, "epoch": ck.get("epoch"), "n_images": len(idxs), "n_rows": len(df),
            "macro_over_organelles": macro, "undefined_counts": undefined,
            "max_abs_change_under_affine_rescaling_of_prediction": inv,
            "assumptions": "SSIM data_range (1.0 or joint range), window and K1/K2 unpublished. "
                           "MAE, cosine and Euclidean definitions assumed, see official_metrics.py"}
    json.dump(info, open(out / "official_score_info.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
