"""Re-score a trained checkpoint on the local validation split with the current proxy and the official-style proxy.

Usage: python score_official.py --checkpoint OUT_RUN/best.pt --out OUT_SCORE [--subset N]
       python score_official.py --checkpoint RUN_A/best.pt RUN_B/best.pt RUN_C/best.pt --out OUT_SCORE --window hann --overlap 128

Several checkpoints: every model and the mean of their predictions (ensemble) are scored in one pass, each in its own
subfolder of --out (named after the run folder and the checkpoint file, plus "ensemble"). --tta averages each model
over the 8 flips and 90-degree rotations of the input. --window and --overlap set the tile blending (default: the
Week 2 and 3 setting, linear with 64 px).

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

from config import MODALITIES, ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import score_image
from official_metrics import METRIC_COLUMNS, official_metrics
from predict import WINDOWS, predict_full
from unet import unet_from_checkpoint


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


def predict_tta(model, x, tile, overlap, mod, window, tta):
    """Prediction of one model, optionally averaged over the 8 flips and 90-degree rotations of the input."""
    if not tta:
        return predict_full(model, x, tile, overlap, mod=mod, window=window)
    acc = None
    for k in range(4):
        for flip in (False, True):
            xt = torch.rot90(x, k, dims=(2, 3))
            if flip:
                xt = torch.flip(xt, dims=(3,))
            p = predict_full(model, xt.contiguous(), tile, overlap, mod=mod, window=window)
            if flip:
                p = p[:, :, ::-1]
            p = np.rot90(p, -k, axes=(1, 2))
            acc = p.copy() if acc is None else acc + p
    return acc / 8.0


def variant_name(path):
    p = Path(path)
    return f"{p.parent.name}_{p.stem}"


def invariance_check(raw, pred, n_images):
    """Affine rescaling of a prediction must not change the official-style scores (prediction is normalized by the evaluator)."""
    diffs = []
    for gt, p in zip(raw[:n_images], pred[:n_images]):
        a, b = official_metrics(gt, p), official_metrics(gt, p * 37.0 + 11.0)
        diffs.append(max(abs(a[k] - b[k]) for k in METRIC_COLUMNS if np.isfinite(a[k]) and np.isfinite(b[k])))
    return float(max(diffs)) if diffs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, nargs="+", help="one or more checkpoints (several: also their ensemble)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=0, help="score only the first N validation images (0 = all)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--modality-label", choices=["true", "unknown"], default="true", help="for a modality-conditioned model")
    ap.add_argument("--resume", action="store_true", help="continue from partial_rows.csv in --out")
    ap.add_argument("--window", choices=WINDOWS, default="linear", help="tile blending weight (see predict.py)")
    ap.add_argument("--overlap", type=int, default=None, help="tile overlap in px (default: config tile_overlap)")
    ap.add_argument("--tta", action="store_true", help="average over the 8 flips and 90-degree rotations")
    a = ap.parse_args()
    kw = {}
    if a.data_root:
        kw["data_root"] = Path(a.data_root)
    if a.pairs_csv:
        kw["pairs_csv"] = Path(a.pairs_csv)
    cfg = ChallengeConfig(**kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overlap = cfg.tile_overlap if a.overlap is None else a.overlap
    models, epochs = [], []
    for c in a.checkpoint:
        ck = torch.load(c, map_location=device)
        models.append(unet_from_checkpoint(ck, device, len(MODALITIES) + 1))
        epochs.append(ck.get("epoch"))
    names = [variant_name(c) for c in a.checkpoint]
    assert len(set(names)) == len(names), f"checkpoint names must be unique: {names}"
    single = len(models) == 1
    variants = names if single else names + ["ensemble"]
    ds = LMC2Dataset(cfg, "val", train=False)
    idxs = range(len(ds)) if not a.subset else range(min(a.subset, len(ds)))
    out = Path(a.out)
    outs = {v: out if single else out / v for v in variants}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)

    rows = {v: [] for v in variants}
    raw_keep = {v: [] for v in variants}
    pred_keep = {v: [] for v in variants}
    done = None
    if a.resume:
        for v in variants:
            f = outs[v] / "partial_rows.csv"
            prev = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["study", "image_id"])
            keys = set(zip(prev.study, prev.image_id))
            done = keys if done is None else done & keys
        for v in variants:  # keep only images scored for every variant, so all variants cover the same images
            f = outs[v] / "partial_rows.csv"
            if f.exists():
                prev = pd.read_csv(f)
                rows[v] = [r for r in prev.to_dict("records") if (r["study"], r["image_id"]) in done]
        print(f"resuming with {len(done)} images already scored", flush=True)
    done = done or set()
    for n, idx in enumerate(idxs):
        r = ds.df.iloc[idx]
        if (int(r.study), int(r.image_id)) in done:
            continue
        item = retry(lambda: ds[idx])
        x = item["input"][None].to(device)
        preds = {}
        for name, model in zip(names, models):
            mod = None
            if model.n_mod:
                mod = item["modality_id"] if a.modality_label == "true" else model.n_mod - 1
            preds[name] = predict_tta(model, x, cfg.tile_size, overlap, mod, a.window, a.tta)
        if not single:
            preds["ensemble"] = np.mean([preds[nm] for nm in names], axis=0)
        obs = item["observed"].numpy()
        raws = {}
        for k, organelle in enumerate(ORGANELLES):
            if obs[k]:
                path = cfg.data_root / f"Study_{int(r.study):02d}" / r[f"{organelle}_file"]
                raws[k] = retry(lambda: tifffile.imread(path)).astype(np.float64)
        for v in variants:
            pred = preds[v]
            proxy = {x_["organelle"]: x_ for x_ in score_image(pred, item["target"].numpy(), obs)}
            for k, organelle in enumerate(ORGANELLES):
                if not obs[k]:
                    continue
                raw = raws[k]
                assert raw.shape == pred[k].shape, f"shape mismatch {r.study} {r.image_id} {organelle}"
                row = {"study": int(r.study), "image_id": int(r.image_id), "modality": r.modality, "organelle": organelle,
                       "proxy_ssim": proxy[organelle]["ssim"], "proxy_pcc": proxy[organelle]["pcc"]}
                row.update(official_metrics(raw, pred[k]))
                rows[v].append(row)
                if len(raw_keep[v]) < 5:
                    raw_keep[v].append(raw)
                    pred_keep[v].append(pred[k].astype(np.float64))
        if (n + 1) % 50 == 0:
            for v in variants:
                pd.DataFrame(rows[v]).to_csv(outs[v] / "partial_rows.csv", index=False)
            print(f"{n + 1}/{len(idxs)} images", flush=True)

    pd.set_option("display.width", 200)
    cols = ["proxy_ssim", "proxy_pcc", *METRIC_COLUMNS]
    overview = {}
    for v in variants:
        df = pd.DataFrame(rows[v])
        o = outs[v]
        df.to_csv(o / "per_image_official_scores.csv", index=False)
        summ = df.groupby("organelle")[cols].agg(["mean", "std"])
        summ.insert(0, ("n", "observed"), df.groupby("organelle").size())
        summ = summ.reindex(ORGANELLES)
        summ.to_csv(o / "summary_official.csv")
        by_mod = df.groupby(["modality", "organelle"])[cols].mean().assign(n=df.groupby(["modality", "organelle"]).size())
        by_mod.reset_index().to_csv(o / "by_modality_official.csv", index=False)
        by_study = df.groupby(["study", "organelle"])[cols].mean().assign(n=df.groupby(["study", "organelle"]).size())
        by_study.reset_index().to_csv(o / "by_study_official.csv", index=False)
        macro = {c: float(df.groupby("organelle")[c].mean().reindex(ORGANELLES).mean()) for c in cols}
        undefined = {c: int(df[c].isna().sum()) for c in cols}
        inv = invariance_check(raw_keep[v], pred_keep[v], 5)
        is_ens = v == "ensemble"
        cks = a.checkpoint if is_ens else [a.checkpoint[names.index(v)]]
        uses_mod = any(m.n_mod for m in (models if is_ens else [models[names.index(v)]]))
        info = {"checkpoint": cks[0] if len(cks) == 1 else cks,
                "modality_label": a.modality_label if uses_mod else "not used",
                "epoch": epochs[names.index(v)] if not is_ens else epochs,
                "window": a.window, "overlap": overlap, "tta": a.tta,
                "n_images": len(idxs), "n_rows": len(df),
                "macro_over_organelles": macro, "undefined_counts": undefined,
                "max_abs_change_under_affine_rescaling_of_prediction": inv,
                "assumptions": "SSIM data_range (1.0 or joint range), window and K1/K2 unpublished. "
                               "MAE, cosine and Euclidean definitions assumed, see official_metrics.py"}
        json.dump(info, open(o / "official_score_info.json", "w"), indent=2)
        overview[v] = macro
        print(f"===== {v}")
        print(summ.round(4).to_string())
        print(json.dumps(info, indent=2))
    if not single:
        ov = pd.DataFrame(overview).T[["off_ssim_dr1", "off_ssim_drjoint", "off_pcc", "off_mae", "off_cos", "off_euc"]]
        ov.to_csv(out / "overview_macro.csv")
        print("===== macro over organelles, all variants")
        print(ov.round(4).to_string())


if __name__ == "__main__":
    main()
