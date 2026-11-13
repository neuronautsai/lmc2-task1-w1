"""Compare output intensity scales offline against the raw 16-bit validation targets (review item 2).

Usage (Colab, from the repo folder):
  python compare_output_scale.py --checkpoint RUN/best.pt --pairs-csv PAIRS_EXCLUDED --out OUT [--n-stats 300] [--resume]

The model predicts normalized intensities p in [0, 1]. Three ways of writing them are compared:
  a  uint8:   round(255 * p), what the v0.1.0 container writes
  b  uint16:  round(lo + p * (hi - lo)), lo and hi per organelle from the raw training targets
  c  float32: lo + p * (hi - lo), same mapping without rounding
lo and hi are the medians, over training images with that target, of the raw target's 2nd and 99.8th percentile
(the percentiles the evaluator uses, see official_metrics.py). They are computed once on up to --n-stats training
images per organelle (random, seed 0) and stored in target_stats.json, so inference needs no test target.

Scores per validation image and observed organelle:
  - official-style metrics (official_metrics.py: both images percentile normalized) for option a on every image, and
    for all three options on the first --n-invariance images, compared with the unrounded prediction p, to confirm
    that SSIM, PCC and cosine distance do not change with the output scale (only rounding can change them);
  - raw-scale metrics for every option, computed on the raw target and the written values without any normalization:
    mae_raw (mean absolute difference), euc_raw (L2 norm of the difference), cos_raw (1 minus cosine similarity).
    These are an assumption about how the portal computes MAE and Euclidean distance: its values (thousands and
    millions for 16-bit targets) are not compatible with normalized images. They are labelled as such.
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
from official_metrics import P_MAX, P_MIN, official_metrics
from predict import predict_full
from score_official import retry
from unet import UNet

OPTIONS = ("a_uint8", "b_uint16", "c_float32")


def target_stats(cfg, out, n_stats):
    path = out / "target_stats.json"
    if path.exists():
        return json.load(open(path))
    df = load_pairs(cfg, "train")
    rng = np.random.default_rng(0)
    stats = {}
    for o in ORGANELLES:
        d = df[df[f"{o}_obs"].astype(bool)]
        d = d.iloc[rng.permutation(len(d))[:n_stats]]
        assert len(d), f"no training target for {o}"
        lo, hi, mx = [], [], []
        for _, r in d.iterrows():
            raw = retry(lambda: tifffile.imread(cfg.data_root / f"Study_{int(r.study):02d}" / r[f"{o}_file"]))
            p = np.percentile(raw, [P_MIN, P_MAX])
            lo.append(float(p[0]))
            hi.append(float(p[1]))
            mx.append(float(raw.max()))
        stats[o] = {"n_images": len(d), "lo_median": float(np.median(lo)), "hi_median": float(np.median(hi)),
                    "lo_q25_q75": [float(np.percentile(lo, 25)), float(np.percentile(lo, 75))],
                    "hi_q25_q75": [float(np.percentile(hi, 25)), float(np.percentile(hi, 75))],
                    "max_median": float(np.median(mx))}
        print(f"{o}: {len(d)} training targets, median p2 {stats[o]['lo_median']:.1f}, median p99.8 {stats[o]['hi_median']:.1f}",
              flush=True)
    json.dump(stats, open(path, "w"), indent=2)
    return stats


def write_options(p, lo, hi):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    mapped = lo + p * (hi - lo)
    return {"a_uint8": np.round(255.0 * p).astype(np.uint8),
            "b_uint16": np.clip(np.round(mapped), 0, 65535).astype(np.uint16),
            "c_float32": mapped.astype(np.float32)}


def raw_metrics(gt, out):
    g, o = gt.ravel().astype(np.float64), out.ravel().astype(np.float64)
    ng, no = np.linalg.norm(g), np.linalg.norm(o)
    cos = np.nan if (ng == 0 or no == 0) else 1.0 - float(g @ o) / (ng * no)
    return {"mae_raw": float(np.abs(g - o).mean()), "euc_raw": float(np.linalg.norm(g - o)), "cos_raw": cos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n-stats", type=int, default=300)
    ap.add_argument("--n-invariance", type=int, default=20)
    ap.add_argument("--subset", type=int, default=0, help="score only the first N validation images (0 = all)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    kw = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = ChallengeConfig(**kw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = target_stats(cfg, out, a.n_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    assert ck["cfg"].get("modality_mode") != "film", "use an unconditioned checkpoint"
    model = UNet(base=int(ck["cfg"]["base_channels"]), n_mod=0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    val = load_pairs(cfg, "val")
    if a.subset:
        val = val.iloc[:a.subset]

    partial = out / "partial_rows.csv"
    rows, done = [], set()
    if a.resume and partial.exists():
        prev = pd.read_csv(partial)
        rows, done = prev.to_dict("records"), set(zip(prev.study, prev.image_id))
        print(f"resuming with {len(done)} images already scored", flush=True)
    inv = {o: 0.0 for o in OPTIONS}
    n_inv = 0
    for n, (_, r) in enumerate(val.iterrows()):
        if (int(r.study), int(r.image_id)) in done:
            continue
        base = cfg.data_root / f"Study_{int(r.study):02d}"
        raw_in = np.squeeze(retry(lambda: tifffile.imread(base / r.in_file))).astype(np.float32)
        x = torch.from_numpy(normalize(raw_in, cfg.p_low, cfg.p_high))[None, None].to(device)
        pred = predict_full(model, x, cfg.tile_size, cfg.tile_overlap)
        check = n_inv < a.n_invariance
        for k, o in enumerate(ORGANELLES):
            if not bool(r[f"{o}_obs"]):
                continue
            gt = np.squeeze(retry(lambda: tifffile.imread(base / r[f"{o}_file"]))).astype(np.float64)
            assert gt.shape == pred[k].shape, f"shape mismatch {r.study} {r.image_id} {o}"
            written = write_options(pred[k], stats[o]["lo_median"], stats[o]["hi_median"])
            off_a = official_metrics(gt, written["a_uint8"])
            row = {"study": int(r.study), "image_id": int(r.image_id), "modality": r.modality, "organelle": o,
                   "gt_mean": float(gt.mean()), **off_a}
            for opt in OPTIONS:
                row.update({f"{m}_{opt}": v for m, v in raw_metrics(gt, written[opt]).items()})
            if check:
                ref = official_metrics(gt, np.clip(pred[k].astype(np.float64), 0.0, 1.0))
                for opt in OPTIONS:
                    off = off_a if opt == "a_uint8" else official_metrics(gt, written[opt])
                    d = max(abs(off[m] - ref[m]) for m in ("off_pcc", "off_ssim_dr1", "off_ssim_drjoint", "off_cos")
                            if np.isfinite(off[m]) and np.isfinite(ref[m]))
                    inv[opt] = max(inv[opt], d)
            rows.append(row)
        n_inv += int(check)
        if (n + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"{n + 1}/{len(val)} images", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_output_scale.csv", index=False)
    lines = []
    for m in ("mae_raw", "euc_raw", "cos_raw"):
        t = pd.DataFrame({opt: df.groupby("organelle")[f"{m}_{opt}"].mean() for opt in OPTIONS}).reindex(ORGANELLES)
        t.loc["macro"] = t.mean()
        lines += [f"\n{m} (raw scale, assumed portal definition), mean over images", t.to_string(float_format=lambda v: f"{v:,.4f}" if m == "cos_raw" else f"{v:,.1f}")]
    off = df.groupby("organelle")[["off_ssim_dr1", "off_pcc", "off_cos", "off_mae", "off_euc"]].mean().reindex(ORGANELLES)
    lines += ["\nOfficial-style metrics (normalized, option a), mean over images", off.round(4).to_string()]
    lines += [f"\nInvariance check on the first {n_inv} images: max |change| of official-style SSIM, PCC and cosine "
              "distance against the unrounded prediction: " + ", ".join(f"{opt} {v:.2e}" for opt, v in inv.items())]
    lines += ["\nTarget stats used for b and c (median p2 and p99.8 of raw training targets):"]
    lines += [f"  {o}: lo {stats[o]['lo_median']:.1f}, hi {stats[o]['hi_median']:.1f} (n={stats[o]['n_images']})" for o in ORGANELLES]
    lines += [f"\nValidation images: {df[['study', 'image_id']].drop_duplicates().shape[0]}, rows: {len(df)}. "
              f"Mean raw target value per organelle: " + ", ".join(f"{o} {v:,.0f}" for o, v in df.groupby('organelle').gt_mean.mean().reindex(ORGANELLES).items())]
    text = "\n".join(lines)
    (out / "output_scale_summary.txt").write_text(text)
    json.dump({"checkpoint": a.checkpoint, "epoch": ck.get("epoch"), "pairs_csv": str(cfg.pairs_csv),
               "n_images": int(df[["study", "image_id"]].drop_duplicates().shape[0]), "invariance_max_abs": inv,
               "target_stats": stats}, open(out / "output_scale_info.json", "w"), indent=2)
    print(text)


if __name__ == "__main__":
    main()
