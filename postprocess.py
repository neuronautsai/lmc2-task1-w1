"""Per-organelle post-processing of predictions (#333 item 9 and the Week 1 TV ablation), and its sweep.

Operations on one organelle's prediction p in [0, 1], applied in this order and before the output mapping:
  tv      skimage.restoration.denoise_tv_chambolle(p, weight=tv), the TV problem of the issue with weight lambda
  offset  max(p - offset, 0) / (1 - offset): background level moved to 0
  gamma   p ** gamma
Settings are a dict per organelle, for example {"Tubulin": {"tv": 0.03}, "Actin": {"gamma": 1.25}}. A missing organelle
or key means no change. The container reads the same dict from postprocess.json next to the weights.

Sweep (python postprocess.py ...): predicts every image once with the given checkpoints (mean of the models, optional
8-fold TTA, Hann tiling), applies the unchanged prediction and every single setting of the grid to each observed
organelle, and scores each with the official-style metrics against the raw target. One setting is chosen per organelle
by the rule of the issue: the highest SSIM among the settings that raise SSIM and do not lower PCC, else no change.
--split heldout scores the held-out training studies of a run (heldout_studies.json written by train.py), which is
the honest choice for models that did not train on them. On the validation split the choice is optimistic, because it
is made on the reported images.

Usage:
  python postprocess.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --tta --out OUT
  python postprocess.py --checkpoint RUN/best.pt --split heldout --heldout-json RUN/heldout_studies.json --out OUT
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import ORGANELLES

DEFAULT_GRID = {"tv": [0.01, 0.03, 0.1], "offset": [0.02, 0.05, 0.1], "gamma": [0.8, 1.25, 1.5]}


def apply_postprocess(p, settings):
    """p: 2D prediction in [0, 1] of one organelle, settings: dict with optional keys tv, offset, gamma."""
    q = np.clip(np.asarray(p, dtype=np.float32), 0.0, 1.0)
    if settings.get("tv"):
        from skimage.restoration import denoise_tv_chambolle
        q = denoise_tv_chambolle(q, weight=float(settings["tv"])).astype(np.float32)
    if settings.get("offset"):
        b = float(settings["offset"])
        assert 0.0 <= b < 1.0, f"offset {b} outside [0, 1)"
        q = np.maximum(q - b, 0.0) / (1.0 - b)
    if settings.get("gamma"):
        q = np.clip(q, 0.0, 1.0) ** float(settings["gamma"])
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def apply_all(pred, plan):
    """pred: (4, H, W) in [0, 1], plan: {organelle: settings}. Returns a new array."""
    out = np.array(pred, dtype=np.float32, copy=True)
    for k, organelle in enumerate(ORGANELLES):
        if plan.get(organelle):
            out[k] = apply_postprocess(out[k], plan[organelle])
    return out


def check_plan(plan):
    for organelle, settings in plan.items():
        if organelle not in ORGANELLES:
            raise ValueError(f"postprocess: unknown organelle {organelle!r}")
        bad = set(settings) - set(DEFAULT_GRID)
        if bad:
            raise ValueError(f"postprocess: unknown setting(s) {sorted(bad)} for {organelle}")
    return plan


def variants_from(grid):
    out = [("none", {})]
    for key in ("tv", "offset", "gamma"):
        out += [(f"{key}={v:g}", {key: float(v)}) for v in grid[key]]
    return out


def choose(summary):
    """summary: rows (organelle, variant) with off_ssim_dr1 and off_pcc means. Returns {organelle: (variant, settings)}."""
    chosen = {}
    for organelle, g in summary.groupby("organelle"):
        base = g[g.variant == "none"].iloc[0]
        ok = g[(g.variant != "none") & (g.off_ssim_dr1 > base.off_ssim_dr1) & (g.off_pcc >= base.off_pcc)]
        chosen[organelle] = ok.sort_values("off_ssim_dr1").iloc[-1].variant if len(ok) else "none"
    return chosen


def main():
    import torch
    import tifffile
    from config import MODALITIES, ChallengeConfig
    from dataset import eval_dataset
    from official_metrics import METRIC_COLUMNS, official_metrics
    from predict import WINDOWS, predict_tta
    from score_official import retry
    from unet import unet_from_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True, help="several checkpoints are averaged (ensemble)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "heldout"], default="val")
    ap.add_argument("--heldout-json", default=None, help="heldout_studies.json of a run, with --split heldout")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--subset", type=int, default=0, help="first N images only (0 = all)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--window", choices=WINDOWS, default="hann")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--tv", type=float, nargs="*", default=DEFAULT_GRID["tv"])
    ap.add_argument("--offset", type=float, nargs="*", default=DEFAULT_GRID["offset"])
    ap.add_argument("--gamma", type=float, nargs="*", default=DEFAULT_GRID["gamma"])
    ap.add_argument("--workers", type=int, default=4, help="threads for the settings of one image")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    kw = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = ChallengeConfig(**kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [unet_from_checkpoint(torch.load(c, map_location=device, weights_only=False), device, len(MODALITIES) + 1)
              for c in a.checkpoint]
    variants = variants_from({"tv": a.tv, "offset": a.offset, "gamma": a.gamma})
    ds = eval_dataset(cfg, a.split, a.heldout_json)
    idxs = range(len(ds)) if not a.subset else range(min(a.subset, len(ds)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "partial_rows.csv"
    rows, done = [], set()
    if a.resume and part.exists():
        prev = pd.read_csv(part)
        rows, done = prev.to_dict("records"), set(zip(prev.study, prev.image_id))
        print(f"resuming with {len(done)} images already scored", flush=True)
    pool = ThreadPoolExecutor(max(1, a.workers))

    def score(raw, p, name, settings):
        r = {"variant": name}
        r.update(official_metrics(raw, apply_postprocess(p, settings)))
        return r

    for n, idx in enumerate(idxs):
        r = ds.df.iloc[idx]
        if (int(r.study), int(r.image_id)) in done:
            continue
        item = retry(lambda: ds[idx])
        x = item["input"][None].to(device)
        preds = []
        for m in models:
            mod = item["modality_id"] if getattr(m, "n_mod", 0) else None
            preds.append(predict_tta(m, x, cfg.tile_size, a.overlap, mod=mod, window=a.window, n_tta=8 if a.tta else 1))
        pred = np.mean(preds, axis=0)
        for k, organelle in enumerate(ORGANELLES):
            if not bool(item["observed"][k]):
                continue
            path = cfg.data_root / f"Study_{int(r.study):02d}" / r[f"{organelle}_file"]
            raw = retry(lambda: tifffile.imread(path)).astype(np.float64)
            for res in pool.map(lambda v: score(raw, pred[k], *v), variants):
                rows.append({"study": int(r.study), "image_id": int(r.image_id), "modality": r.modality,
                             "organelle": organelle, **res})
        if (n + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"{n + 1}/{len(idxs)} images", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_postprocess.csv", index=False)
    cols = list(METRIC_COLUMNS)
    summary = df.groupby(["organelle", "variant"])[cols].mean().reset_index()
    summary["n"] = df.groupby(["organelle", "variant"]).size().values
    summary.to_csv(out / "summary_postprocess.csv", index=False)
    chosen = choose(summary)
    settings = dict(variants)
    plan = {o: settings[v] for o, v in chosen.items() if v != "none"}
    json.dump(plan, open(out / "postprocess_selected.json", "w"), indent=2)
    # macro over organelles: unchanged prediction, and the per-organelle choice
    base = summary[summary.variant == "none"].set_index("organelle")[cols].reindex(ORGANELLES)
    sel = pd.DataFrame([summary[(summary.organelle == o) & (summary.variant == chosen.get(o, "none"))][cols].iloc[0]
                        for o in ORGANELLES if o in chosen], index=[o for o in ORGANELLES if o in chosen])
    macro = pd.DataFrame({"none": base.mean(), "selected per organelle": sel.mean()}).T
    macro.to_csv(out / "macro_postprocess.csv")
    info = {"checkpoints": a.checkpoint, "split": a.split, "heldout_json": a.heldout_json, "tta": a.tta,
            "window": a.window, "overlap": a.overlap, "n_images": len(set(zip(df.study, df.image_id))),
            "grid": {"tv": a.tv, "offset": a.offset, "gamma": a.gamma}, "chosen": chosen,
            "rule": "highest SSIM among settings with SSIM above and PCC not below the unchanged prediction",
            "selection_note": "optimistic on the validation split (chosen on the reported images)" if a.split == "val"
                              else "chosen on held-out training studies"}
    json.dump(info, open(out / "postprocess_info.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    piv = summary.pivot(index="variant", columns="organelle", values=["off_ssim_dr1", "off_pcc"])
    print("official-style SSIM (data range 1) and PCC per organelle and setting")
    print(piv.round(4).to_string())
    print("chosen per organelle:", json.dumps(chosen))
    print("macro over organelles")
    print(macro[["off_ssim_dr1", "off_pcc", "off_mae", "off_cos", "off_euc"]].round(4).to_string())


if __name__ == "__main__":
    main()
