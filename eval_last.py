"""Score the last-epoch weights (last.pt) of every Week 3 run on the full local validation split.

Same evaluation as the end of train.py (evaluate_model, summarize), only the checkpoint differs: train.py scores best.pt,
which is picked by the maximum macro SSIM on the 100-image validation subset. This checks whether that rule hides the
effect of the learning-rate decay. A film model is also scored with the modality label withheld.

Usage: python eval_last.py --root OUTPUTS_W3 --prepared-dir /content/prepared
Writes ROOT/<run>/eval_last/ (summary.csv, per_image_scores.csv, eval_info.json) and ROOT/w3_best_vs_last.txt.
Finished runs are skipped on a rerun, so it can be restarted after a disconnect.
"""
import argparse
import dataclasses
import json
import re
from pathlib import Path

import pandas as pd
import torch

from config import MODALITIES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import summarize
from predict import evaluate_model
from unet import UNet

CFG = ("base", "cos", "cosfilm")


def ms(x):
    x = x.dropna()
    if not len(x):
        return "n/a"
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f} (n={len(x)})" if len(x) > 1 else f"{x.mean():.4f} (n=1)"


def score_run(run, cfg, val_ds, device):
    out = run / "eval_last"
    if (out / "eval_info.json").exists():
        return json.load(open(out / "eval_info.json"))
    ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
    n_mod = (len(MODALITIES) + 1) if ck["cfg"].get("modality_mode") == "film" else 0
    model = UNet(base=int(ck["cfg"]["base_channels"]), n_mod=n_mod).to(device)
    model.load_state_dict(ck["model"])
    df, sec = evaluate_model(model, val_ds, cfg, device)
    t = summarize(df)
    info = {"epoch": ck["epoch"], "macro_ssim": float(t.attrs["macro_ssim"]), "macro_pcc": float(t.attrs["macro_pcc"]),
            "inference_seconds_per_image": sec}
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_image_scores.csv", index=False)
    t.to_csv(out / "summary.csv", index=False)
    if n_mod:
        du, _ = evaluate_model(model, val_ds, cfg, device, label="unknown")
        tu = summarize(du)
        du.to_csv(out / "per_image_scores_unknown_modality.csv", index=False)
        info.update(macro_ssim_unknown_modality=float(tu.attrs["macro_ssim"]),
                    macro_pcc_unknown_modality=float(tu.attrs["macro_pcc"]))
    json.dump(info, open(out / "eval_info.json", "w"), indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--pairs-csv", default=None, help="default: the path in config.py")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        raise SystemExit(f"{root} not found. Is Google Drive mounted?")
    over = {"prepared_dir": Path(a.prepared_dir)}
    if a.pairs_csv:
        over["pairs_csv"] = Path(a.pairs_csv)
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = LMC2Dataset(cfg, "val", train=False)
    rows = []
    for run in sorted(root.glob("*_s*")):
        m = re.fullmatch(r"(\w+)_s(\d+)", run.name)
        if not m or not (run / "run_info.json").exists() or not (run / "last.pt").exists():
            continue
        best = json.load(open(run / "run_info.json"))
        print("scoring", run.name, flush=True)
        last = score_run(run, cfg, val_ds, device)
        rows.append({"config": m.group(1), "seed": int(m.group(2)), "best_epoch": best["best_epoch"],
                     "best_ssim": best["macro_ssim"], "best_pcc": best["macro_pcc"], "last_epoch": last["epoch"],
                     "last_ssim": last["macro_ssim"], "last_pcc": last["macro_pcc"],
                     "last_ssim_unknown": last.get("macro_ssim_unknown_modality"),
                     "last_pcc_unknown": last.get("macro_pcc_unknown_modality")})
    df = pd.DataFrame(rows).sort_values(["config", "seed"])
    df.to_csv(root / "w3_best_vs_last.csv", index=False)
    cfgs = [c for c in CFG if (df.config == c).any()]
    L = ["Every run (full validation split, macro over organelles)", df.round(4).to_string(index=False), "",
         "Per configuration, mean ± std over seeds"]
    L.append(pd.DataFrame({c: {k: ms(df[df.config == c][k]) for k in
                                ("best_ssim", "best_pcc", "last_ssim", "last_pcc", "last_ssim_unknown", "last_pcc_unknown")}
                           for c in cfgs}).T.to_string())
    L.append("")
    p = df.pivot(index="seed", columns="config", values=["last_ssim", "last_pcc"])
    for x, y in (("cos", "base"), ("cosfilm", "cos"), ("cosfilm", "base")):
        for k in ("last_ssim", "last_pcc"):
            if (k, x) in p and (k, y) in p:
                d = (p[(k, x)] - p[(k, y)]).dropna()
                if len(d):
                    L.append(f"{x} minus {y}, {k}: " + ", ".join(f"seed {s}: {v:+.4f}" for s, v in d.items())
                             + f" | mean {d.mean():+.4f}, gains {(d > 0).sum()} of {len(d)}")
    text = "\n".join(L)
    (root / "w3_best_vs_last.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
