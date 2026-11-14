"""Ceiling of the local proxy for a model trained with another target rule (review item 4).

The local proxy (evaluate.score_image) compares the prediction, clipped to [0, 1], with validation targets normalized by
the declared rule (p1 to p99.5, clipped). A model trained on targets with another rule (for example p2 to p99.8, not
clipped) learns a different intensity mapping, so even a perfect prediction of its own training target does not reach
SSIM 1 and PCC 1 on the proxy. This script scores exactly that perfect prediction: for every validation image and
observed organelle, the raw target normalized with the training rule is scored against the declared-rule target.
The gap to 1 is the bias of the proxy against that rule. The official-style scorer normalizes the prediction itself,
so it has no such bias.

Usage: python proxy_ceiling.py --pairs-csv PAIRS --out OUT [--data-root DIR] [--target-p-low 2 --target-p-high 99.8 --no-target-clip]
"""
import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from config import ORGANELLES, ChallengeConfig
from dataset import LMC2Dataset
from evaluate import score_image, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--target-p-low", type=float, default=2.0)
    ap.add_argument("--target-p-high", type=float, default=99.8)
    ap.add_argument("--no-target-clip", action="store_true")
    a = ap.parse_args()
    over = {k: Path(v) for k, v in (("data_root", a.data_root), ("pairs_csv", a.pairs_csv)) if v}
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    rule = (a.target_p_low, a.target_p_high, not a.no_target_clip)
    ds = LMC2Dataset(cfg, "val", train=False)  # declared rule, raw TIFFs
    rows = []
    for idx in range(len(ds)):
        r = ds.df.iloc[idx]
        item = ds[idx]
        target, observed = item["target"].numpy(), item["observed"].numpy()
        perfect = target.copy()
        for k, o in enumerate(ORGANELLES):
            if observed[k]:
                perfect[k] = ds._read(int(r.study), r[f"{o}_file"], rule)
        for s in score_image(perfect, target, observed):
            s.update(study=int(r.study), image_id=int(r.image_id), modality=r.modality)
            rows.append(s)
        if (idx + 1) % 100 == 0:
            print(f"{idx + 1}/{len(ds)} images", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image_proxy_ceiling.csv", index=False)
    t = summarize(df)
    t.to_csv(out / "proxy_ceiling_summary.csv", index=False)
    info = {"training_rule": {"p_low": rule[0], "p_high": rule[1], "clip": rule[2]},
            "declared_rule": {"p_low": cfg.p_low, "p_high": cfg.p_high, "clip": True},
            "n_images": len(ds), "n_rows": len(df),
            "macro_ssim_ceiling": float(t.attrs["macro_ssim"]), "macro_pcc_ceiling": float(t.attrs["macro_pcc"])}
    json.dump(info, open(out / "proxy_ceiling_info.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
