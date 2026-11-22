"""Official-style SSIM and PCC per transmitted-light modality (#333 item 8, first step), from score_official.py output.

Reads SCORE_DIR/per_image_official_scores.csv (one score_official.py variant folder, for example the ensemble) and prints,
per modality, the number of images, per-organelle means and the macro mean over the organelles observed in that modality.

Usage: python modality_report.py --score-dir W6/ens_base_hann128_tta/ensemble [--out modality_report.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from config import MODALITIES, ORGANELLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.score_dir) / "per_image_official_scores.csv")
    cols = ["off_ssim_dr1", "off_pcc", "off_mae", "off_cos", "off_euc"]
    per = df.groupby(["modality", "organelle"])[cols].mean()
    per["n"] = df.groupby(["modality", "organelle"]).size()
    macro = per[cols].groupby("modality").mean()
    macro["images"] = df.groupby("modality")[["study", "image_id"]].apply(lambda g: len(g.drop_duplicates()))
    macro["organelles"] = per.groupby("modality").size()
    macro = macro.reindex([m for m in MODALITIES if m in macro.index])
    pd.set_option("display.width", 200)
    print("per modality and organelle")
    print(per.reindex([(m, o) for m in MODALITIES for o in ORGANELLES if (m, o) in per.index]).round(4).to_string())
    print("macro over the organelles observed in each modality")
    print(macro.round(4).to_string())
    if a.out:
        per.reset_index().to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
