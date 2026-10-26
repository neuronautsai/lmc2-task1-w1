"""Local proxy metrics (not official challenge scores).

SSIM: skimage structural_similarity, data_range=1, default window, on normalized images.
PCC: scipy pearsonr per image and channel, undefined (NaN) when prediction or target is constant, never replaced.
Scores are per image and observed channel, never pooled across images. Missing targets are not scored.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

from config import ORGANELLES


def score_image(pred, target, observed):
    """pred, target: (4, H, W) arrays in [0, 1], observed: (4,) bool. Returns a list of dicts."""
    rows = []
    for k, organelle in enumerate(ORGANELLES):
        if not bool(observed[k]):
            continue
        p = np.clip(np.asarray(pred[k], dtype=np.float64), 0.0, 1.0)
        t = np.asarray(target[k], dtype=np.float64)
        ssim = structural_similarity(t, p, data_range=1.0)
        pcc = np.nan if (p.max() == p.min() or t.max() == t.min()) else pearsonr(t.ravel(), p.ravel())[0]
        rows.append({"organelle": organelle, "ssim": ssim, "pcc": pcc})
    return rows


def summarize(df):
    """Per organelle table: observed targets, defined PCC count, mean and std of SSIM and PCC, plus macro mean."""
    out = []
    for organelle in ORGANELLES:
        d = df[df.organelle == organelle]
        out.append({
            "organelle": organelle,
            "n_observed": len(d),
            "n_defined_pcc": int(d.pcc.notna().sum()),
            "ssim_mean": d.ssim.mean(), "ssim_std": d.ssim.std(),
            "pcc_mean": d.pcc.mean(), "pcc_std": d.pcc.std(),
        })
    t = pd.DataFrame(out)
    t.attrs["macro_ssim"] = t.ssim_mean.mean()
    t.attrs["macro_pcc"] = t.pcc_mean.mean()
    return t


def breakdown(df, by):
    """Mean SSIM and PCC by an extra column (modality, study) and organelle."""
    return df.groupby([by, "organelle"]).agg(n=("ssim", "size"), n_pcc=("pcc", "count"),
                                              ssim=("ssim", "mean"), pcc=("pcc", "mean")).reset_index()
