"""Official-style metric proxy for LMC2 (not official scores).

Follows what the organizers stated in the forum topic "Evaluation metrics and normalization" (read on 8 November 2026):
ground truth and prediction are each percentile normalized on their own, in float, with pmin 2 and pmax 99.8,
(image - p_low) / (p_high - p_low), no clipping and no uint16 cast. A constant image (p_low == p_high) is returned unchanged.
Metrics named there: PCC, SSIM, MAE, cosine distance, Euclidean distance, all computed per image and organelle.

Not published, so these are assumptions and are labelled as such in the output columns:
- SSIM data_range, window, K1, K2. Two data_range variants are reported: 1.0 (ssim_dr1) and the joint range of the
  normalized ground truth and prediction (ssim_drjoint). Window and K1/K2 are the skimage defaults.
- Exact definitions of MAE, cosine distance and Euclidean distance. Used here: mean absolute difference, 1 minus cosine
  similarity of the flattened images, and the L2 norm of the difference of the flattened images.
The ground truth must be the raw image. Scoring an already clipped and rescaled target would not match the official input.
"""
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

P_MIN, P_MAX = 2.0, 99.8
METRIC_COLUMNS = ("off_pcc", "off_ssim_dr1", "off_ssim_drjoint", "off_mae", "off_cos", "off_euc")


def percentile_normalization(image, pmin=P_MIN, pmax=P_MAX):
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [pmin, pmax])
    if low == high:
        return image
    return (image - low) / (high - low)


def official_metrics(gt_raw, pred):
    """gt_raw, pred: 2D arrays of the same shape. Returns a dict of the metrics above (NaN where undefined)."""
    gt_raw, pred = np.asarray(gt_raw), np.asarray(pred)
    assert gt_raw.shape == pred.shape, f"shape mismatch {gt_raw.shape} vs {pred.shape}"
    g, p = percentile_normalization(gt_raw), percentile_normalization(pred)
    gf, pf = g.ravel(), p.ravel()
    pcc = np.nan if (gf.min() == gf.max() or pf.min() == pf.max()) else pearsonr(gf, pf)[0]
    joint = max(g.max(), p.max()) - min(g.min(), p.min())
    ssim_dr1 = structural_similarity(g, p, data_range=1.0)
    ssim_drjoint = structural_similarity(g, p, data_range=joint) if joint > 0 else np.nan
    ng, npred = np.linalg.norm(gf), np.linalg.norm(pf)
    cos = np.nan if (ng == 0 or npred == 0) else 1.0 - float(gf @ pf) / (ng * npred)
    return {"off_pcc": pcc, "off_ssim_dr1": ssim_dr1, "off_ssim_drjoint": ssim_drjoint,
            "off_mae": float(np.abs(gf - pf).mean()), "off_cos": cos, "off_euc": float(np.linalg.norm(gf - pf))}
