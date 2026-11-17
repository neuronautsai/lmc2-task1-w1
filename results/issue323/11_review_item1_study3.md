## Study 3 check and exclusion (review item 1)

Code: `check_duplicates.py` (commit `41e734a`). Every usable image with two or more observed targets (2,064 of 2,725, training and validation) was checked for target channels that are identical after normalization or correlate with r ≥ 0.98 on a 4x subsampled grid. Every flagged pair was then compared on the raw TIFFs.

**Duplicates found:** 4 images, all in Study 3, all Nucleus = Actin, all raw pixel-identical: images 53, 55, 56 and 57. No other study and no other channel pair was flagged.

**Study 3 in the split:** all 8 usable Study 3 images (53 to 60, BF, with Nucleus, Mitochondria and Actin observed) are in the local validation split. None is in the training split, so training was not affected by the duplicates. This answers the open question in the Week 3 inspection comment. Images 54, 58, 59 and 60 have non-duplicated Actin files. As suggested in the review, all Study 3 Actin targets are excluded, not only the four duplicates.

**Exclusion:** `pairs_excluded.csv` marks Study 3 Actin as not observed and is passed with `--pairs-csv`. New counts of observed targets (usable images with at least one observed target):

| Split | Images | Nucleus | Mitochondria | Tubulin | Actin |
|---|---|---|---|---|---|
| Train | 2,193 | 2,077 | 1,048 | 393 | 414 (unchanged) |
| Local validation | 532 | 522 | 326 | 128 | 213 (was 221) |

No image drops out, because every Study 3 image keeps its Nucleus and Mitochondria targets.

**Week 3 scores without Study 3 Actin** (best.pt, recomputed from the existing per-image scores, mean ± std over seeds 42, 123, 456):

| Config | Macro SSIM before | Macro SSIM after | Macro PCC before | Macro PCC after | Actin SSIM after | Actin PCC after |
|---|---|---|---|---|---|---|
| `base` | 0.4947 ± 0.0086 | 0.4970 ± 0.0087 | 0.5946 ± 0.0178 | 0.5953 ± 0.0178 | 0.4452 ± 0.0041 | 0.5063 ± 0.0169 |
| `cos` | 0.4861 ± 0.0033 | 0.4882 ± 0.0033 | 0.5804 ± 0.0053 | 0.5809 ± 0.0053 | 0.4355 ± 0.0010 | 0.4907 ± 0.0145 |
| `cosfilm` | 0.4873 ± 0.0021 | 0.4896 ± 0.0021 | 0.5673 ± 0.0256 | 0.5678 ± 0.0257 | 0.4297 ± 0.0026 | 0.4834 ± 0.0176 |

Macro SSIM rises by about 0.002 for every configuration and the ranking does not change. All further runs and scores use `pairs_excluded.csv`.

Command (from the repo folder):
```
python check_duplicates.py --out OUT/dup_check --exclude 3:Actin --w3-root OUT/w3
```
Outputs: `duplicate_targets.csv`, `study3_images.csv`, `pairs_excluded.csv`, `w3_rescored_excluded.csv`.
