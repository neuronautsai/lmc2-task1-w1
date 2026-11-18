## SSIM term in the loss (review item 5), final result over seeds 42 and 123

**Status:** done with two seeds. Seed 456 was not trained (reason under Reading).

**Setup:** as in the interim comment. Loss = (1 - w) L1 + w (1 - SSIM) with w = 0.84, single-scale SSIM computed like the local proxy, all other settings copied from the Week 3 `base` checkpoint of the same seed, `pairs_excluded.csv`, 50 epochs. Code: commit `56ddb83`. The base runs of both seeds were rescored with the same pairs file.

Mean over seeds 42 and 123, full local validation split (532 images, 1189 image and organelle rows), macro over organelles. Arrows give the better direction.

| Metric | base `best.pt` | SSIM term `best.pt` | base `last.pt` | SSIM term `last.pt` |
|---|---|---|---|---|
| Official-style SSIM, data range 1 ↑ | 0.4190 | 0.4170 | 0.4041 | 0.3863 |
| Official-style SSIM, joint range ↑ | 0.5239 | 0.5185 | 0.5101 | 0.4837 |
| Official-style PCC ↑ | 0.5800 | 0.5480 | 0.5883 | 0.5360 |
| MAE ↓ | 0.1738 | 0.1780 | 0.1751 | 0.1938 |
| Cosine distance ↓ | 0.2615 | 0.2818 | 0.2593 | 0.2896 |
| Euclidean distance ↓ | 425.1 | 457.8 | 423.8 | 467.7 |
| Proxy SSIM | 0.4958 | 0.5206 | 0.4807 | 0.4928 |
| Proxy PCC | 0.5920 | 0.5607 | 0.6005 | 0.5483 |

Selected `best.pt` epochs: base 48 (seed 42) and 46 (seed 123), SSIM term 27 (seed 42) and 37 (seed 123).

Difference SSIM term minus base, per seed:

| Metric | seed 42 `best.pt` | seed 123 `best.pt` | seed 42 `last.pt` | seed 123 `last.pt` |
|---|---|---|---|---|
| Official-style SSIM, data range 1 ↑ | -0.0128 | +0.0088 | -0.0158 | -0.0196 |
| Official-style SSIM, joint range ↑ | -0.0089 | -0.0019 | -0.0209 | -0.0320 |
| Official-style PCC ↑ | -0.0535 | -0.0106 | -0.0764 | -0.0282 |
| MAE ↓ | +0.0001 | +0.0082 | +0.0233 | +0.0141 |
| Cosine distance ↓ | +0.0338 | +0.0067 | +0.0435 | +0.0172 |
| Euclidean distance ↓ | +25.5 | +39.8 | +44.5 | +43.2 |

Official-style PCC per organelle, `best.pt`:

| Organelle | n | seed 42 base | seed 42 SSIM term | seed 123 base | seed 123 SSIM term |
|---|---|---|---|---|---|
| Nucleus | 522 | 0.6587 | 0.4408 | 0.5988 | 0.4670 |
| Mitochondria | 326 | 0.6199 | 0.5933 | 0.6023 | 0.5978 |
| Tubulin | 128 | 0.6217 | 0.6420 | 0.5614 | 0.6477 |
| Actin | 213 | 0.4859 | 0.4963 | 0.4914 | 0.4989 |

Official-style SSIM (data range 1) per organelle, seed 123 `best.pt`, base to SSIM term: Nucleus 0.5488 to 0.4385, Mitochondria 0.4866 to 0.5094, Tubulin 0.2558 to 0.3272, Actin 0.3136 to 0.3649.

**Check without the checkpoint-selection images:** `best.pt` is chosen on a fixed 100-image validation subset (numpy `default_rng(42)`, as in `train.py`), and the tables above include those images. Without them (432 images), macro official-style SSIM and PCC of `best.pt`: seed 42 base 0.4356 and 0.5927, SSIM term 0.4207 and 0.5406. Seed 123 base 0.3998 and 0.5610, SSIM term 0.4057 and 0.5509. Removing the subset lowers SSIM by 0.0033 to 0.0045 for the SSIM-term runs and by 0.0004 to 0.0017 for base, so the selection is slightly optimistic for the SSIM term. The direction of every comparison stays the same.

**Reading**
- PCC, MAE, cosine and Euclidean distance are worse with the SSIM term in all four comparisons (MAE equal for seed 42 `best.pt`). Joint-range SSIM is lower in all four.
- Data-range-1 SSIM is higher only for seed 123 `best.pt` (+0.009). This is smaller than the seed spread of base itself (0.4368 for seed 42 against 0.4012 for seed 123).
- The per-organelle pattern repeats in both seeds: Nucleus PCC drops strongly (-0.218 and -0.132), Tubulin PCC goes up (+0.020 and +0.086) and Actin PCC slightly (+0.010 and +0.008). For seed 123, the SSIM of Mitochondria, Tubulin and Actin goes up while Nucleus SSIM drops by 0.110.
- Proxy SSIM goes up in all four comparisons while the official-style metrics go down, the same proxy and official-style divergence as in item 4.
- **Decision:** the SSIM term at w = 0.84 is not adopted. The `base` setting stays.
- Seed 456 was not trained: PCC and the three error metrics are worse in all four comparisons, so a third seed could not make this setting adoptable.
- Limits: two seeds, one weight, single-scale SSIM. A negative result at w = 0.84 does not rule out an SSIM term in general. The per-organelle pattern suggests a possible follow-up (SSIM term only for Tubulin and Actin), not tested.

**Commands**
```
python train.py --out W5/ssim84_s123 --config-from W3/base_s123/best.pt --pairs-csv PAIRS_EXCLUDED --prepared-dir /content/prepared --ssim-weight 0.84 --resume
python eval_last.py --root W5 --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED
python score_official.py --checkpoint W5/ssim84_s123/best.pt --out W5/ssim84_s123/official_best --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W5/ssim84_s123/last.pt --out W5/ssim84_s123/official_last --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W3/base_s123/best.pt --out W3/base_s123/official_excl_best --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W3/base_s123/last.pt --out W3/base_s123/official_excl_last --pairs-csv PAIRS_EXCLUDED --resume
```
