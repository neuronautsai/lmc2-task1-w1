## SSIM term in the loss (review item 5), interim result after seed 42

**Status:** seed 42 trained and scored. Seed 123 trained, its official-style scoring is next. Seed 456 not started.

**Setup:** loss = (1 - w) L1 + w (1 - SSIM) with w = 0.84. The value is the L1 and MS-SSIM mixing weight of Zhao et al. 2017 (arXiv 1511.08861), used here as a starting value, not tuned. This run uses single-scale SSIM computed like the local proxy (skimage defaults: 7 x 7 uniform window, data range 1), checked against skimage within 6e-6. All other settings are copied from the Week 3 `base` checkpoint of the same seed (`train.py --config-from`), with `pairs_excluded.csv`, 50 epochs. A second checkpoint `best_mean.pt` is selected on the mean of macro SSIM and PCC of the 100-image subset. `--resume` now refuses a run whose settings differ from the stored ones. Commit `56ddb83`. The seed 42 run was interrupted at epoch 8 and resumed: model and optimizer were restored, but the data order after the resume differs from an uninterrupted run.

Seed 42, full local validation split (532 images, 1189 image and organelle rows), macro over organelles. Arrows give the better direction.

| Metric | base `best.pt` (ep. 48) | SSIM term `best.pt` (ep. 27) | base `last.pt` (ep. 50) | SSIM term `last.pt` (ep. 50) | SSIM term `best_mean.pt` (ep. 38) |
|---|---|---|---|---|---|
| Official-style SSIM, data range 1 ↑ | 0.4368 | 0.4240 | 0.4019 | 0.3861 | 0.4042 |
| Official-style SSIM, joint range ↑ | 0.5371 | 0.5282 | 0.5025 | 0.4816 | 0.5038 |
| Official-style PCC ↑ | 0.5966 | 0.5431 | 0.5973 | 0.5209 | 0.5556 |
| MAE ↓ | 0.1703 | 0.1704 | 0.1798 | 0.2031 | 0.1751 |
| Cosine distance ↓ | 0.2506 | 0.2844 | 0.2544 | 0.2979 | 0.2793 |
| Euclidean distance ↓ | 420.0 | 445.5 | 433.7 | 478.2 | 444.5 |
| Proxy SSIM | 0.5043 | 0.5200 | 0.4777 | 0.4918 | 0.5110 |
| Proxy PCC | 0.6088 | 0.5558 | 0.6099 | 0.5329 | 0.5688 |

Official-style PCC per organelle, `best.pt`: Nucleus 0.6587 to 0.4408, Mitochondria 0.6199 to 0.5933, Tubulin 0.6217 to 0.6420, Actin 0.4859 to 0.4963.

**Reading**
- `last.pt`: all six official-style metrics are worse than base. `best.pt`: all worse except MAE, which is equal. `best_mean.pt` is also below base `best.pt` on all six, though base has no `best_mean.pt` to compare like with like.
- The largest change is Nucleus PCC (-0.218 for `best.pt`). Tubulin and Actin PCC go up slightly.
- The proxy SSIM goes up (+0.016 for `best.pt`, +0.014 for `last.pt`) while the official-style SSIM goes down. One possible reason, not tested: the SSIM term is computed on the unnormalized prediction with data range 1, like the proxy, so it also rewards intensity calibration, which the official-style scorer removes by normalizing the prediction.
- Seed 123, proxy only so far (`best.pt`, epoch 37): macro SSIM 0.5212, macro PCC 0.5656, Nucleus PCC 0.4774. The low Nucleus PCC repeats.
- Limits: one seed on the official-style scorer and one weight. A negative result at w = 0.84 does not rule out an SSIM term in general.

**Next:** official-style scoring of seed 123 and of base seed 123 with the same pairs file, then the decision on seed 456 and the final item 5 comment.

**Commands**
```
python train.py --out W5/ssim84_s42 --config-from W3/base_s42/best.pt --pairs-csv PAIRS_EXCLUDED --prepared-dir /content/prepared --ssim-weight 0.84 --resume
python eval_last.py --root W5 --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED
python score_official.py --checkpoint W5/ssim84_s42/best.pt --out W5/ssim84_s42/official_best --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W5/ssim84_s42/last.pt --out W5/ssim84_s42/official_last --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W5/ssim84_s42/best_mean.pt --out W5/ssim84_s42/official_best_mean --pairs-csv PAIRS_EXCLUDED --resume
```
