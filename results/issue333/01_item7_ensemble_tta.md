## Item 7: ensemble with 8-fold TTA

**Setup:** the three Week 3 `base` checkpoints (seeds 42, 123, 456, `best.pt`), Hann window with 128 px overlap, `pairs_excluded.csv`, full local validation split (532 images, 1189 image and organelle rows). TTA averages each model over the 8 flips and 90-degree rotations of the input, then the ensemble averages the three models. Code: commit `5731dfa`, `score_official.py --tta`. The rows without TTA are the run in the #323 ensemble comment, same tiles and pairs file.

Macro over organelles, official-style scorer (SSIM with data range 1). Arrows give the better direction.

| Configuration | SSIM ↑ | SSIM joint range ↑ | PCC ↑ | MAE ↓ | Cosine dist. ↓ | Euclidean dist. ↓ |
|---|---|---|---|---|---|---|
| seed 42 | 0.4390 | 0.5389 | 0.5988 | 0.1710 | 0.2490 | 420.2 |
| seed 42 + TTA | **0.4459** | 0.5424 | 0.6056 | 0.1715 | 0.2447 | 420.1 |
| seed 123 | 0.4052 | 0.5142 | 0.5653 | 0.1739 | 0.2705 | 425.4 |
| seed 123 + TTA | 0.4137 | 0.5179 | 0.5725 | 0.1764 | 0.2662 | 428.9 |
| seed 456 | 0.4343 | 0.5384 | 0.5882 | **0.1647** | 0.2588 | **419.3** |
| seed 456 + TTA | 0.4453 | **0.5462** | 0.5959 | 0.1652 | 0.2541 | 420.0 |
| 3-seed ensemble | 0.4404 | 0.5392 | 0.6076 | 0.1692 | 0.2448 | 420.5 |
| 3-seed ensemble + TTA | 0.4453 | 0.5420 | **0.6116** | 0.1688 | **0.2422** | 420.1 |

Gain from TTA:

| Gain from TTA | SSIM | SSIM joint | PCC | MAE | Cosine | Euclidean |
|---|---|---|---|---|---|---|
| seed 42 | +0.0069 | +0.0036 | +0.0068 | +0.0005 | -0.0044 | -0.1 |
| seed 123 | +0.0085 | +0.0036 | +0.0072 | +0.0024 | -0.0043 | +3.5 |
| seed 456 | +0.0110 | +0.0078 | +0.0077 | +0.0005 | -0.0047 | +0.7 |
| 3-seed ensemble | +0.0050 | +0.0028 | +0.0041 | -0.0004 | -0.0026 | -0.4 |

Per organelle, official-style SSIM and PCC of the ensemble:

| Organelle | n | ensemble | ensemble + TTA |
|---|---|---|---|
| Nucleus | 522 | 0.5946 and 0.6721 | 0.5982 and 0.6789 |
| Mitochondria | 326 | 0.5242 and 0.6234 | 0.5258 and 0.6252 |
| Tubulin | 128 | 0.2975 and 0.6159 | 0.3035 and 0.6216 |
| Actin | 213 | 0.3451 and 0.5189 | 0.3537 and 0.5208 |

**Reading**
- TTA raises SSIM, joint-range SSIM and PCC and lowers the cosine distance for every single model and for the ensemble. MAE goes up slightly for the single models (+0.0005 to +0.0024) and down slightly for the ensemble.
- With the ensemble, TTA adds less (+0.005 SSIM, +0.004 PCC) than for a single model (+0.007 to +0.011 SSIM), so part of what TTA gives overlaps with what averaging seeds gives.
- The ensemble with TTA improves every organelle's SSIM and PCC over the ensemble without TTA.
- A single model with TTA reaches the SSIM of the ensemble with TTA (seed 42: 0.4459, seed 456: 0.4453, ensemble: 0.4453), but the ensemble keeps the highest PCC and the lowest cosine distance. The best single seed is picked after looking at the results.
- Cost: 8 forward passes per model, so the ensemble with TTA is 24 passes per tile. Inference time per 2048 x 2048 case has not been measured yet. The limit in #333 is 5 minutes per case and 16 GB of RAM, to be confirmed on the final-phase page. The setting for the Mock submission is chosen after that measurement, with 4-fold TTA (flips only) or fewer members as fallbacks.

**Command**
```
python score_official.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --out W6/ens_base_hann128_tta --pairs-csv PAIRS_EXCLUDED --window hann --overlap 128 --tta --resume
```
