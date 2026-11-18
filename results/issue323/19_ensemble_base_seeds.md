## Ensemble of the three base seeds (new task: ensemble approaches)

**Setup:** the Week 3 `base` checkpoints of seeds 42, 123 and 456 (`best.pt`, epochs 48, 46 and 41), no retraining. The ensemble is the mean of the three predictions per pixel. Every model and the ensemble are scored in one pass with the same tiles, Hann window with 128 px overlap (review item 3), `pairs_excluded.csv`, full local validation split (532 images, 1189 image and organelle rows). Code: commit `5731dfa` (`score_official.py` takes several checkpoints and writes each model and the ensemble to its own folder).

Macro over organelles, official-style scorer. Arrows give the better direction.

| Metric | seed 42 | seed 123 | seed 456 | mean of the three | ensemble |
|---|---|---|---|---|---|
| SSIM, data range 1 ↑ | 0.4390 | 0.4052 | 0.4343 | 0.4262 | **0.4404** |
| SSIM, joint range ↑ | 0.5389 | 0.5142 | 0.5384 | 0.5305 | **0.5392** |
| PCC ↑ | 0.5988 | 0.5653 | 0.5882 | 0.5841 | **0.6076** |
| MAE ↓ | 0.1710 | 0.1739 | **0.1647** | 0.1699 | 0.1692 |
| Cosine distance ↓ | 0.2490 | 0.2705 | 0.2588 | 0.2594 | **0.2448** |
| Euclidean distance ↓ | 420.2 | 425.4 | **419.3** | 421.6 | 420.5 |

Per organelle, official-style SSIM (data range 1) and PCC:

| Organelle | n | seed 42 | seed 123 | seed 456 | ensemble |
|---|---|---|---|---|---|
| Nucleus | 522 | 0.6126 and 0.6643 | 0.5557 and 0.6064 | 0.5819 and 0.6299 | 0.5946 and **0.6721** |
| Mitochondria | 326 | 0.5185 and 0.6222 | 0.4928 and 0.6035 | **0.5348** and 0.6156 | 0.5242 and **0.6234** |
| Tubulin | 128 | 0.2903 and **0.6224** | 0.2575 and 0.5574 | 0.2930 and 0.5875 | **0.2975** and 0.6159 |
| Actin | 213 | 0.3345 and 0.4865 | 0.3147 and 0.4938 | 0.3274 and **0.5198** | **0.3451** and 0.5189 |

Hann window against the earlier linear blending (64 px), same checkpoints and pairs file, official-style scorer:

| Change, Hann 128 minus linear 64 | SSIM dr1 | SSIM joint | PCC | MAE | Cosine | Euclidean |
|---|---|---|---|---|---|---|
| seed 42 `best.pt` | +0.0022 | +0.0018 | +0.0022 | +0.0007 | -0.0016 | +0.2 |
| seed 123 `best.pt` | +0.0040 | +0.0035 | +0.0018 | -0.0034 | -0.0019 | -4.9 |

**Reading**
- Against the mean of the three single models, the ensemble gains +0.014 SSIM, +0.023 PCC and -0.015 cosine distance, about the size of the spread between seeds.
- Against the best single model (seed 42), the ensemble is better in SSIM (+0.0014), PCC (+0.0087) and cosine distance (-0.0043). Seed 456 keeps the lowest MAE and Euclidean distance. The single model with the best score was picked after looking at the results, so this comparison favours the single model.
- Per organelle, the ensemble has the best Nucleus and Mitochondria PCC and the best Tubulin and Actin SSIM. Nucleus SSIM stays below seed 42 (0.5946 against 0.6126).
- The gain is largest in PCC, which fits averaging smoothing out errors that differ between seeds.
- Hann with 128 px overlap improves SSIM and PCC over the linear blending for both seeds scored before, on the official-style scorer too. This confirms the item 3 result outside the local proxy.
- Limits: one ensemble of three seeds, so there is no spread for the ensemble itself. The three members were selected on the 100-image validation subset. Inference time in the container grows about threefold with three models, and the Grand Challenge time limit has not been checked yet.
- Next: the same ensemble with 8-fold test-time augmentation (flips and 90-degree rotations), already supported by `--tta`.

**Command**
```
python score_official.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --out W6/ens_base_hann128 --pairs-csv PAIRS_EXCLUDED --window hann --overlap 128 --resume
```
