## Training targets p2 to p99.8 without clipping (review item 4)

**Result in one line:** no consistent gain from one seed. On the official-style scorer the run scores lower than the Week 3 base for `best.pt` and about the same for `last.pt`. The proxy shows an SSIM gain that the official-style scores do not confirm. My proposal is to keep the declared target rule for the method change (item 5).

**Why the output head had to change:** with a sigmoid output and an L1 loss, a target above 1 gives the same gradient as a target of exactly 1, and a target below 0 the same as 0, for every output in (0, 1). Removing the clipping alone would therefore not change training. To let the model learn the bright tail, this run uses a linear output head. So the run changes two things together: the target rule (p2 to p99.8, not clipped) and the output head. The run closest to "keep everything else fixed" (sigmoid head with p2 to p99.8 targets, which equals clipped p2 to p99.8 targets) was not done. If the head change is what hurts, only that run would show it.

**Setup:** all other settings copied from the Week 3 `base` seed 42 checkpoint (`train.py --config-from`). Inputs and the validation targets used for checkpoint selection and the proxy keep the declared rule (p1 to p99.5, clipped). The pairs file is `pairs_excluded.csv` (Study 3 Actin excluded). Study 3 is entirely in the validation split, so the training data are the same as for the Week 3 base: 2193 training images, 532 validation images, 1189 image and organelle rows. 50 epochs in one session (the later `--resume` only ran the final evaluation). `best.pt` is epoch 45, selected on proxy SSIM of the 100-image subset as in Week 3, and `last.pt` is epoch 50. Training time 4.2 h. Code at commit `bc56abe`. `run_info.json` now records package and CUDA versions.

**Proxy bias:** a perfect prediction of the new training target, scored by the proxy against the declared-rule target, reaches macro SSIM 0.9711 and macro PCC 0.9975 (`proxy_ceiling.py`, all 532 images). Yet the trained model gains proxy SSIM over the base (table below) while its official-style SSIM goes down. One possible reason, not tested: the proxy clips the prediction to [0, 1] and compares it without normalization, so it also rewards intensity calibration, and the two target rules calibrate differently. The official-style scorer normalizes the prediction p2 to p99.8 itself (affine invariance checked, maximum change below 1e-13), so it is the fair comparison here.

Macro over organelles, full local validation split, official-style scorer (`score_official.py`). Week 3 `base` seed 42 rescored with the same pairs file. Arrows give the better direction.

| Metric | base `best.pt` (ep. 48) | p2 linear `best.pt` (ep. 45) | base `last.pt` (ep. 50) | p2 linear `last.pt` (ep. 50) |
|---|---|---|---|---|
| SSIM, data range 1 ↑ | 0.4368 | 0.4126 | 0.4019 | 0.3994 |
| SSIM, joint data range ↑ | 0.5371 | 0.5191 | 0.5025 | 0.5071 |
| PCC ↑ | 0.5966 | 0.5907 | 0.5973 | 0.6028 |
| MAE ↓ | 0.1703 | 0.1669 | 0.1798 | 0.1725 |
| Cosine distance ↓ | 0.2506 | 0.2571 | 0.2544 | 0.2474 |
| Euclidean distance ↓ | 420.0 | 413.6 | 433.7 | 424.0 |
| Proxy SSIM | 0.5043 | 0.5313 | 0.4777 | 0.5152 |
| Proxy PCC | 0.6088 | 0.6028 | 0.6099 | 0.6153 |

Official-style SSIM (data range 1) per organelle:

| Organelle | n | base `best.pt` | p2 linear `best.pt` | base `last.pt` | p2 linear `last.pt` |
|---|---|---|---|---|---|
| Nucleus | 522 | 0.6097 | 0.5815 | 0.5714 | 0.5661 |
| Mitochondria | 326 | 0.5129 | 0.4850 | 0.4645 | 0.4319 |
| Tubulin | 128 | 0.2901 | 0.2792 | 0.2580 | 0.2970 |
| Actin | 213 | 0.3346 | 0.3047 | 0.3138 | 0.3025 |

**Reading**
- `best.pt`: official-style SSIM is lower for every organelle (macro -0.024 with data range 1, -0.018 with the joint range) and PCC is lower by 0.006. MAE and Euclidean distance are slightly lower, cosine distance slightly higher. The four organelles come from the same model, so they are not four independent confirmations.
- `last.pt`: the macro values are close (SSIM -0.003 with data range 1, +0.005 with the joint range, PCC +0.006, MAE, cosine and Euclidean distance a little better), but the organelles move in opposite directions: Tubulin +0.039 and Mitochondria -0.033. The joint data range is set by the minimum and maximum, so outliers of the unbounded output widen it and can raise this SSIM variant, which may explain its sign.
- The proxy SSIM gain (+0.027 for `best.pt`, +0.038 for `last.pt`) is 3 to 4 times the Week 3 proxy seed spread of `base` (±0.0086), but the official-style scores do not follow it. The proxy is not a valid way to compare runs trained with different target rules, and checkpoint selection on the proxy is affected in the same way, so `best.pt` of this run may not be its best epoch on the official-style scorer.
- Single seed. The Week 3 spread above is a proxy number for single runs (3 seeds, before the Study 3 exclusion). The spread of a difference between two runs is larger, and the seed spread of the official-style scores was not measured, so none of the official-style differences can be called significant.
- The scored output is the raw network output. The container clips the output to [0, 1] before writing it (`inference.py`), so a submitted linear-head model would lose the bright tail this run is about. Submitting it would need a change in `inference.py`.

**Proposal:** keep the declared target rule and the sigmoid head for item 5, since this run gives no reason to change them. The sigmoid run with p2 to p99.8 targets is still open if you want the effect of the target rule alone (one more run, about 4.2 h).

**Commands**
```
python prepare_dataset.py build --out /content/prepared_p2 --splits train --pairs-csv PAIRS_EXCLUDED --target-p-low 2 --target-p-high 99.8 --no-target-clip --shards-dir SHARDS
python train.py --out W4/p2lin_s42 --config-from W3/base_s42/best.pt --pairs-csv PAIRS_EXCLUDED --prepared-dir VAL_PREPARED --train-prepared-dir /content/prepared_p2 --target-p-low 2 --target-p-high 99.8 --no-target-clip --output-activation linear --resume
python proxy_ceiling.py --out W4/proxy_ceiling --pairs-csv PAIRS_EXCLUDED --no-target-clip
python eval_last.py --root W4 --prepared-dir VAL_PREPARED --pairs-csv PAIRS_EXCLUDED
python score_official.py --checkpoint RUN/best.pt --out RUN/official_best --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint RUN/last.pt --out RUN/official_last --pairs-csv PAIRS_EXCLUDED --resume
```
