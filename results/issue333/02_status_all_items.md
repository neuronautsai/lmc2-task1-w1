## Status of every item, with the earlier Week 1 results collected here

The Week 1 results of this task were posted on #323 before the move. They are collected below so that this issue holds the whole record, and from now on every result and code change is posted here. Each earlier result keeps its full comment as a file in the repo, linked in the tables. Records of this issue: [`results/issue333/`](https://github.com/neuronautsai/lmc2-task1-w1/tree/main/results/issue333).

### Week 1 result table

Official-style scorer (p2 to p99.8 normalization of target and prediction, no clipping, SSIM with data range 1), full local validation split with Study 3 Actin excluded (532 images, 1,189 image and organelle rows), macro over the four organelles. Rows are cumulative unless marked.

| Configuration | SSIM | PCC | MAE | Cosine dist. | Euclidean dist. | Inference s/image | Record |
|---|---|---|---|---|---|---|---|
| base seed 42, v0.1.0 pipeline (linear blending, 64 px) | 0.4368 | 0.5966 | 0.1703 | 0.2506 | 420.0 | not measured | [item 4](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/15_review_item4_target_rule.md) |
| + uint16 output mapping | unchanged, see note | unchanged, see note | see note | unchanged, see note | see note | not measured | [offline](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/12_review_item2_output_scale_offline.md), [Mock](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/13_review_item2_output_scale_mock.md) |
| + Hann blending, overlap 64 | not run | | | | | | |
| + Hann blending, overlap 128 | 0.4390 | 0.5988 | 0.1710 | 0.2490 | 420.2 | not measured | [ensemble](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/19_ensemble_base_seeds.md) |
| p2 to p99.8 unclipped targets (separate run, linear head, linear 64 blending) | 0.4126 | 0.5907 | 0.1669 | 0.2571 | 413.6 | not measured | [item 4](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/15_review_item4_target_rule.md) |
| 3-seed ensemble (Hann 128) | 0.4404 | 0.6076 | 0.1692 | 0.2448 | 420.5 | not measured | [ensemble](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/19_ensemble_base_seeds.md) |
| 3-seed ensemble + 8x TTA | 0.4453 | 0.6116 | 0.1688 | 0.2422 | 420.1 | not measured | [item 7](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue333/01_item7_ensemble_tta.md) |
| best above + TV, λ = 0.01, 0.03, 0.1 | pending | | | | | | code ready, see item 9 |

Notes:
- Output mapping: the rescaling does not change official-style SSIM, PCC and cosine distance (largest per-image change 1.5e-7 for float32 and 1.8e-2 for uint16, from rounding, on the first 20 images). On the raw target scale, the macro MAE goes from 1,524.6 to 1,311.9 and the macro Euclidean distance from 4,472,187 to 3,951,777. On the Mock phase the float32 version scored worse (table below), so the container is back on uint8.
- Hann with 64 px overlap was not run. At 128 px the window shape changed the local proxy macro SSIM by at most 0.0001, and Hann had the lowest seam index (1.07 against 1.45 for the linear ramp).
- Inference time per image is measured next with `time_inference.py` on a T4, at the largest training image side (3,000 px), for 1, 4 and 8 TTA variants.
- Study 3: 4 Actin targets are pixel-identical to Nucleus (images 53, 55, 56, 57). All 8 usable Study 3 images are in the local validation split, and all Study 3 Actin targets are excluded. Valid counts: train 2,077 / 1,048 / 393 / 414 and validation 522 / 326 / 128 / 213 (Nucleus, Mitochondria, Tubulin, Actin). [Record](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/11_review_item1_study3.md).
- Leakage check: no study, image id or identical array in both splits, and 3 near-duplicate candidates turned out to be different images. The 100 checkpoint-selection images make the validation scores optimistic by at most 0.0045 SSIM and 0.0068 PCC. [Record](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/18_leakage_check.md).

### Mock submissions

Portal macro is the mean over the four organelles of the per-organelle portal values. The Mock phase scores 2 cases.

| # | Submission ID | Tag / commit | Changes vs v0.1.0 | Portal macro SSIM | Portal macro PCC | Rank | Quota left |
|---|---|---|---|---|---|---|---|
| 1 | `67c619bd-bee7-44c2-af3b-85c3476ce62b` | `v0.1.0` / `c53165b` | none | 0.3508 | 0.3612 | 19 | 4 of 5 |
| 2 | `4f010e6d-d855-40e8-94b1-db433d9e580b` | `v0.2.0` / `ada3abc` | float32 output in the target range | 0.3601 | 0.3350 | 25 | 3 of 5 |

### Change summary so far

| Change | Macro SSIM delta | Macro PCC delta | Kept? |
|---|---|---|---|
| Output mapping (float32) | 0 locally, Mock worse | 0 locally, Mock worse | no, uint8 |
| Hann blending, 128 px (seed 42) | +0.0022 | +0.0022 | yes |
| Unclipped normalization, p2 to p99.8 (best.pt) | -0.0242 | -0.0059 | no |
| SSIM loss, w = 0.84 (mean over seeds 42 and 123, best.pt) | -0.0020 | -0.0320 | no ([record](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/17_review_item5_ssim_loss_final.md)) |
| 3-seed ensemble (against seed 42) | +0.0014 | +0.0087 | yes |
| 8x TTA on the ensemble | +0.0050 | +0.0041 | pending the time check |
| TV post-processing | pending | pending | |
| Pretrained encoder | running | running | |
| SSL encoder | code ready | | |

### The 13 proposed items

| # | Item | Status | Code |
|---|---|---|---|
| 1 | Focal plane of the challenge inputs | open | `defocus_eval.py` (`9629a6f`), container log of the OME z fields and a sharpness value per input (`8b95bf2`) |
| 2 | Pretrained encoder | Run A training now: ResNet34, ImageNet weights, encoder learning rate 0.1x, seed 42, held-out selection | `train.py --arch smp` (`51432af`), Run B `ssl_pretrain.py` and `--ssl-checkpoint` (`8b95bf2`) |
| 3 | Per-organelle selection of checkpoints and members | run queued (base seeds and SSIM-term seeds) | `score_official.py --groups` (`51432af`), `--split heldout` and container `members.json` (`8b95bf2`) |
| 4 | Loss aligned with the official metric | runs after item 2 | `--pcc-weight`, `--ssim-norm`, `--loss-term-organelles` (`51432af`) |
| 5 | Checkpoint selection on held-out training studies | in use by Run A and by a base run with the same rule | `--select-on heldout` (`51432af`) |
| 6 | EMA, longer training, cosine schedule | applied to the item 2 configuration after Run A | `--ema-decay` (`51432af`), `--epochs`, `--lr-schedule` |
| 7 | Ensemble with TTA and time check | scored ([record](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue333/01_item7_ensemble_tta.md)), time check next | container ensemble and `LMC2_TTA` (`b262271`), `time_inference.py` (`ce63016`) |
| 8 | Modality conditioning | per-modality report of the ensemble next | `modality_report.py` (`8b95bf2`), `--modality film` from #323 Week 3 |
| 9 | Per-organelle post-processing | sweep queued on the ensemble | `postprocess.py` and container `postprocess.json` (`8b95bf2`) |
| 10 | Patch size study | after items 2 to 4 | `--patch-size 128` |
| 11 | BM3D input channel | BM3D time per 2048 x 2048 image measured first | `bm3d_input.py`, `--extra-input bm3d` (`8b95bf2`) |
| 12 | Literature review | done | [`results/literature_review.md`](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/literature_review.md) |
| 13 | Mock submission before 1 December | after the time check, uint8 output | container `v0.1.0` scheme, CI on every push |

Item 1 so far: training uses the best-focus transmitted-light slice. The first edition defined its metrics "for each (5) deviations of the focus plane" ([Zenodo 10687569](https://zenodo.org/records/10687569)). The Light My Cells 2 pages I read do not state the focal plane of the test inputs. `defocus_eval.py` scores the validation images with the slice at best focus + d (d from -8 to +8) to show how sensitive the model is. Whether Grand Challenge shows the container log to participants, and so the z fields of the Mock inputs, will be checked at the next Mock submission.

The new code was tested on a synthetic set only: default training reproduces the earlier history exactly, SSL pretraining resumes, SSL and BM3D fine-tuning run, and the container with `members.json`, `postprocess.json` and an smp model passes CI (`4ed5304`). No new scores come from it yet.

### Commands queued on Colab

```
python time_inference.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --size 3000 --tta 1 4 8
python train.py --out W5/smp_r34_s42_v2 --seed 42 --arch smp --encoder resnet34 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python train.py --out W5/base_heldout_s42 --seed 42 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python defocus_eval.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --manifest data_manifest_full.csv --pairs-csv PAIRS_EXCLUDED --out W5/defocus_base3
python score_official.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt W5/ssim84_s123/best.pt W5/ssim84_s42/best.pt --window hann --overlap 128 --groups base3=0,1,2 ssim2=3,4 all5=0,1,2,3,4 --out W5/groups --pairs-csv PAIRS_EXCLUDED
```
