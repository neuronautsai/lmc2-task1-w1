## Week 2: pretrained encoder, held-out selection, ensemble with TTA, member groups, modality, post-processing and focal plane

Items 1, 2, 3, 5, 7, 8 and 9 of the proposal comment. All runs keep the Week 1 pipeline fixed: targets p1 to p99.5 clipped, Study 3 Actin excluded (`pairs_excluded.csv`), Hann blending with 128 px overlap, frozen split. Checkpoints are selected on held-out training studies (item 5): GroupShuffleSplit with seed 42 over the training groups of `results/split_manifest.csv` holds out groups g2, s16, s30, s56, s66, s74 and s83, that is studies 7, 8, 16, 30, 56, 66, 74 and 83 (205 images). 1,988 images are left for training. Every epoch, 100 of the held-out images are scored with the official-style scorer (mean of macro SSIM and PCC), and the best epoch becomes `best.pt`. The local validation split is only used for the scores below.

Run A (item 2): U-Net decoder on an ImageNet-pretrained ResNet34 (`segmentation_models_pytorch` 0.6.0, `in_channels=1`, so the library sums the first-layer weights over the three colour channels), sigmoid output, encoder learning rate 0.1x the decoder rate (1e-5 and 1e-4), 50 epochs, everything else as the base run. The same configuration was trained with seeds 42, 123 and 456. The reference is the base U-Net trained with the same held-out rule and seed 42 ("Week 1 pipeline" row).

### Week 2 result table

Official-style scorer (p2 to p99.8 normalization of target and prediction, no clipping, SSIM with data range 1), full local validation split (532 images, 1,189 image and organelle rows: Nucleus 522, Mitochondria 326, Tubulin 128, Actin 213), best checkpoint on the held-out training studies, Hann 128 blending, no TTA unless marked.

| Run | Encoder | Pretraining | Seeds | Nucleus SSIM / PCC | Mito SSIM / PCC | Tubulin SSIM / PCC | Actin SSIM / PCC | Macro SSIM | Macro PCC |
|---|---|---|---|---|---|---|---|---|---|
| Week 1 pipeline | U-Net, 32 ch | none | 42 | 0.5807 / 0.6790 | 0.4934 / 0.6198 | 0.2927 / 0.6141 | 0.2941 / 0.4994 | 0.4152 | 0.6030 |
| A | ResNet34 | ImageNet | 42 | 0.6525 / 0.7644 | 0.5085 / 0.6586 | 0.3732 / 0.6482 | 0.3725 / 0.5157 | 0.4767 | 0.6467 |
| B | | masked-image SSL | | not run | | | | | |
| C | | | | not run | | | | | |
| Selected, ensemble | ResNet34 | ImageNet | 42, 123, 456 | 0.6619 / 0.7825 | 0.5133 / 0.6790 | 0.3963 / 0.6953 | 0.3877 / 0.5247 | 0.4898 | 0.6704 |
| Selected, ensemble + 8x TTA | ResNet34 | ImageNet | 42, 123, 456 | 0.6681 / 0.7917 | 0.5256 / 0.6854 | 0.4102 / 0.7011 | 0.3998 / 0.5314 | 0.5009 | 0.6774 |

Seed-level macro scores of the selected configuration, mean ± std over seeds 42, 123 and 456: SSIM 0.4678 ± 0.0113 and PCC 0.6520 ± 0.0075 without TTA, SSIM 0.4844 ± 0.0099 and PCC 0.6661 ± 0.0076 with 8x TTA. For comparison, the three Week 3 base seeds (all training data, proxy selection): SSIM 0.4261 ± 0.0183, PCC 0.5841 ± 0.0171.

All rows, macro over the four organelles:

| Configuration | SSIM | SSIM joint range | PCC | MAE | Cosine dist. | Euclidean dist. |
|---|---|---|---|---|---|---|
| Week 3 base seed 42 | 0.4390 | 0.5389 | 0.5988 | 0.1710 | 0.2490 | 420.2 |
| Week 3 base seed 123 | 0.4052 | 0.5142 | 0.5653 | 0.1739 | 0.2705 | 425.4 |
| Week 3 base seed 456 | 0.4343 | 0.5384 | 0.5882 | 0.1647 | 0.2588 | 419.3 |
| Week 3 base ensemble | 0.4404 | 0.5392 | 0.6076 | 0.1692 | 0.2448 | 420.5 |
| Week 3 base ensemble + 8x TTA ([item 7](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue333/01_item7_ensemble_tta.md)) | 0.4453 | 0.5420 | 0.6116 | 0.1688 | 0.2422 | 420.1 |
| Base, held-out selection, seed 42 | 0.4152 | 0.5196 | 0.6030 | 0.1790 | 0.2455 | 435.4 |
| ResNet34 seed 42 | 0.4767 | 0.5848 | 0.6467 | 0.1394 | 0.2141 | 368.5 |
| ResNet34 seed 123 | 0.4550 | 0.5588 | 0.6607 | 0.1644 | 0.2035 | 424.7 |
| ResNet34 seed 456 | 0.4716 | 0.5788 | 0.6487 | 0.1398 | 0.2097 | 377.3 |
| ResNet34 ensemble | 0.4898 | 0.5938 | 0.6704 | 0.1447 | 0.1972 | 385.0 |
| ResNet34 seed 42 + 8x TTA | 0.4918 | 0.5959 | 0.6620 | 0.1398 | 0.2044 | 368.3 |
| ResNet34 seed 123 + 8x TTA | 0.4732 | 0.5718 | 0.6749 | 0.1653 | 0.1958 | 425.2 |
| ResNet34 seed 456 + 8x TTA | 0.4883 | 0.5906 | 0.6614 | 0.1406 | 0.2015 | 378.5 |
| ResNet34 ensemble + 8x TTA | 0.5009 | 0.6016 | 0.6774 | 0.1456 | 0.1927 | 386.5 |
| Week 3 base and ResNet34 ensembles together (6 models) | 0.4692 | 0.5661 | 0.6663 | 0.1594 | 0.2061 | 403.3 |

Local proxy (p1 to p99.5, clipped, SSIM with data range 1), macro, from the final evaluation of `train.py`:

| Run | Proxy SSIM | Proxy PCC |
|---|---|---|
| Base, held-out selection, seed 42 | 0.4920 | 0.6192 |
| ResNet34 seed 42 | 0.5572 | 0.6576 |
| ResNet34 seed 123 | 0.5615 | 0.6702 |
| ResNet34 seed 456 | 0.5513 | 0.6611 |
| ResNet34 ensemble | 0.5696 | 0.6837 |
| ResNet34 ensemble + 8x TTA | 0.5714 | 0.6908 |

The two ensemble rows of the proxy table come from `score_official.py`, which computes both scorers on the same predictions (macro of the per-organelle proxy means).

| Run | Parameters | Training h | Pretraining h | Inference s/image | Peak GPU memory |
|---|---|---|---|---|---|
| Base, held-out selection | 7,765,508 | 3.43 | none | 0.33 | 3.70 GB |
| A, ResNet34 seed 42 | 24,430,532 | 5.47 (two sessions, the first one slower) | ImageNet weights, no own pretraining | 0.21 | 1.51 GB |
| ResNet34 seed 123 | 24,430,532 | 2.71 | same | 0.22 | 1.41 GB |
| ResNet34 seed 456 | 24,430,532 | 2.80 | same | 0.21 | 1.41 GB |
| Selected ensemble + TTA | 3 x 24,430,532 | | | pending, T4 timing | pending |

Training time and inference time are from `run_info.json` (NVIDIA L4, single model, no TTA, final evaluation of `train.py`). The time per case of the ensemble with TTA on the Grand Challenge hardware (T4, largest training image side 3,000 px) is measured next with `time_inference.py`.

**Reading**
- The pretrained encoder raises every organelle's SSIM and PCC over the base U-Net trained with the same data and selection rule: macro SSIM +0.0615 and PCC +0.0437 for seed 42. The seed-level spread of the ResNet34 runs (±0.0113 SSIM, ±0.0075 PCC) is far smaller than this gap.
- One ResNet34 model (0.4767 / 0.6467) scores above the Week 3 base ensemble with 8x TTA (0.4453 / 0.6116).
- The ResNet34 ensemble adds +0.0220 SSIM and +0.0184 PCC over the mean of the single seeds, and 8x TTA adds another +0.0111 SSIM and +0.0070 PCC. MAE and Euclidean distance of the ensemble lie between the seeds, because seed 123 has a higher MAE (0.1644).
- Against the v0.1.0 base seed 42 (linear blending, official-style SSIM per organelle in the [item 4 record](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue323/15_review_item4_target_rule.md): Nucleus 0.6097, Mitochondria 0.5129, Tubulin 0.2901, Actin 0.3346), the ResNet34 ensemble with TTA is higher on every organelle's SSIM. Per-organelle PCC of v0.1.0 was not recorded. Against base seed 42 with Hann 128 (PCC 0.6643, 0.6222, 0.6224, 0.4865) it is higher on every organelle's PCC as well. Mitochondria SSIM is the closest margin (0.5256 against 0.5129).
- The base run with held-out selection (0.4152 / 0.6030) scores below the Week 3 base seed 42 (0.4390 / 0.5988) on SSIM. It trains on 205 fewer images and selects on another rule, so the Week 3 base rows are not a like-for-like reference for Run A. The "Week 1 pipeline" row is.
- Run A selected epoch 40, seed 123 epoch 45 and seed 456 epoch 47 of 50. The held-out score of Run A was still rising slowly at epoch 50 (block means over 8 epochs from epoch 18: 0.436, 0.443, 0.441, 0.451), with large epoch-to-epoch noise from 100 images of 8 studies.

### Item 3: per-organelle choice of ensemble members

Groups scored on the validation split (best variant by the mean of SSIM and PCC per organelle, chosen on the reported images, so optimistic):

| Organelle | Without TTA | With 8x TTA (ResNet34 seeds only) |
|---|---|---|
| Nucleus | ResNet34 ensemble, 0.7222 | ResNet34 ensemble, 0.7299 |
| Mitochondria | ResNet34 ensemble, 0.5962 | ResNet34 ensemble, 0.6055 |
| Tubulin | ResNet34 ensemble, 0.5458 | ResNet34 ensemble, 0.5557 |
| Actin | base and ResNet34 together, 0.4599 (ResNet34 ensemble 0.4562) | ResNet34 ensemble, 0.4656 |

The ResNet34 ensemble is the best group for three organelles. For Actin, adding the Week 3 base models gains 0.0037 on the mean of SSIM and PCC, chosen on the same images. The base models were trained on the held-out studies, so this choice cannot be checked there. The container supports a member list per organelle (`members.json`), but the selected configuration uses the same three ResNet34 models for every organelle.

### Item 8: modality

Official-style scores of the ensembles without TTA, macro over the organelles observed in each modality (Phase contrast has no Tubulin targets):

| Modality | Images | ResNet34 ensemble SSIM / PCC | Week 3 base ensemble SSIM / PCC |
|---|---|---|---|
| Bright field | 73 | 0.4640 / 0.6736 | 0.4324 / 0.6022 |
| DIC | 238 | 0.4468 / 0.6528 | 0.3893 / 0.5792 |
| Phase contrast | 221 | 0.5748 / 0.6934 | 0.5517 / 0.6406 |

- DIC is the weakest modality and gains the most from the pretrained encoder (+0.0575 SSIM, +0.0736 PCC). Per organelle, DIC is also below Phase contrast for Nucleus (SSIM 0.5755 against 0.7768) and Mitochondria (0.4172 against 0.5497).
- Modality conditioning (FiLM, `--modality film`) did not help the base U-Net in #323 Week 3. A conditioned ResNet34 run has not been trained.

### Item 9: post-processing, chosen on held-out training studies

`postprocess.py` on the ResNet34 ensemble with 8x TTA, 205 images of the held-out studies (not seen in training by these models). Settings swept one at a time per organelle: TV with weight 0.01, 0.03, 0.1, background offset 0.02, 0.05, 0.1, gamma 0.8, 1.25, 1.5. Keep rule of the issue: the highest SSIM among the settings that raise SSIM and do not lower PCC.

| Organelle | Unchanged SSIM / PCC | Chosen setting | Chosen SSIM / PCC |
|---|---|---|---|
| Nucleus | 0.6346 / 0.7806 | TV 0.1 | 0.6359 / 0.7813 |
| Mitochondria | 0.3822 / 0.6094 | gamma 1.25 | 0.4408 / 0.6099 |
| Tubulin | 0.2669 / 0.4815 | TV 0.1 | 0.2702 / 0.4830 |
| Actin | 0.4044 / 0.4222 | TV 0.1 | 0.4108 / 0.4260 |
| Macro | 0.4220 / 0.5734 | | 0.4394 / 0.5750 |

TV on all four organelles (Week 1 table rows, here on the held-out studies, macro of the 4-decimal per-organelle means): weight 0.01 gives 0.4234 / 0.5737, 0.03 gives 0.4239 / 0.5740, 0.1 gives 0.4242 / 0.5744.

- TV changes SSIM by at most +0.0064 per organelle, as the issue expected. Weight 0.1 is the largest value of the grid and was chosen for three organelles.
- Gamma 1.25 on Mitochondria raises its SSIM by 0.0586 at equal PCC. Gamma 1.5 raises it further (0.4764) but lowers PCC (0.6067), so the rule rejects it. Gamma 1.25 would also raise Nucleus SSIM (0.6523) but lowers its PCC (0.7776).
- These choices are made on 205 images of 8 studies. They are applied to the validation split next, and kept only if they hold there.

### Item 1: focal plane

`defocus_eval.py` with ResNet34 seed 42: 120 validation images drawn with seed 0, the transmitted-light slice at best focus + d as input (d in slice index steps of the training stacks), best-focus fluorescence targets. 54 of the 120 images have a slice at every offset. Official-style macro on those 54 images:

| Offset | -8 | -4 | -2 | -1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| SSIM | 0.4448 | 0.4756 | 0.4850 | 0.4854 | 0.4825 | 0.4854 | 0.4765 | 0.4726 | 0.4481 |
| PCC | 0.5642 | 0.5657 | 0.5980 | 0.6037 | 0.6168 | 0.6147 | 0.6079 | 0.5814 | 0.4979 |

- Within ±2 slices of best focus, SSIM stays within 0.006 of its value at best focus and PCC drops by at most 0.0188. At ±4 slices PCC drops by 0.035 to 0.051 and at ±8 slices by 0.053 to 0.119.
- The model tolerates small focus errors and degrades for large ones. Whether the challenge inputs are at best focus is still open. The container now writes the OME-XML z fields and a sharpness value of each input to its log, to be read at the next Mock submission if Grand Challenge shows the log to participants.

### Next

- Time the ResNet34 ensemble with 1, 4 and 8 TTA variants on a T4 at 3,000 px against the 5 minute and 16 GB limits.
- Apply the post-processing chosen on the held-out studies to the validation split.
- Mock submission 3 of 5 before the phase closes on 1 December: ResNet34 ensemble, the TTA setting that fits the time limit, post-processing only if it holds on the validation split, uint8 output. The ResNet34 weights are baked into the model files. Their source, the ImageNet weights that `segmentation_models_pytorch` 0.6.0 downloads for `resnet34` (Hugging Face repository `smp-hub/resnet34.imagenet`), is disclosed on the challenge forum before the final deadline.
- Item 4 (normalized SSIM and PCC terms) and item 6 (EMA, longer training with cosine schedule) on the ResNet34 configuration, as far as GPU time allows before the freeze on 2 December.

### Reproducibility

- Code: trainings with `train.py` from `51432af` to `75744d3` (identical for these options, checked on a synthetic set). Scores and focal plane with `75744d3` (Run A score, `defocus_eval.py`) and `fb9f606` (all other scores, member groups, modality, post-processing). Notebooks: [`notebooks/lmc2_333_step1_runs.ipynb`](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/notebooks/lmc2_333_step1_runs.ipynb), [`notebooks/lmc2_333_step2_scores.ipynb`](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/notebooks/lmc2_333_step2_scores.ipynb).
- Environment (Colab): NVIDIA L4, Python 3.13.16, torch 2.14.1+cu130, CUDA 13.0, cuDNN 92800, numpy 2.4.6, scikit-image 0.25.2, scipy 1.16.3, pandas 2.2.3, segmentation-models-pytorch 0.6.0. The CPU count was not recorded in `run_info.json` and will be added.
- Data: frozen split `results/split_manifest.csv`, `pairs_excluded.csv` (Study 3 Actin excluded), held-out studies listed above (`heldout_studies.json` of each run). Seeds 42, 123, 456.

Commands:

```
python train.py --out W5/smp_r34_s42_v2 --seed 42 --arch smp --encoder resnet34 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python train.py --out W5/smp_r34_s123 --seed 123 --arch smp --encoder resnet34 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python train.py --out W5/smp_r34_s456 --seed 456 --arch smp --encoder resnet34 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python train.py --out W5/base_heldout_s42 --seed 42 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W5/smp_r34_s42_v2/best.pt W5/smp_r34_s123/best.pt W5/smp_r34_s456/best.pt W5/base_heldout_s42/best.pt W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --groups smp3=0,1,2 base3=4,5,6 all6=0,1,2,4,5,6 --window hann --overlap 128 --out W5/score_all_hann128 --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint W5/smp_r34_s42_v2/best.pt W5/smp_r34_s123/best.pt W5/smp_r34_s456/best.pt --groups smp3=0,1,2 --window hann --overlap 128 --tta --out W5/score_smp3_hann128_tta --pairs-csv PAIRS_EXCLUDED --resume
python modality_report.py --score-dir W5/score_all_hann128/smp3
python modality_report.py --score-dir W5/score_all_hann128/base3
python postprocess.py --checkpoint W5/smp_r34_s42_v2/best.pt W5/smp_r34_s123/best.pt W5/smp_r34_s456/best.pt --tta --split heldout --heldout-json W5/smp_r34_s42_v2/heldout_studies.json --out W5/postprocess_smp3_tta_heldout --pairs-csv PAIRS_EXCLUDED --resume
python defocus_eval.py --checkpoint W5/smp_r34_s42_v2/best.pt --manifest data_manifest_full.csv --pairs-csv PAIRS_EXCLUDED --out W5/defocus_smp_r34_s42
```
