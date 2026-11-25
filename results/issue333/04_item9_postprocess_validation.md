## Item 9: post-processing chosen on the held-out studies, checked on the validation split

The settings chosen on the held-out training studies in the [Week 2 record](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/results/issue333/03_week2_encoder_ensemble.md) (TV 0.1 for Nucleus, Tubulin and Actin, gamma 1.25 for Mitochondria) were applied to the full local validation split. Model: the ResNet34 ensemble (seeds 42, 123, 456) with 8x TTA and Hann 128 blending. Only the unchanged prediction and these two settings were scored, each on every organelle. Official-style scorer (p2 to p99.8, SSIM with data range 1), 532 images, 1,189 image and organelle rows (Nucleus 522, Mitochondria 326, Tubulin 128, Actin 213).

A setting is kept if it passes the rule of the issue on both splits: SSIM goes up and PCC does not go down.

### Per organelle, SSIM / PCC

| Organelle | Unchanged | TV 0.1 | Gamma 1.25 | Chosen on held-out | Holds on validation |
|---|---|---|---|---|---|
| Nucleus | 0.6681 / 0.7917 | 0.6674 / 0.7918 | 0.7052 / 0.7887 | TV 0.1 | no, SSIM -0.0006 |
| Mitochondria | 0.5256 / 0.6854 | 0.5258 / 0.6857 | 0.5807 / 0.6890 | gamma 1.25 | yes, SSIM +0.0552, PCC +0.0036 |
| Tubulin | 0.4102 / 0.7011 | 0.4104 / 0.7009 | 0.4670 / 0.7036 | TV 0.1 | no, PCC -0.0001 |
| Actin | 0.3998 / 0.5314 | 0.4040 / 0.5310 | 0.4138 / 0.5282 | TV 0.1 | no, PCC -0.0004 |

### Macro over the four organelles

| Plan | SSIM | SSIM joint range | PCC | MAE | Cosine dist. | Euclidean dist. |
|---|---|---|---|---|---|---|
| Unchanged | 0.5009 | 0.6016 | 0.6774 | 0.1456 | 0.1927 | 386.5 |
| As chosen on held-out (TV 0.1 for N, T, A, gamma 1.25 for M) | 0.5156 | 0.6145 | 0.6782 | 0.1398 | 0.1921 | 375.1 |
| Kept: gamma 1.25 for Mitochondria only | 0.5147 | 0.6150 | 0.6783 | 0.1400 | 0.1921 | 375.6 |
| Not used, chosen on validation (gamma 1.25 for M and T) | 0.5289 | 0.6304 | 0.6789 | 0.1340 | 0.1914 | 366.3 |
| Not used, gamma 1.25 for all four | 0.5417 | 0.6441 | 0.6774 | 0.1269 | 0.1940 | 353.9 |

- Gamma 1.25 on Mitochondria holds: +0.0552 SSIM here against +0.0586 on the held-out studies, with PCC up on both splits. It is kept.
- TV 0.1 does not carry over. On validation it changes SSIM by -0.0006 to +0.0042 and PCC by -0.0004 to +0.0004, and it fails the rule for all three organelles it was chosen for. On the held-out studies its gains were at most +0.0064, and 0.1 was the largest weight of the grid. TV is dropped.
- Gamma 1.25 would also raise Tubulin (+0.0567 SSIM, +0.0025 PCC) on validation. On the held-out studies the rule did not select it for Tubulin, so using it now would be a choice made on the reported images. The same applies to gamma on all four organelles, which leaves macro PCC unchanged but lowers PCC for Nucleus (-0.0029) and Actin (-0.0032). Both rows are listed only for reference.

### Mock submission 3

- Weights: the three ResNet34 checkpoints, `postprocess.json` = `{"Mitochondria": {"gamma": 1.25}}`, no `members.json` (all three models for every organelle), uint8 output.
- TTA: the largest setting that fits the 5 minute limit on a T4 at 3,000 px, from `time_inference.py`. The measurement is running.
- Local validation macro of this configuration with 8x TTA: SSIM 0.5147, PCC 0.6783.

### Reproducibility

- Code `f88950f`, notebook [`notebooks/lmc2_333_step3_postprocess_val.ipynb`](https://github.com/neuronautsai/lmc2-task1-w1/blob/main/notebooks/lmc2_333_step3_postprocess_val.ipynb), NVIDIA L4.
- Command:

```
python postprocess.py --checkpoint W5/smp_r34_s42_v2/best.pt W5/smp_r34_s123/best.pt W5/smp_r34_s456/best.pt --tta --split val --tv 0.1 --offset --gamma 1.25 --out W5/postprocess_smp3_tta_val_check --pairs-csv PAIRS_EXCLUDED --resume
```

`--offset` without values leaves the offset out of the grid.
