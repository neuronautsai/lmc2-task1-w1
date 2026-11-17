## Week 3 results: cosine LR decay and modality-aware conditioning (3 seeds)

**Summary.** Neither change improved on the Week 2 baseline on the local validation split. This holds for both checkpoint rules tested (best subset SSIM and last epoch). The baseline (constant learning rate, no conditioning) has the highest mean macro SSIM and PCC.

### Setup
- Three configurations, each trained with seeds 42, 123 and 456 (9 runs):
  - `base`: Week 2 U-Net, constant learning rate 1e-4.
  - `cos`: same, with cosine learning-rate decay per epoch down to 1% of the base rate.
  - `cosfilm`: `cos` plus modality-aware conditioning (FiLM). A learned embedding of the modality (BF, DIC, PC, plus an unknown token) gives a scale and shift after every encoder and decoder block, zero-initialized. The label is hidden with probability 0.3 during training. 7,862,788 parameters against 7,765,508.
- Everything else as in Week 2: 50 epochs, AdamW, observed-channel L1, batch of 4 images with 4 crops of 256, prepared float16 dataset.
- Evaluation: local proxy (p1 to p99.5 normalization, clipped, SSIM with data_range 1, and PCC) on the full local validation split (532 images: Nucleus 522, Mitochondria 326, Tubulin 128, Actin 221 observed targets). Macro means over the four organelles. Std is over the three seeds.
- Hardware: Google Colab, NVIDIA L4, 12 CPUs. Code: commit `a4bc300` (training options) and `d866508` (`aggregate_w3.py`, `eval_last.py`) in https://github.com/neuronautsai/lmc2-task1-w1.

### Two checkpoint rules
`best.pt` is the Week 2 rule: maximum macro SSIM on a fixed 100-image subset of the validation split. The `cos` runs were selected early (epochs 20, 23, 38, against 41 to 48 for `base`), so I also scored the last-epoch weights (`last.pt`, epoch 50) of every run. `best.pt` numbers are slightly optimistic because the subset is part of the split they are scored on.

| | best.pt SSIM | best.pt PCC | last.pt SSIM | last.pt PCC |
|---|---|---|---|---|
| base | 0.4947 ± 0.0086 | 0.5946 ± 0.0178 | 0.4837 ± 0.0095 | 0.5983 ± 0.0098 |
| cos | 0.4861 ± 0.0033 | 0.5804 ± 0.0053 | 0.4724 ± 0.0076 | 0.5973 ± 0.0111 |
| cosfilm | 0.4873 ± 0.0021 | 0.5673 ± 0.0256 | 0.4772 ± 0.0058 | 0.5806 ± 0.0147 |

Paired differences per seed (42, 123, 456):

| Comparison | Rule | SSIM | PCC |
|---|---|---|---|
| cos minus base | best.pt | -0.0133, -0.0028, -0.0097 (mean -0.0086) | -0.0239, +0.0000, -0.0187 (mean -0.0142) |
| cos minus base | last.pt | +0.0036, -0.0175, -0.0199 (mean -0.0113) | -0.0013, +0.0080, -0.0095 (mean -0.0009) |
| cosfilm minus cos | best.pt | +0.0007, +0.0028, +0.0001 (mean +0.0012) | -0.0081, +0.0129, -0.0439 (mean -0.0131) |
| cosfilm minus cos | last.pt | +0.0049, +0.0105, -0.0010 (mean +0.0048) | -0.0129, -0.0172, -0.0202 (mean -0.0168) |

### Per run
| Run | best epoch | best.pt SSIM | best.pt PCC | last.pt SSIM | last.pt PCC |
|---|---|---|---|---|---|
| base 42 | 48 | 0.5020 | 0.6082 | 0.4753 | 0.6092 |
| base 123 | 46 | 0.4852 | 0.5744 | 0.4817 | 0.5904 |
| base 456 | 41 | 0.4970 | 0.6011 | 0.4941 | 0.5953 |
| cos 42 | 23 | 0.4887 | 0.5844 | 0.4789 | 0.6078 |
| cos 123 | 20 | 0.4824 | 0.5744 | 0.4641 | 0.5984 |
| cos 456 | 38 | 0.4873 | 0.5824 | 0.4742 | 0.5857 |
| cosfilm 42 | 29 | 0.4894 | 0.5762 | 0.4838 | 0.5950 |
| cosfilm 123 | 36 | 0.4852 | 0.5873 | 0.4746 | 0.5812 |
| cosfilm 456 | 20 | 0.4874 | 0.5384 | 0.4731 | 0.5655 |

### Per organelle (best.pt, mean ± std over seeds)
| SSIM | Nucleus | Mitochondria | Tubulin | Actin |
|---|---|---|---|---|
| base | 0.6190 ± 0.0046 | 0.5947 ± 0.0139 | 0.3289 ± 0.0158 | 0.4363 ± 0.0038 |
| cos | 0.6113 ± 0.0130 | 0.5823 ± 0.0121 | 0.3238 ± 0.0049 | 0.4272 ± 0.0008 |
| cosfilm | 0.6158 ± 0.0087 | 0.5543 ± 0.0073 | 0.3584 ± 0.0067 | 0.4208 ± 0.0027 |

| PCC | Nucleus | Mitochondria | Tubulin | Actin |
|---|---|---|---|---|
| base | 0.6375 ± 0.0299 | 0.6308 ± 0.0105 | 0.6065 ± 0.0313 | 0.5035 ± 0.0165 |
| cos | 0.6201 ± 0.0087 | 0.6147 ± 0.0046 | 0.5979 ± 0.0131 | 0.4888 ± 0.0141 |
| cosfilm | 0.5617 ± 0.0541 | 0.6252 ± 0.0058 | 0.6008 ± 0.0266 | 0.4816 ± 0.0172 |

### Modality label withheld (cosfilm)
| | best.pt SSIM | best.pt PCC | last.pt SSIM | last.pt PCC |
|---|---|---|---|---|
| true label | 0.4873 ± 0.0021 | 0.5673 ± 0.0256 | 0.4772 ± 0.0058 | 0.5806 ± 0.0147 |
| label withheld | 0.4869 ± 0.0136 | 0.5522 ± 0.0215 | 0.4695 ± 0.0053 | 0.5644 ± 0.0162 |

### Cost
| | training (h) | inference (s per image) | peak GPU memory (GB) |
|---|---|---|---|
| base | 4.18 ± 0.04 | 0.336 ± 0.003 | 3.70 |
| cos | 4.24 ± 0.03 | 0.341 ± 0.007 | 3.70 |
| cosfilm | 4.44 ± 0.05 | 0.366 ± 0.000 | 4.13 |

Peak memory is the maximum over seeds. Runs resumed after a Colab disconnect only measure the part after the resume (two cosfilm runs report 0.54 GB because only the final evaluation ran after the resume), so the mean is not reported.

### Observations
- Cosine decay did not help under either rule. With last.pt its PCC gap to the baseline closes (mean -0.0009), but SSIM stays lower in two of three seeds.
- FiLM gives no consistent macro gain. Its PCC is lower than `cos` in all three seeds with last.pt. One organelle effect is larger than the seed std: Tubulin SSIM is higher (0.358 against 0.324 for `cos`) and Mitochondria SSIM is lower (0.554 against 0.582).
- Withholding the label lowers PCC in all three seeds, so the model uses the label. Whether the label is available at inference in the challenge is still unanswered in the forum.
- On average over seeds, in all three configurations SSIM falls and PCC rises between the best.pt epoch and epoch 50 (not in every single run). The choice of checkpoint therefore trades one metric against the other, and a rule based on subset SSIM alone favors early epochs.
- Seed variation is of the same size as the differences between configurations (for example, base best.pt macro SSIM ranges from 0.4852 to 0.5020 across seeds).

### Not done yet
- Official-style rescoring (`score_official.py`) of the Week 3 checkpoints.
- Week 3 weights and per-image CSVs are not yet in the shared Drive folder. Update: done, linked in the Week 3 inspection comment.
- Submission container and validation submission. Update: done, see the Week 3 validation submission comment.
