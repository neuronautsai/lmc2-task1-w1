## Week 2 results: Light My Cells 2, Task 1 (Best Focus)

Status: training, full-image tiled evaluation and single-image inference are done for Baseline A (constant control) and Baseline B (four-output U-Net from scratch, seed 42). All numbers are local proxy metrics on the local validation split, not official scores.

### 1. Setup
- **Input rule:** one transmitted-light plane per image at its best-z slice, priority BF, then DIC, then PC (arbitrary but deterministic). 21 images without a defined transmitted best-z slice are excluded, all of them in the training split. Usable inputs: 2,196 train and 532 local validation. 3 train images have no fluorescence target and are dropped from training, which leaves 2,193.
- **Normalization (declared rule):** every image is scaled by its own 1st and 99.5th percentile and clipped to [0, 1]. Targets use the same rule. The local metrics therefore score normalized fluorescence, not absolute intensity.
- **Split:** frozen Week 1 split (GroupShuffleSplit, seed 42, 80/20 by study, merged groups 2/3/4, 7/8 and 11/12/13). Local validation has 19 studies and 532 images.
- **Baseline A:** per-pixel mean of the normalized training targets per organelle, predicted as a constant image for every validation image.
- **Baseline B:** U-Net with 1 input channel and 4 sigmoid outputs, depth 4, 32 base channels, GroupNorm(8), GELU, transposed convolution upsampling, 7,765,508 parameters. Loss: L1 over observed channels only (missing ground-truth channels contribute nothing). AdamW, learning rate 1e-4 constant (no schedule), bf16 autocast, batch of 4 images with 4 random 256x256 crops each, paired random flips and 90-degree rotations, 50 epochs, seed 42.
- **Inference:** full images in 512x512 tiles with 64 px overlap and linear blending, reflect padding to a multiple of 16.
- **Checkpoint rule (fixed before comparing):** maximum macro-average SSIM over the four organelles on a fixed 100-image validation subset (drawn once with seed 42). The selected checkpoint is then scored on the entire local validation split.
- **Metric proxy:** `skimage` SSIM (`data_range=1`, default window) and `scipy` Pearson correlation per image and observed channel, predictions clipped to [0, 1]. Undefined PCC is left as NaN and counted, never replaced.

### 2. Per-organelle results on the full local validation split (532 images)

Baseline A (constant training mean). PCC is undefined for a constant prediction, so it is not reported.

| Organelle | N observed | N defined PCC | SSIM mean ± std | PCC |
|---|---|---|---|---|
| Nucleus | 522 | 0 | 0.316 ± 0.132 | undefined |
| Mitochondria | 326 | 0 | 0.329 ± 0.104 | undefined |
| Tubulin | 128 | 0 | 0.261 ± 0.171 | undefined |
| Actin | 221 | 0 | 0.333 ± 0.089 | undefined |
| **Macro average** | | | **0.310** | undefined |

Baseline B (U-Net, checkpoint of epoch 48).

| Organelle | N observed | N defined PCC | SSIM mean ± std | PCC mean ± std |
|---|---|---|---|---|
| Nucleus | 522 | 522 | 0.623 ± 0.180 | 0.660 ± 0.201 |
| Mitochondria | 326 | 326 | 0.612 ± 0.138 | 0.638 ± 0.173 |
| Tubulin | 128 | 128 | 0.340 ± 0.222 | 0.615 ± 0.186 |
| Actin | 221 | 221 | 0.440 ± 0.148 | 0.505 ± 0.207 |
| **Macro average** | | | **0.504** | **0.605** |

Mean ± standard deviation are over images, per observed channel, never pooled across images or organelles.

### 3. Breakdown by modality (Baseline B)

| Modality | Organelle | N | N defined PCC | SSIM | PCC |
|---|---|---|---|---|---|
| BF | Actin | 55 | 55 | 0.470 | 0.569 |
| BF | Mitochondria | 24 | 24 | 0.538 | 0.679 |
| BF | Nucleus | 73 | 73 | 0.610 | 0.511 |
| BF | Tubulin | 38 | 38 | 0.241 | 0.579 |
| DIC | Actin | 142 | 142 | 0.424 | 0.474 |
| DIC | Mitochondria | 81 | 81 | 0.578 | 0.579 |
| DIC | Nucleus | 228 | 228 | 0.510 | 0.655 |
| DIC | Tubulin | 90 | 90 | 0.382 | 0.631 |
| PC | Actin | 24 | 24 | 0.468 | 0.545 |
| PC | Mitochondria | 221 | 221 | 0.632 | 0.655 |
| PC | Nucleus | 221 | 221 | 0.745 | 0.715 |

Modality and study were not separated in this analysis, so these differences should not be read as pure modality effects.

<details><summary>Breakdown by study (Baseline B, 19 validation studies, 48 study and organelle cells)</summary>

| Study | Organelle | N | N defined PCC | SSIM | PCC |
|---|---|---|---|---|---|
| 2 | Mitochondria | 19 | 19 | 0.375 | 0.588 |
| 2 | Nucleus | 19 | 19 | 0.164 | 0.489 |
| 3 | Actin | 8 | 8 | 0.190 | 0.423 |
| 3 | Mitochondria | 8 | 8 | 0.331 | 0.445 |
| 3 | Nucleus | 8 | 8 | 0.279 | 0.202 |
| 4 | Mitochondria | 31 | 31 | 0.569 | 0.445 |
| 4 | Nucleus | 31 | 31 | 0.548 | 0.291 |
| 4 | Tubulin | 31 | 31 | 0.063 | 0.390 |
| 10 | Mitochondria | 8 | 8 | 0.685 | 0.767 |
| 10 | Nucleus | 8 | 8 | 0.581 | 0.649 |
| 19 | Mitochondria | 23 | 23 | 0.757 | 0.740 |
| 19 | Nucleus | 23 | 23 | 0.665 | 0.722 |
| 21 | Nucleus | 23 | 23 | 0.732 | 0.779 |
| 21 | Tubulin | 23 | 23 | 0.560 | 0.781 |
| 27 | Mitochondria | 197 | 197 | 0.647 | 0.655 |
| 27 | Nucleus | 197 | 197 | 0.771 | 0.720 |
| 31 | Nucleus | 10 | 10 | 0.685 | 0.058 |
| 31 | Tubulin | 10 | 10 | 0.324 | 0.552 |
| 37 | Actin | 16 | 16 | 0.583 | 0.776 |
| 37 | Nucleus | 16 | 16 | 0.462 | 0.390 |
| 37 | Tubulin | 16 | 16 | 0.498 | 0.698 |
| 39 | Actin | 10 | 10 | 0.623 | 0.762 |
| 39 | Tubulin | 10 | 10 | 0.627 | 0.744 |
| 40 | Actin | 10 | 10 | 0.684 | 0.850 |
| 40 | Nucleus | 10 | 10 | 0.453 | 0.806 |
| 40 | Tubulin | 10 | 10 | 0.529 | 0.809 |
| 42 | Actin | 18 | 18 | 0.423 | 0.465 |
| 42 | Nucleus | 18 | 18 | 0.683 | 0.659 |
| 42 | Tubulin | 18 | 18 | 0.241 | 0.608 |
| 44 | Actin | 11 | 11 | 0.712 | 0.845 |
| 44 | Nucleus | 11 | 11 | 0.705 | 0.621 |
| 53 | Actin | 8 | 8 | 0.485 | 0.638 |
| 53 | Mitochondria | 8 | 8 | 0.584 | 0.612 |
| 53 | Nucleus | 8 | 8 | 0.538 | 0.450 |
| 57 | Actin | 11 | 11 | 0.339 | 0.575 |
| 57 | Mitochondria | 11 | 11 | 0.338 | 0.682 |
| 57 | Nucleus | 11 | 11 | 0.299 | 0.483 |
| 60 | Actin | 13 | 13 | 0.577 | 0.520 |
| 60 | Mitochondria | 13 | 13 | 0.659 | 0.632 |
| 60 | Nucleus | 13 | 13 | 0.721 | 0.825 |
| 75 | Actin | 98 | 98 | 0.346 | 0.343 |
| 75 | Nucleus | 98 | 98 | 0.487 | 0.803 |
| 76 | Actin | 8 | 8 | 0.524 | 0.706 |
| 76 | Mitochondria | 8 | 8 | 0.599 | 0.825 |
| 76 | Nucleus | 8 | 8 | 0.462 | 0.707 |
| 82 | Actin | 10 | 10 | 0.469 | 0.462 |
| 82 | Nucleus | 10 | 10 | 0.709 | 0.556 |
| 82 | Tubulin | 10 | 10 | 0.156 | 0.553 |

Cells with 8 to 31 images are noisy.
</details>

### 4. Training and cost
- Best epoch: 48 of 50. Macro SSIM on the 100-image selection subset: 0.514. Same checkpoint on the full local validation split: 0.504. The subset score is 0.010 higher than the full-split score, so the selection is optimistic and the full-split number is the one to quote.
- Training time: 15,780 s (4.38 h) on one NVIDIA A100-SXM4-40GB, about 315 s per epoch. The loader was the bottleneck (full-resolution reads and percentile scaling per image), the GPU step took about 0.04 s per 16 crops in a separate timing test.
- Inference: 0.296 s per image on average over the full local validation split (tiled, GPU, forward passes only). Peak GPU memory: 3.68 GB (`torch.cuda.max_memory_allocated`).
- Training curves: [curves.png](https://github.com/neuronautsai/lmc2-task1-w1/blob/2e13ca070d1440b90d7fd6bc50c94efe27ea49b4/results/week2/curves.png). Training loss falls from 0.201 to 0.087 and is still slowly decreasing at epoch 50. Validation subset SSIM plateaus between about 0.46 and 0.51 after epoch 20 and fluctuates by about 0.03 to 0.04 between neighbouring epochs. PCC is much more stable. The epoch 48 choice sits on top of this noise, epochs 45 and 48 are at the same level.

### 5. End-to-end inference demonstration
`infer.py` takes one raw transmitted-light plane and writes a 4-channel float32 OME-TIFF (axes CYX, order Nucleus, Mitochondria, Tubulin, Actin, same height and width as the input). Run on a 1024x1024 DIC validation image (Study_02, image 43, z7), it produced a 4 x 1024 x 1024 output in 0.59 s with the epoch 48 checkpoint (single call, includes GPU warm-up). The output is in the normalized [0, 1] scale. Mapping predictions back to the intensity scale the organizers expect is an open question (see section 8).

### 6. Failure modes
Examples below are the lowest, median and highest SSIM image per organelle, not a random sample. Figures: [Nucleus](https://github.com/neuronautsai/lmc2-task1-w1/blob/2e13ca070d1440b90d7fd6bc50c94efe27ea49b4/results/week2/grid_Nucleus.png), [Mitochondria](https://github.com/neuronautsai/lmc2-task1-w1/blob/2e13ca070d1440b90d7fd6bc50c94efe27ea49b4/results/week2/grid_Mitochondria.png), [Tubulin](https://github.com/neuronautsai/lmc2-task1-w1/blob/2e13ca070d1440b90d7fd6bc50c94efe27ea49b4/results/week2/grid_Tubulin.png), [Actin](https://github.com/neuronautsai/lmc2-task1-w1/blob/2e13ca070d1440b90d7fd6bc50c94efe27ea49b4/results/week2/grid_Actin.png). Each row shows input, ground truth, prediction and absolute error.
1. **Blurry, dim predictions that lose fine structure.** Mitochondria punctate signal (Study 27 median case) becomes a smooth cytoplasm fill and Tubulin filaments (Study 31 median case) are not resolved. Intensity is generally under-predicted, for example the bright nuclei in the best Nucleus case (Study 42, SSIM 0.907).
2. **Non-zero background.** In the lowest Tubulin case (Study 4, SSIM 0.023) the ground truth is black with one bright cell while the prediction is grey across the whole image. In the median Actin case (Study 75) the model draws dark discs at nucleus positions instead of cell outlines.
3. **Sparse objects are missed or invented.** In the lowest Nucleus case (Study 2) the real nuclei are missed and blurry bright patches appear on debris. In the median Nucleus case (Study 31) SSIM is 0.686 while PCC is -0.009: one bright nucleus is missed and a few small dots are placed in wrong positions, so SSIM is carried by the dark background. This matches the Week 1 observation that SSIM alone does not separate a wrong prediction from a good one on sparse channels.
4. **Weakest channels and studies.** Tubulin has the lowest SSIM (0.340) and the largest spread, Actin has the lowest PCC (0.505). Lowest study cells: Study 4 Tubulin (0.063, N=31), Study 82 Tubulin (0.156, N=10), Study 2 Nucleus (0.164, N=19), Study 3 Actin (0.190, N=8). Study 75 holds 98 of the 221 Actin images with SSIM 0.346 and PCC 0.343, so it pulls the Actin mean down.
5. **Possible input and target mismatch, not verified.** In the lowest Actin case (Study 3) the ground truth shows a cluster of small bright round cells, while the input shows elongated cells that do not seem to overlap with them. Study 3 is also weak for Nucleus (0.279) and Mitochondria (0.331). To be checked against the raw files before drawing conclusions.

### 7. Limitations
- Single seed (42). Seeds 123 and 456 are planned for Week 3, so no variance across seeds is reported yet.
- Local proxy metrics on normalized images, the organizers' implementation is not published.
- Modality shift: the training inputs are BF-heavy (about 70 percent) while the local validation set contains many DIC and PC images. Per-modality numbers are reported above for this reason.
- Checkpoint chosen on a 100-image subset of the validation split (selection bias described in section 4). No separate test split exists locally.
- Constant learning rate with no decay and no early stopping, training was not converged at epoch 50.
- GroupNorm statistics are computed per tile, so tiled and whole-image outputs differ slightly (mean absolute difference 0.0047 on a randomly initialized network in a synthetic test, not measured on the trained model).

### 8. Open questions
Same list as Week 1. In addition: which intensity scale (raw uint16, per-image normalized or another) do the official metrics expect for the output channels?

### Key result
A four-output U-Net trained from scratch reaches macro SSIM 0.504 and macro PCC 0.605 on the full local validation split (532 images), against macro SSIM 0.310 and undefined PCC for the constant control. All four organelles improve over the control, Tubulin the least (SSIM 0.261 to 0.340).

### Takeaway
The model localizes cells and large structures but produces blurry, under-bright output with an elevated background, and Tubulin and Actin remain weak. The noisy validation SSIM and the unconverged loss suggest trying a decaying learning rate before changing the architecture. Study 3 and Study 4 should be checked for input and target alignment.

### Reproducibility
- **Environment:** Google Colab, one NVIDIA A100-SXM4-40GB, torch 2.11.0+cu130, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-image 0.25.2, tifffile 2026.10.15. Python 3.13.16 was recorded in the Week 1 session and was not printed again in the training session.
- **Seed:** 42 (split, subset draw, training).
- **Code:** https://github.com/neuronautsai/lmc2-task1-w1
  - Training and Baseline A were run at commit `0d6ff7fceff0b83f1e2c5e259c0c0f255b1d7083`.
  - Single-image inference and figures: `ff7034d40315e6ed8697f2648c7572121e517443` (`infer.py` and `make_figures.py` were added after the training run and do not change `train.py`).
  - Figures committed in `2e13ca070d1440b90d7fd6bc50c94efe27ea49b4`.
- **Artifacts:** training curves and example grids are in `results/week2/`. Weights (`best.pt`, `last.pt`), `history.csv`, `per_image_scores.csv`, `summary.csv`, `by_modality.csv`, `by_study.csv` and `run_info.json` are on the author's Google Drive and are not in the repository.
- **Commands:**
  - `python baseline_constant.py --out OUT_CONSTANT`
  - `python train.py --out OUT_RUN --seed 42 --epochs 50 --workers 12 --resume`
  - `python make_figures.py --run OUT_RUN`
  - `python infer.py --checkpoint OUT_RUN/best.pt --input IMAGE.tif --out PRED.ome.tiff`
