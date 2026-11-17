## Official-style rescoring of the Week 2 U-Net (seed 42)

Follows the plan in my previous comment. All numbers are local proxy scores on the full local validation split (532 images, 1,197 image and organelle pairs), not official scores.

### Setup
- Model: epoch 48 checkpoint of the seed 42 run (best.pt), same tiled inference as Week 2.
- Official-style columns: ground truth is the RAW target image, ground truth and prediction are each percentile normalized on their own with p2 and p99.8, float, no clipping, as stated by the organizers in the forum.
- Current proxy columns (p1 to p99.5, clipped, data_range 1) are computed in the same pass for comparison. They reproduce the Week 2 table exactly (macro SSIM 0.504, macro PCC 0.605, same N per organelle).
- Unpublished and therefore assumed: SSIM data_range, window and K1/K2 (two data_range variants are reported), and the exact definitions of MAE, cosine and Euclidean distance (mean absolute difference, 1 minus cosine similarity, L2 norm of the difference, all on the normalized flattened images).
- No undefined values in any metric. Scaling a prediction by an affine map changed no score by more than 6e-14.
- Code: https://github.com/neuronautsai/lmc2-task1-w1 commit 281444a, files official_metrics.py and score_official.py. Environment: Google Colab, NVIDIA L4, 12 CPUs, torch version not recorded in this session. Seed 42.
- Command: python score_official.py --checkpoint OUT_RUN/best.pt --out OUT_RUN/official --resume
- Outputs (per image CSV, summary, by modality, by study) are in the shared Drive folder from my previous comment, subfolder official.

### Per organelle, mean over images

| Organelle | N | proxy SSIM | proxy PCC | official-style PCC | SSIM (data_range 1) | SSIM (joint range) | MAE | Cosine dist. | Euclidean dist. |
|---|---|---|---|---|---|---|---|---|---|
| Nucleus | 522 | 0.623 | 0.660 | 0.650 | 0.604 | 0.676 | 0.080 | 0.270 | 270.6 |
| Mitochondria | 326 | 0.612 | 0.638 | 0.621 | 0.511 | 0.593 | 0.127 | 0.278 | 362.7 |
| Tubulin | 128 | 0.340 | 0.615 | 0.603 | 0.285 | 0.378 | 0.209 | 0.291 | 420.6 |
| Actin | 221 | 0.440 | 0.505 | 0.498 | 0.324 | 0.488 | 0.265 | 0.184 | 633.6 |
| Macro | | 0.504 | 0.605 | 0.593 | 0.431 | 0.534 | 0.170 | 0.256 | 421.9 |

Standard deviations over images are in summary_official.csv.

### Reading
- PCC changes little between the two normalizations (0.605 to 0.593).
- SSIM depends strongly on the unpublished data_range: macro 0.431 with data_range 1 and 0.534 with the joint range of the two normalized images, against 0.504 for the old proxy. The spread (about 0.10) is larger than most differences I expect between methods, so the organizers' SSIM settings matter for comparing local and portal scores. Per organelle the ordering stays the same, Tubulin has the lowest SSIM and Actin the lowest PCC.
- Euclidean distance grows with image size, so its mean over images mixes different image sizes and should be read with care.
- The model was trained on targets clipped at p99.5, while the official-style ground truth is not clipped, so bright pixels above the clipped range are a possible extra source of error. I have not tested this.
- Not done: modality and study breakdowns were saved but not interpreted here.
