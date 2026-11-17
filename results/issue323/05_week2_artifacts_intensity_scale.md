## Week 2 artifacts and intensity scale update

### 1. Artifacts (point 5)

Shared Drive folder, view only for anyone with the link: (Drive link in issue #323)

Contents of the seed 42 run (training commit 0d6ff7fceff0b83f1e2c5e259c0c0f255b1d7083): best.pt (29.6 MB, epoch 48), last.pt (88.9 MB), history.csv, per_image_scores.csv, summary.csv, by_modality.csv, by_study.csv, run_info.json, curves.png, grid_Nucleus/Mitochondria/Tubulin/Actin.png and example_prediction.ome.tiff.

### 2. Output intensity scale (point 2)

I did not open a new forum topic because the forum topic "Evaluation metrics and normalization" already answers most of it (organizer replies, checked on 8 November). What the organizer stated there:

- Ground truth and prediction are percentile normalized independently, per image, before the metrics.
- After a correction announced in that topic, the first-edition normalization is restored: float, pmin 2 and pmax 99.8, `(image - p_low) / (p_high - p_low)`, no clipping to [0, 1], no uint16 cast. A constant image (p_low equal to p_high) is returned unchanged. The evaluation code was to be updated so that Mock and final phases use the same function. I have not seen the updated code.
- Metrics are PCC, SSIM, MAE, cosine distance and Euclidean distance, all five for every organelle. Ranking is the average of the per-metric ranks.
- Still unanswered when I checked: SSIM data_range, window size, K1 and K2, and how data_range is set when normalized values fall outside [0, 1] (asked by another participant).

What this means for us (my reading, not stated by the organizers):

- Since the prediction is normalized by the evaluator, its absolute output scale should not matter as long as the output is not constant and is not clipped or quantized. This is an inference from the formula, not tested on the portal.
- Our local proxy is different from the official one: p1 to p99.5, clipped to [0, 1], data_range 1, only SSIM and PCC. The local numbers are therefore not directly comparable to portal scores.

Planned next step: add an official-style scoring mode (p2 and p99.8 on both ground truth and prediction, float, no clipping, plus MAE, cosine and Euclidean) next to the current proxy and re-score the existing seed 42 predictions with both before the first portal submission. Not done yet.

### 3. Study 3 forum report

A forum topic reports that the Nucleus and Actin files of Study_03 images 53, 55, 56 (z0 only) and 57 are pixel-identical, and the organizer confirmed it as an export error that was being fixed. I have not checked which of these images fall in our split or whether this explains the weak Study 3 scores from Week 2. The suspected input and target mismatch noted in Week 2 is still unverified.
