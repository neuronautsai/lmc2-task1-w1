## Weighted tile blending (review item 3)

**Correction to the review point:** the tile blending in `predict_full` was already weighted, with a linear ramp from 0.05 to 1 over the 64 px overlap at every tile edge. Because the ramp starts at 0.05 and not near zero, the weight of a new tile jumps from 0 to 0.05 on every line where a tile starts or ends. Where the other tile has weight 1, the blended output steps by about 0.05 / 1.05 of the difference between the two tile predictions. This is my explanation for the bands seen in the Week 3 inspection, read from the code, not tested separately.

**Change:** `predict_full` got a `window` option: `linear` (the old ramp), `hann`, and `gaussian` (sigma = tile / 8, as in nnU-Net). Hann and Gaussian weights are floored at 1e-3, so their jump at a tile boundary is 1e-3 instead of 0.05. The default stays `linear` with 64 px overlap, so the container and all earlier results are unchanged. Commit `3162ad3`, comparison script `compare_blending.py`.

**Setup:** `base` seed 42 `best.pt` (epoch 48), local validation split with Study 3 Actin excluded (`pairs_excluded.csv`), 532 images, 1189 image and organelle rows per setting, 512 px tiles, local proxy SSIM and PCC (targets normalized p1 to p99.5 in the dataset, prediction clipped to [0, 1]). Only the four settings below were run, so there is no Hann or Gaussian run with 64 px overlap.

**Seam index:** the mean absolute step between neighbouring pixels across the tile boundary lines, divided by the same mean over all lines of the image. 1 means the boundary lines look like any other line, values above 1 mean larger steps on the boundaries. Images that fit in one tile have no boundary lines and are left out of the seam means. "Near-empty" is the 10 % of image and organelle rows with the lowest target mean (target mean <= 0.0407).

| Setting | Macro SSIM | Macro PCC | Seam index, all | Seam index, near-empty |
|---|---|---|---|---|
| linear, 64 px (current) | 0.5044 | 0.6088 | 1.45 | 1.57 |
| linear, 128 px | 0.5060 | 0.6129 | 1.46 | 1.56 |
| hann, 128 px | 0.5061 | 0.6111 | 1.07 | 1.08 |
| gaussian, 128 px | 0.5060 | 0.6101 | 1.12 | 1.23 |

Paired difference to the current setting, computed from `per_image_blending.csv`: for each image the difference is averaged over its organelle rows, then averaged over images, with a 95 % percentile bootstrap interval over images (2000 resamples). This is a different average from the macro scores above, which is why the macro SSIM gain is 0.0016 to 0.0017 and the paired gain is about 0.0020.

| Setting | SSIM difference | PCC difference | Rows with higher SSIM |
|---|---|---|---|
| linear, 128 px | +0.0020 [+0.0016, +0.0024] | +0.0051 [+0.0032, +0.0073] | 67 % |
| hann, 128 px | +0.0021 [+0.0017, +0.0025] | +0.0036 [+0.0018, +0.0056] | 66 % |
| gaussian, 128 px | +0.0020 [+0.0016, +0.0024] | +0.0028 [+0.0010, +0.0048] | 64 % |

**Reading**
- Going from 64 to 128 px overlap gains about 0.002 SSIM, and at 128 px the window shape barely changes SSIM. The gain is consistent (the intervals exclude zero) but small. For PCC the window shape matters more: linear gains the most, Gaussian the least.
- For the linear ramp, a larger overlap does not change the seam index. Hann brings it from 1.45 to 1.07 and on the near-empty rows from 1.57 to 1.08, Gaussian less. Whether Hann or Gaussian would also reduce the seams at 64 px was not tested.
- Study 4 Tubulin SSIM moves from 0.0575 to 0.0594 with Hann (31 images). The seam index of that subset was not computed separately, but the score gain there is as small as elsewhere, so the low score of this subset is not mainly explained by the bands.
- Limits: local proxy only, one checkpoint, and the bootstrap over images ignores the correlation of images within a study. Run time is not reported: after a resume only 32 images were timed, and the settings always ran in the same order, so the timings are not comparable.

**Proposal:** make Hann with 128 px overlap the default in `predict_full` and in the container. It reduces the seam index the most and gains as much SSIM as the other 128 px settings. The trade-off is PCC: linear with 128 px has the larger PCC gain (+0.0051 against +0.0036), and the two were not compared with each other directly. I would not spend one of the 3 remaining Mock submissions on a local gain of 0.002 SSIM alone. It would go into the container together with the next model change (items 4 and 5).
