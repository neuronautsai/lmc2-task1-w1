## Output intensity scale, offline comparison (review item 2)

Code: `compare_output_scale.py` (commit `9c6cb36`). Model: `base` seed 42 `best.pt` (the submitted checkpoint), full local validation split with Study 3 Actin excluded (532 images, 1,189 image and organelle rows). Ground truth: the raw 16-bit target files.

**Options**
- (a) uint8, `round(255 p)`, what `v0.1.0` writes.
- (b) uint16, `round(lo + p (hi - lo))` per organelle.
- (c) float32, the same mapping without rounding.

`lo` and `hi` are the medians of the raw target's p2 and p99.8 over 300 random training images per organelle (seed 0):

| Organelle | lo (median p2) | hi (median p99.8) |
|---|---|---|
| Nucleus | 115.0 | 934.0 |
| Mitochondria | 106.0 | 1,386.5 |
| Tubulin | 379.5 | 4,741.5 |
| Actin | 604.0 | 10,888.4 |

**Scale-invariant metrics.** Official-style SSIM, PCC and cosine distance (both images percentile normalized, `official_metrics.py`) were compared with the unrounded prediction on the first 20 images. Maximum absolute change: (c) 1.5e-7, (b) 1.8e-2, (a) 3.4e-2. So the rescaling itself changes nothing, and the differences in (a) and (b) come from rounding. Only the maximum was recorded, not the mean change.

**MAE and Euclidean distance.** How the portal computes these two is not published. The official-style versions on normalized images are around 0.1 to 0.3 (MAE) and 280 to 620 (Euclidean), far from the portal values (thousands and millions). Raw-scale versions (raw target against the written values, no normalization) are in the right range, so they are used here as the assumed portal definition.

| Organelle | MAE (a) | MAE (b) and (c) | change | Euclidean (a) | Euclidean (b) and (c) | change |
|---|---|---|---|---|---|---|
| Nucleus | 1,530.4 | 1,383.9 | -9.6 % | 5,833,512 | 5,613,780 | -3.8 % |
| Mitochondria | 1,084.8 | 942.5 | -13.1 % | 3,878,019 | 3,638,755 | -6.2 % |
| Tubulin | 1,190.6 | 1,244.7 | +4.5 % | 3,221,193 | 2,826,043 | -12.3 % |
| Actin | 2,292.7 | 1,676.6 | -26.9 % | 4,956,024 | 3,728,528 | -24.8 % |
| Macro | 1,524.6 | 1,311.9 | -14.0 % | 4,472,187 | 3,951,777 | -11.6 % |

(b) and (c) differ only after the first decimal.

**Reading**
- With (a), MAE is almost equal to the mean raw target value (Nucleus 1,553, Mitochondria 1,107, Tubulin 1,216, Actin 2,363). The uint8 output is close to zero on the target scale, which supports the scale explanation of the portal numbers.
- The fixed mapping helps less than expected: macro MAE -14 %, Euclidean -12 %, and Tubulin MAE gets worse. The validation targets are brighter than the typical training target (mean Nucleus 1,553 against a median p99.8 of 934), so one mapping per organelle cannot follow the per-study intensity scale. The portal MAE values of the two Mock cases (Nucleus 9,321, Mitochondria 11,288) suggest even brighter targets there, so the portal gain may be smaller than the local one.
- Raw-scale cosine distance gets worse with (b) and (c) for Nucleus (0.221 to 0.276) and Mitochondria (0.239 to 0.260), because it is not invariant to the offset `lo`. Whether the portal cosine distance is normalized is also unknown.

**Interface check before a submission.** In a local test (libvips 8.15), panimg's preview step aborted on uint16 and float32 outputs, tiled and untiled, and passed on uint8. Grand Challenge may use a different libvips build. Next step: add an output mode to `inference.py` and check (c) with an algorithm try-out job on Grand Challenge, which does not count as a challenge submission. Only if that works, one Mock submission with (c).

Outputs: `target_stats.json`, `per_image_output_scale.csv`, `output_scale_summary.txt`, `output_scale_info.json`.
