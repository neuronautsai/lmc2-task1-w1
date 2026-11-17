## Output intensity scale, Mock submission result (review item 2, continued)

**Container:** `inference.py` got an output mode (`LMC2_OUTPUT_MODE`, `uint8` or `float32`). `float32` writes `lo + p (hi - lo)` per organelle with the values from `output_scale.json` (option c of the offline comparison). Tag `v0.2.0` (commit `ada3abc`). CI runs the container in both modes.

**Correction to my previous comment:** the panimg abort on non-uint8 outputs was specific to the local libvips build. In CI (pyvips-binary, Python 3.12) the float32 outputs were read with DZI previews, and on Grand Challenge an algorithm try-out job with a synthetic input succeeded and imported all four float32 outputs.

**Submission:** `4f010e6d-d855-40e8-94b1-db433d9e580b` (evaluation `99a8a09a-4b7e-4753-a259-afb341856615`), Best Focus Mock, same model as before (`base` seed 42 `best.pt`), succeeded. 3 of 5 submissions left.

| Organelle | Metric | v0.1.0 (uint8) | v0.2.0 (float32) | change |
|---|---|---|---|---|
| Nucleus | SSIM | 0.4062 | 0.3763 | -0.0299 |
| Nucleus | PCC | 0.4353 | 0.3816 | -0.0537 |
| Nucleus | MAE | 9,321.09 | 10,190.79 | +9.3 % |
| Nucleus | Euclidean distance | 23,979,539 | 27,234,418 | +13.6 % |
| Nucleus | Cosine distance | 0.4094 | 0.4528 | +0.0434 |
| Mitochondria | SSIM | 0.3468 | 0.3617 | +0.0149 |
| Mitochondria | PCC | 0.2448 | 0.2305 | -0.0143 |
| Mitochondria | MAE | 11,288.30 | 11,565.38 | +2.5 % |
| Mitochondria | Euclidean distance | 27,690,166 | 28,435,586 | +2.7 % |
| Mitochondria | Cosine distance | 0.4201 | 0.4288 | +0.0087 |
| Actin | SSIM | 0.4513 | 0.5024 | +0.0511 |
| Actin | PCC | 0.6192 | 0.5864 | -0.0328 |
| Tubulin | SSIM | 0.1988 | 0.2000 | +0.0012 |
| Tubulin | PCC | 0.1454 | 0.1415 | -0.0039 |

**Reading**
- The change did not help. MAE, Euclidean distance and cosine distance got worse for both organelles where they are reported, and PCC went down for all four organelles. Leaderboard position of this evaluation: 25 (the first one was 19 when I posted it, other teams may have submitted since).
- This contradicts the assumption of the offline comparison (MAE and Euclidean distance on the raw scale, prediction not normalized). Under that assumption MAE should have gone down when the output moved from 0 to 255 to the target range. So the portal does something else with the prediction, and I cannot tell what from two cases.
- PCC is invariant to an affine rescaling, and both submissions ran the same model on the same two cases, so the PCC change can only come from rounding (v0.1.0 rounds to 256 levels) or from a non-linear step in the evaluator, for example a cast or a clip that depends on the data type.
- Decision: the active algorithm image is set back to `v0.1.0` (uint8). Before another submission on the output scale, I would ask on the forum how the prediction is processed for MAE and Euclidean distance. The remaining items of the review (weighted tile blending, the p2 to p99.8 training run, then one method change) do not depend on it and come next.
