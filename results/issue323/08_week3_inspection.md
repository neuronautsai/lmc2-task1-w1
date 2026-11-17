## Week 3 inspection: modalities, studies and the same examples for every model

Code: `w3_inspect.py` (commit `5441b54`). Tables are best.pt scores on the full local validation split, averaged over images within each seed, then over the three seeds. Figures use the seed 42 best.pt of `base`, `cos` and `cosfilm` (true modality label) on the same images, picked by a fixed rule from the `base` scores only: the median-SSIM image of each modality and the lowest-SSIM image of each organelle. Display range is [0, 1] for every panel.

Artifacts (Week 3 best.pt weights, per-image CSVs, last-epoch scores, tables, figures): (Drive link in issue #323) (`analysis/` holds the figures and `by_modality_w3.csv`, `by_study_w3.csv`).

### By modality (SSIM / PCC)
| Modality | Organelle | base | cos | cosfilm |
|---|---|---|---|---|
| BF | Nucleus | 0.616 / 0.496 | 0.628 / 0.493 | 0.585 / 0.424 |
| BF | Mitochondria | 0.537 / 0.671 | 0.529 / 0.665 | 0.524 / 0.644 |
| BF | Tubulin | 0.241 / 0.574 | 0.242 / 0.572 | 0.243 / 0.555 |
| BF | Actin | 0.471 / 0.562 | 0.451 / 0.546 | 0.449 / 0.534 |
| DIC | Nucleus | 0.504 / 0.627 | 0.504 / 0.659 | 0.483 / 0.541 |
| DIC | Mitochondria | 0.554 / 0.554 | 0.555 / 0.544 | 0.545 / 0.558 |
| DIC | Tubulin | 0.366 / 0.620 | 0.358 / 0.609 | 0.407 / 0.620 |
| DIC | Actin | 0.419 / 0.474 | 0.414 / 0.461 | 0.405 / 0.453 |
| PC | Nucleus | 0.739 / 0.696 | 0.717 / 0.622 | 0.764 / 0.628 |
| PC | Mitochondria | 0.616 / 0.654 | 0.598 / 0.635 | 0.561 / 0.648 |
| PC | Actin | 0.457 / 0.542 | 0.449 / 0.519 | 0.452 / 0.531 |

The validation split has no PC image with a Tubulin target. The FiLM Tubulin SSIM gain reported above comes from DIC (0.407 against 0.366 for `base`), and its Nucleus PCC loss comes from BF and DIC.

### Weakest studies (base, SSIM / PCC, images in the validation split)
| Study | Organelle | base | images |
|---|---|---|---|
| 2 | Nucleus | 0.159 / 0.468 | 19 |
| 3 | Actin | 0.197 / 0.430 | 8 |
| 3 | Nucleus | 0.263 / 0.243 | 8 |
| 4 | Tubulin | 0.060 / 0.399 | 31 |
| 31 | Nucleus | 0.634 / 0.092 | 10 |

The same studies are weak for all three configurations. The full table is in `by_study_w3.csv`.

### What the figures show
- **Study 3, Actin target is a copy of the Nucleus target.** In both Study 3 cases (images 53 and 55) the Actin target shows the same nuclei as the Nucleus target, not an actin pattern. This matches the duplicate-file report for Study 3 in the challenge forum. Study 3 Actin scores (8 validation images) are therefore not meaningful. The Study 3 images in the training split have not been checked yet, if they are affected the Actin channel was partly trained on nucleus images.
- **Study 2, Nucleus target shows only part of the nuclei.** In image 52 the Nucleus target has two nuclei, while the input and the Mitochondria target show more cells. All three models predict a nucleus for each cell, which the metric counts as errors. Input and targets look aligned in this case.
- **Study 4, mostly empty field.** In image 75 a single small cell sits in an empty field. The cell is in the right place in every prediction, but the predicted background has straight horizontal and vertical bands. They look like tile boundaries of the tiled inference (not verified), and with an almost black target they dominate the SSIM (0.02 to 0.09 for Tubulin).
- **Median cases (BF study 42, DIC study 75, PC study 27).** Nuclei are found in DIC and PC, but as speckled blobs in PC, with some nucleus-like spots where there is no nucleus. Filamentous structures are lost in all three models: mitochondria and tubulin come out as diffuse cytoplasm, and DIC actin comes out as the inverse of the nuclei instead of the cell outlines.
- No input-target misalignment was visible in these seven cases. This is not a check of the whole split.

### Consequences
- Study 3 Actin should be excluded from training and scoring until the files are corrected. This is the first change for the next experiment.
- The background bands in near-empty images point to the tile blending of `predict_full` as a second thing to check.
