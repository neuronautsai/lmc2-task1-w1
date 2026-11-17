## Week 1 results: Light My Cells 2, Task 1 (Best Focus)

Status: data access, inventory, metadata checks, frozen split and local metric proxy are done. Code, figures and split manifest are in the repository linked under Reproducibility.

### 1. Registration, rules and access
- Selected task: Task 1 (single transmitted-light image to best-focus fluorescence). Task 2 (stack to MIP) is not started.
- Registration: join request accepted, training database reachable through the registered challenge account.
- Lab team: the Teams page lists 11 teams, none identifiable as our lab. No team created or joined yet, to be decided with the advisor.
- Timeline (challenge timeline graphic): training database released 31 July, validation from 21 September, test evaluation 1 to 13 December, results 18 December.
- Submission limits seen on the Best Focus - Mock phase page: 5 submissions in 70 days, 5 minutes per case, max 16 GB RAM, no GPU or NVIDIA T4 (16 GiB), closes 1 December 2026 01:59 (Europe/Istanbul). The real validation phase quota has not been seen yet.
- Rules page: Docker only, fully automated, offline execution, average runtime of 10 minutes or less. The mock phase page says 5 minutes per case (see open questions).
- Output contract: all four organelle channels, same size as the input, OME-TIFF.
- External data and pretrained weights are allowed. All of them must be disclosed and public resources must be linked on the forum before the submission deadline. Submissions relying on undisclosed private resources can be excluded from the final ranking.
- Metrics: at least SSIM and PCC, ranking by averaged metric ranks. Metrics are computed only where the ground-truth channel exists. Implementation details are not published.

### 2. Problem statement (draft, to be reviewed)
Transmitted-light images (BF, PC, DIC) show label-free contrast. Fluorescence images show one labeled structure each: nucleus, mitochondria, tubulin and actin. The task is image-to-image regression from one transmitted-light plane to the best-focus fluorescence image of each structure. Task 2 predicts maximum-intensity projections from a z-stack instead.

### 3. Data access and inventory
- Source: official Seafile share linked from the challenge Database page (URL not posted here). 108 files, about 563 GB, 87 study archives (Study_01 to Study_87) plus README and a study-info archive. The README text says studies 51 to 76 were added on 10 September, the folder now contains up to Study_87. The issue text mentions 76 studies.
- Split archives: Study_26 (z01, z02, zip), Study_27 (z01 to z08, zip), Study_74 (z01 to z05, zip), Study_75 (z01 to z04, zip), extracted with 7-Zip.
- No checksums are supplied by the organizers. Own SHA-256 recorded for two test archives:
  - Study_86.zip `482fb09cde4a7fb3f880cffed81b3215bc03a004caef579c7116996e2b6d1887` (331,927,318 bytes)
  - Study_01.zip `b368cbb0769a5f492f918aa1123b9d8b497f05a015858835766b26cc9f1b1ff6` (711,833,982 bytes)
- Study-info TSVs (`all_files` and `best_focus` per study) list 87 studies, 2,749 images, 155,406 files: BF 32,852, PC 13,627, DIC 10,412, Phase 228 (Study_59 only), Nucleus 51,296, Mitochondria 27,262, Actin 11,727, Tubulin 8,002.
- Input and target channel combinations differ per study (26 combinations in the TSVs, 25 if Phase is treated as PC). Missing channels are absent, not zero images.
- Subset kept for Task 1: all transmitted-light slices plus the best-z slice of each fluorescence channel. 62,287 files listed, 62,077 available (all in Study_59, see open questions). Stored on Drive, originals untouched.
- Inventory per study: `inventory_by_study.csv`. Full manifest with metadata: `data_manifest_full.csv` (155,406 rows, columns include study, image_id, channel, z, filename, best_z, nZ, kind, on_drive, needed, shape, axes, dtype, PhysicalSizeX, PhysicalSizeXUnit, PhysicalSizeY).

### 4. Metadata checks (62,077 available files)
- All files: axes `YX`, dtype `uint16`, no read errors.
- 16 distinct image shapes (for example 2048x2048, 512x512, 980x1016, 2044x2048, 1200x1200). Two studies (82 and 85) contain two shapes each.
- Pixel size is present in all files, unit micrometer.
- Within the same image, no shape mismatch and no pixel-size mismatch between channels (0 of 2,748 images with files).
- Image ids are globally unique (2,749 ids). Image id ranges of studies 2, 3, 4, 7, 8, 11, 12 and 13 are interleaved.

### 5. Intensity observations (60 random files per channel, pooled)
- Fluorescence channels are strongly right-skewed with most pixels near background. Median p50 per image: Nucleus 194, Mitochondria 230, Tubulin 1,063, Actin 2,308. Median p99: 1,475, 1,635, 4,108, 6,749.
- Intensity scales differ strongly between studies (median per-image maximum 3,062 for Nucleus against values up to 65,535 in the pooled histogram). A small spike appears at 65,535 in several channels, its cause was not checked.
- BF shows several histogram peaks, consistent with study-specific baselines (not verified per study).
- Otsu foreground fraction in one random example: Nucleus 6 percent, Mitochondria 6 percent, Tubulin 30 percent, Actin 40 percent. Sparse targets will be dominated by background.
- Figures (in `results/`, axis titles are in Turkish, Calisma = study, kanal = channel, siyah = var = black means present): [example_grid.png](https://github.com/neuronautsai/lmc2-task1-w1/blob/9b2a4bc8df62c1f835a085cec5c4fc462ad2ba0d/results/example_grid.png) (Study_01 and Study_86 only), [intensity_histograms.png](https://github.com/neuronautsai/lmc2-task1-w1/blob/9b2a4bc8df62c1f835a085cec5c4fc462ad2ba0d/results/intensity_histograms.png), [fg_bg_examples.png](https://github.com/neuronautsai/lmc2-task1-w1/blob/9b2a4bc8df62c1f835a085cec5c4fc462ad2ba0d/results/fg_bg_examples.png), [channel_coverage.png](https://github.com/neuronautsai/lmc2-task1-w1/blob/9b2a4bc8df62c1f835a085cec5c4fc462ad2ba0d/results/channel_coverage.png).

### 6. Local split
GroupShuffleSplit, seed 42, test size 0.2, grouped by study. Studies 2, 3, 4 / 7, 8 / 11, 12, 13 are merged into three groups as a precaution because their image id ranges interleave (assumption, relation not verified). 87 studies form 82 groups, none crosses the split. Patches are extracted only after splitting.

| Split | Studies | Groups | Inputs | Nucleus targets | Mitochondria targets | Tubulin targets | Actin targets |
|---|---|---|---|---|---|---|---|
| Train | 68 | 65 | 2212 | 2083 | 1052 | 397 | 416 |
| Local validation | 19 | 17 | 532 | 524 | 326 | 128 | 221 |

Counts use files available on Drive. Manifest: [results/split_manifest.csv](https://github.com/neuronautsai/lmc2-task1-w1/blob/9b2a4bc8df62c1f835a085cec5c4fc462ad2ba0d/results/split_manifest.csv).

### 7. Metric checks (local proxy, not official scores)
Proxy: `skimage.metrics.structural_similarity` (default window) and `scipy.stats.pearsonr`. Target min-max scaled to [0, 1], prediction scaled with the target range and clipped to [0, 1], `data_range=1`. A constant target is not scored, a constant prediction gives undefined PCC (NaN, not replaced). Test image: one Nucleus image (Study_01).

| Metric check | SSIM | PCC | Expected behavior and explanation |
|---|---|---|---|
| Identical nonconstant pair | 1.0000 | 1.0000 | Perfect structure and correlation. |
| Added noise (5 percent of range) | 0.3224 | 0.6864 | SSIM drops strongly because noise breaks local structure in dark background. |
| Blur (Gaussian, sigma 3) | 0.9089 | 0.9766 | Global structure is kept, fine detail is lost. |
| Intensity rescaling x0.5 | 0.5913 | 1.0000 | PCC ignores linear scaling, SSIM penalizes the brightness difference. |
| Intensity rescaling x1.5 | 0.8653 | 0.9954 | Clipping at the target maximum lowers PCC slightly. |
| Constant image (prediction) | 0.8761 | undefined | PCC has zero variance and is undefined (1 of 1). SSIM is high because background dominates. |

Also observed: a different Nucleus image gave SSIM 0.7706 and PCC 0.0208. SSIM alone does not separate a wrong or constant prediction from a good one on sparse fluorescence images, and a high PCC does not guarantee calibrated intensity.

### 8. Open questions for the organizers
1. Study_59: TSV lists 684 files (channel `Phase`), the archive holds 107 files (channel `PC`). Only 3 of 24 best-z fluorescence targets listed in the TSV are in the archive.
2. Four images have no transmitted-light file listed.
3. Official evaluation code, SSIM `data_range` and window, scaling, clipping and constant-image handling.
4. Runtime limit: 10 minutes on average (rules page) or 5 minutes per case (mock phase page).
5. Are image id ranges interleaved across studies 2 to 4, 7 to 8 and 11 to 13 a sign of related acquisitions?
6. Quota of the real validation phase.

### Key result
All 87 study archives were processed. 62,077 of 62,287 listed Task 1 files are available, metadata is consistent across channels of the same image, and a study-level, group-aware split is frozen.

### Takeaway
Image sizes and intensity scales vary strongly between studies, and SSIM is dominated by background on sparse channels. Week 2 needs a declared normalization rule, tile-based evaluation of full images and per-channel reporting of both SSIM and PCC.

### Reproducibility
- Environment: Google Colab, Python 3.13.16, numpy 2.1.3, pandas 2.2.3, tifffile 2026.10.15, scipy 1.16.3, scikit-image 0.25.2, scikit-learn 1.6.1, 7-Zip 23.01.
- Seed: 42 (split), random sampling for figures uses numpy default_rng(42). Seeds 42, 123, 456 are reserved for Weeks 2 and 3.
- Dataset version: Seafile share as of 25 and 26 October 2026, README dated 10 September 2026.
- Code: `download_data.py`, `inspect_data.py`, `make_splits.py`. Repository: https://github.com/neuronautsai/lmc2-task1-w1 (public). Commit hash: `9b2a4bc8df62c1f835a085cec5c4fc462ad2ba0d`.
- Commands: `python inspect_data.py manifest --info INFO_DIR --out OUT`, `python download_data.py --base-url URL_TEMPLATE --root ROOT --out OUT --manifest OUT/manifest.csv`, `python inspect_data.py scan --manifest OUT/manifest.csv --out OUT`, `python make_splits.py --manifest OUT/data_manifest_full.csv --out split_manifest.csv`. The scripts were assembled from the Colab cells that produced these results and were not rerun end to end as scripts.
