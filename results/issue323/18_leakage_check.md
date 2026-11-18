## Leakage check of the local split

**Question:** can information from the local validation images reach training, or inflate the reported validation scores?

**Method** (`leakage_check.py`, commit `d63c188`), on `pairs_excluded.csv` and the prepared arrays (2,193 training and 532 validation images):
1. Split integrity: studies and image ids present in both splits.
2. Exact duplicates: md5 of every channel (input and observed targets) of all 2,725 prepared arrays. The arrays are normalized per image, so copies that differ only by a linear intensity change are caught too.
3. Near duplicates: every input downsampled to 64 x 64 and z-scored, highest correlation of each validation input with any training input over the 8 flips and 90-degree rotations. Candidates above 0.95 were read from the raw files, compared again at 256 x 256 and inspected visually.
4. Selection subset: macro official-style SSIM and PCC with and without the 100 validation images used to pick `best.pt`.

**Results**

| Check | Result |
|---|---|
| Studies in both splits | 0 |
| Image ids in both splits | 0 |
| Identical arrays across splits (any channel) | 0 |
| Validation inputs with correlation above 0.95 to a training input (64 x 64) | 3 of 532 |

| Validation image | Closest training image | Correlation 64 x 64 | Correlation 256 x 256 | Visual check |
|---|---|---|---|---|
| Study 39, image 2074 (DIC, 2048 x 2048) | Study 35, image 2023 (DIC, 2048 x 2048) | 0.963 | 0.838 | different cells and field of view |
| Study 39, image 2077 (DIC, 2048 x 2048) | Study 35, image 2028 (DIC, 2048 x 2048) | 0.962 | 0.795 | different cells and field of view |
| Study 40, image 2083 (DIC, 2048 x 2048) | Study 26, image 1328 (BF, 980 x 1016) | 0.958 | 0.903 | different cells, modality and image size |

The correlation falls when the resolution rises, so what the pairs share is the low-frequency illumination background, not image content.

Selection subset, over the 11 scored checkpoints of the Week 3 base runs and review items 4 and 5: removing the 100 selection images changes macro official-style SSIM by -0.0045 to +0.0005 and macro PCC by -0.0068 to -0.0001. No comparison changes direction. The per-run numbers for item 5 are in the item 5 comment.

**Reading**
- No leakage between the local training and validation splits was found.
- The validation scores of `best.pt` are slightly optimistic, because checkpoint selection uses 100 of the 532 validation images: by at most 0.0045 SSIM and 0.0068 PCC here. Method decisions (blending, normalization, loss) were also taken on this split. The hidden challenge test set is not affected.
- Limits: the near-duplicate test looks at the input only, at 64 x 64, and can miss a copy that is cropped or rescaled differently. Duplicates within the training split were not checked, as they cannot leak into validation.

**Commands**
```
python leakage_check.py --pairs-csv PAIRS_EXCLUDED --prepared-dir /content/prepared --outputs OUT
python leakage_check.py --pairs-csv PAIRS_EXCLUDED --data-root DATA --view 39:2074,35:2023 --view 39:2077,35:2028 --view 40:2083,26:1328 --fig leakage_pairs.png
```
The checks were run as notebook cells. `leakage_check.py` holds the same code and was tested on a synthetic set with a planted duplicate.
