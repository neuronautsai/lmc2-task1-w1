### Week 2 follow-up: correction and additions

**Correction (excluded images).** The Week 2 comment says the 21 excluded images are all in the training split. This is wrong. 19 are in the training split and 2 are in the local validation split (both in Study 3). Counts from `pairs.csv` and `split_manifest.csv`:

| Split | Images in split manifest | Excluded (empty `in_file`) | Usable inputs |
|---|---|---|---|
| Train | 2,215 | 19 | 2,196 |
| Local validation | 534 | 2 | 532 |
| Total | 2,749 | 21 | 2,728 |

Excluded images by study: Study 59 (12), Study 24 (5), Study 11 (1), Study 12 (1), Study 3 (2, validation). All reported validation results are on the 532 usable validation images.

**Reconciliation with Week 1.** The Week 1 split table lists 2,212 train and 532 validation inputs (2,744 images), which are images with a transmitted-light file available on Drive. Week 2 uses images with a defined best-z transmitted plane (2,728). The difference of 16 images is entirely in the training split. This is consistent with 5 of the 21 excluded images having no transmitted-light file at all (3 train, 2 validation) and 16 train images having a file but no defined best-z plane. This split of the 21 was inferred from the count differences and not traced image by image. The Week 1 text says four images have no listed transmitted-light file while the Week 1 table implies five, the fifth was not traced.

**Addition (overfit check on a tiny subset).** Before the full run, the training code was run on 8 training images and scored on the same 8 images (30 epochs, 60 optimizer steps, learning rate 1e-3, commit `28bba2fde6ba789c61eed7e7bd571564e9038f99`). Training loss fell from 0.368 to 0.156, macro SSIM rose from 0.138 to 0.282 and macro PCC from 0.027 to 0.336 (Nucleus, Mitochondria and Tubulin only, since none of the 8 images has an Actin target). The metrics were still rising at epoch 30, so this check shows that pairing, masking and the loss work, but it does not show memorization of the subset.
