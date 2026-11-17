## Week 3 validation submission (Best Focus, Mock phase)

| Submission ID | Task | Commit / model | Runtime | Remaining quota |
|---|---|---|---|---|
| `67c619bd-bee7-44c2-af3b-85c3476ce62b` (evaluation `9109c77e-17c9-4b68-b068-b6b9ce64d378`) | Task 1, Best Focus, Mock | tag `v0.1.0` (commit `c53165b`), U-Net `base` seed 42 best.pt | 2 algorithm jobs, both succeeded, submission to result about 7 min | 4 of 5 in the 70-day window (phase closes 1 December 2026) |

**Portal scores (official, Mock phase, kept separate from the local proxy):**

| Organelle | SSIM | PCC | MAE | Euclidean distance | Cosine distance |
|---|---|---|---|---|---|
| Nucleus | 0.4062 | 0.4353 | 9321.09 | 23979538.81 | 0.4094 |
| Mitochondria | 0.3468 | 0.2448 | 11288.30 | 27690165.51 | 0.4201 |
| Actin | 0.4513 | 0.6192 | not reported | not reported | not reported |
| Tubulin | 0.1988 | 0.1454 | not reported | not reported | not reported |

Leaderboard position 19 at the time of writing. The Mock phase ran 2 cases, so these numbers are not comparable with the local validation means.

**Container and checks**
- `Dockerfile` and `inference.py` follow the organizers' template I/O. Weights are uploaded as the algorithm model, not baked into the image. The image was built by Grand Challenge from the GitHub tag.
- Outputs are uint8 OME-TIFFs with the input OME-XML (pixel type set to uint8), written with tifffile. A local test with libvips 8.15 aborted on 16-bit tiled outputs in panimg, uint8 passed.
- CI (GitHub Actions) builds the image and runs it with `--network none` on synthetic OME-TIFF inputs, then checks names, shapes, dtype, OME-XML and a panimg read.
- `check_real_inputs.py` ran `inference.py` on 20 real validation images: all outputs present, channel order confirmed, and the uint8 rounding changed per-image scores by at most 0.0032 SSIM and 0.00016 PCC.
- The challenge OME-XML contains a non-ASCII unit (the micro sign), which tifffile rejects as a str description. Fixed in `c53165b` by writing UTF-8 bytes.

**Open point from the portal metrics:** MAE and Euclidean distance are in the thousands and millions, so the evaluator does not seem to normalize predictions before these two metrics, unlike SSIM and PCC. Our outputs are on a 0 to 255 scale while the targets in our local split are 16-bit, so these two metrics are penalized by the output scale. Writing predictions in the target intensity range is a cheap fix to test before the next submission. This is an inference from the numbers, not confirmed by the organizers.
