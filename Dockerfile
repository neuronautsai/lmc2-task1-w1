# Grand Challenge algorithm container for Light My Cells 2, Task 1 (Best Focus).
# Built from the repository root. Weights are not in the image: they are uploaded to Grand Challenge as the
# algorithm model tarball and appear under /opt/ml/model at run time.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user requirements-submission.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements-submission.txt

COPY --chown=user:user config.py dataset.py evaluate.py predict.py unet.py ome_utils.py postprocess.py inference.py output_scale.json /opt/app/

# Output mode: uint8 is the v0.1.0 behaviour, float32 writes predictions on a typical raw target scale (see inference.py).
# Tiling: Hann window with 128 px overlap. TTA: 1 (none), 4 (flips) or 8 (flips and rotations). Every *.pt in the model
# tarball is an ensemble member. Optional members.json and postprocess.json next to the weights (see inference.py).
ENV LMC2_OUTPUT_MODE=uint8 LMC2_WINDOW=hann LMC2_OVERLAP=128 LMC2_TTA=1

ENTRYPOINT ["python", "inference.py"]
