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

COPY --chown=user:user config.py dataset.py evaluate.py predict.py unet.py ome_utils.py inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
