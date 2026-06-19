# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_DATA_DIR=/data \
    APP_VENV=/opt/app-venv \
    DYNAGAN_DIR=/opt/Dynagan \
    DYNAGAN_PYTHON=/opt/dynagan-venv/bin/python \
    DEEPDRR_MODE=python

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ffmpeg \
    build-essential \
    ca-certificates \
    git \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libgles2 \
    freeglut3-dev \
    python3 \
    python3-dev \
    python3-venv \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
  && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9

RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ \
  && printf '%s\n' \
    '{' \
    '  "file_format_version": "1.0.0",' \
    '  "ICD": {' \
    '    "library_path": "libEGL_nvidia.so.0"' \
    '  }' \
    '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /app
COPY requirements ./requirements

RUN --mount=type=cache,target=/root/.cache/pip \
  python3 -m venv /opt/app-venv \
  && /opt/app-venv/bin/python -m pip install --upgrade pip "setuptools<70" wheel \
  && /opt/app-venv/bin/python -m pip install -r requirements/app.txt

# The default image is self-contained for end users: it installs the app,
# DeepDRR, and Dynagan during the build.
ARG INSTALL_DEEPDRR=true
ARG INSTALL_DYNAGAN=true
ARG DYNAGAN_PRETRAINED_URL=https://ubocloud.univ-brest.fr/s/fT6saSzZjGCbmAS/download/pretrained_model.pth
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DYNAGAN" = "true" ]; then \
      python3.9 -m venv /opt/dynagan-venv && \
      /opt/dynagan-venv/bin/python -m pip install --upgrade pip "setuptools<70" wheel numpy==1.23.2 && \
      /opt/dynagan-venv/bin/python -m pip install --dry-run --prefer-binary --no-build-isolation -r requirements/dynagan.txt && \
      /opt/dynagan-venv/bin/python -m pip install --prefer-binary --no-build-isolation -r requirements/dynagan.txt && \
      /opt/dynagan-venv/bin/python -m pip check && \
      /opt/dynagan-venv/bin/python -c "import torch, torchvision; print('Dynagan torch:', torch.__version__)"; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DEEPDRR" = "true" ]; then \
      git clone https://github.com/arcadelab/deepdrr.git /opt/deepdrr && \
      /opt/app-venv/bin/python -m pip install --no-cache-dir -r requirements/deepdrr.txt && \
      /opt/app-venv/bin/python -m pip install --no-cache-dir "/opt/deepdrr[cuda11x]" pycuda && \
      /opt/app-venv/bin/python -c "from cuda import cudart; from numba import guvectorize, int64; from pyparsing import alphas; from deepdrr import Volume, MobileCArm; from deepdrr.projector import Projector; print('DeepDRR imports OK')"; \
    fi
RUN if [ "$INSTALL_DYNAGAN" = "true" ]; then \
      git clone https://github.com/cyiheng/Dynagan.git /opt/Dynagan && \
      rm -f /opt/Dynagan/datasets/imagesTs/*.nii.gz && \
      rm -f /opt/Dynagan/datasets/tumor/*.nii.gz && \
      mkdir -p /opt/Dynagan/checkpoints/pretrained_model && \
      curl -L --fail --retry 3 "$DYNAGAN_PRETRAINED_URL" -o /opt/Dynagan/checkpoints/pretrained_model/latest_net_G.pth && \
      test -s /opt/Dynagan/checkpoints/pretrained_model/latest_net_G.pth && \
      test -f /opt/Dynagan/test_3D.py; \
    fi

COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
RUN /opt/app-venv/bin/python -m pip install --no-deps .

RUN mkdir -p /data/jobs
EXPOSE 8080

CMD ["/opt/app-venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
