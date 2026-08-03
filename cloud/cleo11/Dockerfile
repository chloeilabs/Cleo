FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace

# System deps for datasets / git-backed hub access
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY cleo1 ./cleo1
COPY configs ./configs
COPY cloud/cleo11/entrypoint.sh /usr/local/bin/cleo11-entrypoint

# Keep the CUDA torch from the base image; only add Hub extras + the project.
RUN pip install --no-cache-dir "datasets==4.0.0" "huggingface_hub==0.34.4" && \
    pip install --no-cache-dir --no-deps -e . && \
    chmod +x /usr/local/bin/cleo11-entrypoint

VOLUME ["/workspace/data/cleo11", "/workspace/artifacts/cleo11"]

ENTRYPOINT ["cleo11-entrypoint"]
