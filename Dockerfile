FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ghostscript \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Make pip resilient to a flaky/slow connection (large wheels below).
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=10

# docling/easyocr/sentence-transformers pull torch (+ torchvision, via docling's
# layout/table-detection model) in transitively and would otherwise resolve a
# multi-GB CUDA wheel that goes unused here — this container's own torch
# workloads (OCR, reranker) are CPU-only by design, the GPU is reserved for
# the Ollama service. torch and torchvision MUST be installed together from
# the same index: installing torch alone here and letting torchvision come in
# later from the default PyPI index (via requirements.txt) pulls a torchvision
# build compiled against a different torch version, breaking compiled ops
# like torchvision::nms at runtime.
#
# Separate RUN layer from requirements.txt on purpose: these are the two
# largest/slowest downloads in the image, so if the requirements.txt layer
# below fails (or a later `docker compose build --build-arg`/code change
# invalidates it), this layer stays cached instead of re-downloading torch.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN mkdir -p data/uploads data/sessions

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
