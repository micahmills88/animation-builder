# Animation-service: FastAPI app on top of the upstream Kimodo image.
#
# The upstream image (kimodo:1.0) is built by the text-encoder service in
# docker-compose.yml. Use scripts/start.sh on first run to build in the
# right order.
#
# The Kimodo motion-model checkpoint is downloaded LAZILY at runtime on
# first generation request. That keeps the image small and the build fast
# and also means you don't need a HuggingFace token to BUILD — only to
# RUN (and only if the text-encoder's LLM2Vec model is gated on HF).
#
# fbxsdkpy is NOT installed in this image. The FBX SDK is x86-only, so on
# aarch64 hosts (the GB10) the actual retarget runs in a sidecar
# `retarget-worker` container under qemu-user emulation. This service POSTs
# arrays + template FBX to RETARGET_URL and writes the returned FBX to disk.
# Worker code lives in worker/, not app/.
FROM kimodo:1.0

ENV HF_HOME=/opt/hf-cache
RUN mkdir -p ${HF_HOME}

WORKDIR /workspace/animation-service

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY worker/ ../worker/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/workspace/animation-service:/workspace
ENV OUTPUT_ROOT=/workspace/animation-service/outputs

EXPOSE 7870
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7870"]
