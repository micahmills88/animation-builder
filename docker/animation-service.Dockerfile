# Animation-service: FastAPI app on top of the upstream Kimodo image.
#
# The upstream image (kimodo:1.0) is built by the text-encoder service in
# docker-compose.yml. Use scripts/start.sh on first run to build in the
# right order, or build manually in sequence: text-encoder first, then
# animation-service.
#
# The Kimodo motion-model checkpoint is downloaded LAZILY at runtime on
# first generation request. That keeps the image small and the build fast
# and also means you don't need a HuggingFace token to BUILD — only to
# RUN (and only if the text-encoder's LLM2Vec model is gated on HF).
FROM kimodo:1.0

# HF cache is persisted via a named docker volume (see compose).
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p ${HF_HOME}

WORKDIR /workspace/animation-service

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# FBX SDK Python bindings. Autodesk doesn't publish on PyPI; Inria's GitLab
# package index hosts a Linux build of `fbxsdkpy`. One pip install and we
# can read/write FBX natively with textures/materials preserved on
# round-trip — the entire "no-Blender" trick of the pipeline.
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple \
        fbxsdkpy==2020.1.post2

COPY app/ ./app/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/workspace/animation-service:/workspace
ENV OUTPUT_ROOT=/workspace/animation-service/outputs

EXPOSE 7870
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7870"]
