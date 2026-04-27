# Local override of upstream/Dockerfile — bumps the NGC PyTorch base image
# from the original 24.10-py3 (x86 only) to a multi-arch tag so the build
# works on aarch64. docker-compose.yml uses repo root as build context and
# points at this file, so the upstream submodule itself stays untouched.
#
# 26.03-py3 selected because it's already cached on the GB10 from other
# ML workloads. PyTorch 2.11 + CUDA 13 + Blackwell sm_121 all native.
# If you swap to a newer tag, verify kimodo's loose `numpy>=1.23,<2`
# constraint still resolves — that's the historic friction point.
#
# Differences from upstream/Dockerfile:
#   * FROM 26.03-py3 (multi-arch) instead of 24.10-py3 (x86 only)
#   * Installs from upstream/docker_requirements.in directly so pip can
#     resolve aarch64 wheels. The pinned x86 lockfile
#     (upstream/docker_requirements.txt) was compiled with
#     --python-platform x86_64-manylinux2014 and won't resolve here.
#   * COPY paths are relative to repo root, not upstream/.

FROM nvcr.io/nvidia/pytorch:26.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      cmake build-essential \
      gosu \
      libsimde-dev \
    && rm -rf /var/lib/apt/lists/*

# Some NGC base images ship a broken /usr/local/bin/cmake shim that shadows
# the system cmake and breaks MotionCorrection's CMake build. Strip it.
RUN rm -f /usr/local/bin/cmake || true

COPY upstream/docker_requirements.in /workspace/docker_requirements.in
COPY upstream/setup.py /workspace/setup.py
COPY upstream/pyproject.toml /workspace/pyproject.toml
COPY upstream/kimodo /workspace/kimodo
COPY upstream/kimodo-viser /workspace/kimodo-viser
COPY upstream/MotionCorrection /workspace/MotionCorrection

# aarch64 patches:
#
# 1. MotionCorrection: upstream's SIMD.h includes <immintrin.h> and uses
#    Intel intrinsics (__m128, _mm_*) that don't exist on ARM. We patch
#    SIMD.h to branch on architecture and pull in `simde` (SIMD Everywhere)
#    on aarch64 — a header-only library that maps Intel intrinsics to NEON
#    at compile time. With SIMDE_ENABLE_NATIVE_ALIASES the rest of the
#    code (Matrix.inl, Quaternion.inl) keeps using __m128 / _mm_* unchanged.
#    Combined with the CMakeLists patch below (drop -msse4.1/-mavx on
#    aarch64), MotionCorrection builds and we keep the post-processing
#    pass (foot-contact correction, motion smoothing).
#
# 2. MotionCorrection CMakeLists: gates the x86 -msse4.1/-mavx flags so
#    they're only set on x86 hosts.
#
# 3. scenepic: 1.1.2 source tarball is missing dist/scenepic.min.js so the
#    source build dies in CMake. They ship binary wheels only for x86_64.
#    kimodo declares scenepic in pyproject.toml deps but doesn't actually
#    import it (it's used by upstream visualization scripts we don't touch).
#    Strip from both pyproject.toml and the .in.
COPY docker/patches/MotionCorrection-SIMD.h /workspace/MotionCorrection/src/cpp/Math/SIMD.h
COPY docker/patches/MotionCorrection-CMakeLists.txt /workspace/MotionCorrection/CMakeLists.txt
RUN sed -i '/scenepic/d' /workspace/pyproject.toml \
 && sed -i '/^scenepic/d' /workspace/docker_requirements.in

# Install Python deps from the loose-pinned .in source (not the x86 lockfile).
# torch is intentionally absent from the .in — we use the NGC image's tested
# build instead.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && SKIP_MOTION_CORRECTION_IN_SETUP=1 python -m pip install -r docker_requirements.in

COPY upstream/kimodo/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
# Strip CRLF — the upstream submodule's .gitattributes doesn't enforce LF on
# .sh files, so a Windows checkout leaves bash seeing `bash\r` as the
# interpreter and exiting 127 immediately. Our parent .gitattributes can't
# reach inside the submodule's working tree.
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint \
 && chmod +x /usr/local/bin/docker-entrypoint

ENTRYPOINT ["docker-entrypoint"]
CMD ["bash"]
