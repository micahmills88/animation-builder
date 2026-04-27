# Retarget worker — FEX-Emu sidecar.
#
# Architecture:
#   Stage 1 builds an x86_64 Python environment with fbxsdkpy + the worker
#   code installed normally. This is the "rootfs" the worker actually runs
#   in. It's an x86 image so the install steps see x86 libs/wheels.
#
#   Stage 2 is aarch64 native (where FEX lives — FEX is itself an ARM64
#   binary). It installs FEX from the official PPA, then COPYs Stage 1's
#   filesystem in as /opt/fex-rootfs/. ENTRYPOINT launches uvicorn via
#   FEXBash, which sets up the x86 environment and runs the x86 python
#   under FEX-Emu binary translation.
#
# Why this beats the previous (--platform linux/amd64 + qemu binfmt) setup:
#   - Single container, FEX runs in its own aarch64 environment with proper
#     ARM64 libs (FEX is dynamically linked; this avoided the rabbit hole)
#   - No host-level binfmt registration needed
#   - FEX is faster than qemu-user for CPU-bound code (binary recompiler vs
#     qemu's TCG splatter-JIT) — community benchmarks show 2-3x
#   - The HTTP boundary (POST /retarget) is unchanged so animation-service
#     doesn't know or care which emulator is under the hood.

# ---------------------------------------------------------------------------
# Stage 1: build the x86 Python environment that FEX will run.
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.10-slim-bookworm AS x86-env

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Vendored fbxsdkpy install — drop the pre-built .so + dist-info into
# site-packages so `import fbx` works. (Inria's PyPI is unreliable; we keep
# the wheel in vendor/ so builds are deterministic.)
COPY vendor/fbxsdkpy/ /usr/local/lib/python3.10/site-packages/

RUN pip install --no-cache-dir \
        "numpy>=1.26,<2" \
        "scipy>=1.11,<2" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.9"

# CPU-only torch for motion_correction's Python wrapper (motion_postprocess.py)
# which calls .detach().cpu() on its tensor inputs. We never run anything on
# GPU here — the worker is CPU-bound emulation. CPU-only torch is ~200 MB
# vs the full CUDA build's ~2 GB.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4,<2.7"

# motion_correction (Kimodo's foot-contact correction + smoothing C++ extension).
# Build it natively for x86 here — no simde, no NEON guessing — so the math
# is byte-identical to what kimodo expects upstream. The animation-service
# monkey-patches motion_correction.motion_postprocess.correct_motion to POST
# to /postprocess on this worker instead of running the C++ in-process.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential git && \
    rm -rf /var/lib/apt/lists/*
COPY upstream/MotionCorrection /tmp/MotionCorrection
RUN cd /tmp/MotionCorrection && pip install --no-cache-dir . && rm -rf /tmp/MotionCorrection

COPY worker/ /workspace/worker/

# ---------------------------------------------------------------------------
# Stage 2: aarch64 native + FEX-Emu, hosting Stage 1 as the x86 rootfs.
# ---------------------------------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# fex-emu-armv8.4 is the right variant for GB10 (Neoverse V2 implements
# Armv9-A, backwards-compatible with all Armv8 variants — pick the highest
# instruction-set version FEX targets for the best codegen).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl && \
    add-apt-repository -y ppa:fex-emu/fex && \
    apt-get update && \
    apt-get install -y --no-install-recommends fex-emu-armv8.4 && \
    rm -rf /var/lib/apt/lists/*

# Bring the x86 environment in as FEX's RootFS. Everything in stage 1 ends
# up under /opt/fex-rootfs/ — usr/, lib/, bin/, workspace/, etc.
COPY --from=x86-env / /opt/fex-rootfs/

# FEX reads its config from ~/.fex-emu/Config.json. Tell it where the
# rootfs lives so it can resolve x86 dynamic-linker + library paths.
RUN mkdir -p /root/.fex-emu && \
    echo '{"Config":{"RootFS":"/opt/fex-rootfs"}}' > /root/.fex-emu/Config.json

# Build-time linkability check — fail loudly here if FEX can't load the
# vendored fbxsdkpy or the freshly-built motion_correction. FEXBash drops
# into the x86 bash inside the rootfs.
RUN FEXBash -c "python -c 'import fbx; m = fbx.FbxManager.Create(); m.Destroy(); print(\"fbxsdkpy under FEX OK\")'"
RUN FEXBash -c "python -c 'from motion_correction import motion_postprocess; print(\"motion_correction under FEX OK:\", motion_postprocess.correct_motion)'"

WORKDIR /opt/fex-rootfs/workspace
EXPOSE 9560

# FEX does dynamic library translation, NOT filesystem chroot. The x86
# python process sees the OUTER aarch64 container's filesystem — so paths
# must be addressed from there (/opt/fex-rootfs/...), not as if they were
# inside the rootfs. PYTHONPATH points at the actual location.
ENV PYTHONPATH=/opt/fex-rootfs/workspace
CMD ["FEXBash", "-c", "exec /opt/fex-rootfs/usr/local/bin/python -m uvicorn worker.server:app --host 0.0.0.0 --port 9560"]
