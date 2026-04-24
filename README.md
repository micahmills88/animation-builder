# animation-builder

A game-developer-friendly UI on top of NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion model. Type a prompt, upload your Mixamo-rigged FBX, and generate a single animated FBX that drops straight into Unity or Unreal. The built-in viewer lets you scrub, auto-trim a clean loop, blend the seam, regenerate the whole clip in place, and download the result.

```
  prompt + your Mixamo-rigged FBX
           │
           ▼
   animation-service  ───►  Kimodo motion NPZ (in-process)  ───►  FBX SDK retarget  ───►  FBX
     (FastAPI :7870)                                              (textures embedded)
```

## Why this exists

Kimodo is a strong open text-to-motion model but ships as a research tool — single-clip generation in a Gradio demo, with motion as BVH, and no way to point it at your character. This project wraps it in a workflow that a game developer actually cares about:

- Bring your own FBX. The output uses your mesh, your rig, your skin weights, and your embedded textures. No white-character import.
- **NPZ-direct retarget via the Autodesk FBX SDK.** No Blender in the pipeline. No BVH intermediate. Textures and materials survive the round-trip.
- A browser viewer with per-frame pose-distance visualization, draggable trim markers, exhaustive pairwise auto-trim, cyclic-blend loop closure, and in-place regenerate.

## Requirements

- NVIDIA GPU with ~17 GB VRAM (RTX 4090 class). `nvidia-container-toolkit` installed.
- Docker Compose v2 with BuildKit.
- Linux or Windows host.
- A HuggingFace read-only token (see below).

### HuggingFace token setup

The text encoder (LLM2Vec) wraps a gated Meta LLaMA model, so HuggingFace will refuse to serve the weights without an authenticated user who has agreed to Meta's license. You only need to do this once:

1. Create a HuggingFace account if you don't have one.
2. Go to <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp> (or whichever LLM2Vec variant you'll be using) and click the "Agree and access repository" button. This sends a request to Meta and **approval can take up to ~30 minutes** (sometimes instant, sometimes half an hour). You'll get an email when it's granted. The stack won't be able to download the weights until then.
3. Create a read-only access token at <https://huggingface.co/settings/tokens> (role: **Read** is enough, you do NOT need Write).
4. Paste the token into `.env` as `HF_TOKEN=hf_...` when you do the Quick Start below.

The token is only read at model-download time. Once the weights are cached in the Docker volume, the stack runs fully offline.

## Quick start

```bash
git clone --recursive https://github.com/micahmills88/animation-builder.git
cd animation-builder

# Set up your HF token:
cp .env.example .env
# Edit .env and replace the HF_TOKEN placeholder with your actual token.

./scripts/start.sh
```

`scripts/start.sh` handles the ordered build:

1. Clones `nv-tlabs/kimodo-viser` into `upstream/` (upstream's Dockerfile expects it as a sibling; it can't be a submodule-inside-a-submodule).
2. Copies `.env.example` to `.env` if you haven't already.
3. Builds the upstream `kimodo:1.0` image (CUDA, PyTorch, Kimodo Python package).
4. Builds `animation-service` on top.
5. Starts all containers.

The first run downloads ~18 GB of model weights from HuggingFace (LLM2Vec + Kimodo SOMA). After that the named Docker volume caches them.

## What you get

Once everything is up:

- **http://localhost:7870** — the animation builder UI (this project)
- **http://localhost:7860** — upstream Kimodo's unmodified single-clip Viser demo, for comparison

All per-animation data (uploaded template, raw Kimodo NPZ, retargeted FBX, metadata) lives inside a named Docker volume — nothing touches your host filesystem. This is deliberate: you can run the whole stack headless on a cloud GPU or a GB10 and drive it entirely through the web UI (upload template → generate → trim → download FBX). If you want to back up or migrate the volume:

```bash
# Export everything to outputs.tgz on your host:
docker run --rm \
    -v animation-builder_outputs:/d \
    -v "$(pwd)":/b \
    alpine tar czf /b/outputs.tgz -C /d .

# Or copy a specific job out:
docker cp animation-service:/workspace/animation-service/outputs/jobs/<id> ./<id>
```

## Generating an animation

In the UI:

1. Type a prompt describing the motion. The prompt is handed to Kimodo verbatim — no personality composition, no prompt massaging. Be direct: `A person jogs forward at a steady pace, arms relaxed, even stride`.
2. Set seed and duration.
3. Upload a Mixamo-rigged FBX. Export from Mixamo as **"T-pose with skin"** so textures are embedded.
4. Click *Generate*.

From the CLI:

```bash
curl -X POST http://localhost:7870/api/animations \
  -F 'prompt=A person jogs forward at a steady pace, arms relaxed, even stride' \
  -F 'seed=42' \
  -F 'duration_s=10' \
  -F 'template_fbx=@/path/to/MixamoRigged.fbx'
```

## Editing in the viewer

Click **View / Edit** on any completed animation to open the editor. You get:

- A 3D viewer that plays the retargeted FBX on loop, with a thin playhead across the top showing exactly where the current frame is and where the loop seam sits.
- A timeline bar shaded by per-frame pose distance to the start marker. Cyclical motions (walk, jog, idle) show up as repeating gradient patterns — visually obvious stride cycles.
- Draggable **start** and **end** markers plus an **Auto-trim** button that does exhaustive pairwise search for the best loop pair.
- Two seam-closure modes you can use independently or together:
  - **Blend last N frames** (green). Retroactively smoothstep-eases the last N frames of the kept slice toward the start-frame pose. Clip length unchanged. Right for cyclic motions.
  - **Add bridge N frames** (orange). Appends N synthesized SLERP frames after the end. Clip grows by N. Right for one-way motions (arm raise, door open) where there's no natural cycle to blend into.
- **Regenerate** — edit the prompt / duration / seed and replace the clip in place.

A seam-preview panel on the right overlays the start-frame pose (blue) and end-frame pose (yellow) so you can rotate the skeleton around and eyeball loop quality directly.

## Known quirks

- **Kimodo's prompt handling is not literal.** "A person jogs for a long time" is just as likely to produce eight seconds of standing still followed by two seconds of jogging as it is to produce ten seconds of jogging. Describe the motion you want; don't describe duration. The `duration_s` field controls clip length.
- **Motion quality is bounded by the training data.** Kimodo-SOMA is trained on academic mocap (AMASS-style), not game animation. Expect the occasional mid-sprint pirouette, hands clipping through hips, or weirdly floaty jog. Treat every generation as a draft and use the trim + blend + regenerate loop to curate.
- **One generation at a time.** The GPU queue is single-slot. Concurrent requests wait.
- **Model reload on switch.** Switching between SOMA-RP and SOMA-SEED reloads Kimodo weights (~10 s). The text encoder stays loaded either way.

## Smoke test

```bash
TEMPLATE_FBX=/path/to/MixamoRigged.fbx ./scripts/smoke_test.sh
```

Brings up the stack, generates a single idle clip, asserts the FBX lands on disk. Useful for verifying a fresh clone works end-to-end.

## Project layout

```
app/
  main.py              FastAPI app (endpoints) + static UI mount
  kimodo_runtime.py    in-process Kimodo singleton (loads once, generates many)
  npz_to_fbx.py        Kimodo NPZ → retargeted FBX via FBX SDK
  seam_blend.py        cyclic blend + bridge-frame generation
  storage.py           JSON-file-per-job metadata
  ui/                  browser UI (vanilla JS + three.js)
docker/
  animation-service.Dockerfile
scripts/
  start.sh             staged docker compose build + up
  smoke_test.sh        end-to-end stack test
  phase1_smoke.py      stand-alone one-shot CLI (runs inside the container)
docs/
  KIMODO_REFERENCE.md  file-and-line-level audit of upstream Kimodo
upstream/              git submodule → nv-tlabs/kimodo
```

## References

- Upstream Kimodo: <https://github.com/nv-tlabs/kimodo>
- Kimodo docs: <https://research.nvidia.com/labs/sil/projects/kimodo/docs/index.html>
- ComfyUI-Kimodo (reference retargeter that ours is adapted from): <https://github.com/jtydhr88/ComfyUI-Kimodo>
- FBX SDK Python bindings: Inria's GitLab PyPI — `pip install fbxsdkpy==2020.1.post2 --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple`

## License

Apache-2.0. See `LICENSE`. Third-party attributions (Autodesk FBX SDK, NVIDIA Kimodo, LLM2Vec / Meta LLaMA, ComfyUI-Kimodo, etc.) are in `NOTICE`.

This product contains Autodesk(R) FBX(R) SDK technology via the `fbxsdkpy` package. Autodesk, FBX, and the Autodesk logo are trademarks of Autodesk, Inc. The upstream Kimodo code under `upstream/` ships under its own Apache-2.0 license (`upstream/LICENSE`). Kimodo model weights are distributed under the NVIDIA Open Model License.
