"""FastAPI app — single-animation Kimodo generator + static UI + per-clip editor.

One prompt → one clip. The prompt is handed to Kimodo verbatim — no slot
catalog, no personality composition, no constraints. The user iterates by
editing prompt/duration/seed and hitting Regenerate, or by scrubbing trim
markers in the viewer and hitting Save.

Endpoints:
    GET    /                               redirect to /ui/
    GET    /api/models                     allowed Kimodo model ids
    POST   /api/animations                 multipart/form-data — start a job
    GET    /api/animations                 list all jobs (summary)
    GET    /api/animations/{id}            full job detail
    DELETE /api/animations/{id}            remove workdir
    POST   /api/animations/{id}/rerun      re-run with identical inputs (new job)
    GET    /api/animations/{id}/clip       current FBX (trimmed if saved)
    GET    /api/animations/{id}/editor-data pose data + trim state
    POST   /api/animations/{id}/trim       update trim markers (save=true re-exports)
    POST   /api/animations/{id}/regenerate re-run in place with new prompt/duration/seed
    GET    /api/animations/{id}/template   uploaded template FBX

Singleton: ``KimodoRuntime`` is loaded once at startup and reused across
jobs. When a job requests a different model, we reload on demand. All
job execution runs on a single-worker ThreadPoolExecutor (GPU-bound).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.kimodo_runtime import DEFAULT_MODEL_NAME, KimodoRuntime
from app.storage import (
    AnimationJob,
    job_workdir,
    list_jobs,
    load_job,
    save_job,
    to_summary,
    utcnow_iso,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("animation-service")

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/workspace/animation-service/outputs")).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Animation-Builder Single-Clip Generator")

UI_DIR = Path(__file__).resolve().parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

_executor = ThreadPoolExecutor(max_workers=1)  # GPU-bound, one generation at a time

_runtime: KimodoRuntime | None = None
_runtime_lock = Lock()


AVAILABLE_MODELS = [
    {"id": "Kimodo-SOMA-RP-v1.1",   "label": "SOMA RP v1.1 (realistic, default)"},
    {"id": "Kimodo-SOMA-SEED-v1.1", "label": "SOMA SEED v1.1 (stylized)"},
]
_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}

MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0


def _get_or_load_runtime(model_name: str) -> KimodoRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None or (
            _runtime.resolved_model.lower() != model_name.lower()
            and model_name not in _runtime.resolved_model
        ):
            log.info("loading Kimodo runtime: %s", model_name)
            _runtime = KimodoRuntime(model_name=model_name)
            log.info("Kimodo runtime ready: %s", _runtime.resolved_model)
    return _runtime


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "unnamed"


def _validate_generation_params(prompt: str, duration_s: float, model: str) -> None:
    if not prompt:
        raise HTTPException(400, "prompt is required")
    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        raise HTTPException(
            400, f"duration_s must be in [{MIN_DURATION_S}, {MAX_DURATION_S}] seconds"
        )
    if model not in _MODEL_IDS:
        raise HTTPException(400, f"unknown model {model!r}; choose from {sorted(_MODEL_IDS)}")


def _generate_and_retarget(meta: AnimationJob) -> None:
    """Run Kimodo + retarget to FBX for a single job. Mutates meta in place
    and persists on each state transition. Re-used by the initial generate
    path and the regenerate-in-place path."""
    from app.npz_to_fbx import retarget_npz_to_fbx

    workdir = job_workdir(OUTPUT_ROOT, meta.id)
    fbx_dir = workdir / "fbx"
    npz_dir = workdir / "npz"
    fbx_dir.mkdir(exist_ok=True)
    npz_dir.mkdir(exist_ok=True)

    template_path = workdir / (meta.template_fbx or "")
    if not meta.template_fbx or not template_path.exists():
        raise FileNotFoundError(f"Template FBX missing on disk: {template_path}")

    runtime = _get_or_load_runtime(meta.model)
    info = runtime.skeleton_info

    log.info(
        "[%s] generate seed=%d duration=%.2fs prompt=%r",
        meta.id, meta.seed, meta.duration_s, meta.prompt,
    )
    npz = runtime.generate(
        prompt=meta.prompt,
        duration_s=meta.duration_s,
        seed=meta.seed,
        constraint_lst=[],
    )
    total_frames = int(npz["global_rot_mats"].shape[0])

    npz_path = npz_dir / "clip.npz"
    payload = {k: v for k, v in npz.items()
               if isinstance(v, np.ndarray) and not k.startswith("__")}
    np.savez_compressed(npz_path, **payload)

    fbx_out = fbx_dir / "clip.fbx"
    retarget_npz_to_fbx(
        posed_joints=npz["posed_joints"],
        global_rot_mats=npz["global_rot_mats"],
        joint_names=info.joint_names,
        joint_parents=info.joint_parents,
        neutral_joints=info.neutral_joints,
        fps=info.fps,
        target_fbx_path=template_path,
        output_fbx_path=fbx_out,
        skeleton_name=info.skeleton_name,
    )

    meta.total_frames = total_frames
    meta.trim_start_frame = 0
    meta.trim_end_frame = total_frames - 1
    meta.saved = False
    meta.fbx_size = fbx_out.stat().st_size
    # Regenerate paths reuse the same AnimationJob — reset seam settings
    # so a fresh clip doesn't inherit stale blend/bridge values.
    meta.blend_frames = 0
    meta.bridge_frames = 0
    log.info(
        "[%s] done → %s (%d bytes, %d frames)",
        meta.id, fbx_out, meta.fbx_size, total_frames,
    )


def _run_job_wrapper(job_id: str) -> None:
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        log.error("job %s vanished before generation could start", job_id)
        return
    try:
        _generate_and_retarget(meta)
        meta.status = "completed"
        meta.finished_at = utcnow_iso()
        meta.error = None
        save_job(OUTPUT_ROOT, meta)
    except Exception as exc:
        log.exception("job %s failed", job_id)
        meta.status = "failed"
        meta.error = str(exc)
        meta.finished_at = utcnow_iso()
        save_job(OUTPUT_ROOT, meta)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models() -> list[dict[str, str]]:
    return AVAILABLE_MODELS


@app.post("/api/animations")
async def create_animation(
    prompt: str = Form(...),
    seed: int = Form(42),
    duration_s: float = Form(10.0),
    model: str = Form(DEFAULT_MODEL_NAME),
    template_fbx: UploadFile = File(...),
) -> dict[str, Any]:
    prompt = (prompt or "").strip()
    _validate_generation_params(prompt, duration_s, model)
    if template_fbx is None or not template_fbx.filename:
        raise HTTPException(
            400,
            "template_fbx is required — upload a Mixamo-rigged FBX with embedded textures",
        )

    job_id = uuid.uuid4().hex
    workdir = job_workdir(OUTPUT_ROOT, job_id)
    workdir.mkdir(parents=True, exist_ok=True)

    template_filename = _safe_filename(Path(template_fbx.filename).stem) + ".fbx"
    dest = workdir / template_filename
    dest.write_bytes(await template_fbx.read())
    log.info("template uploaded: %s (%d bytes)", dest, dest.stat().st_size)

    meta = AnimationJob(
        id=job_id,
        prompt=prompt,
        seed=seed,
        duration_s=float(duration_s),
        model=model,
        template_fbx=template_filename,
        created_at=utcnow_iso(),
        finished_at=None,
        status="running",
        error=None,
    )
    save_job(OUTPUT_ROOT, meta)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_job_wrapper, job_id)
    return {"id": job_id, "status": "running"}


@app.post("/api/animations/{job_id}/rerun")
async def rerun_animation(job_id: str) -> dict[str, Any]:
    import shutil

    old = load_job(OUTPUT_ROOT, job_id)
    if not old:
        raise HTTPException(404, "Unknown job id")
    if not old.template_fbx:
        raise HTTPException(409, "original job has no template FBX saved")
    old_template_path = job_workdir(OUTPUT_ROOT, job_id) / old.template_fbx
    if not old_template_path.exists():
        raise HTTPException(410, f"original template missing on disk: {old_template_path.name}")

    new_job_id = uuid.uuid4().hex
    new_workdir = job_workdir(OUTPUT_ROOT, new_job_id)
    new_workdir.mkdir(parents=True, exist_ok=True)
    new_template_path = new_workdir / old.template_fbx
    shutil.copyfile(old_template_path, new_template_path)

    meta = AnimationJob(
        id=new_job_id,
        prompt=old.prompt,
        seed=old.seed,
        duration_s=old.duration_s,
        model=old.model,
        template_fbx=old.template_fbx,
        created_at=utcnow_iso(),
        finished_at=None,
        status="running",
        error=None,
    )
    save_job(OUTPUT_ROOT, meta)
    log.info("rerunning job %s → new job %s", job_id, new_job_id)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_job_wrapper, new_job_id)
    return {"id": new_job_id, "status": "running", "based_on": job_id}


@app.get("/api/animations")
async def animations() -> list[dict[str, Any]]:
    return [to_summary(m) for m in list_jobs(OUTPUT_ROOT)]


@app.get("/api/animations/{job_id}")
async def animation_detail(job_id: str) -> dict[str, Any]:
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")
    return to_summary(meta)


@app.delete("/api/animations/{job_id}")
async def delete_animation(job_id: str) -> dict[str, Any]:
    import shutil

    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")
    workdir = job_workdir(OUTPUT_ROOT, job_id)
    if workdir.exists():
        shutil.rmtree(workdir)
    return {"id": job_id, "deleted": True}


# ---------------------------------------------------------------------------
# Clip — FBX + editor data + trim + in-place regenerate
# ---------------------------------------------------------------------------

@app.get("/api/animations/{job_id}/clip")
async def clip_fbx(job_id: str) -> FileResponse:
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")
    fbx = job_workdir(OUTPUT_ROOT, job_id) / "fbx" / "clip.fbx"
    if not fbx.exists():
        raise HTTPException(404, "clip not ready")
    # Prevent the browser from serving stale bytes after Save/Regenerate
    # re-exports the file. The client also cache-busts via a ?v= query,
    # but setting must-revalidate here is belt-and-suspenders.
    return FileResponse(
        path=str(fbx),
        media_type="model/fbx",
        filename=f"{job_id}.fbx",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@app.get("/api/animations/{job_id}/editor-data")
async def clip_editor_data(job_id: str) -> dict[str, Any]:
    """Pose data + trim state the browser editor needs.

    ``posed_joints`` is returned flattened so the browser can recompute
    per-frame pose-distance (root-relative Euclidean) for the slider
    gradients against any reference frame (start OR end).
    """
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")
    if meta.status != "completed" or meta.total_frames is None:
        raise HTTPException(409, f"job not ready: status={meta.status}")

    npz_path = job_workdir(OUTPUT_ROOT, job_id) / "npz" / "clip.npz"
    if not npz_path.exists():
        raise HTTPException(410, "raw NPZ no longer on disk")

    with np.load(npz_path) as data:
        posed_joints = data["posed_joints"]         # (T, J, 3)

    T, J = posed_joints.shape[:2]

    runtime = _get_or_load_runtime(meta.model)
    joint_parents = list(runtime.skeleton_info.joint_parents)

    return {
        "job_id": job_id,
        "total_frames": int(T),
        "num_joints": int(J),
        "fps": float(runtime.skeleton_info.fps),
        "trim_start_frame": int(meta.trim_start_frame or 0),
        "trim_end_frame": int(meta.trim_end_frame or (T - 1)),
        "saved": bool(meta.saved),
        "blend_frames": int(meta.blend_frames or 0),
        "bridge_frames": int(meta.bridge_frames or 0),
        "joint_parents": joint_parents,
        "prompt": meta.prompt,
        "seed": meta.seed,
        "duration_s": float(meta.duration_s),
        "model": meta.model,
        "posed_joints": posed_joints.astype(np.float32).reshape(-1).tolist(),
    }


def _reexport_trimmed_fbx(
    job_id: str,
    start_frame: int,
    end_frame: int,
    blend_frames: int,
    bridge_frames: int,
) -> int:
    """Re-export the FBX for this job using the given trim + seam settings.

    Order of operations:
        1. Slice the raw NPZ to [start_frame, end_frame].
        2. If ``blend_frames`` > 0: smoothstep-ease the last N frames of
           the slice IN PLACE toward the start-frame pose (cyclic blend).
           Clip length unchanged. Right for cyclic motions.
        3. If ``bridge_frames`` > 0: SLERP N new frames between (possibly
           blended) end and start and APPEND them to the slice. Clip
           grows by N. Right for one-way motions.
        4. Retarget the combined sequence onto the template FBX.

    Each step is optional. Typical usage sets exactly one of
    ``blend_frames`` or ``bridge_frames`` — both can coexist, but if
    blend already closes the loop the bridge becomes redundant.
    """
    from app.npz_to_fbx import retarget_npz_to_fbx

    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")
    workdir = job_workdir(OUTPUT_ROOT, job_id)
    npz_path = workdir / "npz" / "clip.npz"
    template_path = workdir / (meta.template_fbx or "")
    if not npz_path.exists():
        raise HTTPException(410, "raw NPZ missing")
    if not template_path.exists():
        raise HTTPException(410, "template FBX missing — cannot re-export")

    runtime = _get_or_load_runtime(meta.model)
    info = runtime.skeleton_info

    with np.load(npz_path) as data:
        posed_joints = data["posed_joints"][start_frame:end_frame + 1].copy()
        global_rot_mats = data["global_rot_mats"][start_frame:end_frame + 1].copy()
        root_positions_slice = data["root_positions"][start_frame:end_frame + 1].copy()

    # Step 2 — retroactive cyclic blend.
    if blend_frames > 0:
        from app.seam_blend import apply_cyclic_blend

        diag = apply_cyclic_blend(
            global_rot_mats=global_rot_mats,
            posed_joints=posed_joints,
            root_positions=root_positions_slice,
            neutral_joints=np.asarray(info.neutral_joints),
            joint_parents=list(info.joint_parents),
            blend_frames=blend_frames,
        )
        log.info(
            "[%s] cyclic blend: eased last %d frames toward start pose "
            "(rot_delta=%.6f, root_delta=%.6f, posed_delta=%.6f)",
            job_id,
            diag["blend_frames_applied"],
            diag.get("last_to_first_rot_delta", -1.0),
            diag.get("last_to_first_root_delta", -1.0),
            diag.get("last_to_first_posed_delta", -1.0),
        )

    # Step 3 — appended bridge frames.
    if bridge_frames > 0:
        from app.seam_blend import build_bridge_frames

        # End/start of the (possibly blended) slice:
        global_rot_end = global_rot_mats[-1]
        global_rot_start = global_rot_mats[0]
        root_pos_end = root_positions_slice[-1]
        root_pos_start = root_positions_slice[0]

        bridge_posed, bridge_global = build_bridge_frames(
            global_rot_mats_end=global_rot_end,
            global_rot_mats_start=global_rot_start,
            root_position_end=root_pos_end,
            root_position_start=root_pos_start,
            clip_length_frames=global_rot_mats.shape[0],
            neutral_joints=np.asarray(info.neutral_joints),
            joint_parents=list(info.joint_parents),
            n_bridge=bridge_frames,
        )
        posed_joints = np.concatenate([posed_joints, bridge_posed], axis=0)
        global_rot_mats = np.concatenate([global_rot_mats, bridge_global], axis=0)
        log.info(
            "[%s] bridge: appended %d frames (clip now %d frames)",
            job_id, bridge_frames, posed_joints.shape[0],
        )

    fbx_out = workdir / "fbx" / "clip.fbx"
    retarget_npz_to_fbx(
        posed_joints=posed_joints,
        global_rot_mats=global_rot_mats,
        joint_names=info.joint_names,
        joint_parents=info.joint_parents,
        neutral_joints=info.neutral_joints,
        fps=info.fps,
        target_fbx_path=template_path,
        output_fbx_path=fbx_out,
        skeleton_name=info.skeleton_name,
    )
    return fbx_out.stat().st_size


def _regenerate_in_place(
    job_id: str, prompt: str, duration_s: float, seed: int,
) -> dict[str, Any]:
    """Regenerate this job's NPZ+FBX with edited prompt/duration/seed.
    Trim resets to full range; saved flips false."""
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")

    meta.prompt = prompt
    meta.duration_s = float(duration_s)
    meta.seed = int(seed)
    meta.status = "running"
    meta.error = None
    save_job(OUTPUT_ROOT, meta)

    try:
        _generate_and_retarget(meta)
        meta.status = "completed"
        meta.finished_at = utcnow_iso()
    except Exception as exc:
        log.exception("regenerate %s failed", job_id)
        meta.status = "failed"
        meta.error = str(exc)
        meta.finished_at = utcnow_iso()
        save_job(OUTPUT_ROOT, meta)
        raise HTTPException(500, f"regenerate failed: {exc}")

    save_job(OUTPUT_ROOT, meta)
    return to_summary(meta)


@app.post("/api/animations/{job_id}/regenerate")
async def animation_regenerate(
    job_id: str,
    prompt: str = Form(...),
    duration_s: float = Form(...),
    seed: int = Form(...),
    model: str | None = Form(None),
) -> dict[str, Any]:
    prompt = (prompt or "").strip()
    if model is None:
        # If caller didn't send a model, keep the existing one.
        existing = load_job(OUTPUT_ROOT, job_id)
        if not existing:
            raise HTTPException(404, "Unknown job id")
        model = existing.model
    _validate_generation_params(prompt, duration_s, model)

    # Persist model change (if any) before queuing.
    existing = load_job(OUTPUT_ROOT, job_id)
    if existing and existing.model != model:
        existing.model = model
        save_job(OUTPUT_ROOT, existing)

    return await asyncio.get_event_loop().run_in_executor(
        _executor, _regenerate_in_place, job_id, prompt, duration_s, seed,
    )


@app.post("/api/animations/{job_id}/trim")
async def animation_trim(
    job_id: str,
    trim_start_frame: int = Form(...),
    trim_end_frame: int = Form(...),
    save: bool = Form(False),
    blend_frames: int = Form(0),
    bridge_frames: int = Form(0),
) -> dict[str, Any]:
    """Update trim markers and seam settings. When ``save=True`` the FBX
    is re-exported with the (optionally) cyclic-blended slice plus any
    bridge frames appended, and ``saved`` flips True. When False the
    markers are stored but the FBX stays as-is — cheap preview path."""
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta:
        raise HTTPException(404, "Unknown job id")
    T = meta.total_frames
    if T is None:
        raise HTTPException(409, "clip has no raw frame data")
    if not (0 <= trim_start_frame < trim_end_frame < T):
        raise HTTPException(
            400,
            f"invalid trim [{trim_start_frame}, {trim_end_frame}] for clip length {T}",
        )
    kept = trim_end_frame - trim_start_frame + 1
    if not (0 <= blend_frames <= max(0, kept - 1)):
        raise HTTPException(
            400,
            f"blend_frames must be in [0, {max(0, kept - 1)}] — can't blend "
            "more frames than exist in the kept slice (minus one anchor).",
        )
    if not (0 <= bridge_frames <= 600):
        raise HTTPException(400, "bridge_frames must be in [0, 600]")

    meta.trim_start_frame = int(trim_start_frame)
    meta.trim_end_frame = int(trim_end_frame)
    meta.blend_frames = int(blend_frames)
    meta.bridge_frames = int(bridge_frames)

    if save:
        new_size = await asyncio.get_event_loop().run_in_executor(
            _executor, _reexport_trimmed_fbx,
            job_id, meta.trim_start_frame, meta.trim_end_frame,
            meta.blend_frames, meta.bridge_frames,
        )
        meta.fbx_size = int(new_size)
        meta.saved = True
        log.info(
            "[%s] saved: trim [%d, %d] blend=%d bridge=%d → %d bytes",
            job_id, meta.trim_start_frame, meta.trim_end_frame,
            meta.blend_frames, meta.bridge_frames, new_size,
        )

    save_job(OUTPUT_ROOT, meta)
    return to_summary(meta)


@app.get("/api/animations/{job_id}/template")
async def job_template(job_id: str) -> FileResponse:
    meta = load_job(OUTPUT_ROOT, job_id)
    if not meta or not meta.template_fbx:
        raise HTTPException(404, "template not found")
    fbx = job_workdir(OUTPUT_ROOT, job_id) / meta.template_fbx
    if not fbx.exists():
        raise HTTPException(404, "template no longer on disk")
    return FileResponse(path=str(fbx), media_type="model/fbx", filename=fbx.name)


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/", status_code=307)
