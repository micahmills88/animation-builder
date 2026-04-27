"""FBX retarget + motion_correction worker — x86 FEX-emulated sidecar.

Both endpoints exist because Autodesk's FBX SDK and Kimodo's MotionCorrection
C++ extension are x86 only (well, MotionCorrection has hardcoded SSE/AVX
intrinsics that don't compile cleanly on aarch64 without simde — and our
simde port produced wrong cross-product results in the IK math, breaking
foot orientation). Both run cleanly under FEX-Emu binary translation here.

POST /retarget multipart/form-data
    template_fbx : binary  — uploaded Mixamo-rigged template
    arrays_npz   : binary  — np.savez_compressed of posed_joints, global_rot_mats
    meta         : json    — joint_names/parents/neutral_joints/fps/etc.
returns: application/octet-stream — the retargeted FBX bytes

POST /postprocess multipart/form-data
    arrays_npz   : binary  — np.savez_compressed of hip_translations,
                              rotations, contacts, hip_translations_input,
                              rotations_input
    meta         : json    — working_rig, constraint_masks, contact_threshold,
                              root_margin, has_double_ankle_joints
returns: application/octet-stream — np.savez_compressed of corrected
         hip_translations + rotations (in-place result of correct_motion)
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from worker.retarget import retarget_npz_to_fbx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("retarget-worker")

WORK_DIR = Path(os.environ.get("WORKER_TMP", "/tmp/retarget"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FBX Retarget Worker")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retarget")
async def retarget(
    template_fbx: UploadFile = File(...),
    arrays_npz: UploadFile = File(...),
    meta: str = Form(...),
) -> Response:
    """Retarget one clip and stream the FBX bytes back to the caller."""
    req_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        params = json.loads(meta)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"meta is not valid JSON: {e}")

    required = ("joint_names", "joint_parents", "neutral_joints", "fps")
    for k in required:
        if k not in params:
            raise HTTPException(400, f"meta missing required key {k!r}")

    job_dir = WORK_DIR / req_id
    job_dir.mkdir()
    try:
        template_path = job_dir / "template.fbx"
        template_path.write_bytes(await template_fbx.read())
        npz_path = job_dir / "arrays.npz"
        npz_path.write_bytes(await arrays_npz.read())
        out_path = job_dir / "out.fbx"

        with np.load(npz_path) as data:
            posed_joints = data["posed_joints"]
            global_rot_mats = data["global_rot_mats"]

        log.info(
            "[%s] retarget %d frames, %d joints, template=%d bytes",
            req_id,
            posed_joints.shape[0],
            posed_joints.shape[1],
            template_path.stat().st_size,
        )

        retarget_npz_to_fbx(
            posed_joints=posed_joints,
            global_rot_mats=global_rot_mats,
            joint_names=list(params["joint_names"]),
            joint_parents=list(params["joint_parents"]),
            neutral_joints=np.asarray(params["neutral_joints"], dtype=np.float32),
            fps=float(params["fps"]),
            target_fbx_path=str(template_path),
            output_fbx_path=str(out_path),
            skeleton_name=str(params.get("skeleton_name", "kimodo_soma")),
            yaw_offset_deg=float(params.get("yaw_offset_deg", 0.0)),
            force_scale=float(params.get("force_scale", 0.0)),
            mapping=params.get("mapping"),
            preserve_root_xz=bool(params.get("preserve_root_xz", False)),
        )

        fbx_bytes = out_path.read_bytes()
        elapsed = time.time() - t0
        log.info("[%s] done in %.2fs (%d output bytes)", req_id, elapsed, len(fbx_bytes))
        return Response(
            content=fbx_bytes,
            media_type="application/octet-stream",
            headers={
                "X-Request-Id": req_id,
                "X-Elapsed-Seconds": f"{elapsed:.3f}",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[%s] retarget failed", req_id)
        raise HTTPException(500, f"retarget failed: {exc}")
    finally:
        # Best-effort cleanup; never let cleanup errors mask the real result.
        try:
            for p in job_dir.iterdir():
                p.unlink(missing_ok=True)
            job_dir.rmdir()
        except Exception:
            pass


@app.post("/postprocess")
async def postprocess(
    arrays_npz: UploadFile = File(...),
    meta: str = Form(...),
) -> Response:
    """Run kimodo's motion_correction.correct_motion on the provided arrays.

    Animation-service monkey-patches motion_correction's Python entry point
    to call this endpoint instead of running the C++ extension natively
    on aarch64 (where our simde port produces wrong rotations). All inputs
    follow the original correct_motion signature; the returned NPZ contains
    the corrected hip_translations + rotations only — animation-service
    copies them back into its own arrays in place.
    """
    req_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        params = json.loads(meta)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"meta is not valid JSON: {e}")

    required = ("working_rig", "constraint_masks", "contact_threshold",
                "root_margin", "has_double_ankle_joints")
    for k in required:
        if k not in params:
            raise HTTPException(400, f"meta missing required key {k!r}")

    try:
        with np.load(io.BytesIO(await arrays_npz.read())) as data:
            hip_translations = data["hip_translations"].copy()
            rotations = data["rotations"].copy()
            contacts = data["contacts"].copy()
            hip_translations_input = data["hip_translations_input"].copy()
            rotations_input = data["rotations_input"].copy()
    except Exception as e:
        raise HTTPException(400, f"arrays_npz parse failed: {e}")

    # Reconstruct working_rig as the SimpleNamespace list correct_motion expects.
    working_rig = [SimpleNamespace(**joint) for joint in params["working_rig"]]

    # constraint_masks: dict[str, list[float]] over the time axis. Kimodo's
    # default for unconstrained generation is all zeros.
    masks = {k: np.asarray(v, dtype=np.float32) for k, v in params["constraint_masks"].items()}

    try:
        import torch
        from motion_correction import motion_postprocess
    except ImportError as e:
        raise HTTPException(500, f"motion_correction or torch not installed in worker: {e}")

    # motion_postprocess.correct_motion expects torch tensors (it calls
    # .detach().cpu().flatten()). Wrap our numpy arrays once at the boundary;
    # masks too (each value is a per-frame float32 mask).
    hip_translations_t = torch.from_numpy(hip_translations)
    rotations_t = torch.from_numpy(rotations)
    contacts_t = torch.from_numpy(contacts)
    hip_translations_input_t = torch.from_numpy(hip_translations_input)
    rotations_input_t = torch.from_numpy(rotations_input)
    masks_t = {k: torch.from_numpy(v.copy()) for k, v in masks.items()}

    log.info(
        "[%s] postprocess B=%d T=%d J=%d contacts=%d",
        req_id,
        hip_translations_t.shape[0],
        hip_translations_t.shape[1],
        rotations_t.shape[2],
        contacts_t.shape[-1],
    )
    try:
        motion_postprocess.correct_motion(
            hip_translations_t,
            rotations_t,
            contacts_t,
            hip_translations_input_t,
            rotations_input_t,
            masks_t,
            float(params["contact_threshold"]),
            float(params["root_margin"]),
            working_rig,
            bool(params["has_double_ankle_joints"]),
        )
    except Exception as exc:
        log.exception("[%s] postprocess correct_motion failed", req_id)
        raise HTTPException(500, f"correct_motion failed: {exc}")

    out_buf = io.BytesIO()
    np.savez_compressed(
        out_buf,
        hip_translations=hip_translations_t.numpy(),
        rotations=rotations_t.numpy(),
    )

    elapsed = time.time() - t0
    log.info("[%s] postprocess done in %.2fs", req_id, elapsed)
    return Response(
        content=out_buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Request-Id": req_id,
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
        },
    )
