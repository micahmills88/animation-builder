"""Monkey-patch motion_correction.correct_motion to call the FEX worker.

Kimodo's post_process_motion (upstream/kimodo/postprocess.py) does all the
prep work — constructs working_rig, builds constraint masks, slices arrays,
calls forward kinematics — but the actual SIMD math lives in the
motion_correction C++ extension. That extension has hardcoded SSE/AVX
intrinsics; our simde -> NEON port produced wrong cross products and broke
foot orientation.

Fix: keep using kimodo's prep logic unchanged, but redirect the inner
correct_motion call to the worker (which builds motion_correction natively
for x86 and runs it under FEX-Emu — no NEON guessing involved).

Call install_remote_postprocess() once at process startup BEFORE any
generation that uses post_processing. After that, every kimodo
post_process_motion invocation will transparently route through HTTP.
"""

from __future__ import annotations

import io
import json
import logging
import os
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np


log = logging.getLogger("postprocess-remote")

POSTPROCESS_URL = os.environ.get(
    "POSTPROCESS_URL",
    "http://retarget-worker:9560/postprocess",
)
POSTPROCESS_TIMEOUT_S = float(os.environ.get("POSTPROCESS_TIMEOUT_S", "120"))


def _serialize_working_rig(working_rig: list) -> list[dict[str, Any]]:
    """Working rig joints are SimpleNamespace; flatten to plain dicts."""
    out = []
    for joint in working_rig:
        out.append({
            "name": joint.name,
            "parent": joint.parent,
            "t_pose_translation": list(joint.t_pose_translation),
            "t_pose_rotation": list(joint.t_pose_rotation),
            "retarget_tag": joint.retarget_tag,
        })
    return out


def _serialize_masks(masks: dict) -> dict[str, list[float]]:
    """Constraint masks may arrive as torch tensors or numpy; coerce to lists."""
    out = {}
    for k, v in masks.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        out[k] = np.asarray(v).astype(float).tolist()
    return out


def _to_numpy(arr) -> np.ndarray:
    """Accept torch tensor or numpy; return numpy."""
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _http_correct_motion(
    hip_translations,
    rotations,
    contacts,
    hip_translations_input,
    rotations_input,
    constraint_masks,
    contact_threshold,
    root_margin,
    working_rig,
    has_double_ankle_joints=False,
):
    """Drop-in replacement for motion_correction.motion_postprocess.correct_motion.

    Same signature, same in-place semantics: writes corrected values back into
    the caller's hip_translations and rotations tensors so downstream kimodo
    code (which reads them after) sees the post-processed result.
    """
    # Snapshot inputs as numpy.
    hip_t_np = _to_numpy(hip_translations)
    rot_np = _to_numpy(rotations)
    contacts_np = _to_numpy(contacts).astype(np.float32)
    hip_t_in_np = _to_numpy(hip_translations_input)
    rot_in_np = _to_numpy(rotations_input)

    npz_buf = io.BytesIO()
    np.savez_compressed(
        npz_buf,
        hip_translations=hip_t_np.astype(np.float32),
        rotations=rot_np.astype(np.float32),
        contacts=contacts_np,
        hip_translations_input=hip_t_in_np.astype(np.float32),
        rotations_input=rot_in_np.astype(np.float32),
    )

    meta = {
        "working_rig": _serialize_working_rig(working_rig),
        "constraint_masks": _serialize_masks(constraint_masks),
        "contact_threshold": float(contact_threshold),
        "root_margin": float(root_margin),
        "has_double_ankle_joints": bool(has_double_ankle_joints),
    }

    files = {
        "arrays_npz": ("arrays.npz", npz_buf.getvalue(), "application/octet-stream"),
    }
    data = {"meta": json.dumps(meta)}

    try:
        with httpx.Client(timeout=POSTPROCESS_TIMEOUT_S) as client:
            resp = client.post(POSTPROCESS_URL, files=files, data=data)
    except httpx.RequestError as e:
        raise RuntimeError(f"postprocess worker unreachable at {POSTPROCESS_URL}: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"postprocess worker {resp.status_code}: {resp.text[:500]}")

    with np.load(io.BytesIO(resp.content)) as out:
        corrected_hip = out["hip_translations"]
        corrected_rot = out["rotations"]

    # In-place write-back so kimodo's caller sees the result.
    if hasattr(hip_translations, "copy_"):
        # torch tensor — use copy_ to preserve device/dtype on the tensor side.
        import torch
        hip_translations.copy_(torch.from_numpy(corrected_hip).to(hip_translations.dtype))
        rotations.copy_(torch.from_numpy(corrected_rot).to(rotations.dtype))
    else:
        hip_translations[:] = corrected_hip
        rotations[:] = corrected_rot


_INSTALLED = False


def install_remote_postprocess() -> None:
    """Replace motion_correction's correct_motion with the HTTP client.

    Idempotent. Logs a warning and noops if motion_correction can't be
    imported (e.g. the C++ extension wasn't built into the image).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from motion_correction import motion_postprocess
    except ImportError as e:
        log.warning(
            "motion_correction not installed, can't redirect to worker (%s). "
            "Generations with post_processing=True will fail loudly.", e,
        )
        return

    motion_postprocess.correct_motion = _http_correct_motion
    _INSTALLED = True
    log.info(
        "motion_correction.correct_motion redirected to %s (FEX worker)",
        POSTPROCESS_URL,
    )
