"""NPZ -> FBX retargeter — HTTP client for the worker sidecar.

Autodesk's FBX SDK ships only as x86_64 binaries, so on aarch64 hosts (the
GB10) the actual retarget runs in a separate ``retarget-worker`` container
under qemu-user emulation. This module presents the original
``retarget_npz_to_fbx`` signature unchanged, but the body POSTs the inputs
to the worker over HTTP and writes the returned FBX bytes to disk.

The math + fbxsdkpy code now lives in ``worker/retarget.py`` (worker side).
``SOMA_TO_MIXAMO`` is still re-exported here for callers that look it up.

Set ``RETARGET_URL`` to point at the worker (default
``http://retarget-worker:9560/retarget``).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import httpx
import numpy as np


RETARGET_URL = os.environ.get("RETARGET_URL", "http://retarget-worker:9560/retarget")
RETARGET_TIMEOUT_S = float(os.environ.get("RETARGET_TIMEOUT_S", "300"))


# Re-exported so existing imports of ``SOMA_TO_MIXAMO`` from
# ``app.npz_to_fbx`` keep working without code changes elsewhere. Source of
# truth is the worker module so the mapping doesn't drift.
from worker.retarget import SOMA_TO_MIXAMO  # noqa: E402, F401


class FbxSdkMissing(RuntimeError):
    """Retained for backwards compatibility; raised when the worker is unreachable."""


def retarget_npz_to_fbx(
    *,
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
    joint_names: list[str],
    joint_parents: list[int],
    neutral_joints: np.ndarray,
    fps: float,
    target_fbx_path: str | os.PathLike,
    output_fbx_path: str | os.PathLike,
    skeleton_name: str = "kimodo_soma",
    yaw_offset_deg: float = 0.0,
    force_scale: float = 0.0,
    mapping: dict[str, str] | None = None,
    preserve_root_xz: bool = False,
) -> str:
    """POST inputs to the retarget worker, write returned FBX to disk."""
    target_fbx_path = Path(target_fbx_path).resolve()
    output_fbx_path = Path(output_fbx_path).resolve()
    if not target_fbx_path.exists():
        raise FileNotFoundError(f"template FBX not found: {target_fbx_path}")
    output_fbx_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize numpy arrays into a single in-memory NPZ payload — keeps the
    # request to a single multipart upload regardless of clip length.
    npz_buf = io.BytesIO()
    np.savez_compressed(
        npz_buf,
        posed_joints=np.asarray(posed_joints),
        global_rot_mats=np.asarray(global_rot_mats),
    )
    npz_buf.seek(0)

    meta = {
        "joint_names": list(joint_names),
        "joint_parents": [int(p) for p in joint_parents],
        "neutral_joints": np.asarray(neutral_joints).astype(float).tolist(),
        "fps": float(fps),
        "skeleton_name": skeleton_name,
        "yaw_offset_deg": float(yaw_offset_deg),
        "force_scale": float(force_scale),
        "preserve_root_xz": bool(preserve_root_xz),
    }
    if mapping is not None:
        meta["mapping"] = dict(mapping)

    files = {
        "template_fbx": (target_fbx_path.name, target_fbx_path.read_bytes(), "application/octet-stream"),
        "arrays_npz": ("arrays.npz", npz_buf.getvalue(), "application/octet-stream"),
    }
    data = {"meta": json.dumps(meta)}

    try:
        with httpx.Client(timeout=RETARGET_TIMEOUT_S) as client:
            resp = client.post(RETARGET_URL, files=files, data=data)
    except httpx.RequestError as e:
        raise FbxSdkMissing(
            f"retarget worker unreachable at {RETARGET_URL}: {e}"
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"retarget worker {resp.status_code}: {resp.text[:500]}"
        )

    output_fbx_path.write_bytes(resp.content)
    return str(output_fbx_path)


__all__ = ["SOMA_TO_MIXAMO", "FbxSdkMissing", "retarget_npz_to_fbx"]
