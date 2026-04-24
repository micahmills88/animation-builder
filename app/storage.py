"""JSON-file-per-job storage. Lives at outputs/jobs/<id>/metadata.json.

Single-animation schema: one prompt → one clip. No slots, no families,
no personality composition. The prompt is handed to Kimodo verbatim.

Chosen over SQLite because the data is small, single-user, inspectable,
and survives `docker compose down -v` as long as the host bind-mount of
outputs/ is intact.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AnimationJob:
    id: str
    prompt: str
    seed: int
    duration_s: float
    model: str
    template_fbx: str | None        # filename inside the job's workdir, or None
    created_at: str                 # ISO-8601 UTC
    finished_at: str | None
    status: str                     # pending | running | completed | failed
    error: str | None

    # Result fields — populated when generation succeeds. Trim markers are
    # inclusive frame indices carved out of the raw clip for the final
    # export; ``saved`` flips true when the user commits the current trim
    # via the viewer's Save button.
    total_frames: int | None = None
    trim_start_frame: int | None = None
    trim_end_frame: int | None = None
    saved: bool = False
    fbx_size: int | None = None
    # Two independent seam-closure modes (see app/seam_blend.py):
    #
    # ``blend_frames`` — retroactive cyclic blend. Smoothstep-eases the LAST
    # N frames of the kept slice toward the start-frame pose; clip length
    # unchanged; last frame exactly equals start pose so loop wrap is
    # mathematically seamless. Right for cyclic motions (jog, walk, idle).
    #
    # ``bridge_frames`` — appended synthesized frames that SLERP from the
    # end-frame pose to the start-frame pose; clip length grows by N.
    # Right for one-way motions (arm raise → lower, door open → close)
    # where the kept trim isn't a natural cycle.
    #
    # Both can coexist: blend runs first on the slice, then bridge is
    # appended. In practice users typically set exactly one.
    blend_frames: int = 0
    bridge_frames: int = 0


def _job_dir(output_root: Path, job_id: str) -> Path:
    return output_root / "jobs" / job_id


def _meta_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "metadata.json"


def save_job(output_root: Path, meta: AnimationJob) -> None:
    path = _meta_path(output_root, meta.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def load_job(output_root: Path, job_id: str) -> AnimationJob | None:
    path = _meta_path(output_root, job_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return AnimationJob(**{k: v for k, v in raw.items() if k in AnimationJob.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def list_jobs(output_root: Path) -> list[AnimationJob]:
    jobs_dir = output_root / "jobs"
    if not jobs_dir.exists():
        return []
    out: list[AnimationJob] = []
    for sub in jobs_dir.iterdir():
        if not sub.is_dir():
            continue
        meta = load_job(output_root, sub.name)
        if meta:
            out.append(meta)
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_workdir(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id)


def to_summary(meta: AnimationJob) -> dict[str, Any]:
    """Compact dict for the /api/animations list view."""
    return {
        "id": meta.id,
        "prompt": meta.prompt,
        "seed": meta.seed,
        "duration_s": meta.duration_s,
        "model": meta.model,
        "template_fbx": meta.template_fbx,
        "created_at": meta.created_at,
        "finished_at": meta.finished_at,
        "status": meta.status,
        "total_frames": meta.total_frames,
        "trim_start_frame": meta.trim_start_frame,
        "trim_end_frame": meta.trim_end_frame,
        "saved": meta.saved,
        "fbx_size": meta.fbx_size,
        "blend_frames": meta.blend_frames,
        "bridge_frames": meta.bridge_frames,
        "error": meta.error,
    }
