"""Phase-1 smoke: prompt → Kimodo → NPZ → retargeted FBX, end-to-end, one-shot.

Usage:
    python -m scripts.phase1_smoke \\
        --prompt "a person walks forward" \\
        --duration 3.0 \\
        --seed 42 \\
        --template /path/to/MixamoTemplate.fbx \\
        --output /tmp/walk_forward.fbx

Runs entirely in-process (no subprocess, no BVH, no Blender). On the first
invocation the Kimodo model loads (~20-60 s depending on caches). Subsequent
calls reuse the loaded model if the script is extended into a long-lived
process.

This is the Phase-1 validation artifact: if the resulting FBX plays cleanly
on import into Unity with no contortion, Phase 1 is complete.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-shot Kimodo → Mixamo FBX")
    p.add_argument("--prompt", required=True, help="Text prompt, e.g. 'a person walks forward'")
    p.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds (default 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--template", required=True, help="Mixamo-rigged FBX template path")
    p.add_argument("--output", required=True, help="Output FBX path")
    p.add_argument("--model", default=None, help="Kimodo model name (default from env or SOMA-RP-v1.1)")
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--yaw_deg", type=float, default=0.0, help="Additional yaw rotation applied to output (Y-axis, degrees)")
    p.add_argument("--force_scale", type=float, default=0.0, help="Override target-skeleton scale; 0 = auto height ratio")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_parser().parse_args(argv)

    template = Path(args.template)
    if not template.exists():
        print(f"template FBX does not exist: {template}", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    from app.kimodo_runtime import DEFAULT_MODEL_NAME, KimodoRuntime
    from app.npz_to_fbx import retarget_npz_to_fbx

    model_name = args.model or DEFAULT_MODEL_NAME

    t0 = time.time()
    runtime = KimodoRuntime(model_name=model_name)
    logging.info("loaded Kimodo model %s in %.1fs", runtime.resolved_model, time.time() - t0)

    t0 = time.time()
    npz = runtime.generate(
        prompt=args.prompt,
        duration_s=args.duration,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
    )
    logging.info(
        "generated %d frames (%.2fs clip) in %.1fs (shapes: posed_joints=%s, global_rot_mats=%s)",
        npz["posed_joints"].shape[0],
        npz["posed_joints"].shape[0] / runtime.fps,
        time.time() - t0,
        npz["posed_joints"].shape,
        npz["global_rot_mats"].shape,
    )

    t0 = time.time()
    info = runtime.skeleton_info
    retarget_npz_to_fbx(
        posed_joints=npz["posed_joints"],
        global_rot_mats=npz["global_rot_mats"],
        joint_names=info.joint_names,
        joint_parents=info.joint_parents,
        neutral_joints=info.neutral_joints,
        fps=info.fps,
        target_fbx_path=template,
        output_fbx_path=output,
        skeleton_name=info.skeleton_name,
        yaw_offset_deg=args.yaw_deg,
        force_scale=args.force_scale,
    )
    logging.info("retargeted + wrote FBX to %s in %.1fs", output, time.time() - t0)
    print(str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
