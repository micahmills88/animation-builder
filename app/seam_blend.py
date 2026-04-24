"""Retroactive cyclic blending for clean loop closure.

Instead of appending new frames that interpolate end-pose → start-pose (the
"bridge" approach, which extends clip length and is visible as slow-motion),
this module modifies the LAST N frames of the trim slice IN PLACE so they
smoothstep-ease toward the start-frame pose. The exported clip has the same
length as the trim, motion keeps playing through the entire duration, and
the last frame EXACTLY equals the start-frame pose — so loop wrap is
mathematically seamless, not just "close to seamless".

Why this over a bridge:
    * Clip length unchanged — no perceptible "slow-motion wind-down".
    * Last frame = start pose exactly, so loop pop is zero.
    * Motion stays active through the whole clip.
    * Simpler code path: modify the slice in place, retarget once.
    * Avoids the FK-through-local-rotations path that accumulates drift
      over long blend windows.

We interpolate GLOBAL rotation matrices directly (independently per joint),
linear-blend ``root_positions``, and recompute ``posed_joints`` via the
parent-chain formula using the blended global rotations. The retargeter
consumes ``global_rot_mats`` + ``posed_joints[:, hips]`` and nothing else,
so these three signals are all that need to stay consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def _slerp_rotmats_pair(rot_a: np.ndarray, rot_b: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """SLERP between two 3x3 rotation matrices at fractions ``ts`` in [0, 1].
    Returns (len(ts), 3, 3). scipy's Slerp handles shortest-path hemisphere
    and constant angular velocity automatically."""
    rots = R.from_matrix(np.stack([rot_a, rot_b]))
    slerp = Slerp([0.0, 1.0], rots)
    return slerp(ts).as_matrix()


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite smoothstep: 3t² - 2t³, with zero derivative at t=0 and t=1.
    The zero-derivative endpoints matter here: at t=0 the blended frame has
    the natural motion velocity (no kink where blending starts), and at t=1
    the blended pose is locked to start-pose with zero angular velocity,
    which matches the start of the next loop cycle reasonably."""
    return t * t * (3.0 - 2.0 * t)


def apply_cyclic_blend(
    *,
    global_rot_mats: np.ndarray,   # (K, J, 3, 3) — MODIFIED IN PLACE
    posed_joints: np.ndarray,      # (K, J, 3)   — MODIFIED IN PLACE
    root_positions: np.ndarray,    # (K, 3)      — MODIFIED IN PLACE
    neutral_joints: np.ndarray,    # (J, 3)
    joint_parents: list[int],
    blend_frames: int,
) -> dict[str, float]:
    """Ease the last ``blend_frames`` of the slice toward the start pose.

    All three arrays are modified in place. Returns a small diagnostics dict
    so the caller can log how well the blend landed (summed pose delta
    between last frame and first frame — zero means perfect loop closure).

    Blend weight is smoothstep across the blend span:
        k = 0 at the first blend frame (unchanged; w=0)
        k = N-1 at the last frame of the clip (fully replaced with start
        pose; w=1)

    Values of ``blend_frames`` that exceed ``K - 1`` are clamped — we always
    leave at least one unblended anchor frame at the start of the blend span
    so the easing begins smoothly from the clip's natural motion.
    """
    K, J = global_rot_mats.shape[:2]
    if K < 2:
        return {"blend_frames_applied": 0, "last_to_first_delta": 0.0}

    N = int(max(0, min(blend_frames, K - 1)))
    if N == 0:
        return {"blend_frames_applied": 0, "last_to_first_delta": 0.0}

    # Capture the target (start-frame) pose.
    start_global = global_rot_mats[0].copy()
    start_root = root_positions[0].copy()

    # Blend span covers the LAST N frames: indices K-N .. K-1.
    # Weight at span position k = smoothstep(k / (N - 1)).
    # k = 0 → weight 0 (unchanged); k = N-1 → weight 1 (= start pose).
    for k in range(N):
        f = K - N + k
        if N > 1:
            t = k / (N - 1)
        else:
            t = 1.0
        w = float(_smoothstep(np.asarray(t)))

        # SLERP each joint's global rotation toward start.
        for j in range(J):
            out = _slerp_rotmats_pair(global_rot_mats[f, j], start_global[j], np.array([w]))
            global_rot_mats[f, j] = out[0]

        # Linear blend of root position.
        root_positions[f] = (1.0 - w) * root_positions[f] + w * start_root

    # Recompute posed_joints for every blended frame using the parent-chain
    # formula with blended global rotations. This guarantees bone lengths are
    # preserved (no position-lerp stretching) and posed_joints stays
    # consistent with global_rot_mats.
    #
    #    posed[root]  = root_position
    #    posed[j]     = posed[parent(j)] + global_rot[parent(j)] @ (neutral[j] - neutral[parent(j)])
    for k in range(N):
        f = K - N + k
        for j in range(J):
            p = joint_parents[j]
            if p < 0:
                posed_joints[f, j] = root_positions[f]
            else:
                rest_offset = neutral_joints[j] - neutral_joints[p]
                posed_joints[f, j] = posed_joints[f, p] + global_rot_mats[f, p] @ rest_offset

    # Diagnostic: how close is the final frame to frame 0 after the blend?
    # With smoothstep and w=1 at k=N-1, this should be essentially zero for
    # global_rot_mats and root_positions. Tiny residual is possible from
    # scipy's SLERP numerical precision.
    delta_rot = float(np.linalg.norm(global_rot_mats[-1] - global_rot_mats[0]))
    delta_root = float(np.linalg.norm(root_positions[-1] - root_positions[0]))
    delta_posed = float(np.linalg.norm(posed_joints[-1] - posed_joints[0]))

    return {
        "blend_frames_applied": N,
        "last_to_first_rot_delta": delta_rot,
        "last_to_first_root_delta": delta_root,
        "last_to_first_posed_delta": delta_posed,
    }


def build_bridge_frames(
    *,
    global_rot_mats_end: np.ndarray,    # (J, 3, 3)
    global_rot_mats_start: np.ndarray,  # (J, 3, 3)
    root_position_end: np.ndarray,      # (3,)
    root_position_start: np.ndarray,    # (3,)
    clip_length_frames: int,            # (end - start + 1) in the source NPZ
    neutral_joints: np.ndarray,         # (J, 3)
    joint_parents: list[int],
    n_bridge: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build N new bridge frames that SLERP from end-pose to start-pose.

    Returns ``(bridge_posed_joints, bridge_global_rot_mats)`` with shapes
    ``(n_bridge, J, 3)`` and ``(n_bridge, J, 3, 3)``. Caller concatenates
    these after the kept slice before retargeting.

    Use this mode for ONE-WAY motions (character raises an arm and needs
    time to lower it back; a door opens and needs to swing closed) where
    the trim can't naturally be auto-cycled. For cyclic motions (jog,
    walk, idle) prefer ``apply_cyclic_blend`` — it closes the loop without
    adding clip length or visible "slow-motion return" frames.

    ``ts = [1/(N+1), ..., N/(N+1)]`` deliberately excludes 0 and 1 so no
    pose is emitted twice at the seam: the sequence plays ... end →
    bridge[0] → bridge[1] → ... → bridge[N-1] → [wrap] → start ... with
    each step ≈ 1/(N+1) of the total angular gap.

    Root position extrapolates forward at the clip's average per-frame
    velocity (so a jog cycle doesn't "pause" during the bridge). For
    in-place clips (avg velocity ≈ 0) this degenerates to "hold at end".
    """
    if n_bridge < 1:
        raise ValueError("n_bridge must be >= 1")
    J = global_rot_mats_end.shape[0]
    if global_rot_mats_start.shape[0] != J:
        raise ValueError("start/end global_rot_mats must share joint count")

    ts = np.arange(1, n_bridge + 1, dtype=np.float64) / (n_bridge + 1)

    # SLERP each joint's global rotation independently. Chain consistency
    # falls out because we rebuild posed_joints from these globals below.
    bridge_global_rot = np.zeros((n_bridge, J, 3, 3))
    for j in range(J):
        bridge_global_rot[:, j] = _slerp_rotmats_pair(
            global_rot_mats_end[j], global_rot_mats_start[j], ts
        )

    # Root: extrapolate forward at the cycle's average per-frame velocity.
    if clip_length_frames > 1:
        avg_vel = (root_position_end - root_position_start) / float(clip_length_frames - 1)
    else:
        avg_vel = np.zeros(3, dtype=np.float64)
    bridge_root_pos = (
        root_position_end[None, :]
        + avg_vel[None, :] * np.arange(1, n_bridge + 1, dtype=np.float64)[:, None]
    )

    # posed_joints via parent-chain formula using the blended globals.
    bridge_posed = np.zeros((n_bridge, J, 3))
    for k in range(n_bridge):
        for j in range(J):
            p = joint_parents[j]
            if p < 0:
                bridge_posed[k, j] = bridge_root_pos[k]
            else:
                rest_offset = neutral_joints[j] - neutral_joints[p]
                bridge_posed[k, j] = bridge_posed[k, p] + bridge_global_rot[k, p] @ rest_offset

    return bridge_posed, bridge_global_rot


__all__ = ["apply_cyclic_blend", "build_bridge_frames"]
