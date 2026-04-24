"""In-process Kimodo model runtime: load once, generate many.

The current pipeline shells out to ``python -m kimodo.scripts.generate`` per
slot (10-30s cold-start per call, full model reload). This module loads the
model once at service start and exposes a single ``generate`` call returning
Kimodo's native output dict as numpy arrays — no BVH, no subprocess, no disk
IO in the hot path. Disk IO is the caller's job if archival is wanted.

The output dict is identical in shape to what ``save_kimodo_npz`` writes (see
``upstream/kimodo/exports/motion_io.py``), minus the leading num_samples
dimension — we always run ``num_samples=1``. Keys: ``posed_joints`` (T,J,3),
``global_rot_mats`` (T,J,3,3), ``local_rot_mats`` (T,J,3,3), ``root_positions``
(T,3), ``smooth_root_pos`` (T,3), ``foot_contacts`` (T,4) or (T,6),
``global_root_heading`` (T,2).

Skeleton metadata (needed by the retargeter) is exposed via ``skeleton_info``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from kimodo import load_model
from kimodo.tools import seed_everything


DEFAULT_MODEL_NAME = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1").split("/")[-1]
DEFAULT_DIFFUSION_STEPS = int(os.environ.get("KIMODO_DIFFUSION_STEPS", "100"))


@dataclass
class SkeletonInfo:
    """Skeleton metadata needed to build an FBX. All arrays are numpy, host memory."""

    joint_names: list[str]        # length J — SOMA77 joint names in model order
    joint_parents: list[int]      # length J — parent index per joint, -1 for root
    neutral_joints: np.ndarray    # (J, 3) — T-pose joint positions, meters, Y-up +Z fwd
    root_idx: int
    fps: float
    skeleton_name: str


class KimodoRuntime:
    """Singleton-style wrapper around the loaded Kimodo model.

    Thread-safety note: ``generate`` is NOT safe to call concurrently. The
    underlying model holds GPU state and the CLI's queue-size-1 assumption
    carries over here. Callers that want parallelism must serialize.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.resolved_model = load_model(
            model_name,
            device=self.device,
            default_family="Kimodo",
            return_resolved_name=True,
        )

        # The model's native skeleton is SOMA30 (30 joints), but Kimodo's
        # NPZ output is expanded to SOMA77 via `to_SOMASkeleton77` with hand
        # + face joints filled from a rest-pose asset (KIMODO_REFERENCE §3).
        # Retargeting operates on the output, so the skeleton metadata must
        # match the output's joint count. Upstream's BVH export does the
        # same switch at generate.py:388-390.
        #
        # For non-SOMA models (G1, SMPL-X) we fall back to the native
        # skeleton — their outputs match the native joint count directly.
        native = self.model.skeleton
        skel = native
        try:
            from kimodo.skeleton import SOMASkeleton30
            if isinstance(native, SOMASkeleton30):
                skel = native.somaskel77.to(self.device)
        except ImportError:
            pass
        self._output_skeleton = skel

        self.skeleton_info = SkeletonInfo(
            joint_names=list(skel.bone_order_names),
            joint_parents=[int(p) for p in skel.joint_parents.detach().cpu().tolist()],
            neutral_joints=skel.neutral_joints.detach().cpu().numpy().astype(np.float32),
            root_idx=int(skel.root_idx),
            fps=float(self.model.fps),
            skeleton_name=str(skel.name),
        )

    @property
    def fps(self) -> float:
        return self.skeleton_info.fps

    def generate(
        self,
        prompt: str,
        duration_s: float,
        *,
        seed: int | None = None,
        constraint_lst: list | None = None,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        cfg_weight_text: float = 2.0,
        cfg_weight_constraint: float = 2.0,
        post_processing: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate one clip. Single prompt, single sample, no multi-segment splitting.

        ``prompt`` must not contain a period (periods split into separate
        segments per upstream ``generate.py:121-139``). We append one ourselves.
        """
        if "." in prompt.rstrip("."):
            raise ValueError(
                f"prompt must not contain periods (they split into multiple segments): {prompt!r}"
            )
        text = prompt.rstrip(". ").strip() + "."

        num_frames = int(round(duration_s * self.fps))
        if num_frames < 2:
            raise ValueError(f"duration_s={duration_s} too short; got {num_frames} frames")

        if seed is not None:
            seed_everything(seed)

        output = self.model(
            [text],
            [num_frames],
            constraint_lst=constraint_lst or [],
            num_denoising_steps=diffusion_steps,
            num_samples=1,
            multi_prompt=True,
            num_transition_frames=5,
            post_processing=post_processing,
            return_numpy=True,
            cfg_type="separated",
            cfg_weight=[cfg_weight_text, cfg_weight_constraint],
        )

        return {
            k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v)
            for k, v in output.items()
        }
