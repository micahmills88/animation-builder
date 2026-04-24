# Kimodo Source-Code Reference

A concrete, file-and-line-level audit of NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo)
CLI, demo UI, model, and export pipeline. Written against the code in the
`animation-service` container at `/workspace/kimodo`. Everything here is grounded in the repo;
every non-trivial claim has a file:line citation.

All CLI references in this document target `python -m kimodo.scripts.generate`
(source: `kimodo/scripts/generate.py`).

---

## 1. The CLI Surface (`kimodo/scripts/generate.py`)

Argparse definitions at `kimodo/scripts/generate.py:19-118`. Main routing logic at `:266-422`.

### `prompt` (positional, optional `nargs="?"`, default `None`)
*File: `:21-27`.*

Free-text prompt. **Split on `.` only** (see Section 2). If omitted, `--input_folder` must be set
(checked at `:232-233`). Not sanitized by the CLI itself — sanitization happens inside the model
(`kimodo/model/kimodo_model.py:147` via `sanitize_texts`).

### `--model` (str, default `DEFAULT_MODEL = "kimodo-soma-rp"`)
*File: `:28-33`; default from `kimodo/model/registry.py:149`.*

Resolved by `resolve_model_name(..., default_family="Kimodo")` inside `load_model`
(`kimodo/scripts/generate.py:273-278`). Accepts:
- Short keys: `kimodo-soma-rp`, `kimodo-soma-rp-v1.1`, `kimodo-soma-seed`, `kimodo-smplx-rp`,
  `kimodo-g1-rp`, `kimodo-g1-seed` (`kimodo/model/registry.py:15-23`, `:132`).
- Full display names: `Kimodo-SOMA-RP-v1.1`, etc.
- Partials: `SOMA`, `SOMA-RP`, `SEED` (resolves to `Kimodo-<skel>-<ds>-latest`).
- Case-insensitive. When multiple versions exist, the versionless alias points to the **latest**
  (`:124-127`).

Checkpoints come from HuggingFace by default via `snapshot_download`
(`kimodo/model/load_model.py:36-58`). Override with env var `CHECKPOINT_DIR` to use a local
directory; otherwise online check runs every time. Set `LOCAL_CACHE=true` to avoid the
online-first behavior (`kimodo/model/load_model.py:45-58`).

Text-encoder selection is separate and not exposed on the CLI: `TEXT_ENCODER_MODE=auto|local|api`,
`TEXT_ENCODER_URL` (default `http://127.0.0.1:9550/`), `TEXT_ENCODER_FP32`
(`kimodo/model/load_model.py:83-100`). `auto` tries the API and falls back to local LLM2Vec.

### `--duration` (str, default `"5.0"`)
*File: `:34-39`; parsing at `:121-139`.*

**A string**, not a float. Space-separated values map to per-prompt durations. Strict
length check: `assert len(durations) == len(texts)` (`:135`). No fractional frames —
`int(duration_sec * fps)` (`:132, 136`). FPS is read from the loaded model (see §10).

### `--num_samples` (int, default `1`)
*File: `:40-45`.*

Passed to `model(...)` as `num_samples`. When `> 1`, output is written to a folder
(see `_output_dir_and_path`, `:155-164`). UI equivalent: `gui_num_samples_slider`
(1-10, default 1, `kimodo/demo/ui.py:353-360`). Hidden when `HF_MODE` env is set.

### `--diffusion_steps` (int, default `100`)
*File: `:46-51`.*

DDIM steps. Passed as `num_denoising_steps` to `Kimodo.__call__`
(`kimodo/model/kimodo_model.py:321`, `:379`). UI range is 2-1000, step 10
(`kimodo/demo/ui.py:372-378`). Higher = slower & generally better; diminishing returns above ~100.

### `--num_transition_frames` (int, default `5`)
*File: `:52-57`.*

Controls how consecutive multi-prompt segments are blended. Only active when the prompt
string contains `.` (producing multiple segments). UI uses
`NB_TRANSITION_FRAMES = 5` from `kimodo/demo/config.py:40`. UI range 1-10
(`kimodo/demo/ui.py:407-413`). Demo folder "Transitions" is **hidden by default** because
`SHOW_TRANSITION_PARAMS = False` (`kimodo/demo/config.py:38`, `kimodo/demo/ui.py:402-405`).
See §2 for what this actually does.

### `--constraints` (str, default `None`)
*File: `:58-63`.*

Path to a saved constraint list JSON — format produced/consumed by
`kimodo.constraints.{load,save}_constraints_lst`. If `--input_folder` is also used,
the CLI falls back to `<input_folder>/constraints.json` when this is not set
(`:250-253`).

### `--output` (str, default `"output"`)
*File: `:64-69`.*

**Stem**, not a full path. Behavior:
- `num_samples == 1` → `<output>.npz` (and `.bvh`, `.csv`, `_amass.npz` as applicable).
  `_single_file_path` adds missing extensions and creates parent dirs
  (`:142-152`).
- `num_samples > 1` → directory `<output>/` containing `<basename>_00.npz`, `_01.npz`, ...
  (`:342-350`, `:407-422`). Uses `_output_dir_and_path` (`:155-164`).

**Gotcha**: for SMPL-X models, the AMASS file is written as `<output>_amass.npz` (single) or
`<out_dir>/amass.npz` (batch) so it does not clobber the Kimodo NPZ (`:352-364`).

### `--bvh` (flag, default False)
*File: `:70-74`.*

**Silently ignored unless skeleton is SOMA** (explicit `"somaskel" not in skeleton.name`
check at `:382`). So you can pass `--bvh` to a G1 or SMPL-X model and it will print
*"BVH export is only supported for SOMA skeletons. Skipping --bvh."* and exit without
BVH output. `SOMASkeleton30` output is transparently expanded to `somaskel77` for BVH
(`:388-390`, and `kimodo/skeleton/definitions.py:247-256`).

### `--bvh_standard_tpose` (flag, default False)
*File: `:75-79`; consumed at `kimodo/exports/bvh.py:99-104`.*

**If False (default): exports in the "BONES-SEED rest pose" frame** — the original training-
data rest pose, using `skeleton.bvh_neutral_joints` and applying `skeleton.from_standard_tpose`
to shift the rotations into that frame. **If True: exports with the standard identity T-pose**
(uses `skeleton.neutral_joints` and leaves rotations alone).

This is load-bearing for retargeting:
- If your downstream rig expects a standard T-pose (arms straight out, identity global
  rotations at rest), pass `--bvh_standard_tpose`.
- If you are retargeting via matching joint orientations in a software that already has
  the BONES-SEED rest pose as reference, omit it.

See §3 and §12 for full details.

### `--no-postprocess` (flag, default False)
*File: `:80-84`; consumed at `:316`.*

Disables `kimodo.postprocess.post_process_motion`. **Ignored entirely for G1** —
`use_postprocess = False if "g1" in resolved_model else (not args.no_postprocess)` (`:316`).
See §8.

### `--seed` (int, default None)
*File: `:85-90`; consumed at `:303-304` via `seed_everything`.*

When None, the model uses whatever torch global RNG state is already in effect. Demo UI
defaults seed to `42` (`kimodo/demo/ui.py:369`).

### `--input_folder` (str, default None)
*File: `:91-96`; parsing at `:245-263`.*

Folder must contain `meta.json`. Optional `constraints.json` in the same folder is picked up
automatically if `--constraints` isn't set (`:250-253`). `meta.json` values override CLI values
for `num_samples`, `diffusion_steps`, `seed` (`:258-260`) — CLI values become fallbacks only.
See §6.

### `--cfg_type` (str, default `argparse.SUPPRESS`)
*File: `:97-107`; resolution at `:167-226`.*

Choices are `CFG_TYPES = ["nocfg", "regular", "separated"]`
(`kimodo/model/cfg.py:10`). Default is `SUPPRESS` — the key is **not present** in the
namespace unless passed, which triggers the resolution cascade in `resolve_cfg_kwargs`.
See §7.

### `--cfg_weight` (float, `nargs="*"`, default `argparse.SUPPRESS`)
*File: `:108-117`; resolution at `:167-226`.*

Zero, one, or two floats. When only `--cfg_weight` is supplied:
- 1 float → implicitly `regular`
- 2 floats → implicitly `separated`

When neither `--cfg_type` nor `--cfg_weight` are passed, `resolve_cfg_kwargs` returns `{}`
— the model uses its own internal default (`separated`, with weights `[2.0, 2.0]`; see §7).

### Flags the demo uses but the CLI does not expose

| Demo control | Default | Notes |
|---|---|---|
| `share_transition` | False | `kimodo/demo/ui.py:415-418`, hidden by default. |
| `percentage_transition_override` | 0.10 | `kimodo/demo/ui.py:420-426` (value/100). |
| `root_margin` | 0.04 m | `kimodo/demo/ui.py:445-453`. CLI always uses 0.04 (`kimodo/model/kimodo_model.py:361`). |
| `real_robot_rotations` | False | G1 only (`kimodo/demo/ui.py:462-467`). |
| `first_heading_angle` | 0.0 | Hardcoded in CLI (`kimodo/model/kimodo_model.py:259, 474`). |
| `gui_use_soma_layer_checkbox` | False | UI-only render toggle. |

---

## 2. Prompt Parsing Semantics

`get_texts_and_num_frames_from_prompt` (`kimodo/scripts/generate.py:121-139`):

```python
texts = [text.strip() for text in prompt.split(".")]
texts = [text + "." for text in texts if text]
```

Key behaviors:
- **Only `.` splits.** Not `?`, not `!`, not newlines. An exclamation-terminated prompt
  (`"Run fast! A person sits."`) becomes **two** segments: `"Run fast! A person sits."`
  and `"..."` wait — no, the whole string is split on `.` only, so
  `"Run fast! A person sits."` → `["Run fast! A person sits", ""]` → **after filter** →
  `["Run fast! A person sits."]` (one segment).
- **Every period produces a segment.** `"Mr. Smith walks."` → `["Mr", "Smith walks", ""]` →
  `["Mr.", "Smith walks."]` — two segments. If you passed `--duration 5.0` with a
  single-value string, **both segments get 5 seconds** (`:132`), silently doubling the
  output length. This is exactly the "4× duration" footgun in task #27.
- Trailing `.` produces an empty string which is filtered out (`:124`). So `"walk."` →
  `["walk"]` → `["walk."]` — one segment.
- **Multiple consecutive periods**: `"walk.. then run."` → `["walk", "", " then run", ""]` →
  `["walk.", "then run."]` (two segments).

### Multi-prompt generation

`Kimodo._multiprompt` (`kimodo/model/kimodo_model.py:123-341`). The CLI always passes
`multi_prompt=True` (`:323`), so *any* prompt containing `.` is treated as a sequence.
What actually happens per segment (pseudo-summary of the loop at `:165-311`):

1. Generate segment A fully, then start segment B.
2. From A's tail, take the last `num_transition_frames` (plus `int(prev_len * 10%)` when
   `share_transition=True`) and build a `FullBodyConstraintSet` that conditions B's first
   frames (`:194-243`).
3. Sample B.
4. Linearly blend A's last `num_transition_frames` with B's first `num_transition_frames`
   (`alpha = linspace(1, 0, N)`) — this produces the shared transition (`:284-304`).
5. Concatenate: `[A[:-N], shared_transition, B[N:]]`.

So segments **are stitched with a real crossfade, not just concatenated**. The first
segment's heading is 0 rad; each subsequent segment starts at A's final heading
(`:255`). The 2D root position chains through via `smooth_root_2d` (`:211`, `:246-249`).

### Text encoder scope

Each segment is encoded independently. `_generate` receives a list of `N_samples` copies
of the *current segment's* text (`:166`, `:271`). There is no "full paragraph" encoding.

### Duration vs prompt count mismatch

**Hard assertion** (`:135`): if `--duration` has spaces and the split count ≠ prompt count,
the script crashes with `AssertionError: The number of durations should match the number
of prompts`. For a single `--duration 5.0` value, all prompts get 5 s.

### Multi-prompt + constraints

`constraint.crop_move(current_frame, current_frame + num_frame)` (`:179`) re-times each
constraint so that a keyframe originally at absolute frame 120 with a 60-frame first
segment lands at local frame 60 of the second. Constraints are applied to every segment
they fall within; transition regions get their own synthesized `FullBodyConstraintSet`
from segment A's tail (`:216-223`).

---

## 3. BVH Export (`kimodo/exports/bvh.py`)

Main entry: `save_motion_bvh` → `motion_to_bvh` (`:62-211`). Called from
`kimodo/scripts/generate.py:399` (single) / `:415` (batch).

### Joint ordering

`joint_names = list(skeleton.bone_order_names)` (`:106`) — for SOMA77, 77 joints in the
order defined at `kimodo/skeleton/definitions.py:86-164`. SOMA30 is transparently
upgraded to SOMA77 (`:94-97`) via `to_SOMASkeleton77` which fills hand/face joints with
`relaxed_hands_rest_pose` (`kimodo/skeleton/definitions.py:247-256`).

### Channel order

`_ROOT_CHANNELS = ["Xposition","Yposition","Zposition","Zrotation","Yrotation","Xrotation"]`
(`:124-131`). **ZYX** rotation order — so rotations applied in Blender/other importers
must match. `_JOINT_CHANNELS = ["Zrotation","Yrotation","Xrotation"]` (`:132`) — same
ZYX for all non-root.

### Unit system and up-axis

- **Centimeters.** Scaled explicitly: `neutral = neutral * 100`, `root_xyz = root_xyz * 100`
  (`:135-136`). The original SEED BVH data is in cm, so this matches.
- **Y-up, +Z forward** (Kimodo internal convention — see `kimodo/exports/smplx.py:54-55`
  and `:17-38` for the Y-up→Z-up AMASS conversion matrix). BVH carries no up-axis header;
  Blender defaults to Y-up which aligns correctly.

### Root joint structure

BVH gets a **wrapper "Root" joint** above `"Hips"` (or whatever `bone_order_names[root_idx]`
is). `root_wrapper = bvhio.BvhJoint("Root", offset=(0,0,0))` with full 6-channel
`_ROOT_CHANNELS` (`:162-164`). Its keyframes are all identity/zero (`:186-188`). **Hips**
then gets the actual root motion on its 6 channels, with offset set to `skeleton.neutral_joints[root_idx] * 100`
or `(0, 100, 0)` as a fallback when the neutral pelvis is at origin (`:138-142`).

This means the BVH has TWO joints with position+rotation channels — if you're parsing
for "the pelvis", match by name (`"Hips"` for SOMA, not `"Root"`).

### `standard_tpose` True vs False

```python
if standard_tpose:
    neutral = skeleton.neutral_joints.detach().cpu().numpy()
else:
    local_rot_mats, _ = skeleton.from_standard_tpose(local_rot_mats)
    neutral = skeleton.bvh_neutral_joints.detach().cpu().numpy()
```
(`kimodo/exports/bvh.py:99-104`)

Two things change together:
1. **Joint offsets** — `neutral_joints` (standard T-pose geometry) vs `bvh_neutral_joints`
   (BONES-SEED rest-pose geometry, loaded from `bvh_joints.p` asset).
2. **Local rotations** — when False, `from_standard_tpose` composes each local rotation
   with the inverse of the skeleton's `global_rot_offsets` so that IDENTITY local rotations
   correspond to the BONES-SEED rest pose rather than the standard T-pose
   (`kimodo/skeleton/transforms.py:91-106`, `kimodo/skeleton/base.py:79-82`).

**Consequence**: `standard_tpose=False` is only self-consistent if you interpret the
resulting BVH rest pose as "BONES-SEED arms-slightly-lowered" rather than the symbolic
T-pose. If your retarget source expects a strict T-pose at frame 0, use `--bvh_standard_tpose`.

### FPS

Read directly from `model.fps` at the caller (`kimodo/scripts/generate.py:404, 420`).
Passed as `frameTime = 1.0 / fps` to `bvhio.BvhContainer` (`kimodo/exports/bvh.py:200`).
Kimodo models are 30 FPS (`kimodo/motion_rep/reps/base.py:50-56`, loaded from each model's
`config.yaml`). There is **no resampling on BVH export** — the file has exactly `T` frames
at the model's native FPS.

### Numerical precision / loopability

- `bvhio.writeBvh(..., percision=6)` (note misspelling in bvhio API), 6 decimal places
  (`:204`).
- **End Site blocks are stripped** by string post-processing (`_strip_end_site_blocks`,
  `:20-46`) to match original-format BVH that some importers expect.
- There is **no loop post-processing** — frame 0 and frame N-1 are not guaranteed to
  match. See §11.

---

## 4. Other Output Formats

### Kimodo NPZ

Written by `save_kimodo_npz` (`kimodo/exports/motion_io.py:202-204`) after
`complete_motion_dict` produces the keys (`:133-186`). Shapes per single sample
(batched writes strip the leading 1):

| Key | Shape | Meaning | Coord frame |
|---|---|---|---|
| `posed_joints` | `(T, J, 3)` | World-space joint positions, **meters** | Y-up, +Z forward |
| `global_rot_mats` | `(T, J, 3, 3)` | World-space joint orientations | — |
| `local_rot_mats` | `(T, J, 3, 3)` | Parent-local joint rotations | — |
| `root_positions` | `(T, 3)` | Pelvis world position, meters | Y-up, +Z forward |
| `smooth_root_pos` | `(T, 3)` | Low-pass filtered root trajectory | same |
| `foot_contacts` | `(T, 4)` (SOMA30), `(T, 6)` (SOMA77), etc. | Contact labels | booleans cast to float |
| `global_root_heading` | `(T, 2)` | `[cos θ, sin θ]` of heading | — |

`J` values by model: SOMA → 77 (internally 30, output expanded; `kimodo/skeleton/definitions.py:264-283`),
G1 → 34, SMPL-X → 22. **All NPZ output is in world/global frame** — `posed_joints` includes
the root offset. Units are meters (`kimodo/exports/motion_io.py:146`).

### G1 CSV

Written only for `kimodo-g1-rp` (`kimodo/scripts/generate.py:366-378`). Format: MuJoCo
`qpos` rows, comma-separated, no header, shape `(T, 36)` = 3 root translation + 4 root
quaternion (wxyz by default) + 29 hinge angles (`kimodo/exports/motion_io.py:228-233`,
`kimodo/exports/mujoco.py:27-40`).

**Coordinate system flip**: MuJoCo is Z-up +X forward, Kimodo is Y-up +Z forward — the
converter matrix is at `kimodo/exports/mujoco.py:76-80`. The `mujoco_rest_zero` flag
controls whether joint angles are relative to T-pose rest (`True`) or raw
(`False`); the CLI writes with default `False` (`kimodo/exports/motion_io.py:215-226`). If
you load a CSV back in, this flag **must match** the one used on write.

### AMASS NPZ

Written only for `kimodo-smplx-rp` (`kimodo/scripts/generate.py:352-364`). Keys
(`kimodo/exports/smplx.py:238-251`, a commented reference block):

- `trans` `(T, 3)` — pelvis translation in AMASS coords (z-up by default), **meters minus pelvis
  offset** so SMPL-X FK adds it back (`kimodo/exports/smplx.py:57-58`).
- `root_orient` `(T, 3)` — axis-angle of root rotation.
- `pose_body` `(T, 63)` — 21 body joints × 3 axis-angle components.
- `pose_hand` `(T, 90)` — filled from `mean_hands.npy` (`:179-189`).
- `pose_jaw` `(T, 3)`, `pose_eye` `(T, 6)` — zeros.
- `betas` `(16,)`, `num_betas=16`, `gender="neutral"`, `surface_model_type="smplx"`,
  `mocap_frame_rate`, `mocap_time_length`.

**`z_up=True` is the default** (`kimodo/exports/smplx.py:46, 198`) → output is rotated by the
matrix in `kimodo_y_up_to_amass_coord_rotation_matrix` (`:17-38`). Set `z_up=False` only if
you want raw Kimodo Y-up axes in the AMASS file — downstream AMASS tooling usually
assumes Z-up.

---

## 5. Model Variants

Short keys and HF repos (`kimodo/model/registry.py:15-23`):

| Short key | HF repo | FPS | Skeleton |
|---|---|---|---|
| `kimodo-soma-rp` | `nvidia/Kimodo-SOMA-RP-v1.1` (latest) | 30 | somaskel30 → somaskel77 output |
| `kimodo-soma-rp-v1` | `nvidia/Kimodo-SOMA-RP-v1` | 30 | same |
| `kimodo-soma-seed` | `nvidia/Kimodo-SOMA-SEED-v1.1` | 30 | same |
| `kimodo-smplx-rp` | `nvidia/Kimodo-SMPLX-RP-v1` | 30 | smplx22 |
| `kimodo-g1-rp` | `nvidia/Kimodo-G1-RP-v1` | 30 | g1skel34 |
| `kimodo-g1-seed` | `nvidia/Kimodo-G1-SEED-v1` | 30 | g1skel34 |

### RP vs SEED

- **RP (Rigplay)** — current default. Larger/cleaner internal dataset.
- **SEED** — BONES-SEED dataset. Different motion style/distribution. Registered as
  `dataset_ui_label = "SEED"` (`kimodo/model/registry.py:44-46`).

Behaviorally identical at the API level — same skeleton, same output dict, same FPS, same
constraint semantics. The difference is in generated motion style.

### Model-specific flags (CLI behavior)

- **G1 always disables postprocessing** (`kimodo/scripts/generate.py:316`).
- **BVH only runs for SOMA** (`:382-384`).
- **AMASS NPZ only for SMPL-X** (`:352`).
- **CSV only for G1** (`:366`).

### Checkpoint caching

`snapshot_download` from `huggingface_hub`. Default behavior **hits network on every run**
unless `LOCAL_CACHE=true` is set (`kimodo/model/load_model.py:45-58`). With
`CHECKPOINT_DIR` set, the script first looks for a subdirectory matching the display
name (e.g. `$CHECKPOINT_DIR/Kimodo-SOMA-RP-v1.1/`), then falls back to the short key
(backward compat), then downloads (`:157-167`).

---

## 6. The `meta.json` Input Format

Parsed by `parse_prompts_from_meta` (`kimodo/meta.py:32-79`) plus the keys pulled in
`kimodo/scripts/generate.py:245-263`.

**Recognized keys**:

| Key | Type | Source |
|---|---|---|
| `text` | str | single-prompt form (`meta.py:52-62`) |
| `duration` | float (seconds) | single-prompt form |
| `texts` | list[str] | multi-prompt form (`meta.py:65-78`) |
| `durations` | list[float] (seconds) | multi-prompt form; length must match `texts` |
| `num_samples` | int | `generate.py:258` |
| `diffusion_steps` | int | `generate.py:259` |
| `seed` | int | `generate.py:260` |
| `cfg` | object | `generate.py:213-223` |

**`cfg` block** (`resolve_cfg_kwargs`, `generate.py:213-224`):

```json
"cfg": {
  "enabled": true,
  "text_weight": 2.0,
  "constraint_weight": 2.0
}
```

When `enabled=false`, CLI uses `cfg_type=nocfg`. When `enabled=true` or missing, CLI uses
`cfg_type=separated` with `[text_weight, constraint_weight]` (defaults 2.0, 2.0 if
either key is missing).

**CLI flags override meta.json** — explicit `--cfg_type` / `--cfg_weight` take priority
(`:188-211`). Precedence: CLI explicit > `meta.cfg` > model internal default (separated, 2.0/2.0).

### Demo UI vs CLI parity

All the meta.json keys are written by the demo's "Save Example" flow
(`kimodo/demo/ui.py:1681-1698`). Reverse (loading) is at `:2071-2084`. Fields that exist
in the UI but **are not currently round-tripped to meta.json**:

- `num_transition_frames`, `share_transition`, `percentage_transition_override`
  (Transitions folder).
- `root_margin` (Post Processing).
- `postprocess_checkbox` (Post Processing Enable).

If your CLI invocation wants to match a specific demo session, these defaults apply:
- `num_transition_frames=5`
- `share_transition=false`
- `percentage_transition_override=0.10`
- `root_margin=0.04`
- `postprocess=true` (off for G1)

Your existing pipeline is likely missing the `cfg` block. Without it, the CLI falls through
to the model's internal default (`separated` with `[2.0, 2.0]`), which happens to match the
demo's defaults — so absent `cfg`, behavior should still match. But explicitly passing
`--cfg_type separated --cfg_weight 2.0 2.0` is worth doing for repeatability.

---

## 7. Classifier-Free Guidance (CFG)

Three modes in `CFG_TYPES = ["nocfg", "regular", "separated"]` (`kimodo/model/cfg.py:10`).
Logic at `kimodo/model/cfg.py:59-131`.

### Default when neither flag is passed

`argparse.SUPPRESS` means the argparse namespace doesn't even contain `cfg_type` /
`cfg_weight`. `resolve_cfg_kwargs` returns `{}` (`generate.py:226`). The model then
uses its init default, which is **`cfg_type="separated"`** (`kimodo/model/kimodo_model.py:34`,
also `kimodo/model/cfg.py:16`). The `cfg_weight` default passed through `__call__` is
**`[2.0, 2.0]`** (`kimodo/model/kimodo_model.py:350`).

**So: the CLI default IS CFG on, separated mode, 2.0/2.0.** This is the same as the demo
default.

### Mode internals

- **`nocfg`** (`kimodo/model/cfg.py:59-69`) — single forward pass with the text conditioning
  intact. No uncond subtraction. Cheapest, generally lowest quality for prompt adherence
  but smoother.

- **`regular`** (`:70-93`) — two forward passes stacked in batch (cond + uncond). Standard
  classifier-free guidance: `out = out_uncond + w * (out_text - out_uncond)`. One scalar
  weight. Text and constraint conditioning are fused — they are both zeroed out in the
  uncond path (text features zeroed at `:73`, motion mask zeroed at `:75`).

- **`separated`** (`:94-129`) — three forward passes: text-only, constraint-only,
  uncond. `out = out_uncond + w_text*(out_text - out_uncond) + w_constraint*(out_constraint - out_uncond)`.
  Two weights. Allows tuning text vs constraint adherence independently — e.g. if
  constraints are dominating too much, lower `w_constraint`; if the motion isn't following
  the text, raise `w_text`. **3× the compute of `nocfg`, 1.5× of `regular`.**

### Demo's `cfg` block → CLI mapping

The demo's `{enabled: true, text_weight: 2.0, constraint_weight: 2.0}` is equivalent to:

```
--cfg_type separated --cfg_weight 2.0 2.0
```

`regular` is **not exposed in the demo UI at all** — the demo only toggles between
"separated" (enabled) and "nocfg" (disabled) (`kimodo/demo/ui.py:2871`:
`cfg_type="separated" if gui_cfg_checkbox.value else "nocfg"`).

### Quality impact (from code, not benchmarks)

- `separated [2.0, 2.0]` → robust default, respects both prompt and constraints.
- Higher weights (3.0-5.0) → more faithful but risk of motion artifacts / unnatural poses.
- `0.0` for a weight effectively removes that channel's guidance but keeps the extra
  forward pass cost — set `cfg_type=regular` if you only want text guidance.

---

## 8. Post-Processing

`post_process_motion` (`kimodo/postprocess.py:181-346`). Called from
`Kimodo._multiprompt` (`kimodo/model/kimodo_model.py:322-332`) after inverse motion
decoding but before SOMA30→SOMA77 upgrade. Non-multi-prompt path at `:511-522`.

### What `--no-postprocess` disables

The entire `motion_postprocess.correct_motion` call (`kimodo/postprocess.py:314-327`).
This is C++/native code from an optional package `motion_correction` — if it's not
installed, postprocess raises `RuntimeError` (`:307-313`) so you need
`pip install -e .` with the motion_correction extra.

What it does:
- Builds a "working rig" from skeleton neutral positions (`:112-178`), snapping the lowest
  toe joint to ground plane plus `above_ground_offset` (0.02 m SOMA, 0.007 m others).
- Calls the native motion corrector with: hip translations, rotations as quaternions,
  foot-contact labels, per-constraint target keyframes, constraint-type masks, threshold
  `contact_threshold=0.5`, `root_margin=0.04` (CLI never exposes these).
- The corrector modifies `hip_translations_corrected` and `rotations_corrected` **in place**
  (`:316-327`).
- Then re-runs FK to regenerate `posed_joints` and `global_rot_mats` (`:332-337`).

### What it modifies

**Both** root translation (hip Y and XZ) and per-bone rotations (to enforce foot contact).
It can shift the root up/down/horizontally (bounded by `root_margin` when constraints are
active) and rotate leg/foot joints to pin contact frames.

### Interactions with BVH export

Runs **before** BVH export and before SOMA30→SOMA77 upgrade. The rotations written to BVH
are the corrected local rotations; the root translation written is the corrected hip
position. Disabling postprocessing can produce visible foot-skate in the BVH.

---

## 9. Demo Settings and CLI Gap Analysis

Source files: `kimodo/demo/ui.py`, `kimodo/demo/generation.py`, `kimodo/demo/config.py`.

### Full list of user-adjustable controls

| Control | Type | Default | File:Line | CLI flag? |
|---|---|---|---|---|
| Num Samples | slider 1-10 | 1 | ui.py:353 | `--num_samples` |
| SOMA layer | checkbox | False | ui.py:362 | (rendering only) |
| Seed | number | 42 | ui.py:369 | `--seed` |
| Denoising Steps | slider 2-1000 | 100 | ui.py:372 | `--diffusion_steps` |
| CFG Enable | checkbox | True | ui.py:380 | `--cfg_type nocfg` when False |
| Text Weight | slider 0-5 step 0.1 | 2.0 | ui.py:386 | `--cfg_weight X Y` first |
| Constraint Weight | slider 0-5 step 0.1 | 2.0 | ui.py:394 | `--cfg_weight X Y` second |
| Transition frames | slider 1-10 | 5 | ui.py:407 | `--num_transition_frames` |
| Override previous frames | checkbox | False | ui.py:415 | **CLI always = True** (§ gap) |
| Percentage overriding frames | slider 0-30 | 10 | ui.py:420 | **CLI always = 10%** (§ gap) |
| Postprocess Enable | checkbox | True | ui.py:439 | `--no-postprocess` inverts |
| Root Margin | number, step 0.01 | 0.04 | ui.py:445 | **CLI always = 0.04** (§ gap) |
| Real robot rotations (G1) | checkbox | False | ui.py:462 | **Not exposed** |
| Gizmo space | Local/World | Local | ui.py:471 | (editing only) |
| Dense Path | checkbox | False | ui.py:492 | (constraint semantics) |

### Silent defaults where CLI differs from UI

1. **`share_transition`**: UI default is False (`ui.py:417`); CLI default inside
   `Kimodo.__call__` is **True** (`kimodo/model/kimodo_model.py:357`). So if you are
   reproducing what a user sees in the demo, your CLI runs actually blend
   `num_transition_frames + 10%` into the previous segment, while the demo by default
   does not. **This is a real discrepancy.** There is no CLI flag to change it.

2. **`percentage_transition_override`**: Hardcoded 0.10 in CLI
   (`kimodo_model.py:358`); matches UI default 10% when `share_transition=true`, but only
   takes effect at all when share is on.

3. **`root_margin`**: CLI uses 0.04 m (`kimodo_model.py:361`); UI default 0.04 m matches.

4. **`first_heading_angle`**: CLI passes 0 (model default). UI doesn't expose this at all
   — both default to facing +Z.

---

## 10. Coordinate Conventions

### SOMA skeleton T-pose and identity rotations

`SkeletonBase.__init__` asserts `neutral_joints[0] == 0` for the root
(`kimodo/skeleton/base.py:112-113`). The standard T-pose is defined by the data in
`joints.p` (neutral) plus `standard_t_pose_global_offsets_rots.p` (which maps the
"baked rest" to true identity, `:79-82`).

`global_rot_offsets` is what `to_standard_tpose` / `from_standard_tpose` apply
(`kimodo/skeleton/transforms.py:76-106`). Interpretation:

- At standard T-pose, **all local rotations are identity**, and all global rotations are
  identity (for a human in canonical Y-up +Z-forward pose — arms out, palms down).
- The BONES-SEED rest pose differs from standard T-pose by `global_rot_offsets`. When
  loaded from BVH (`standard_tpose=False`), the `from_standard_tpose` transform re-expresses
  the Kimodo-space rotations into the BONES-SEED frame.

### Up axis / forward axis

**Y-up, +Z forward** for all Kimodo internals (`kimodo/exports/smplx.py:54`,
`kimodo/exports/mujoco.py:3` header). Scale: **meters** (`kimodo/exports/motion_io.py:146`).

### Root motion representation per format

| Format | Root translation | Root rotation | Units |
|---|---|---|---|
| Kimodo NPZ | `root_positions` key, meters, Y-up +Z-fwd | First joint of `local_rot_mats` / `global_rot_mats` | meters |
| BVH | Hips XYZ channels (**cm**), ZYX Euler | Hips ZYX rotation channels on local rotations | centimeters |
| AMASS NPZ | `trans` key, **Z-up by default**, pelvis offset subtracted | `root_orient` axis-angle | meters |
| G1 CSV (MuJoCo) | Columns 0-2, Z-up +X-fwd | Columns 3-6 quaternion wxyz | meters |

### Scale conventions

- Kimodo internal → **meters**.
- BVH file output → **centimeters** (`kimodo/exports/bvh.py:135-136`).
- AMASS → **meters**.
- G1 CSV → **meters**.

---

## 11. Loopability

**Kimodo has no built-in loop awareness.** Nothing in the model or CLI attempts to
align frame 0 with frame N-1. The diffusion process samples a random noise tensor
at `kimodo_model.py:581` and denoises unconditionally on time boundaries; there is no
`loop=True` flag anywhere in the codebase (grep `loop` finds nothing in the generation
path).

You can get close to a loop via:

1. **Constraints**: pin the first and last poses to match with a `FullBodyConstraintSet`.
   Constraint CFG weight ≥ 2 helps.
2. **Post-hoc**: write your own loop finisher — slerp the last K frames toward the first
   K, or find a pair of near-matching frames and trim.

Note that multi-prompt transition blending (§2) is **not** the same as loop blending —
it only crossfades between successive segments, not between the end and start of the
whole sequence.

---

## 12. Footgun Inventory

1. **Period-split prompt** — passing `"A. B. C."` with `--duration 5` yields three
   5-second segments (15 s total), not 5 s. *Fix*: pre-strip periods from your prompt,
   or always pass space-separated durations matching your segment count.
   *File*: `kimodo/scripts/generate.py:123-137`.

2. **BVH rest pose** — default (`--bvh_standard_tpose` omitted) exports in BONES-SEED
   frame, which is NOT the usual straight-arms T-pose. *Fix*: pass
   `--bvh_standard_tpose` whenever your retarget target assumes a standard T-pose at frame 0.
   *File*: `kimodo/exports/bvh.py:99-104`.

3. **BVH silently skipped for non-SOMA** — `--bvh` with G1 or SMPL-X models prints a
   warning and produces no file. *Fix*: check `resolved_model` and branch on your side.
   *File*: `kimodo/scripts/generate.py:382-384`.

4. **`share_transition` default mismatch** — CLI defaults `share_transition=True` but
   demo defaults False, and the flag is not exposed. *Fix*: accept the CLI behavior, or
   patch `Kimodo.__call__` to expose it. *File*: `kimodo/model/kimodo_model.py:357`.

5. **CFG "nocfg" with `--cfg_weight`** — passing `--cfg_type nocfg --cfg_weight 2.0` is a
   hard error. *File*: `kimodo/scripts/generate.py:184-185`.

6. **`--cfg_weight` count mismatch** — `regular` wants 1 float, `separated` wants 2. Auto-
   detection exists only when `--cfg_type` is absent. Passing `--cfg_type separated
   --cfg_weight 2.0` raises. *File*: `generate.py:192-200`.

7. **Postprocess requires native package** — without `motion_correction` installed,
   postprocess crashes. *Fix*: `pip install -e .` (with the repo's full extras) or pass
   `--no-postprocess`. *File*: `kimodo/postprocess.py:307-313`.

8. **G1 postprocess forced off** — `--no-postprocess` is implied for any model matching
   `"g1" in name`, even if the flag isn't passed. *File*: `generate.py:316`.

9. **Unit surprise in BVH** — joint offsets are cm, all other Kimodo outputs are m.
   *Fix*: when round-tripping Kimodo NPZ ↔ BVH via `bvh_to_kimodo_motion`, the converter
   already handles this (`kimodo/exports/bvh.py:286-297`), but external parsers must.

10. **AMASS z_up default** — `convert_save_npz` defaults `z_up=True`
    (`kimodo/exports/smplx.py:198`). If you feed the resulting file into SMPL-X code that
    expects Y-up, motion will appear rotated 90°.

11. **Output stem vs folder semantics** — with one sample, `--output foo` writes `foo.npz`.
    With multiple, it writes `foo/foo_00.npz`. AMASS uses `foo_amass.npz` vs `foo/amass.npz`.
    Mixing these is common; always check `n_samples` to pick the right glob.
    *File*: `generate.py:334-364`.

12. **Online HF check on every run** — without `LOCAL_CACHE=true`, `snapshot_download`
    hits the network even when the model is already cached. *Fix*: set `LOCAL_CACHE=true`
    in your container env. *File*: `kimodo/model/load_model.py:45-58`.

13. **Text encoder auto-fallback latency** — `TEXT_ENCODER_MODE=auto` probes the API
    then falls back to loading 8B LLM2Vec locally on failure; this adds seconds and ~16
    GB VRAM. Pin to `local` or `api` explicitly. *File*: `kimodo/model/load_model.py:83-100`.

14. **SOMA30 → SOMA77 expansion uses `relaxed_hands_rest_pose`** — hand/face joints in
    the output are *not* driven by the model; they come from a static asset. If your
    pipeline cares about finger animation from a SOMA model, you cannot get it.
    *File*: `kimodo/skeleton/definitions.py:247-256`.

15. **BVH adds a "Root" wrapper joint** — parsing the BVH tree naively and treating the
    root as the body root will pick up a stationary wrapper. The real pelvis is named
    `"Hips"`. *File*: `kimodo/exports/bvh.py:162-166`.

16. **Postprocess modifies root translation** — if your constraint logic assumes root
    trajectory is untouched by the generator, `--no-postprocess` may be required to
    preserve it exactly.
    *File*: `kimodo/postprocess.py:314-327`.

17. **`num_transition_frames` is ignored for single-prompt generation** — it only has
    effect when the prompt contains `.` producing ≥2 segments. Passing it with a plain
    prompt is silently no-op. *File*: `kimodo/model/kimodo_model.py:189-243` (only inside
    `_multiprompt`).

18. **FPS is per-model** — always read `model.fps` (comes from each model's config.yaml
    via `denoiser.motion_rep.fps`, `kimodo/model/kimodo_model.py:49`). Do not hardcode
    30 fps; while all current release models are 30, future variants may not be.
