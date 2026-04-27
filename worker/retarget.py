"""NPZ -> FBX retargeter (worker-side, x86 emulated under qemu).

Math + fbxsdkpy calls — same code that used to live in app/npz_to_fbx.py
before the GB10 deployment split. Autodesk only ships the FBX SDK as
x86_64 binaries, so this module runs inside a --platform linux/amd64
container and the application talks to it over HTTP.

Public surface: ``SOMA_TO_MIXAMO`` and ``retarget_npz_to_fbx``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# SOMA (kimodo native 77-joint) -> Mixamo humanoid bone names.
SOMA_TO_MIXAMO: dict[str, str] = {
    "hips":           "mixamorig:hips",
    "spine1":         "mixamorig:spine",
    "spine2":         "mixamorig:spine1",
    "chest":          "mixamorig:spine2",
    "neck1":          "mixamorig:neck",
    "head":           "mixamorig:head",
    "leftshoulder":   "mixamorig:leftshoulder",
    "leftarm":        "mixamorig:leftarm",
    "leftforearm":    "mixamorig:leftforearm",
    "lefthand":       "mixamorig:lefthand",
    "rightshoulder":  "mixamorig:rightshoulder",
    "rightarm":       "mixamorig:rightarm",
    "rightforearm":   "mixamorig:rightforearm",
    "righthand":      "mixamorig:righthand",
    "leftleg":        "mixamorig:leftupleg",
    "leftshin":       "mixamorig:leftleg",
    "leftfoot":       "mixamorig:leftfoot",
    "lefttoebase":    "mixamorig:lefttoebase",
    "rightleg":       "mixamorig:rightupleg",
    "rightshin":      "mixamorig:rightleg",
    "rightfoot":      "mixamorig:rightfoot",
    "righttoebase":   "mixamorig:righttoebase",
    "lefthandthumb1":   "mixamorig:lefthandthumb1",
    "lefthandthumb2":   "mixamorig:lefthandthumb2",
    "lefthandthumb3":   "mixamorig:lefthandthumb3",
    "lefthandindex1":   "mixamorig:lefthandindex1",
    "lefthandindex2":   "mixamorig:lefthandindex2",
    "lefthandindex3":   "mixamorig:lefthandindex3",
    "lefthandindex4":   "mixamorig:lefthandindex4",
    "lefthandmiddle1":  "mixamorig:lefthandmiddle1",
    "lefthandmiddle2":  "mixamorig:lefthandmiddle2",
    "lefthandmiddle3":  "mixamorig:lefthandmiddle3",
    "lefthandmiddle4":  "mixamorig:lefthandmiddle4",
    "lefthandring1":    "mixamorig:lefthandring1",
    "lefthandring2":    "mixamorig:lefthandring2",
    "lefthandring3":    "mixamorig:lefthandring3",
    "lefthandring4":    "mixamorig:lefthandring4",
    "lefthandpinky1":   "mixamorig:lefthandpinky1",
    "lefthandpinky2":   "mixamorig:lefthandpinky2",
    "lefthandpinky3":   "mixamorig:lefthandpinky3",
    "lefthandpinky4":   "mixamorig:lefthandpinky4",
    "righthandthumb1":  "mixamorig:righthandthumb1",
    "righthandthumb2":  "mixamorig:righthandthumb2",
    "righthandthumb3":  "mixamorig:righthandthumb3",
    "righthandindex1":  "mixamorig:righthandindex1",
    "righthandindex2":  "mixamorig:righthandindex2",
    "righthandindex3":  "mixamorig:righthandindex3",
    "righthandindex4":  "mixamorig:righthandindex4",
    "righthandmiddle1": "mixamorig:righthandmiddle1",
    "righthandmiddle2": "mixamorig:righthandmiddle2",
    "righthandmiddle3": "mixamorig:righthandmiddle3",
    "righthandmiddle4": "mixamorig:righthandmiddle4",
    "righthandring1":   "mixamorig:righthandring1",
    "righthandring2":   "mixamorig:righthandring2",
    "righthandring3":   "mixamorig:righthandring3",
    "righthandring4":   "mixamorig:righthandring4",
    "righthandpinky1":  "mixamorig:righthandpinky1",
    "righthandpinky2":  "mixamorig:righthandpinky2",
    "righthandpinky3":  "mixamorig:righthandpinky3",
    "righthandpinky4":  "mixamorig:righthandpinky4",
}

_HEIGHT_KEYWORDS = (
    "hips", "spine", "neck", "head", "arm", "leg", "foot", "ankle",
    "knee", "shoulder", "elbow", "pelvis", "mixamo",
)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.sum(q**2))


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _mat3_to_quat_wxyz(m33: np.ndarray) -> np.ndarray:
    q = R.from_matrix(m33).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _fbx_mat_to_np(fbx_mat) -> np.ndarray:
    m = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = fbx_mat.Get(i, j)
    return m


def _fbx_mat_rot_to_quat_wxyz(fbx_mat) -> np.ndarray:
    m = _fbx_mat_to_np(fbx_mat)
    m33 = m[:3, :3].T
    q = R.from_matrix(m33).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


class _Bone:
    __slots__ = (
        "name", "parent_name",
        "local_matrix", "world_matrix", "head",
        "has_skeleton_attr", "rest_rotation",
        "world_animation", "world_location_animation",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.parent_name: str | None = None
        self.local_matrix = np.eye(4)
        self.world_matrix = np.eye(4)
        self.head = np.zeros(3)
        self.has_skeleton_attr = False
        self.rest_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.world_animation: dict[int, np.ndarray] = {}
        self.world_location_animation: dict[int, np.ndarray] = {}


class _Skeleton:
    def __init__(self, name: str = "skeleton") -> None:
        self.name = name
        self.bones: dict[str, _Bone] = {}
        self.node_rest_rotations: dict[str, np.ndarray] = {}
        self.fps = 30.0
        self.frame_start = 0
        self.frame_end = 0

    def add(self, bone: _Bone) -> None:
        self.bones[bone.name.lower()] = bone

    def find(self, name: str) -> _Bone | None:
        lo = name.lower()
        if lo in self.bones:
            return self.bones[lo]
        if ":" in lo:
            stripped = lo.split(":")[-1]
            if stripped in self.bones:
                return self.bones[stripped]
        for bname, bone in self.bones.items():
            if ":" in bname and bname.split(":")[-1] == lo:
                return bone
        return None


def _build_source_skeleton(
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
    neutral_joints: np.ndarray,
    joint_names: list[str],
    joint_parents: list[int],
    fps: float,
    skeleton_name: str,
) -> _Skeleton:
    if posed_joints.ndim != 3:
        raise ValueError(f"posed_joints must be (T, J, 3), got {posed_joints.shape}")
    if global_rot_mats.shape[:2] != posed_joints.shape[:2]:
        raise ValueError(
            f"shape mismatch: posed_joints {posed_joints.shape}, global_rot_mats {global_rot_mats.shape}"
        )
    T, J, _ = posed_joints.shape
    if len(joint_names) != J or len(joint_parents) != J:
        raise ValueError(
            f"joint_names ({len(joint_names)}) / joint_parents ({len(joint_parents)}) "
            f"must match J={J}"
        )

    identity_q = np.array([1.0, 0.0, 0.0, 0.0])
    skel = _Skeleton(skeleton_name)
    skel.fps = float(fps)
    skel.frame_start = 0
    skel.frame_end = T - 1

    for i, name in enumerate(joint_names):
        bone = _Bone(name)
        pidx = joint_parents[i]
        bone.parent_name = joint_names[pidx] if pidx >= 0 else None
        bone.rest_rotation = identity_q.copy()
        bone.head = np.asarray(neutral_joints[i], dtype=np.float64).copy()
        bone.world_matrix = np.eye(4)
        bone.world_matrix[3, :3] = bone.head
        for f in range(T):
            bone.world_animation[f] = _mat3_to_quat_wxyz(global_rot_mats[f, i])
            bone.world_location_animation[f] = np.asarray(posed_joints[f, i], dtype=np.float64)
        skel.add(bone)
        skel.node_rest_rotations[name] = bone.rest_rotation
    return skel


def _collect_target_skeleton(node, scene, skel: _Skeleton, parent_name: str | None = None) -> None:
    import fbx
    from fbx import FbxTime

    attr = node.GetNodeAttribute()
    node_name = node.GetName()
    is_bone = False
    if attr:
        atype = attr.GetAttributeType()
        if atype in (3, 4):
            is_bone = True
        elif atype == 2 and (node.GetChildCount() > 0 or parent_name):
            is_bone = True

    name_lo = node_name.lower()
    if any(k in name_lo for k in (
        "hips", "hip", "spine", "neck", "head", "arm", "leg", "foot",
        "ankle", "knee", "shoulder", "elbow", "pelvis", "joint", "mixamo",
        "thigh", "forearm", "hand", "finger", "clavicle", "collar", "toe",
        "thumb", "index", "middle", "ring", "pinky", "upleg", "wrist", "chest",
    )):
        is_bone = True

    # Read rest rotation from the live transform with no animation stack — for
    # Mixamo exports where the mesh was bound while the character was in its
    # original (A-pose) Meshy source image, BindPose is the A-pose rotation but
    # the user's "T-pose character download" puts the bones in T-pose via
    # LclRotation defaults. Using BindPose as t_rest there composes source
    # motion onto an A-pose reference and produces off-axis arm/leg twists.
    # The caller has already disabled the animation stack (see _load_target_fbx),
    # so EvaluateGlobalTransform here matches what Mixamo's web preview shows.
    t_eval = FbxTime()
    global_mat = node.EvaluateGlobalTransform(t_eval)
    skel.node_rest_rotations[node_name] = _fbx_mat_rot_to_quat_wxyz(global_mat)

    if is_bone:
        existing = skel.find(node_name)
        is_real = attr is not None and attr.GetAttributeType() in (3, 4)
        if existing:
            if is_real and not existing.has_skeleton_attr:
                skel.bones.pop(existing.name.lower(), None)
            else:
                is_bone = False

    if is_bone:
        bone = _Bone(node_name)
        bone.has_skeleton_attr = bool(attr and attr.GetAttributeType() in (3, 4))
        bone.parent_name = parent_name
        bone.local_matrix = _fbx_mat_to_np(node.EvaluateLocalTransform(t_eval))
        bone.world_matrix = _fbx_mat_to_np(global_mat)
        t_g = global_mat.GetT()
        bone.head = np.array([t_g[0], t_g[1], t_g[2]])
        bone.rest_rotation = skel.node_rest_rotations[node_name]
        skel.add(bone)
        parent_name = node_name

    for i in range(node.GetChildCount()):
        _collect_target_skeleton(node.GetChild(i), scene, skel, parent_name)


def _load_target_fbx(path: str):
    from fbx import FbxManager, FbxScene, FbxImporter, FbxIOSettings

    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSRoot")
    for prop in (
        "Import|AdvOptGrp|Fbx|Material",
        "Import|AdvOptGrp|Fbx|Texture",
        "Import|AdvOptGrp|Fbx|Model",
        "Import|AdvOptGrp|Fbx|Shape",
        "Import|AdvOptGrp|Fbx|Skin",
    ):
        try:
            ios.SetBoolProp(prop, True)
        except Exception:
            pass
    manager.SetIOSettings(ios)
    scene = FbxScene.Create(manager, "Scene")
    importer = FbxImporter.Create(manager, "")
    if not importer.Initialize(path, -1, ios):
        err = importer.GetStatus().GetErrorString()
        raise RuntimeError(f"cannot open FBX {path!r}: {err}")
    if not importer.Import(scene):
        err = importer.GetStatus().GetErrorString()
        raise RuntimeError(f"FBX import failed for {path!r}: {err}")
    importer.Destroy()

    stack = scene.GetCurrentAnimationStack()
    scene.SetCurrentAnimationStack(None)

    skel = _Skeleton(os.path.basename(path))
    _collect_target_skeleton(scene.GetRootNode(), scene, skel)

    scene.SetCurrentAnimationStack(stack)
    return manager, scene, skel


def _skeleton_height(skel: _Skeleton) -> float:
    y_min, y_max = 1e9, -1e9
    found = False
    for bone in skel.bones.values():
        if any(k in bone.name.lower() for k in _HEIGHT_KEYWORDS):
            h = bone.head[1]
            if abs(h) < 1e-6:
                continue
            y_min, y_max = min(y_min, h), max(y_max, h)
            found = True
    return (y_max - y_min) if found and y_max > y_min else 1.0


def _retarget(
    src: _Skeleton,
    tgt: _Skeleton,
    mapping: dict[str, str],
    yaw_offset_deg: float = 0.0,
    force_scale: float = 0.0,
    preserve_root_xz: bool = False,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    yaw_q_raw = R.from_euler("y", yaw_offset_deg, degrees=True).as_quat()
    yaw_q = np.array([yaw_q_raw[3], yaw_q_raw[0], yaw_q_raw[1], yaw_q_raw[2]])

    active: list[tuple[_Bone, _Bone, np.ndarray]] = []
    mapped_src, mapped_tgt = set(), set()
    for s_key, t_key in mapping.items():
        s = src.find(s_key)
        t = tgt.find(t_key)
        if not s or not t:
            continue
        if s.name in mapped_src or t.name in mapped_tgt:
            continue
        offset = _quat_mul(_quat_inv(s.rest_rotation), t.rest_rotation)
        active.append((s, t, offset))
        mapped_src.add(s.name)
        mapped_tgt.add(t.name)
    if not active:
        raise RuntimeError("no bone pairs matched between source and target skeletons")

    src_h = _skeleton_height(src)
    tgt_h = _skeleton_height(tgt)
    scale = force_scale if force_scale > 1e-4 else (tgt_h / src_h if src_h > 0.01 else 1.0)

    frames = range(src.frame_start, src.frame_end + 1)
    tgt_world_rots: dict[str, dict[int, np.ndarray]] = {}
    out_rots: dict[str, dict[int, np.ndarray]] = {}
    out_locs: dict[str, dict[int, np.ndarray]] = {}

    for s, t, off in active:
        tgt_world_rots[t.name] = {}
        for f in frames:
            sw = s.world_animation.get(f, s.rest_rotation)
            tw = _quat_mul(sw, off)
            if yaw_offset_deg != 0.0:
                tw = _quat_mul(yaw_q, tw)
            tgt_world_rots[t.name][f] = tw

        if "hips" in s.name.lower() or "hips" in t.name.lower():
            out_locs[t.name] = {}
            t_rest_local_pos = t.local_matrix[3, :3]
            pname = t.parent_name
            s_p_0 = s.world_location_animation.get(
                src.frame_start, s.world_matrix[3, :3]
            )
            yaw_rot = R.from_quat([yaw_q[1], yaw_q[2], yaw_q[3], yaw_q[0]])

            for f in frames:
                s_p = s.world_location_animation.get(f, s.world_matrix[3, :3])
                delta_source = s_p - s_p_0
                delta_target = delta_source * scale
                if not preserve_root_xz:
                    delta_target = np.array([0.0, delta_target[1], 0.0])
                if yaw_offset_deg != 0.0:
                    delta_target = yaw_rot.apply(delta_target)
                prot = tgt_world_rots.get(pname, {}).get(f)
                if prot is None:
                    prot = tgt.node_rest_rotations.get(
                        pname, np.array([1.0, 0.0, 0.0, 0.0])
                    )
                    if yaw_offset_deg != 0.0:
                        prot = _quat_mul(yaw_q, prot)
                p_rot_inv = R.from_quat([prot[1], prot[2], prot[3], prot[0]]).inv()
                local_disp = p_rot_inv.apply(delta_target)
                out_locs[t.name][f] = t_rest_local_pos + local_disp

    for s, t, _ in active:
        out_rots[t.name] = {}
        pname = t.parent_name
        for f in frames:
            prot = tgt_world_rots.get(pname, {}).get(f)
            if prot is None:
                prot = tgt.node_rest_rotations.get(pname, np.array([1.0, 0.0, 0.0, 0.0]))
                if yaw_offset_deg != 0.0:
                    prot = _quat_mul(yaw_q, prot)
            out_rots[t.name][f] = _quat_mul(_quat_inv(prot), tgt_world_rots[t.name][f])

    return out_rots, out_locs


_ROT_ORDER_MAP = {0: "xyz", 1: "xzy", 2: "yzx", 3: "yxz", 4: "zxy", 5: "zyx"}


def _apply_animation(
    scene,
    out_rots: dict[str, dict[int, np.ndarray]],
    out_locs: dict[str, dict[int, np.ndarray]],
) -> None:
    import fbx
    from fbx import FbxAnimStack, FbxAnimLayer, FbxTime

    tmode = scene.GetGlobalSettings().GetTimeMode()

    try:
        criteria = fbx.FbxCriteria.ObjectType(FbxAnimStack.ClassId)
        for i in range(scene.GetSrcObjectCount(criteria) - 1, -1, -1):
            s = scene.GetSrcObject(criteria, i)
            scene.DisconnectSrcObject(s)
            s.Destroy()
    except Exception:
        pass

    stack = FbxAnimStack.Create(scene, "Take 001")
    layer = FbxAnimLayer.Create(scene, "BaseLayer")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)

    def apply_to_node(node):
        name = node.GetName()
        if name in out_rots:
            _write_rotation_curves(node, layer, out_rots[name], tmode)
        if name in out_locs:
            _write_translation_curves(node, layer, out_locs[name], tmode)
        for i in range(node.GetChildCount()):
            apply_to_node(node.GetChild(i))

    apply_to_node(scene.GetRootNode())

    # Explicitly set the animation stack's time span to match our keyframe
    # range. Without this, three.js FBXLoader reads a stale span from the
    # original input FBX and reports the clip as longer than it really is.
    frame_indices = set()
    for per_frame in out_rots.values():
        frame_indices.update(per_frame.keys())
    for per_frame in out_locs.values():
        frame_indices.update(per_frame.keys())
    if frame_indices:
        first_frame = min(frame_indices)
        last_frame = max(frame_indices)
        start_time = FbxTime()
        start_time.SetFrame(first_frame, tmode)
        stop_time = FbxTime()
        stop_time.SetFrame(last_frame, tmode)
        span = fbx.FbxTimeSpan(start_time, stop_time)
        stack.SetLocalTimeSpan(span)
        stack.SetReferenceTimeSpan(span)
        try:
            scene.GetGlobalSettings().SetTimelineDefaultTimeSpan(span)
        except Exception:
            pass


def _write_rotation_curves(node, layer, frame_quats: dict[int, np.ndarray], tmode) -> None:
    import fbx
    from fbx import FbxTime

    node.LclRotation.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)

    order_str = _ROT_ORDER_MAP.get(node.RotationOrder.Get(), "xyz")
    pv = node.PreRotation.Get()
    pq = R.from_euler("xyz", [pv[0], pv[1], pv[2]], degrees=True).as_quat()
    pre_inv = _quat_inv(np.array([pq[3], pq[0], pq[1], pq[2]]))
    post_v = node.PostRotation.Get()
    post_q = R.from_euler("xyz", [post_v[0], post_v[1], post_v[2]], degrees=True).as_quat()
    post_inv = _quat_inv(np.array([post_q[3], post_q[0], post_q[1], post_q[2]]))

    cx = node.LclRotation.GetCurve(layer, "X", True)
    cy = node.LclRotation.GetCurve(layer, "Y", True)
    cz = node.LclRotation.GetCurve(layer, "Z", True)
    cx.KeyModifyBegin(); cy.KeyModifyBegin(); cz.KeyModifyBegin()
    curve_map = {"x": cx, "y": cy, "z": cz}
    linear = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear

    for f, q_local in frame_quats.items():
        t = FbxTime()
        t.SetFrame(f, tmode)
        q_final = _quat_mul(pre_inv, _quat_mul(q_local, post_inv))
        rot_q = R.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
        e = rot_q.as_euler(order_str.lower(), degrees=True)
        for i_ax, ch in enumerate(order_str.lower()):
            c = curve_map[ch]
            idx = c.KeyAdd(t)[0]
            c.KeySetValue(idx, float(e[i_ax]))
            c.KeySetInterpolation(idx, linear)

    cx.KeyModifyEnd(); cy.KeyModifyEnd(); cz.KeyModifyEnd()


def _write_translation_curves(node, layer, frame_locs: dict[int, np.ndarray], tmode) -> None:
    import fbx
    from fbx import FbxTime

    node.LclTranslation.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
    tx = node.LclTranslation.GetCurve(layer, "X", True)
    ty = node.LclTranslation.GetCurve(layer, "Y", True)
    tz = node.LclTranslation.GetCurve(layer, "Z", True)
    tx.KeyModifyBegin(); ty.KeyModifyBegin(); tz.KeyModifyBegin()
    linear = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear

    for f, loc in frame_locs.items():
        t = FbxTime()
        t.SetFrame(f, tmode)
        for c, val in zip((tx, ty, tz), loc):
            idx = c.KeyAdd(t)[0]
            c.KeySetValue(idx, float(val))
            c.KeySetInterpolation(idx, linear)

    tx.KeyModifyEnd(); ty.KeyModifyEnd(); tz.KeyModifyEnd()


def _save_fbx(manager, scene, output_path: str) -> None:
    import fbx
    from fbx import FbxExporter, FbxIOSettings

    ios = manager.GetIOSettings() or FbxIOSettings.Create(manager, "IOSRoot")
    if manager.GetIOSettings() is None:
        manager.SetIOSettings(ios)

    for attr in ("EXP_FBX_EMBEDDED", "EXP_FBX_MATERIAL", "EXP_FBX_TEXTURE"):
        const = getattr(fbx, attr, None)
        if const is not None:
            try:
                ios.SetBoolProp(const, True)
            except Exception:
                pass
    for prop in (
        "Export|AdvOptGrp|Fbx|Material",
        "Export|AdvOptGrp|Fbx|Texture",
        "Export|AdvOptGrp|Fbx|Embedded",
        "Export|AdvOptGrp|Fbx|Model",
        "Export|AdvOptGrp|Fbx|Animation",
        "Export|AdvOptGrp|Fbx|Shape",
        "Export|AdvOptGrp|Fbx|Skin",
    ):
        try:
            ios.SetBoolProp(prop, True)
        except Exception:
            pass

    fmt = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    exporter = FbxExporter.Create(manager, "")
    if not exporter.Initialize(output_path, fmt, ios):
        err = exporter.GetStatus().GetErrorString()
        raise RuntimeError(f"FBX exporter init failed: {err}")
    if not exporter.Export(scene):
        err = exporter.GetStatus().GetErrorString()
        raise RuntimeError(f"FBX export failed: {err}")
    exporter.Destroy()


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
    """Retarget a Kimodo NPZ onto a Mixamo-rigged FBX and save to disk."""
    target_fbx_path = str(Path(target_fbx_path).resolve())
    output_fbx_path = str(Path(output_fbx_path).resolve())
    Path(output_fbx_path).parent.mkdir(parents=True, exist_ok=True)

    src = _build_source_skeleton(
        posed_joints=posed_joints,
        global_rot_mats=global_rot_mats,
        neutral_joints=neutral_joints,
        joint_names=joint_names,
        joint_parents=joint_parents,
        fps=fps,
        skeleton_name=skeleton_name,
    )
    manager, scene, tgt = _load_target_fbx(target_fbx_path)
    try:
        out_rots, out_locs = _retarget(
            src,
            tgt,
            mapping or SOMA_TO_MIXAMO,
            yaw_offset_deg=yaw_offset_deg,
            force_scale=force_scale,
            preserve_root_xz=preserve_root_xz,
        )
        _apply_animation(scene, out_rots, out_locs)
        _save_fbx(manager, scene, output_fbx_path)
    finally:
        manager.Destroy()
    return output_fbx_path


__all__ = ["SOMA_TO_MIXAMO", "retarget_npz_to_fbx"]
