// Single-animation editor.
//
// Layout: two columns — main 3D viewer on the left, seam-preview overlay on
// the right. Header shows prompt + model + seed + duration + frame count.
//
// Below the main viewer: a single timeline bar with draggable start (blue)
// and end (yellow) markers. The track is shaded by per-frame pose distance
// to the start marker's current frame (dark = similar pose, light = different).

import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const params = new URLSearchParams(window.location.search);
const JOB_ID = params.get("job");
const $ = (s) => document.querySelector(s);

if (!JOB_ID) {
  $("#viewer-status").textContent = "No job id in URL (?job=<id>).";
  throw new Error("missing job id");
}

// Auto-trim parameters (client-side only). Backend never trims.
//
// Strategy: exhaustive pairwise search. Compute pose-distance D[i, j] for
// every (i, j) with j - i >= AUTO_MIN_PERIOD_S · fps, keep the AUTO_TOP_K
// closest-pose pairs, then among those pick the one with the LONGEST span.
// This gives a balance of "tight seam" (top-K) and "maximum usable clip"
// (longest span inside that top-K).
//
// Distance metric: summed per-joint Euclidean distance between ROOT-RELATIVE
// joint positions (each frame's joint positions minus that frame's root).
// Measures body shape independent of world translation — exactly what loop
// matching needs — and naturally weights errors by how far a joint would
// actually move.
const AUTO_MIN_PERIOD_S = 1.5;
const AUTO_TOP_K = 20;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  job: null,

  editorData: null,        // { total_frames, num_joints, fps, posed, joint_parents, trim_start_saved, trim_end_saved, prompt, seed, duration_s }
  trimStart: 0,
  trimEnd: 0,

  clipRoot: null,
  mixer: null,
  action: null,

  loadSequenceId: 0,
};

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------
function makeScene(host, opts = {}) {
  const { groundRadius = 20, gridSize = 20, gridDivs = 40 } = opts;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0c0e13);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0e13);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 100000);
  camera.position.set(2.4, 1.5, 3.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.HemisphereLight(0xbfd6ff, 0x0a0a0a, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 1.3);
  key.position.set(4, 6, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.3);
  fill.position.set(-4, 3, -3);
  scene.add(fill);

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(groundRadius, 48),
    new THREE.MeshStandardMaterial({ color: 0x1a1d25, roughness: 0.95 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const grid = new THREE.GridHelper(gridSize, gridDivs, 0x2a2f3a, 0x21252d);
  grid.position.y = 0.01;
  scene.add(grid);

  function resize() {
    const r = host.getBoundingClientRect();
    const w = Math.max(r.width, 200);
    const h = Math.max(r.height, 200);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener("resize", resize);

  return { renderer, scene, camera, controls, ground, grid, resize };
}

const mainCtx = makeScene($("#viewer-canvas"));
// Overlay shows both start-frame and end-frame skeletons — for long clips
// the end-frame skeleton can drift far from start. Double the ground/grid
// base geometry so both skeletons stay on the grid.
const overlayCtx = makeScene($("#overlay-canvas"), {
  groundRadius: 40,
  gridSize: 40,
  gridDivs: 80,
});
mainCtx.controls.autoRotate = false;
mainCtx.controls.autoRotateSpeed = 0.8;

$("#chk-auto-rotate").addEventListener("change", (e) => {
  mainCtx.controls.autoRotate = e.target.checked;
});

const clock = new THREE.Clock();
function tick() {
  const dt = clock.getDelta();
  if (state.mixer) state.mixer.update(dt);
  // Preview-trim clamp. Only apply BEFORE save: when the FBX on disk is
  // still the full raw clip, we clamp mixer time to [trimStart/fps,
  // trimEnd/fps] so the user sees the would-be-exported range. After Save
  // the FBX ITSELF is the trimmed+blended clip — its duration is just
  // (kept_frames + bridge_frames) / fps and state.trimStart is a raw-NPZ
  // frame index that points way past the new shorter clip. Clamping to
  // state.trimStart/fps here would set action.time beyond clip.duration
  // on every tick, killing playback.
  if (state.action && state.editorData && state.editorData.trim_start_saved === null) {
    const { fps } = state.editorData;
    const startT = state.trimStart / fps;
    const endT = state.trimEnd / fps;
    if (state.action.time > endT || state.action.time < startT) {
      state.action.time = startT;
    }
  }
  updatePlayhead();
  mainCtx.controls.update();
  overlayCtx.controls.update();
  mainCtx.renderer.render(mainCtx.scene, mainCtx.camera);
  overlayCtx.renderer.render(overlayCtx.scene, overlayCtx.camera);
  requestAnimationFrame(tick);
}
tick();

// ---------------------------------------------------------------------------
// Playhead scrubber (top of the main viewer)
// ---------------------------------------------------------------------------
//
// Shows the current AnimationAction time as a fraction of the clip duration.
// When the marker jumps from the right edge back to the left, that's the
// loop seam — user can stare at it to catch any pop / hitch / jerk. If the
// clip was saved with a blend or bridge, the corresponding zones at the end
// of the bar are tinted green / orange so the user sees the seam approach.
//
// Click anywhere on the bar to scrub to that position (pause stays handled
// by the default LoopRepeat — the action keeps playing from the scrubbed
// time).

function currentClipDuration() {
  if (!state.action) return 0;
  const clip = state.action.getClip ? state.action.getClip() : null;
  if (clip && clip.duration) return clip.duration;
  // Fallback from editor-data if needed.
  if (state.editorData) {
    return state.editorData.total_frames / state.editorData.fps;
  }
  return 0;
}

function updatePlayhead() {
  const marker = $("#playhead-marker");
  const label = $("#playhead-label");
  if (!marker || !label) return;
  const duration = currentClipDuration();
  if (!state.action || duration <= 0) {
    marker.style.left = "0%";
    label.textContent = "—";
    return;
  }
  const t = state.action.time;
  const pct = Math.max(0, Math.min(100, (t / duration) * 100));
  marker.style.left = `${pct}%`;
  const fps = state.editorData?.fps || 30;
  const frame = Math.max(0, Math.round(t * fps));
  const totalFrames = Math.max(1, Math.round(duration * fps));
  label.textContent = `${frame} / ${totalFrames}f · ${t.toFixed(2)}/${duration.toFixed(2)}s`;
}

function updatePlayheadZones() {
  // Paint blend + bridge zones at the RIGHT end of the playhead bar based on
  // the clip's SAVED seam settings (state.editorData.blend_frames_saved /
  // bridge_frames_saved). These are the zones that exist in the currently-
  // playing FBX clip. Pre-save edits don't change these until after Save.
  const blendEl = $("#playhead-blend");
  const bridgeEl = $("#playhead-bridge");
  if (!blendEl || !bridgeEl || !state.editorData) return;

  const duration = currentClipDuration();
  const fps = state.editorData.fps || 30;
  const totalFrames = Math.max(1, Math.round(duration * fps));

  const blendSaved = state.editorData.blend_frames_saved || 0;
  const bridgeSaved = state.editorData.bridge_frames_saved || 0;

  // The exported FBX is (kept_frames + bridge_frames). Bridge occupies the
  // very last bridgeSaved frames. Blend occupies the last blendSaved frames
  // of the KEPT region, i.e. frames (kept - blendSaved .. kept) of the FBX.
  // Represented as percentages of the total clip.
  if (bridgeSaved > 0 && totalFrames > 0) {
    const widthPct = (bridgeSaved / totalFrames) * 100;
    bridgeEl.style.display = "";
    bridgeEl.style.left = `${100 - widthPct}%`;
    bridgeEl.style.width = `${widthPct}%`;
  } else {
    bridgeEl.style.display = "none";
  }
  if (blendSaved > 0 && totalFrames > 0) {
    // Blend sits at the end of the kept region — i.e., right BEFORE the
    // bridge zone (if any).
    const keptEndFrame = totalFrames - bridgeSaved;
    const blendStartFrame = Math.max(0, keptEndFrame - blendSaved);
    const startPct = (blendStartFrame / totalFrames) * 100;
    const widthPct = ((keptEndFrame - blendStartFrame) / totalFrames) * 100;
    blendEl.style.display = "";
    blendEl.style.left = `${startPct}%`;
    blendEl.style.width = `${widthPct}%`;
  } else {
    blendEl.style.display = "none";
  }
}

$("#viewer-playhead").addEventListener("click", (e) => {
  if (!state.action) return;
  const duration = currentClipDuration();
  if (duration <= 0) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  state.action.time = pct * duration;
});

// ---------------------------------------------------------------------------
// Camera fit
// ---------------------------------------------------------------------------
function fitCameraFromBbox(ctx, bbox) {
  if (!isFinite(bbox.min.x) || bbox.isEmpty()) return;
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.01);
  const dist = maxDim * 2.5;
  ctx.camera.position.set(
    center.x + dist * 0.55,
    center.y + size.y * 0.15,
    center.z + dist * 0.85,
  );
  ctx.controls.target.copy(center);
  const gridScale = Math.max(0.25, maxDim / 8);
  ctx.ground.scale.setScalar(gridScale);
  ctx.grid.scale.setScalar(gridScale);
  ctx.camera.near = Math.max(maxDim * 0.001, 0.001);
  ctx.camera.far = dist * 50;
  ctx.camera.updateProjectionMatrix();
  ctx.controls.minDistance = maxDim * 0.25;
  ctx.controls.maxDistance = dist * 8;
  ctx.controls.update();
}

function bboxFromBones(obj) {
  obj.updateMatrixWorld(true);
  const bb = new THREE.Box3();
  bb.makeEmpty();
  obj.traverse((n) => {
    if (n.isBone) {
      const p = new THREE.Vector3();
      n.getWorldPosition(p);
      bb.expandByPoint(p);
    }
  });
  if (!bb.isEmpty()) {
    const s = bb.getSize(new THREE.Vector3());
    bb.expandByScalar(Math.max(s.x, s.y, s.z) * 0.15);
  }
  return bb;
}

function bboxFromPosedJoints(posed, J, frame) {
  const bb = new THREE.Box3();
  bb.makeEmpty();
  const base = frame * J * 3;
  for (let j = 0; j < J; j++) {
    bb.expandByPoint(new THREE.Vector3(
      posed[base + j * 3],
      posed[base + j * 3 + 1],
      posed[base + j * 3 + 2],
    ));
  }
  if (!bb.isEmpty()) {
    const s = bb.getSize(new THREE.Vector3());
    bb.expandByScalar(Math.max(s.x, s.y, s.z) * 0.15);
  }
  return bb;
}

// ---------------------------------------------------------------------------
// Skeleton drawing
// ---------------------------------------------------------------------------
let overlayGroupStart = null;
let overlayGroupEnd = null;

function disposeGroup(group) {
  group.traverse((o) => {
    if (o.geometry) o.geometry.dispose?.();
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => m?.dispose?.());
    }
  });
}

function buildSkeletonGroup(posed, J, parents, frame, colorHex) {
  const group = new THREE.Group();
  const base = frame * J * 3;

  const lineVerts = [];
  for (let j = 0; j < J; j++) {
    const p = parents[j];
    if (p == null || p < 0) continue;
    lineVerts.push(
      posed[base + p * 3], posed[base + p * 3 + 1], posed[base + p * 3 + 2],
      posed[base + j * 3], posed[base + j * 3 + 1], posed[base + j * 3 + 2],
    );
  }
  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
  const lineMat = new THREE.LineBasicMaterial({ color: colorHex });
  group.add(new THREE.LineSegments(lineGeom, lineMat));

  const maxDim = (() => {
    const bb = bboxFromPosedJoints(posed, J, frame);
    const s = bb.getSize(new THREE.Vector3());
    return Math.max(s.x, s.y, s.z, 0.1);
  })();
  const sphereRadius = Math.max(maxDim * 0.012, 0.006);
  const sphereGeom = new THREE.SphereGeometry(sphereRadius, 8, 6);
  const sphereMat = new THREE.MeshBasicMaterial({ color: colorHex });
  for (let j = 0; j < J; j++) {
    const m = new THREE.Mesh(sphereGeom, sphereMat);
    m.position.set(
      posed[base + j * 3],
      posed[base + j * 3 + 1],
      posed[base + j * 3 + 2],
    );
    group.add(m);
  }
  return group;
}

function redrawOverlay() {
  if (overlayGroupStart) { overlayCtx.scene.remove(overlayGroupStart); disposeGroup(overlayGroupStart); overlayGroupStart = null; }
  if (overlayGroupEnd)   { overlayCtx.scene.remove(overlayGroupEnd);   disposeGroup(overlayGroupEnd);   overlayGroupEnd = null; }
  if (!state.editorData) return;
  const { posed, num_joints: J, joint_parents } = state.editorData;
  overlayGroupStart = buildSkeletonGroup(posed, J, joint_parents, state.trimStart, 0x4da6ff);
  overlayGroupEnd   = buildSkeletonGroup(posed, J, joint_parents, state.trimEnd,   0xffcc4d);
  overlayCtx.scene.add(overlayGroupStart);
  overlayCtx.scene.add(overlayGroupEnd);

  // Fit camera ONLY on the first redraw per clip load. Subsequent trim
  // drags rebuild the skeletons but leave the user's orbit/zoom alone —
  // otherwise every slider nudge snaps the camera back to its default
  // framing, making it impossible to orbit around to line up legs/feet.
  if (!state.overlayCameraFitted) {
    const bb = bboxFromPosedJoints(posed, J, state.trimStart);
    bb.union(bboxFromPosedJoints(posed, J, state.trimEnd));
    fitCameraFromBbox(overlayCtx, bb);
    state.overlayCameraFitted = true;
  }
}

// ---------------------------------------------------------------------------
// Pose-distance + stats
// ---------------------------------------------------------------------------
function poseDistanceFromFrame(ed, refFrame) {
  const { total_frames: T, num_joints: J, posed: P } = ed;
  const dist = new Float32Array(T);
  const rB = refFrame * J * 3;
  const rRx = P[rB], rRy = P[rB + 1], rRz = P[rB + 2];
  for (let f = 0; f < T; f++) {
    const fB = f * J * 3;
    const fRx = P[fB], fRy = P[fB + 1], fRz = P[fB + 2];
    let sum = 0;
    for (let j = 0; j < J; j++) {
      const rj = rB + j * 3;
      const fj = fB + j * 3;
      const dx = (P[fj]     - fRx) - (P[rj]     - rRx);
      const dy = (P[fj + 1] - fRy) - (P[rj + 1] - rRy);
      const dz = (P[fj + 2] - fRz) - (P[rj + 2] - rRz);
      sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    dist[f] = sum;
  }
  return dist;
}

function paintTimelineGradient(distArray) {
  let dMin = Infinity, dMax = -Infinity;
  for (const v of distArray) { if (v < dMin) dMin = v; if (v > dMax) dMax = v; }
  const span = dMax - dMin || 1;
  const stops = [];
  for (let f = 0; f < distArray.length; f++) {
    const norm = (distArray[f] - dMin) / span;
    const light = 14 + Math.round(norm * 54);
    const pct = (100 * f) / (distArray.length - 1 || 1);
    stops.push(`hsl(215, 55%, ${light}%) ${pct.toFixed(2)}%`);
  }
  $("#timeline-gradient").style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
}

function updateLabels() {
  if (!state.editorData) return;
  const { total_frames: T, fps } = state.editorData;
  $("#label-start").textContent = String(state.trimStart);
  $("#label-end").textContent = String(state.trimEnd);
  const kept = state.trimEnd - state.trimStart + 1;
  const dur = kept / fps;
  $("#label-kept").textContent = `keeping ${kept} / ${T} frames (${dur.toFixed(2)} s)`;

  const startDist = poseDistanceFromFrame(state.editorData, state.trimStart);
  const seam = startDist[state.trimEnd];
  let bestSeam = Infinity;
  const minGap = Math.max(2, Math.round(AUTO_MIN_PERIOD_S * fps));
  for (let f = state.trimStart + minGap; f < T; f++) {
    if (startDist[f] < bestSeam) bestSeam = startDist[f];
  }
  const bestStr = Number.isFinite(bestSeam) ? bestSeam.toFixed(3) : "—";
  $("#stats-similarity").textContent =
    `seam distance: ${seam.toFixed(3)} m   (best reachable: ${bestStr} m)`;
  $("#stats-meta").textContent = `${fps.toFixed(1)} fps · ${T} total frames`;
  $("#overlay-stats").textContent = `start↔end pose distance: ${seam.toFixed(3)} m`;
}

function updateMarkerPositions() {
  if (!state.editorData) return;
  const T = state.editorData.total_frames;
  const pct = (f) => (f / (T - 1)) * 100;
  const s = pct(state.trimStart);
  const e = pct(state.trimEnd);
  $("#marker-start").style.left = `${s}%`;
  $("#marker-end").style.left = `${e}%`;
  const hl = $("#timeline-highlight");
  hl.style.left = `${s}%`;
  hl.style.width = `${Math.max(0, e - s)}%`;

  // Blend marker sits at the LAST NON-BLENDED frame (= trim_end - blend).
  // When blend = 0 it sits on top of the end marker. Dragging it LEFT
  // extends the blend span one frame at a time. The blend fill to the
  // RIGHT of the marker shows the frames that will be eased toward start.
  const blendN = computeBlendFrames();
  const markerFrame = Math.max(state.trimStart, state.trimEnd - blendN);
  const blendPct = pct(markerFrame);
  $("#marker-blend").style.left = `${blendPct}%`;

  updateSeamIndicators();
}

function keptFrames() {
  if (!state.editorData) return 0;
  return Math.max(0, state.trimEnd - state.trimStart + 1);
}

function computeBlendFrames() {
  // Clamp to [0, kept - 1]: can't blend more than exists, and we always
  // keep one unblended anchor at the start of the blend span.
  const n = parseInt($("#blend-frames").value, 10) || 0;
  const maxN = Math.max(0, keptFrames() - 1);
  return Math.max(0, Math.min(maxN, n));
}

function computeBridgeFrames() {
  const n = parseInt($("#bridge-frames").value, 10) || 0;
  return Math.max(0, Math.min(600, n));
}

function updateSeamIndicators() {
  // Two independent indicators:
  //   blend (green)  — inside the kept highlight, covers the LAST N frames
  //                    of the kept slice. Eased toward start pose on Save.
  //   bridge (orange) — AFTER the end marker. N new frames appended on Save.
  if (!state.editorData) return;
  const T = state.editorData.total_frames;
  const fps = state.editorData.fps || 30;

  // --- Blend indicator ---
  const blendN = computeBlendFrames();
  const blendEl = $("#timeline-blend");
  const blendReadout = $("#blend-readout");
  if (blendReadout) {
    if (blendN > 0) {
      const dur = (blendN / fps).toFixed(2);
      const kept = keptFrames();
      const pct = kept > 0 ? Math.round((blendN / kept) * 100) : 0;
      blendReadout.textContent = `last ${dur} s  ·  ${pct}% of kept`;
    } else {
      blendReadout.textContent = "off";
    }
  }
  if (blendN <= 0 || T <= 1) {
    blendEl.style.display = "none";
  } else {
    const endFrame = state.trimEnd;
    const blendStartFrame = Math.max(state.trimStart, endFrame - blendN + 1);
    const startPct = (blendStartFrame / (T - 1)) * 100;
    const endPct = (endFrame / (T - 1)) * 100;
    blendEl.style.display = "";
    blendEl.style.left = `${startPct}%`;
    blendEl.style.width = `${Math.max(0, endPct - startPct)}%`;
    blendEl.dataset.label = `blend ${blendN}f`;
  }

  // --- Bridge indicator ---
  const bridgeN = computeBridgeFrames();
  const bridgeEl = $("#timeline-bridge");
  const bridgeReadout = $("#bridge-readout");
  if (bridgeReadout) {
    if (bridgeN > 0) {
      const dur = (bridgeN / fps).toFixed(2);
      const kept = keptFrames();
      const pct = kept > 0 ? Math.round((bridgeN / kept) * 100) : 0;
      bridgeReadout.textContent = `+${dur} s  ·  ${pct}% of kept`;
    } else {
      bridgeReadout.textContent = "off";
    }
  }
  if (bridgeN <= 0 || T <= 1) {
    bridgeEl.style.display = "none";
  } else {
    const endPct = (state.trimEnd / (T - 1)) * 100;
    const widthPct = (bridgeN / (T - 1)) * 100;
    bridgeEl.style.display = "";
    bridgeEl.style.left = `${endPct}%`;
    bridgeEl.style.width = `${Math.min(widthPct, Math.max(0, 100 - endPct))}%`;
    bridgeEl.dataset.label = `+${bridgeN}f`;
  }
}

function updateSaveButton() {
  const btn = $("#btn-save");
  if (!state.editorData) {
    btn.disabled = true; btn.textContent = "Save"; return;
  }
  const currentBlend = computeBlendFrames();
  const currentBridge = computeBridgeFrames();
  const {
    trim_start_saved, trim_end_saved,
    blend_frames_saved, bridge_frames_saved,
  } = state.editorData;
  const matches =
    trim_start_saved === state.trimStart &&
    trim_end_saved === state.trimEnd &&
    blend_frames_saved === currentBlend &&
    bridge_frames_saved === currentBridge;
  if (matches && state.job?.saved) {
    btn.textContent = "Saved ✓"; btn.disabled = true;
  } else {
    const tags = [];
    if (currentBlend > 0) tags.push(`blend ${currentBlend}`);
    if (currentBridge > 0) tags.push(`+${currentBridge}`);
    btn.textContent = tags.length ? `Save (${tags.join(", ")})` : "Save";
    btn.disabled = false;
  }
}

function refreshEditorUI() {
  if (!state.editorData) return;
  paintTimelineGradient(poseDistanceFromFrame(state.editorData, state.trimStart));
  updateMarkerPositions();
  updateLabels();
  redrawOverlay();
  updateSaveButton();
}

// ---------------------------------------------------------------------------
// Timeline marker dragging
// ---------------------------------------------------------------------------
let draggingMarker = null;

function frameFromClientX(clientX) {
  if (!state.editorData) return 0;
  const rect = $("#timeline").getBoundingClientRect();
  const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
  const norm = x / rect.width;
  const f = Math.round(norm * (state.editorData.total_frames - 1));
  return Math.max(0, Math.min(state.editorData.total_frames - 1, f));
}

function onMarkerPointerDown(which, e) {
  draggingMarker = which;
  e.preventDefault();
  e.stopPropagation();
}

$("#marker-start").addEventListener("pointerdown", (e) => onMarkerPointerDown("start", e));
$("#marker-end").addEventListener("pointerdown", (e) => onMarkerPointerDown("end", e));
$("#marker-blend").addEventListener("pointerdown", (e) => onMarkerPointerDown("blend", e));

$("#timeline").addEventListener("pointerdown", (e) => {
  if (!state.editorData || draggingMarker) return;
  const f = frameFromClientX(e.clientX);
  const dStart = Math.abs(f - state.trimStart);
  const dEnd = Math.abs(f - state.trimEnd);
  draggingMarker = dStart <= dEnd ? "start" : "end";
  handleDrag(e);
});

function handleDrag(e) {
  if (!draggingMarker || !state.editorData) return;
  const f = frameFromClientX(e.clientX);
  if (draggingMarker === "start") {
    state.trimStart = Math.min(f, state.trimEnd - 1);
  } else if (draggingMarker === "end") {
    state.trimEnd = Math.max(f, state.trimStart + 1);
  } else if (draggingMarker === "blend") {
    // Marker sits at the LAST non-blended frame. blend_frames = trim_end - f.
    //   f = trim_end  → blend = 0 (no blend, marker on end)
    //   f = trim_end - N → blend = N (N frames to the right of marker)
    const clampedF = Math.max(state.trimStart, Math.min(state.trimEnd, f));
    const blendN = Math.max(0, Math.min(keptFrames() - 1, state.trimEnd - clampedF));
    $("#blend-frames").value = String(blendN);
    updateSaveButton();
  }
  refreshEditorUI();
}

window.addEventListener("pointermove", handleDrag);
window.addEventListener("pointerup", () => { draggingMarker = null; });
window.addEventListener("pointercancel", () => { draggingMarker = null; });

// ---------------------------------------------------------------------------
// Auto-trim — exhaustive (i, j) pair search
// ---------------------------------------------------------------------------
function findBestLoopPair(editorData) {
  const { total_frames: T, num_joints: J, posed: P, fps } = editorData;
  const minPeriodFrames = Math.max(2, Math.round(AUTO_MIN_PERIOD_S * fps));
  if (T - minPeriodFrames < 2) {
    return { start: 0, end: T - 1, topList: [] };
  }

  // Pre-compute root-relative joint positions once.
  const R = new Float32Array(T * J * 3);
  for (let f = 0; f < T; f++) {
    const base = f * J * 3;
    const rx = P[base], ry = P[base + 1], rz = P[base + 2];
    for (let j = 0; j < J; j++) {
      const o = base + j * 3;
      R[o]     = P[o]     - rx;
      R[o + 1] = P[o + 1] - ry;
      R[o + 2] = P[o + 2] - rz;
    }
  }

  const topList = [];
  const pushCandidate = (i, j, d) => {
    if (topList.length < AUTO_TOP_K) {
      topList.push({ i, j, d });
      topList.sort((a, b) => a.d - b.d);
    } else if (d < topList[AUTO_TOP_K - 1].d) {
      topList[AUTO_TOP_K - 1] = { i, j, d };
      topList.sort((a, b) => a.d - b.d);
    }
  };

  for (let i = 0; i < T - minPeriodFrames; i++) {
    const iBase = i * J * 3;
    for (let j = i + minPeriodFrames; j < T; j++) {
      const jBase = j * J * 3;
      let sum = 0;
      for (let k = 0; k < J; k++) {
        const o = k * 3;
        const dx = R[iBase + o]     - R[jBase + o];
        const dy = R[iBase + o + 1] - R[jBase + o + 1];
        const dz = R[iBase + o + 2] - R[jBase + o + 2];
        sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      pushCandidate(i, j, sum);
    }
  }

  let best = topList[0];
  for (const c of topList) {
    if (c.j - c.i > best.j - best.i) best = c;
  }
  // j-1 offset so playback wraps end→start as a natural next-frame.
  const endTrimmed = Math.max(best.i + 1, best.j - 1);
  return { start: best.i, end: endTrimmed, topList };
}

function autoTrim() {
  if (!state.editorData) return;
  const btn = $("#btn-reset");
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "searching…";
  requestAnimationFrame(() => {
    const t0 = performance.now();
    const { start, end, topList } = findBestLoopPair(state.editorData);
    const dt = performance.now() - t0;
    state.trimStart = start;
    state.trimEnd = end;
    refreshEditorUI();
    btn.disabled = false;
    btn.textContent = prevText;
    console.log(
      `[auto-trim] chose [${start}, ${end}] (span ${end - start} frames, ` +
      `${((end - start) / state.editorData.fps).toFixed(2)} s) ` +
      `from ${topList.length} top-K candidates in ${dt.toFixed(0)} ms`,
    );
  });
}

$("#btn-reset").addEventListener("click", autoTrim);

// Both input + change — `input` fires on typing/arrows, `change` fires on
// blur / enter. Belt and suspenders against a single missed event breaking
// the indicator.
for (const ev of ["input", "change"]) {
  $("#blend-frames").addEventListener(ev, () => {
    updateSeamIndicators();
    updateSaveButton();
  });
  $("#bridge-frames").addEventListener(ev, () => {
    updateSeamIndicators();
    updateSaveButton();
  });
}

$("#chk-ground").addEventListener("change", (e) => {
  mainCtx.ground.visible = e.target.checked;
  mainCtx.grid.visible = e.target.checked;
});
$("#chk-mesh").addEventListener("change", (e) => {
  if (!state.clipRoot) return;
  state.clipRoot.traverse((n) => { if (n.isMesh || n.isSkinnedMesh) n.visible = e.target.checked; });
});

// ---------------------------------------------------------------------------
// Clip loader
// ---------------------------------------------------------------------------
function clearMainClip() {
  if (state.clipRoot) {
    mainCtx.scene.remove(state.clipRoot);
    disposeGroup(state.clipRoot);
    state.clipRoot = null;
  }
  state.mixer = null;
  state.action = null;
}

function modelShort(model) {
  return model ? model.replace("Kimodo-SOMA-", "").replace("-v1.1", "") : "—";
}

function updateHeader(job) {
  $("#header-prompt").textContent = job.prompt || "—";
  $("#header-model").textContent = modelShort(job.model);
  $("#header-seed").textContent = job.seed ?? "—";
  $("#header-duration").textContent = job.duration_s != null ? `${job.duration_s.toFixed(1)}s` : "—";
  $("#header-frames").textContent = job.total_frames ?? "—";
  document.title = `Editor · ${(job.prompt || "").slice(0, 40) || job.id}`;

  const dl = $("#download-link");
  if (job.status === "completed") {
    dl.href = `/api/animations/${job.id}/clip`;
    dl.style.display = "";
  } else {
    dl.removeAttribute("href");
    dl.style.display = "none";
  }
}

async function loadClip() {
  const myId = ++state.loadSequenceId;
  $("#viewer-status").textContent = "loading…";

  clearMainClip();
  // Camera-fit flags are NOT reset here — both the main-viewer and overlay
  // cameras fit once on the very first loadClip of the page and then stay
  // wherever the user has orbited/zoomed them. Save + Regenerate trigger
  // loadClip but preserve the view; only a full page reload re-fits.
  if (overlayGroupStart) { overlayCtx.scene.remove(overlayGroupStart); disposeGroup(overlayGroupStart); overlayGroupStart = null; }
  if (overlayGroupEnd)   { overlayCtx.scene.remove(overlayGroupEnd);   disposeGroup(overlayGroupEnd);   overlayGroupEnd = null; }

  // Cache-bust the FBX URL so the browser actually refetches after Save or
  // Regenerate. Without this, the browser serves the pre-save FBX from its
  // cache and the viewer keeps playing the untrimmed / un-bridged clip.
  // fbx_size changes on every re-export, so it doubles as a version tag.
  const fbxVer = state.job?.fbx_size ?? Date.now();

  let editorData, fbxObj;
  try {
    const editorP = fetch(`/api/animations/${JOB_ID}/editor-data`).then((r) => {
      if (!r.ok) throw new Error(`editor-data HTTP ${r.status}`);
      return r.json();
    });
    const fbxP = new Promise((resolve, reject) => {
      new FBXLoader().load(`/api/animations/${JOB_ID}/clip?v=${fbxVer}`, resolve, undefined, reject);
    });
    [editorData, fbxObj] = await Promise.all([editorP, fbxP]);
  } catch (err) {
    if (myId !== state.loadSequenceId) return;
    console.error(err);
    $("#viewer-status").textContent = `failed to load: ${err.message ?? err}`;
    return;
  }
  if (myId !== state.loadSequenceId) return;

  state.editorData = {
    total_frames: editorData.total_frames,
    num_joints: editorData.num_joints,
    fps: editorData.fps,
    posed: Float32Array.from(editorData.posed_joints),
    joint_parents: editorData.joint_parents,
    trim_start_saved: editorData.saved ? editorData.trim_start_frame : null,
    trim_end_saved: editorData.saved ? editorData.trim_end_frame : null,
    blend_frames_saved: editorData.saved ? (editorData.blend_frames || 0) : null,
    bridge_frames_saved: editorData.saved ? (editorData.bridge_frames || 0) : null,
    prompt: editorData.prompt || "",
    seed: editorData.seed,
    duration_s: editorData.duration_s || (editorData.total_frames / editorData.fps),
  };
  state.trimStart = editorData.trim_start_frame;
  state.trimEnd = editorData.trim_end_frame;

  $("#regen-prompt").value = state.editorData.prompt;
  $("#regen-duration").value = state.editorData.duration_s.toFixed(1);
  $("#regen-seed").value = String(state.editorData.seed ?? "");

  // Seam defaults:
  //   * Both blend and bridge default to 0 on a fresh unsaved clip.
  //   * If the clip was previously saved with values, restore them.
  // (Intentionally NOT auto-seeding with a percentage — mixing modes
  // should be an explicit choice, and a fresh clip with no seam work
  // should export exactly what the trim says.)
  $("#blend-frames").value = String(Math.min(300, Math.max(0, editorData.blend_frames || 0)));
  $("#bridge-frames").value = String(Math.min(600, Math.max(0, editorData.bridge_frames || 0)));

  state.clipRoot = fbxObj;
  mainCtx.scene.add(fbxObj);
  if (!state.mainCameraFitted) {
    fitCameraFromBbox(mainCtx, bboxFromBones(fbxObj));
    state.mainCameraFitted = true;
  }

  if (fbxObj.animations && fbxObj.animations.length > 0) {
    state.mixer = new THREE.AnimationMixer(fbxObj);
    const clip = fbxObj.animations[0];
    state.action = state.mixer.clipAction(clip);
    state.action.setLoop(THREE.LoopRepeat, Infinity);
    state.action.play();
  }

  updatePlayheadZones();
  refreshEditorUI();
  $("#viewer-status").textContent =
    `${state.editorData.total_frames} frames (${(state.editorData.total_frames / state.editorData.fps).toFixed(1)} s), ` +
    `${editorData.saved ? "previously saved" : "unsaved"}`;
  $("#current-status").textContent = editorData.saved ? "saved" : "unsaved edits";
}

// ---------------------------------------------------------------------------
// Save / Regenerate
// ---------------------------------------------------------------------------
$("#btn-save").addEventListener("click", async () => {
  if (!state.editorData) return;
  const btn = $("#btn-save");
  btn.disabled = true;
  btn.textContent = "saving…";
  try {
    const fd = new FormData();
    fd.append("trim_start_frame", String(state.trimStart));
    fd.append("trim_end_frame", String(state.trimEnd));
    fd.append("blend_frames", String(computeBlendFrames()));
    fd.append("bridge_frames", String(computeBridgeFrames()));
    fd.append("save", "true");
    const r = await fetch(`/api/animations/${JOB_ID}/trim`, {
      method: "POST", body: fd,
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.job = data;
    updateHeader(data);
    await loadClip();
  } catch (err) {
    alert("Save failed: " + (err.message ?? err));
    btn.disabled = false;
    btn.textContent = "Save";
  }
});

$("#btn-regen-reset").addEventListener("click", () => {
  if (!state.editorData) return;
  $("#regen-prompt").value = state.editorData.prompt;
  $("#regen-duration").value = state.editorData.duration_s.toFixed(1);
  $("#regen-seed").value = String(state.editorData.seed ?? "");
});

$("#btn-regen").addEventListener("click", async () => {
  const prompt = $("#regen-prompt").value.trim();
  if (!prompt) {
    alert("Prompt can't be empty.");
    return;
  }
  const durVal = parseFloat($("#regen-duration").value);
  if (!(durVal > 0 && durVal <= 30)) {
    alert("Duration must be between 0.5 and 30 seconds.");
    return;
  }
  const seedVal = parseInt($("#regen-seed").value, 10);
  if (Number.isNaN(seedVal)) {
    alert("Seed must be a number.");
    return;
  }
  const btn = $("#btn-regen");
  btn.disabled = true;
  const prevLabel = btn.textContent;
  btn.textContent = "regenerating…";
  $("#viewer-status").textContent = "regenerating on GPU — this takes a few seconds…";
  try {
    const fd = new FormData();
    fd.append("prompt", prompt);
    fd.append("duration_s", String(durVal));
    fd.append("seed", String(seedVal));
    const r = await fetch(`/api/animations/${JOB_ID}/regenerate`, {
      method: "POST", body: fd,
    });
    if (!r.ok) throw new Error(await r.text());
    const job = await r.json();
    state.job = job;
    updateHeader(job);
    await loadClip();
  } catch (err) {
    alert("Regenerate failed: " + (err.message ?? err));
  } finally {
    btn.disabled = false;
    btn.textContent = prevLabel;
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  try {
    const job = await fetch(`/api/animations/${JOB_ID}`).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    state.job = job;
    updateHeader(job);

    if (job.status !== "completed") {
      $("#viewer-status").textContent =
        job.status === "failed"
          ? `generation failed: ${job.error || "(no detail)"}`
          : `generation in progress (status: ${job.status})…`;
      return;
    }
    await loadClip();
  } catch (err) {
    console.error(err);
    $("#viewer-status").textContent = `init failed: ${err.message ?? err}`;
  }
}
init();
