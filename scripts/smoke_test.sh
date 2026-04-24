#!/usr/bin/env bash
# End-to-end smoke: bring up the stack, generate a single animation from a
# simple prompt against a user-provided Mixamo-rigged FBX template, assert
# an FBX lands on disk.
#
#   TEMPLATE_FBX=/absolute/path/to/MixamoTemplate.fbx ./scripts/smoke_test.sh
#
# Prereqs: docker compose, GPU access, fbxsdkpy installed in the service
# image (handled by docker/animation-service.Dockerfile).

set -euo pipefail
cd "$(dirname "$0")/.."

PROMPT="A person stands in a neutral idle, breathing softly, weight balanced."
DURATION="5.0"

if [ -z "${TEMPLATE_FBX:-}" ]; then
  echo "[smoke] ERROR: set TEMPLATE_FBX=/path/to/MixamoRigged.fbx before running." >&2
  echo "[smoke] Export from Mixamo as 'T-pose with skin' to include textures." >&2
  exit 2
fi
if [ ! -f "$TEMPLATE_FBX" ]; then
  echo "[smoke] ERROR: TEMPLATE_FBX does not exist: $TEMPLATE_FBX" >&2
  exit 2
fi

echo "[smoke] starting stack..."
docker compose up -d --build

cleanup() {
  echo "[smoke] tearing down..."
  docker compose down
}
trap cleanup EXIT

echo "[smoke] waiting for animation-service to respond..."
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:7870/api/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[smoke] requesting one animation..."
RESP=$(curl -fsS -X POST "http://localhost:7870/api/animations" \
  -F "prompt=$PROMPT" \
  -F "duration_s=$DURATION" \
  -F "seed=42" \
  -F "template_fbx=@$TEMPLATE_FBX")
JOB_ID=$(echo "$RESP" | python -c "import json,sys;print(json.load(sys.stdin)['id'])")
echo "[smoke] job id: $JOB_ID"

echo "[smoke] polling for completion (kimodo cold-start takes a minute on first run)..."
for i in $(seq 1 120); do
  STATUS=$(curl -fsS "http://localhost:7870/api/animations/$JOB_ID" | python -c "import json,sys;print(json.load(sys.stdin)['status'])")
  echo "  [$i] status=$STATUS"
  if [ "$STATUS" = "completed" ]; then break; fi
  if [ "$STATUS" = "failed" ]; then
    echo "[smoke] FAILED — job hit error" >&2
    curl -fsS "http://localhost:7870/api/animations/$JOB_ID" >&2
    exit 1
  fi
  sleep 10
done

# Verify via the API (outputs live in a Docker volume, not on the host).
TMP_FBX=$(mktemp --suffix=.fbx)
HTTP_CODE=$(curl -fsS -o "$TMP_FBX" -w "%{http_code}" "http://localhost:7870/api/animations/$JOB_ID/clip" || echo "fail")
if [ "$HTTP_CODE" != "200" ]; then
  echo "[smoke] FAILED — GET /api/animations/$JOB_ID/clip returned $HTTP_CODE" >&2
  rm -f "$TMP_FBX"
  exit 1
fi
SIZE=$(stat -c%s "$TMP_FBX" 2>/dev/null || stat -f%z "$TMP_FBX")
if [ "$SIZE" -lt 1024 ]; then
  echo "[smoke] FAILED — downloaded FBX is only $SIZE bytes, looks wrong" >&2
  rm -f "$TMP_FBX"
  exit 1
fi
rm -f "$TMP_FBX"
echo "[smoke] OK — GET /api/animations/$JOB_ID/clip returned $SIZE bytes of FBX"
echo "[smoke] open http://localhost:7870/ui/view.html?job=$JOB_ID to eyeball the result"
