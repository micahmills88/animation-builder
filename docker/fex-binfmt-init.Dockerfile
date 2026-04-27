# FEX-Emu binfmt-init: drop-in faster replacement for qemu-user (tonistiigi/binfmt).
#
# FEX is a binary recompiler that translates x86 instructions to ARM64 with a
# custom IR — substantially faster than qemu's TCG splatter-JIT (2-3x on
# CPU-bound code, more on SIMD-heavy code). For boring fbxsdkpy serialization
# we expect ~2x throughput vs qemu. No Apple Rosetta involved.
#
# Pattern: this image installs FEX from the official Ubuntu PPA, then writes
# a binfmt_misc registration that points at /usr/bin/FEXInterpreter with the
# F flag (kernel holds an open fd, so the registration survives this
# container's exit).
#
# Run (from compose, see binfmt-init service):
#   docker run --privileged --rm fex-binfmt-init
#
# After this exits, any container started with --platform linux/amd64 on this
# host will run via FEX. To revert to qemu, run tonistiigi/binfmt --install all
# again (overwrites the registration).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common ca-certificates curl && \
    add-apt-repository -y ppa:fex-emu/fex && \
    apt-get update && \
    apt-get install -y --no-install-recommends fex-emu-armv8.4 && \
    rm -rf /var/lib/apt/lists/*

# Sanity check at build time that the interpreter is where we expect.
RUN test -x /usr/bin/FEXInterpreter || (echo "FEXInterpreter not found" && exit 1)

COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e

# binfmt_misc usually isn't mounted inside a container even with --privileged;
# the host's mount doesn't propagate. Mount it ourselves before writing.
if [ ! -e /proc/sys/fs/binfmt_misc/register ]; then
    mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc 2>&1 || {
        echo "[fex-binfmt-init] FATAL: cannot mount binfmt_misc — host kernel support missing or container not --privileged" >&2
        exit 1
    }
fi

# Drop any existing x86_64 binfmt registration so FEX takes precedence.
# Both tonistiigi/binfmt and qemu-user-static register under qemu-x86_64.
for old in /proc/sys/fs/binfmt_misc/qemu-x86_64 /proc/sys/fs/binfmt_misc/FEX-x86_64; do
    if [ -e "$old" ]; then
        echo -1 > "$old" 2>/dev/null || true
    fi
done

# Register FEX for x86_64 ELF.
# MAGIC + MASK identify a 64-bit Intel ELF binary (same matchers tonistiigi
# uses for qemu-x86_64). F flag keeps the interpreter fd alive in the kernel
# so it stays valid after this container exits. OC flags preserve credentials
# and open the binary for the interpreter.
echo ':FEX-x86_64:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x3e\x00:\xff\xff\xff\xff\xff\xfe\xfe\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/bin/FEXInterpreter:OCF' \
    > /proc/sys/fs/binfmt_misc/register

echo "[fex-binfmt-init] FEX registered as x86_64 interpreter"
cat /proc/sys/fs/binfmt_misc/FEX-x86_64 | head -3
EOF
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
