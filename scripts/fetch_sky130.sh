#!/usr/bin/env bash
# Fetch the SkyWater SKY130 device models needed by the paper's DAC examples.
#
# These models are third-party (Apache-2.0, "The SkyWater PDK Authors") and are
# ~15 MB, so they are not vendored here. This script sparse-checks out only the
# two cell libraries and the model index that examples/paper/ptc_hspice needs:
#
#   sky130_fd_pr/cells/nfet_01v8/
#   sky130_fd_pr/cells/pfet_01v8/
#   sky130_fd_pr/models/
#
# Usage:   ./scripts/fetch_sky130.sh [destination]
# Default destination: examples/paper/ptc_hspice/dac_model/sky130_fd_pr
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/examples/paper/ptc_hspice/dac_model/sky130_fd_pr}"
UPSTREAM="https://github.com/google/skywater-pdk-libs-sky130_fd_pr.git"

if [ -d "$DEST/cells/nfet_01v8" ]; then
    echo "SKY130 models already present at: $DEST"
    exit 0
fi

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning SKY130 primitive device models (sparse, shallow)..."
git clone --quiet --depth 1 --filter=blob:none --sparse "$UPSTREAM" "$TMP/sky130_fd_pr"
git -C "$TMP/sky130_fd_pr" sparse-checkout set cells/nfet_01v8 cells/pfet_01v8 models

mkdir -p "$DEST"
cp -r "$TMP/sky130_fd_pr/cells"  "$DEST/"
cp -r "$TMP/sky130_fd_pr/models" "$DEST/"
for f in LICENSE NOTICE README.md; do
    [ -f "$TMP/sky130_fd_pr/$f" ] && cp "$TMP/sky130_fd_pr/$f" "$DEST/" || true
done

echo "SKY130 models installed at: $DEST"
echo "Licence: Apache-2.0, (c) The SkyWater PDK Authors -- see $DEST/LICENSE"
