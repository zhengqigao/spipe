#!/usr/bin/env bash
# Locate (or check) the SKY130 device models the paper's DAC examples need.
#
#   examples/paper/ptc_hspice/dac_model/8bit_DAC/switch.sub says
#       .lib "../sky130_fd_pr/models/sky130.lib.spice" tt
#   and is simulated with HSPICE.
#
# WHY THIS IS NOT A PLAIN `git clone`
# -----------------------------------
# The obvious source, github.com/google/skywater-pdk-libs-sky130_fd_pr, publishes the
# models in *ngspice* syntax: device lines read `l = {l} w = {w}`. HSPICE rejects that
# with "syntax error at or before {l}". Every tag from v0.10.0 to main is like this --
# checked, not assumed. HSPICE needs `l = 'l' w = 'w'`.
#
# The HSPICE-syntax library is produced by open_pdks, which post-processes sky130_fd_pr
# into a simulator-specific PDK tree (sky130A). That is what the decks were written
# against. Cloning the raw repo gives a tree that installs cleanly and then fails inside
# HSPICE, which is worse than not installing it, so this script does not do that.
#
# USAGE
#   ./scripts/fetch_sky130.sh                  # check what is already in place
#   ./scripts/fetch_sky130.sh /path/to/sky130A # install from an open_pdks build
#
# The second form accepts either the sky130A root or the sky130_fd_pr directory inside it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/examples/paper/ptc_hspice/dac_model/sky130_fd_pr"
SOURCE="${1:-}"

# The one file every deck enters through, and the syntax HSPICE requires in it.
LIB_REL="models/sky130.lib.spice"

hspice_flavoured() {
    # $1 = a sky130_fd_pr root. True if the nfet subckt uses HSPICE 'quoted' parameters.
    local probe="$1/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.pm3.spice"
    [ -f "$probe" ] && grep -q "l = 'l'" "$probe"
}

describe() {
    local root="$1"
    if [ ! -f "$root/$LIB_REL" ]; then
        echo "  no $LIB_REL under $root"
    elif hspice_flavoured "$root"; then
        echo "  OK: HSPICE-syntax models ($(du -sh "$root" | cut -f1))"
    else
        echo "  PRESENT BUT UNUSABLE: ngspice-syntax models -- HSPICE will fail on {l}"
    fi
}

if [ -z "$SOURCE" ]; then
    echo "Checking $DEST"
    if [ -f "$DEST/$LIB_REL" ] && hspice_flavoured "$DEST"; then
        describe "$DEST"
        echo "Nothing to do."
        exit 0
    fi
    describe "$DEST"
    cat >&2 <<'MSG'

The paper's DAC examples need SKY130 models in HSPICE syntax. They are not vendored
here (third-party, Apache-2.0, "The SkyWater PDK Authors") and cannot be fetched from
the upstream git repository, which ships ngspice syntax HSPICE cannot parse.

To get them, build open_pdks once:

    git clone https://github.com/RTimothyEdwards/open_pdks
    cd open_pdks && ./configure --enable-sky130-pdk && make && sudo make install

then point this script at the result:

    ./scripts/fetch_sky130.sh $PDK_ROOT/sky130A

Everything else in SPIPE runs without this: the built-in engine needs no PDK, and
examples/derived/ reproduces the same circuits on Level-1 devices (see
examples/README.md for how closely they agree, and where they do not).
MSG
    exit 1
fi

# --- install from a user-supplied tree ------------------------------------------------
[ -d "$SOURCE" ] || { echo "error: $SOURCE is not a directory" >&2; exit 1; }

# Accept either .../sky130A or .../sky130A/libs.ref/sky130_fd_pr or the fd_pr dir itself.
SRC_ROOT=""
for cand in "$SOURCE" "$SOURCE/libs.ref/sky130_fd_pr" "$SOURCE/sky130_fd_pr"; do
    [ -f "$cand/$LIB_REL" ] && { SRC_ROOT="$cand"; break; }
done
[ -n "$SRC_ROOT" ] || {
    echo "error: no $LIB_REL under $SOURCE (tried it, libs.ref/sky130_fd_pr, sky130_fd_pr)" >&2
    exit 1; }

hspice_flavoured "$SRC_ROOT" || {
    echo "error: $SRC_ROOT has ngspice-syntax models ({l} rather than 'l')." >&2
    echo "       HSPICE cannot read these. Use an open_pdks-generated sky130A." >&2
    exit 1; }

echo "Installing from $SRC_ROOT"
mkdir -p "$DEST"
cp -r "$SRC_ROOT/models" "$SRC_ROOT/cells" "$DEST/"
for f in LICENSE NOTICE README.rst README.md; do
    [ -f "$SRC_ROOT/$f" ] && cp "$SRC_ROOT/$f" "$DEST/" || true
done

# Verify rather than assume: every .include the tt corner names must resolve.
missing=0
while read -r inc; do
    [ -f "$DEST/models/$inc" ] || { echo "  missing: models/$inc" >&2; missing=$((missing + 1)); }
done < <(awk '/^\.lib tt$/,/^\.endl/' "$DEST/$LIB_REL" | grep -oE '"[^"]+"' | tr -d '"')
if [ "$missing" -gt 0 ]; then
    echo "error: $missing include(s) named by the tt corner are absent; HSPICE would fail." >&2
    exit 1
fi

echo "SKY130 models installed at: $DEST"
describe "$DEST"
echo "Licence: Apache-2.0, (c) The SkyWater PDK Authors"
