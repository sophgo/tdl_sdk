#!/bin/bash
set -euo pipefail

# Package the CV190X TDL install tree for board deployment.
#
# Two modes, chosen by the target filesystem:
#   (default)      preserve SONAME symbolic links (ext4 / NFS)
#   portable|fat   dereference symlinks to independent file copies (FAT/vfat,
#                  which cannot hold symlinks). No symlinks, no hardlinks.
#
# In both modes the loader finds every SONAME name (libopencv_*.so.4.5,
# libz.so.1, ...) as either a symlink or a real ELF copy, each carrying the
# correct SONAME inside its ELF dynamic section.

usage() {
  echo "Usage: $0 [portable] [install-root] [output-archive]"
  echo "  portable  FAT/vfat-compatible: copy each SONAME name as a real file"
}

MODE=preserve
if [[ "${1:-}" == "portable" || "${1:-}" == "fat" ]]; then
  MODE=portable
  shift
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SDK_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ROOT=${1:-$SDK_ROOT/install/CV190X}
ARCHIVE=${2:-$SDK_ROOT/install/CV190X.tar.gz}
# Make both absolute now: the tar subshell cd's into the install parent,
# so a relative ARCHIVE would resolve against the wrong directory.
ROOT=$(cd "$ROOT" && pwd)
ARCHIVE=$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")
VERIFY="$SCRIPT_DIR/verify_cv190x_package.sh"

fail() {
  echo "package_cv190x: $*" >&2
  exit 1
}

[[ -d "$ROOT" ]] || fail "install root does not exist: $ROOT"
[[ -x "$VERIFY" ]] || fail "verification script is not executable: $VERIFY"

"$VERIFY" "$ROOT"

rm -f "$ARCHIVE"
mkdir -p "$(dirname "$ARCHIVE")"
root_parent=$(dirname "$ROOT")
root_name=$(basename "$ROOT")

(
  cd "$root_parent"
  if [[ "$MODE" == "portable" ]]; then
    # --dereference replaces each symlink with the content of its target;
    # --hard-dereference also turns hardlinks into independent regular files,
    # so every SONAME name becomes a standalone ELF copy. FAT-safe.
    tar --format=gnu --owner=0 --group=0 --numeric-owner \
      --dereference --hard-dereference \
      --gzip --create --file="$ARCHIVE" "$root_name"
  else
    # Preserve symlinks. Do NOT use -h/--dereference: the loader needs the
    # SONAME names represented by the symlink chain.
    tar --format=gnu --owner=0 --group=0 --numeric-owner \
      --gzip --create --file="$ARCHIVE" "$root_name"
  fi
)

"$VERIFY" "$ROOT" "$ARCHIVE"
echo "package_cv190x: created $ARCHIVE (mode=$MODE)"
