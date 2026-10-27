#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <install-root> [archive]"
}

ROOT=${1:-}
ARCHIVE=${2:-}
if [[ -z "$ROOT" ]]; then
  usage >&2
  exit 2
fi

ROOT=$(cd "$ROOT" && pwd)

fail() {
  echo "verify_cv190x_package: $*" >&2
  exit 1
}

have_readelf=1
command -v readelf >/dev/null 2>&1 || have_readelf=0

# A library "name" is acceptable in either of two deployment shapes:
#   (a) a symbolic link that resolves to a real ELF file, or
#   (b) a regular file that is itself the ELF (FAT/vfat deployments cannot
#       hold symlinks, so each SONAME name must be an independent file copy).
# In both cases readelf follows symlinks, so the same ELF/SONAME check applies.
check_name() {
  local path=$1
  local soname=$2
  [[ -e "$path" ]] || fail "missing library name: $path"
  if [[ "$have_readelf" -ne 0 ]]; then
    LC_ALL=C readelf -h "$path" 2>/dev/null | grep -q 'Class:[[:space:]]*ELF64' || \
      fail "not an ELF64 file: $path"
    LC_ALL=C readelf -h "$path" 2>/dev/null | grep -q 'Machine:[[:space:]]*RISC-V' || \
      fail "not a RISC-V ELF: $path"
    LC_ALL=C readelf -d "$path" 2>/dev/null | grep -q "SONAME.*\[$soname\]" || \
      fail "SONAME $soname not found in $path"
  else
    # Board BusyBox may lack readelf; fall back to non-empty + ELF magic.
    [[ -s "$path" ]] || fail "empty library: $path"
    LC_ALL=C od -An -tx1 -N4 "$path" | grep -qi '7f 45 4c 46' || \
      fail "not an ELF file: $path"
  fi
}

check_library_set() {
  local dir=$1
  shift
  local name soname
  while (($#)); do
    name=$1
    soname=$2
    shift 2
    check_name "$dir/$name" "$soname"
  done
}

check_install_tree() {
  local root=$1
  local opencv="$root/sample/3rd/opencv/lib"
  local zlib="$root/sample/3rd/zlib/lib"
  [[ -d "$opencv" ]] || fail "missing OpenCV library directory: $opencv"
  [[ -d "$zlib" ]] || fail "missing zlib library directory: $zlib"

  # Each SONAME name must resolve to an ELF carrying that SONAME. Accept either
  # symlinks (ext4/NFS) or independent file copies (FAT/vfat).
  check_library_set "$opencv" \
    libopencv_core.so        libopencv_core.so.4.5 \
    libopencv_core.so.4.5    libopencv_core.so.4.5 \
    libopencv_core.so.4.5.0 libopencv_core.so.4.5 \
    libopencv_imgproc.so        libopencv_imgproc.so.4.5 \
    libopencv_imgproc.so.4.5    libopencv_imgproc.so.4.5 \
    libopencv_imgproc.so.4.5.0 libopencv_imgproc.so.4.5 \
    libopencv_imgcodecs.so        libopencv_imgcodecs.so.4.5 \
    libopencv_imgcodecs.so.4.5    libopencv_imgcodecs.so.4.5 \
    libopencv_imgcodecs.so.4.5.0 libopencv_imgcodecs.so.4.5

  check_library_set "$zlib" \
    libz.so        libz.so.1 \
    libz.so.1      libz.so.1 \
    libz.so.1.2.11 libz.so.1

  # The CV1900B package must not accidentally contain the previous CV1901C build.
  if grep -RIl 'soc_cv1901c_wevb_0018a' "$root/bin" "$root/lib" \
    "$root/sample" 2>/dev/null | grep -q .; then
    fail "package contains stale CV1901C build metadata"
  fi
}

check_archive() {
  local archive=$1
  local listing
  [[ -f "$archive" ]] || fail "missing archive: $archive"
  listing=$(LC_ALL=C tar -tvzf "$archive") || fail "cannot read archive: $archive"

  # Hard-link entries break both FAT and symlink deployments: the loader may
  # resolve the link name to zero bytes or the wrong file. Reject them.
  if grep -q '^h' <<<"$listing"; then
    fail "archive contains hard-link entries"
  fi

  local path
  local root_name
  root_name=$(basename "$ROOT")
  for path in \
    sample/3rd/opencv/lib/libopencv_core.so \
    sample/3rd/opencv/lib/libopencv_core.so.4.5 \
    sample/3rd/opencv/lib/libopencv_core.so.4.5.0 \
    sample/3rd/opencv/lib/libopencv_imgproc.so \
    sample/3rd/opencv/lib/libopencv_imgproc.so.4.5 \
    sample/3rd/opencv/lib/libopencv_imgproc.so.4.5.0 \
    sample/3rd/opencv/lib/libopencv_imgcodecs.so \
    sample/3rd/opencv/lib/libopencv_imgcodecs.so.4.5 \
    sample/3rd/opencv/lib/libopencv_imgcodecs.so.4.5.0 \
    sample/3rd/zlib/lib/libz.so \
    sample/3rd/zlib/lib/libz.so.1 \
    sample/3rd/zlib/lib/libz.so.1.2.11; do
    # Accept either a symbolic link ('l') or a regular file ('-') for each name.
    grep -Eq "^[-l][^ ]*[[:space:]].*[[:space:]]${root_name}/${path}([[:space:]]|$)" <<<"$listing" || \
      fail "archive entry missing or wrong type: $path"
  done
}

check_install_tree "$ROOT"

if [[ -n "$ARCHIVE" ]]; then
  check_archive "$ARCHIVE"
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  tar -xzf "$ARCHIVE" -C "$tmp"
  check_install_tree "$tmp/$(basename "$ROOT")"
fi

echo "verify_cv190x_package: OK: $ROOT${ARCHIVE:+ and $ARCHIVE}"
