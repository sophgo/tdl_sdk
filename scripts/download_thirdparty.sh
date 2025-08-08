#!/usr/bin/env bash
set -euo pipefail

# Determine architecture from the toolchain file
case "${TOOLCHAIN_FILE:-}" in
  *arm-linux-gnueabihf.cmake   | *arm-none-linux-gnueabihf.cmake)
    ARCHITECTURE="32bit" ;;
  *arm-none-linux-musleabihf.cmake)
    ARCHITECTURE="musl_arm" ;;
  *aarch64-linux-gnu.cmake     | *aarch64-none-linux-gnu.cmake \
  | *aarch64-buildroot-linux-gnu.cmake)
    ARCHITECTURE="64bit" ;;
  *riscv64-unknown-linux-gnu.cmake)
    ARCHITECTURE="glibc_riscv64" ;;
  *riscv64-unknown-linux-musl.cmake)
    ARCHITECTURE="musl_riscv64" ;;
  *)
    echo "ERROR: unsupported toolchain file: ${TOOLCHAIN_FILE}" >&2
    exit 1
    ;;
esac

TARGET_DIR="$(dirname "\$0")/dependency/thirdparty"

mkdir -p "$TARGET_DIR"

# List of packages to fetch
packages=(
  eigen
  googletest
  nlohmannjson
  stb
  curl
  libwebsockets
  openssl
  zlib
)

base_url="https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/${ARCHITECTURE}"

echo
echo "=================================================="
echo "Start downloading third-party packages..."
echo "Target dir: $TARGET_DIR"
echo "Architecture: $ARCHITECTURE"
echo

for pkg in "${packages[@]}"; do
  filename="${pkg}.tar.gz"
  url="${base_url}/${filename}"
  dest="${TARGET_DIR}/${filename}"

  if [[ -f "$dest" ]]; then
    echo "✔ $filename already exists, skipping"
    continue
  fi

  echo -n "⬇️  Downloading $filename ..."
  wget -q -c -O "$dest" "$url"
  if [[ $? -eq 0 ]]; then
    echo " done."
  else
    echo " failed!" >&2
    exit 1
  fi
done

echo "third-party packages download completed."
echo "All third-party packages are in place."
