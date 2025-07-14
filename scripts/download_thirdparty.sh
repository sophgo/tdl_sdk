
#!/bin/bash
CMAKE_TOOLCHAIN_FILE="$1"
# Get the architecture-specific part based on the toolchain file
case "$CMAKE_TOOLCHAIN_FILE" in
  *arm-linux-gnueabihf.cmake   | *arm-none-linux-gnueabihf.cmake)
    ARCHITECTURE="32bit"
    ;;
  *arm-none-linux-musleabihf.cmake)
    ARCHITECTURE="musl"
    ;;
  *aarch64-linux-gnu.cmake     | *aarch64-none-linux-gnu.cmake \
  | *aarch64-buildroot-linux-gnu.cmake)
    ARCHITECTURE="64bit"
    ;;
  *riscv64-unknown-linux-gnu.cmake)
    ARCHITECTURE="glibc_riscv64"
    ;;
  *riscv64-unknown-linux-musl.cmake)
    ARCHITECTURE="musl_riscv64"
    ;;
  *)
    echo "ERROR: unsupported toolchain file: ${TOOLCHAIN_FILE}" >&2
    exit 1
    ;;
esac


TARGET_DIR="$(dirname "\$0")/dependency/thirdparty"

mkdir -p "$TARGET_DIR"

wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/eigen.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/googletest.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/nlohmannjson.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/stb.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/${ARCHITECTURE}/curl.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/${ARCHITECTURE}/libwebsockets.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/${ARCHITECTURE}/openssl.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/${ARCHITECTURE}/zlib.tar.gz"

echo "download thirdparty files done!"

