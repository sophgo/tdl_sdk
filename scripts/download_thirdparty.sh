
#!/bin/bash

TARGET_DIR="$(dirname "\$0")/thirdparty"

mkdir -p "$TARGET_DIR"

wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/eigen.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/googletest.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/nlohmannjson.tar.gz"
wget -P "$TARGET_DIR"  "https://github.com/sophgo/oss/raw/refs/heads/master/oss_release_tarball/32bit/stb.tar.gz"

echo "download thirdparty files done!"

