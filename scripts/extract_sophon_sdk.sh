#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: ./scripts/extract_sophon_sdk.sh zip_file chip_arch"
    exit 1
fi

ZIP_FILE="$1"
CHIP_ARCH="${2^^}"

if [ ! -f "$ZIP_FILE" ]; then
    echo "file '$ZIP_FILE' not exist !"
    exit 1
fi

TARGET_DIR="$(cd "$(dirname "$0")/.." && pwd)/dependency/$CHIP_ARCH"
TMP_DIR="$TARGET_DIR/tmp"

echo "Unzipping files, this may take several minutes ..."
mkdir -p "$TARGET_DIR"
mkdir -p "$TMP_DIR"
unzip -q "$ZIP_FILE" -d "$TMP_DIR"

if [ $? -eq 0 ]; then
    echo "unzip '$ZIP_FILE' to '$TMP_DIR'."
else
    echo "unzip '$ZIP_FILE' failed !"
    exit 1
fi

echo $TMP_DIR

if [ "$CHIP_ARCH" = "BM1688" ]; then
    # BM1688 解压逻辑

    # 动态查找并解压 libsophon 文件
    LIBSOPHON_TAR=$(find "$TMP_DIR" -name "libsophon_soc_*_aarch64.tar.gz" | head -n 1)
    if [ -n "$LIBSOPHON_TAR" ]; then
        tar -xf "$LIBSOPHON_TAR" -C "$TMP_DIR"
        mv "$TMP_DIR"/libsophon_soc_*/opt/sophon/libsophon-* "$TARGET_DIR/libsophon"
    else
        echo "libsophon tar.gz not found!"
        exit 1
    fi

    # 动态查找并解压 sophon-media 文件
    SOPHON_MEDIA_TAR=$(find "$TMP_DIR" -name "sophon-media-soc_*_aarch64.tar.gz" | head -n 1)
    if [ -n "$SOPHON_MEDIA_TAR" ]; then
        tar -xf "$SOPHON_MEDIA_TAR" -C "$TMP_DIR"
        mv "$TMP_DIR"/sophon-media-soc_*/opt/sophon/sophon-ffmpeg_* "$TARGET_DIR/sophon-ffmpeg"
        mv "$TMP_DIR"/sophon-media-soc_*/opt/sophon/sophon-opencv_* "$TARGET_DIR/sophon-opencv"
    else
        echo "sophon-media tar.gz not found!"
        exit 1
    fi

    # 动态查找并解压 libisp 的 deb 文件
    LIBISP_DEV_DEB=$(find "$TMP_DIR" -name "sophon-soc-libisp-dev_*_arm64.deb" | head -n 1)
    LIBISP_DEB=$(find "$TMP_DIR" -name "sophon-soc-libisp_*_arm64.deb" | head -n 1)

    if [ -n "$LIBISP_DEV_DEB" ] && [ -n "$LIBISP_DEB" ]; then
        dpkg -x "$LIBISP_DEV_DEB" "$TMP_DIR/libisp_dev"
        dpkg -x "$LIBISP_DEB" "$TMP_DIR/libisp"

        # 动态查找解压后的目录
        LIBISP_DIR=$(find "$TMP_DIR/libisp/opt/sophon/" -type d -name "sophon-soc-libisp_*" | head -n 1)
        LIBISP_DEV_DIR=$(find "$TMP_DIR/libisp_dev/opt/sophon/" -type d -name "sophon-soc-libisp-dev_*" | head -n 1)
        mv "$LIBISP_DIR" "$TARGET_DIR/sophon-soc-libisp"
        cp -r "$LIBISP_DEV_DIR"/* "$TARGET_DIR/sophon-soc-libisp"

    else
        echo "libisp deb files not found!"
        exit 1
    fi


elif [ "$CHIP_ARCH" = "BM1684X" ]; then
    # BM1684X 解压逻辑
    # 动态查找并解压 libsophon 文件
    LIBSOPHON_TAR=$(find "$TMP_DIR" -name "libsophon_*_aarch64.tar.gz" | head -n 1)
    if [ -n "$LIBSOPHON_TAR" ]; then
        tar -xf "$LIBSOPHON_TAR" -C "$TMP_DIR"
        mv "$TMP_DIR"/libsophon_*/opt/sophon/libsophon-* "$TARGET_DIR/libsophon"
    else
        echo "libsophon tar.gz not found!"
        exit 1
    fi

    # 动态查找并解压 sophon-mw 文件
    SOPHON_MW_TAR=$(find "$TMP_DIR" -name "sophon-mw-soc_*_aarch64.tar.gz" | head -n 1)
    if [ -n "$SOPHON_MW_TAR" ]; then
        tar -xf "$SOPHON_MW_TAR" -C "$TMP_DIR"
        mv "$TMP_DIR"/sophon-mw-soc_*/opt/sophon/sophon-ffmpeg_* "$TARGET_DIR/sophon-ffmpeg"
        mv "$TMP_DIR"/sophon-mw-soc_*/opt/sophon/sophon-opencv_* "$TARGET_DIR/sophon-opencv"
    else
        echo "sophon-mw tar.gz not found!"
        exit 1
    fi


elif [ "$CHIP_ARCH" = "BM1684" ]; then
    # BM1684 解压逻辑
    # 动态查找并解压 libsophon 文件
    LIBSOPHON_TAR=$(find "$TMP_DIR" -name "libsophon_*_aarch64.tar.gz" | head -n 1)
    if [ -n "$LIBSOPHON_TAR" ]; then
        tar -xf "$LIBSOPHON_TAR" -C "$TMP_DIR"
        mv "$TMP_DIR"/libsophon_*/opt/sophon/libsophon-* "$TARGET_DIR/libsophon"
    else
        echo "libsophon tar.gz not found!"
        exit 1
    fi

    # 动态查找并解压 sophon-mw 文件
    SOPHON_MW_TAR=$(find "$TMP_DIR" -name "sophon-mw-soc_*_aarch64.tar.gz" | head -n 1)
    if [ -n "$SOPHON_MW_TAR" ]; then
        tar -xf "$SOPHON_MW_TAR" -C "$TMP_DIR"
        mv "$TMP_DIR"/sophon-mw-soc_*/opt/sophon/sophon-ffmpeg_* "$TARGET_DIR/sophon-ffmpeg"
        mv "$TMP_DIR"/sophon-mw-soc_*/opt/sophon/sophon-opencv_* "$TARGET_DIR/sophon-opencv"
    else
        echo "sophon-mw tar.gz not found!"
        exit 1
    fi

else
    echo "Unsupported CHIP_ARCH: $CHIP_ARCH"
    exit 1
fi


rm -rf "$TMP_DIR"
echo "done"