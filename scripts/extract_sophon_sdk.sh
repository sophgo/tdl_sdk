#!/bin/bash


if [ "$#" -ne 1 ]; then
    echo "usage: ./scripts/extract_sophon_sdk.sh zip_file path"
    exit 1
fi

ZIP_FILE="$1"


if [ ! -f "$ZIP_FILE" ]; then
    echo "file '$ZIP_FILE' not exist !"
    exit 1
fi


TARGET_DIR="$(dirname "\$0")/sophon_sdk"
TMP_DIR="$TARGET_DIR/tmp"

echo "Unzipping files, this may take several minutes ..."

mkdir -p "$TMP_DIR"
unzip -q "$ZIP_FILE" -d "$TMP_DIR"


if [ $? -eq 0 ]; then
    echo " unzip'$ZIP_FILE' to '$TMP_DIR'."
else
    echo " unzip '$ZIP_FILE' failed !"
    exit 1
fi


echo $TMP_DIR

tar -xf "$TMP_DIR/V1.8 sophonsdk_edge_v1.8_ofical_release/sophon-img/libsophon_soc_0.4.10_aarch64.tar.gz" -C "$TMP_DIR"
mv "$TMP_DIR/libsophon_soc_0.4.10_aarch64/opt/sophon/libsophon-0.4.10" "$TARGET_DIR"


tar -xf "$TMP_DIR/V1.8 sophonsdk_edge_v1.8_ofical_release/sophon_media/sophon-media-soc_1.8.0_aarch64.tar.gz" -C "$TMP_DIR"
mv "$TMP_DIR/sophon-media-soc_1.8.0_aarch64/opt/sophon/sophon-ffmpeg_1.8.0" "$TARGET_DIR"
mv "$TMP_DIR/sophon-media-soc_1.8.0_aarch64/opt/sophon/sophon-opencv_1.8.0" "$TARGET_DIR"


dpkg -x "$TMP_DIR/V1.8 sophonsdk_edge_v1.8_ofical_release/sophon-img/bsp-debs/sophon-soc-libisp-dev_1.0.0_arm64.deb"  "$TMP_DIR/libisp_1.0.0"
dpkg -x "$TMP_DIR/V1.8 sophonsdk_edge_v1.8_ofical_release/sophon-img/bsp-debs/sophon-soc-libisp_1.0.0_arm64.deb"  "$TMP_DIR/libisp_1.0.0"

mv "$TMP_DIR/libisp_1.0.0/opt/sophon/sophon-soc-libisp_1.0.0" "$TARGET_DIR"
echo "$TMP_DIR/libisp_1.0.0/opt/sophon/sophon-soc-libisp-dev_1.0.0/*"

cp -r $TMP_DIR/libisp_1.0.0/opt/sophon/sophon-soc-libisp-dev_1.0.0/* "$TARGET_DIR/sophon-soc-libisp_1.0.0"

rm -rf "$TMP_DIR"

echo "done"