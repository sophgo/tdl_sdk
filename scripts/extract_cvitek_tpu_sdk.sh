
#!/bin/bash

if [ -f "scripts/credential.sh" ]; then
  source "scripts/credential.sh"
fi

if [ -z "$FTP_SERVER_IP" ]; then
    FTP_SERVER_IP=218.17.249.213
fi

DEP_DIR=dependency
if [ ! -d "${DEP_DIR}" ]; then
    mkdir -p ${DEP_DIR}
fi
echo "save dir: ${DEP_DIR}"

curl -u cvitek_mlir_2023:"7&2Wd%cu5k" sftp://${FTP_SERVER_IP}/home/tpu_sdk_t4.1.0-23-gb920beb/cvitek_tpu_sdk_x86_64.tar.gz -o ${DEP_DIR}/cvitek_tpu_sdk.tar.gz

tar -xf ${DEP_DIR}/cvitek_tpu_sdk.tar.gz -C ${DEP_DIR}

mv ${DEP_DIR}/cvitek_tpu_sdk ${DEP_DIR}/CMODEL_CVITEK

rm -rf ${DEP_DIR}/cvitek_tpu_sdk.tar.gz

echo "done!"
