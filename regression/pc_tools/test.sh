#!/bin/bash

DATASET_PATH=$1
MLIR_PATH=$2
REGRESSION_FOLDER="regression_result"
RESULT_SERVER="http://qaweb/${REGRESSION_FOLDER}/"

wget -r -np -nH -R index.html* -o log ${RESULT_SERVER}

if [ ! -d ${REGRESSION_FOLDER} ]; then
  echo "Download failed"
  exit -1
fi

# yolov3
YOLOV3_OUTPUT="yolov3_result.json"
python3 eval_coco.py ${DATASET_PATH}/coco/instances_val2017.json ${REGRESSION_FOLDER}/${YOLOV3_OUTPUT}

# mobiledetv2_d0
MOBILE_OUTPUT="mobiledetv2_result.json"
python3 eval_coco.py ${DATASET_PATH}/coco/instances_val2017.json ${REGRESSION_FOLDER}/${MOBILE_OUTPUT}

# thermal
THERMAL_OUTPUT="thermal_result.json"
python3 eval_coco.py ${DATASET_PATH}/thermal_val/annotations.json ${REGRESSION_FOLDER}/${THERMAL_OUTPUT}

# wider face
WIDER_OUTPUT="wider_face_result"
python3 eval_widerface.py  "${REGRESSION_FOLDER}/${WIDER_OUTPUT}" ${MLIR_PATH}

# fr lfw
FR_LFW_OUTPUT="lfw_result.txt"
python3 eval_lfw.py "${REGRESSION_FOLDER}/${FR_LFW_OUTPUT}"

# mask fr
MASK_FR_OUTPUT="mask_fr_result.txt"
python3 eval_lfw.py "${REGRESSION_FOLDER}/${MASK_FR_OUTPUT}"

# rgbir liveness
RGBIR_LIVENESS_OUTPUT="rgbir_liveness_result.txt"
python3 eval_lfw.py "${REGRESSION_FOLDER}/${RGBIR_LIVENESS_OUTPUT}"

rm -rf ${REGRESSION_FOLDER}
rm log
