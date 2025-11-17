#!/bin/bash

# 要运行的模型命令
CMD="/mnt/data/sdk_package/cviruntime/build_sdk/build_cviruntime/tool/model_runner --model /mnt/data/sdk_package/cv181x/yolov8n_det_pet_person_035_384_640_INT8_cv181x.cvimodel"

# 循环 100 次执行
for i in $(seq 1 100)
do
    echo "---- 第 $i 次运行 ----"
    $CMD
    echo "---- 第 $i 次运行结束 ----"
    echo ""
done