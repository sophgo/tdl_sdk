#!/bin/sh

print_usage() {
  echo ""
  echo "Usage: daily_regression.sh [-m] [-d] [-a] [-h]"
  echo ""
  echo "Options:"
  echo -e "\t-m, cvimodel directory (default: /mnt/data/cvimodel)"
  echo -e "\t-d, dataset directory (default: /mnt/data/dataset)"
  echo -e "\t-a, json data directory (default: /mnt/data/asset)"
  echo -e "\t-h, help"
}

while getopts "m:d:a:h?" opt; do
  case ${opt} in
    m)
      model_dir=$OPTARG
      ;;
    d)
      dataset_dir=$OPTARG
      ;;
    a)
      asset_dir=$OPTARG
      ;;
    h)
      print_usage
      exit 0
      ;;
    \?)
      print_usage
      exit 128
      ;;
  esac
done

model_dir=${model_dir:-/mnt/data/cvimodel}
dataset_dir=${dataset_dir:-/mnt/data/dataset}
asset_dir=${asset_dir:-/mnt/data/asset}


if [ -z "$CHIP_ARCH" ]; then

  chip_info=$(devmem 0x300008c)

  if  echo "$chip_info" | grep -q "181"; then
    CHIP_ARCH="CV181X"
  elif  echo "$chip_info" | grep -q "184"; then
    CHIP_ARCH="CV184X"
  elif [ -f "/sys/kernel/debug/ion/cvi_npu_heap_dump/total_mem" ]; then
    CHIP_ARCH="CV186X"
  elif [ -f "/proc/soph/vpss" ]; then
    CHIP_ARCH="BM1688"
  else 
    echo "unkonw CHIP_ARCH!"
    exit
  fi

else
  case "$CHIP_ARCH" in
      CV181X|CV184X|CV186X|BM1688)
          ;;
      *)
          echo "Error: CHIP_ARCH must be one of: CV181X, CV184X, CV186X, BM1688"
          exit 1
          ;;
  esac

fi

echo "CHIP_ARCH: ${CHIP_ARCH}"

det_test_suites="DetectionTestSuite.*"
cls_test_suites="ClassificationTestSuite.*"
face_attribute_cls_test_suites="AttributesTestSuite.*"
kpt_test_suites="KeypointTestSuite.*"
feature_test_suites="FeatureExtraTestSuite.*"
segmentation_suites="SegmentationTestSuite.*"
ocr_test_suites="OcrTestSuite.*"
sot_test_suites="SotTestSuite.*"

det_json=""
cls_json=""
face_attribute_cls_json=""
kpt_json=""
feature_json=""
segmentation_json=""
ocr_json=""
sot_json=""

# det
det_json="${det_json}:mbv2_det_person_256_448_INT8.json"
det_json="${det_json}:mbv2_det_person_512_896_INT8.json"
det_json="${det_json}:mbv2_det_person_256_384_INT8.json"
det_json="${det_json}:mbv2_det_person_896_896_INT8.json"
det_json="${det_json}:ppyoloe_det_coco80_640_640_INT8.json"
det_json="${det_json}:yolov10n_det_coco80_640_640_INT8.json"
det_json="${det_json}:yolov6n_det_coco80_640_640_INT8.json"
det_json="${det_json}:yolov6s_det_coco80_640_640_INT8.json"
det_json="${det_json}:yolov8n_det_coco80_640_640_INT8.json"
det_json="${det_json}:yolov8n_det_face_head_person_pet_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_fire_smoke_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_hand_face_person_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_head_hardhat_576_960_INT8.json"
det_json="${det_json}:yolov8n_det_ir_person_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_ir_person_mbv2_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_license_plate_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_monitor_person_256_448_INT8.json"
det_json="${det_json}:yolov8n_det_overlook_person_256_448_INT8.json"
det_json="${det_json}:yolov8n_det_person_vehicle_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_person_vehicle_mv2_035_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_pet_person_035_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_pet_person_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_traffic_light_384_640_INT8.json"
det_json="${det_json}:yolov8s_det_coco80_640_640_INT8.json"
det_json="${det_json}:yolov8n_det_head_shoulder_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_head_person_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_fire_384_640_INT8.json"
det_json="${det_json}:scrfd_det_face_432_768_INT8.json"
det_json="${det_json}:yolov8n_det_hand_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_hand_mv3_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_bicycle_motor_ebicycle_384_640_INT8.json"
det_json="${det_json}:yolov8n_det_bicycle_motor_ebicycle_mbv2_384_640_INT8.json"

if [ ${CHIP_ARCH} != "CV184X" ]; then  #tpu-milr bug
  det_json="${det_json}:yolov5m_det_coco80_640_640_INT8.json"
  det_json="${det_json}:yolov5s_det_coco80_640_640_INT8.json"
  det_json="${det_json}:yolov7_tiny_det_coco80_640_640_INT8.json"
  # det_json="${det_json}:yolox_m_det_coco80_640_640_INT8.json"
  # det_json="${det_json}:yolox_s_det_coco80_640_640_INT8.json"
fi


#cls
cls_json="${cls_json}:cls_hand_gesture_128_128_INT8.json"
cls_json="${cls_json}:cls_rgbliveness_256_256_INT8.json"
cls_json="${cls_json}:cls_sound_babay_cry_188_40_INT8.json"
cls_json="${cls_json}:cls_sound_nihaoshiyun_126_40_INT8.json"
cls_json="${cls_json}:cls_sound_xiaoaixiaoai_126_40_INT8.json"
cls_json="${cls_json}:cls_keypoint_hand_gesture_1_42_INT8.json"


#face_attribute_cls
face_attribute_cls_json="${face_attribute_cls_json}:cls_attribute_gender_age_glass_mask_112_112_INT8.json"
face_attribute_cls_json="${face_attribute_cls_json}:cls_attribute_gender_age_glass_112_112_INT8.json"
face_attribute_cls_json="${face_attribute_cls_json}:cls_attribute_gender_age_glass_emotion_112_112_INT8.json"
#kpt
kpt_json="${kpt_json}:keypoint_face_v2_64_64_INT8.json"
kpt_json="${kpt_json}:keypoint_hand_128_128_INT8.json"
kpt_json="${kpt_json}:keypoint_license_plate_64_128_INT8.json"
kpt_json="${kpt_json}:keypoint_simcc_person17_256_192_INT8.json"
kpt_json="${kpt_json}:keypoint_yolov8pose_person17_384_640_INT8.json"
kpt_json="${kpt_json}:lstr_det_lane_360_640_MIX.json"


# feature
feature_json="${feature_json}:feature_cviface_112_112_INT8.json"


#segmentation_json
segmentation_json="${segmentation_json}:yolov8n_seg_coco80_640_640_INT8.json"
if [ ${CHIP_ARCH} != "CV184X" ]; then  #tpu-milr bug
  segmentation_json="${segmentation_json}:topformer_seg_person_face_vehicle_384_640_INT8.json"
fi


#ocr
if [ ${CHIP_ARCH} != "CV184X" ]; then  #tpu-milr bug
  ocr_json="${ocr_json}:recognition_license_plate_24_96_MIX.json"
fi

#sot
sot_json="${sot_json}:tracking_feartrack_128_128_256_256_INT8.json"


failed_list="" 
reg_num=0

run_test_main() {
  json_files="$1"
  test_suites="$2"
  json_separated=$(echo "$json_files" | tr ':' ' ')

  echo "${test_suites} to be executed:"
  for json_file in ${json_separated}
  do
      echo  "\t${json_file}"
  done

  echo "----------------------"

  for json_file in ${json_separated}
  do
    full_json_path="${asset_dir}/${json_file}"
    echo "./test_main ${model_dir} ${dataset_dir} ${full_json_path} --gtest_filter=${test_suites}"
    ./test_main "${model_dir}" "${dataset_dir}" "${full_json_path}" --gtest_filter="${test_suites}"
    ret=$?
    if [ "$ret" -ne 0 ]; then
      failed_list="$failed_list ${json_file}"
    fi
    reg_num=$(expr "$reg_num" + 1)
  done

}


echo "----------------------"
echo  "regression setting:"
echo  "model dir: \t\t${model_dir}"
echo  "dataset dir: \t\t${dataset_dir}"
echo  "asset dir: \t\t${asset_dir}"
echo  "CHIP_ARCH: \t\t${CHIP_ARCH}"
echo  "ION size: \t\t${total_ion_size} bytes"

run_test_main "${det_json}" "${det_test_suites}"
run_test_main "${cls_json}" "${cls_test_suites}"
run_test_main "${face_attribute_cls_json}" "${face_attribute_cls_test_suites}"
run_test_main "${kpt_json}" "${kpt_test_suites}"
run_test_main "${feature_json}" "${feature_test_suites}"
run_test_main "${segmentation_json}" "${segmentation_suites}"
run_test_main "${ocr_json}" "${ocr_test_suites}"
run_test_main "${sot_json}" "${sot_test_suites}"

set -- $failed_list
failed_num=$#

if [ ${failed_num} == 0 ]; then
  echo "[${reg_num}/${reg_num}] ALL TEST PASSED"

else
  echo "failed json:"
  for item in $failed_list; do
    echo "$item"
  done
  echo "[${failed_num}/${reg_num}] TEST FAILED"
fi