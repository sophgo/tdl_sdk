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


total_ion_size=30000000
if [ -f "/sys/kernel/debug/ion/cvi_carveout_heap_dump/total_mem" ]; then
  total_ion_size=$(cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/total_mem)
  CHIP_ARCH="CV181X"
elif [ -f "/sys/kernel/debug/ion/cvi_npu_heap_dump/total_mem" ]; then
  total_ion_size=$(cat /sys/kernel/debug/ion/cvi_npu_heap_dump/total_mem)
  CHIP_ARCH="CV186X"
elif [ -f "/proc/soph/vpss" ]; then
  CHIP_ARCH="BM1688"
fi



det_test_suites="DetectionTestSuite.*"
cls_test_suites="ClassificationTestSuite.*"
kpt_test_suites="KeypointTestSuite.*"

det_json=""
cls_json=""
kpt_json=""


# ION requirement >= 20 MB
if [ "$total_ion_size" -gt "20000000" ]; then

  #det
  det_json="${det_json}:mbv2_det_person.json"
  det_json="${det_json}:yolov8n_det_fire.json"
  det_json="${det_json}:yolov8n_det_fire_smoke.json"
  det_json="${det_json}:yolov8n_det_hand_face_person.json"
  det_json="${det_json}:yolov8n_det_hand.json"
  det_json="${det_json}:yolov8n_det_head_hardhat.json"
  det_json="${det_json}:yolov8n_det_head_shoulder.json"
  det_json="${det_json}:yolov8n_det_license_plate.json"
  det_json="${det_json}:yolov8n_det_person_vehicle.json"
  det_json="${det_json}:yolov8n_det_pet_person.json"
  det_json="${det_json}:yolov8n_det_traffic_light.json"
  det_json="${det_json}:yolov8n_det_monitor_person.json"
  det_json="${det_json}:scrfd_det_face.json"
  #cls
  cls_json="${cls_json}:cls_rgbliveness.json"
  cls_json="${cls_json}:cls_sound_babay_cry.json"
  cls_json="${cls_json}:cls_sound_nihaoshiyun.json"
  #kpt
  kpt_json="${kpt_json}:keypoint_hand.json"
  kpt_json="${kpt_json}:keypoint_license_plate.json"
  kpt_json="${kpt_json}:keypoint_yolov8pose_person17.json"
fi

# # ION requirement >= 35 MB
# if [ "$total_ion_size" -gt "35000000" ]; then
#   #det
#   det_json="${det_json}:xxx.json"

#   #cls
#   cls_json="${cls_json}:xxx.json"

#   #kpt
#   kpt_json="${kpt_json}:xxx.json"
# fi


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
    ./test_main "${model_dir}" "${dataset_dir}" "${full_json_path}" --gtest_filter="${test_suites}"
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
run_test_main "${kpt_json}" "${kpt_test_suites}"
