
#!/bin/bash

print_usage() {
  echo ""
  echo "Usage: regression.sh [-r] [-m] [-s] [-d] [-t] [-h] test_type"
  echo ""
  echo "Argument:"
  echo -e "\ttest_type, set value to \"daily\" if it's a daily test"
  echo ""
  echo "Options:"
  echo -e "\t-r, results output directory (default: /mnt/data/result}"
  echo -e "\t-m, cvimodel directory (default: /mnt/data/cvimodel}"
  echo -e "\t-s, sample input images directory (default: /mnt/data/image}"
  echo -e "\t-d, dataset directory (default: /mnt/data/dataset}"
  echo -e "\t-h, help"
}

while getopts "r:m:d:s:h?" opt; do
  case ${opt} in
    r)
      results_dir=$OPTARG
      ;;
    m)
      model_dir=$OPTARG
      ;;
    s)
      sample_image_dir=$OPTARG
      ;;
    d)
      dataset_dir=$OPTARG
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

counter=0
run() {
    result="PASSED"
    echo "@@@@ RUN TEST: $@"
    ./"$@" 2>&1 || result="FAILED"
    counter=$((counter+1))
    echo "@@@@ TEST $counter $1 Done: $result"
}

makedir_if_needed() {
  target_dir="${1}"
  [ ! -d "${target_dir}" ] && echo "Directory \"${target_dir}\" DOES NOT exists. Makes one for test" && mkdir -p ${target_dir}
}

dataset_dir=${dataset_dir:-/mnt/data/dataset}
model_dir=${model_dir:-/mnt/data/cvimodel}
sample_image_dir=${sample_image_dir:-/mnt/data/image}
results_dir=${results_dir:-/mnt/data/result}
test_type=${@:$OPTIND:1}
test_type=${test_type:-None}

echo "----------------------"
echo "regression setting:"
echo -e "dataset dir: \t\t${dataset_dir}"
echo -e "model dir: \t\t${model_dir}"
echo -e "sample image dir: \t${sample_image_dir}"
echo -e "results dir: \t\t${results_dir}"
echo -e "test type: \t\t${test_type}"
echo "----------------------"

# For normal CI check
run sample_init
run sample_read_fr ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/cviface-v3-attribute.cvimodel ${sample_image_dir}/ryan.png
run sample_read_fr_custom ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/cviface-v3-attribute.cvimodel ${sample_image_dir}/ryan.png
run sample_read_dt ${model_dir}/retinaface_mnet0.25_608.cvimodel ${sample_image_dir}/ryan.png
run reg_object_intersect

# For daily build tests
if [[ "$test_type" != "daily" ]]; then
  exit 0
fi

echo "start to do regression test:"
makedir_if_needed ${results_dir}

run reg_wider_face ${model_dir}/retinaface_mnet0.25_608.cvimodel ${dataset_dir}/WIDER_val ${results_dir}/wider_face_result
run reg_lfw ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/cviface-v3-attribute.cvimodel ${dataset_dir}/lfw.txt ${results_dir}/lfw_result_cviface-v3.txt 1
run reg_lfw ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/cviface-v4.cvimodel ${dataset_dir}/lfw.txt ${results_dir}/lfw_result_cviface-v4.txt 0
run reg_lfw ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/cviface-v5-m.cvimodel ${dataset_dir}/lfw.txt ${results_dir}/lfw_result_cviface-v5-m.txt 0

makedir_if_needed ${results_dir}/face_attribute_feature
run reg_face_attribute ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/cviface-v3-attribute.cvimodel ${model_dir}/fqnet-v5_shufflenetv2-softmax.cvimodel ${dataset_dir}/face_zkt_3000 ${results_dir}/face_attribute_feature
run reg_face_quality ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/fqnet-v5_shufflenetv2-softmax.cvimodel ${dataset_dir}/pic2 ${dataset_dir}/neg_14_28
run reg_mask_classification ${model_dir}/mask_classifier.cvimodel ${dataset_dir}/mask_classification_ex_1.0/val/masked ${dataset_dir}/mask_classification_ex_1.0/val/unmasked

run reg_yolov3 ${model_dir}/yolo_v3_416.cvimodel ${dataset_dir}/val2017 ${dataset_dir}/annotations/instances_val2017.json ${results_dir}/yolov3_result.json
run reg_mobiledetv2 ${model_dir}/mobiledetv2-d0.cvimodel ${dataset_dir}/val2017 ${dataset_dir}/annotations/instances_val2017.json ${results_dir}/mobiledetv2_d0_result.json mobiledetv2-d0
run reg_mobiledetv2 ${model_dir}/mobiledetv2-d1.cvimodel ${dataset_dir}/val2017 ${dataset_dir}/annotations/instances_val2017.json ${results_dir}/mobiledetv2_d1_result.json mobiledetv2-d1
run reg_mobiledetv2 ${model_dir}/mobiledetv2-d2.cvimodel ${dataset_dir}/val2017 ${dataset_dir}/annotations/instances_val2017.json ${results_dir}/mobiledetv2_d2_result.json mobiledetv2-d2
run reg_mobiledetv2 ${model_dir}/mobiledetv2-lite.cvimodel ${dataset_dir}/val2017 ${dataset_dir}/annotations/instances_val2017.json ${results_dir}/mobiledetv2_lite_result.json mobiledetv2-lite
run reg_mobiledetv2 ${model_dir}/mobiledetv2-vehicle-d0.cvimodel ${dataset_dir}/val2017 ${dataset_dir}/annotations/instances_val2017_vehicle.json /${results_dir}/mobiledetv2_vehicle_d0_result.json mobiledetv2-vehicle-d0
run reg_thermal ${model_dir}/thermalfd-v1.cvimodel ${dataset_dir}/thermal_val ${dataset_dir}/thermal_val/valid.json ${results_dir}/thermal_result.json
run reg_rgbir_liveness ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/liveness-rgb-ir-f35.cvimodel ${dataset_dir}/face_spoof_RGBIR/ ${dataset_dir}/face_spoof_RGBIR/list_wo_backlight.txt ${results_dir}/rgbir_liveness_result.txt
run reg_mask_fr ${model_dir}/retinaface_mnet0.25_608.cvimodel ${model_dir}/masked-fr-v1-m.cvimodel ${dataset_dir}/mask_fr_images/images ${dataset_dir}/mask_fr_images/pair_list.txt ${results_dir}/mask_fr_result.txt
run reg_reid ${model_dir}/person-reid-v1.cvimodel ${dataset_dir}/Market-1501-v15.09.15/
run reg_face_align ${model_dir}/retinaface_mnet0.25_608.cvimodel ${dataset_dir}/test_data
run reg_es_classification ${model_dir}/es_classification.cvimodel ${dataset_dir}/ESC50/

# run reg_deeplabv3 ${model_dir}/deeplabv3_mobilenetv2_1x1_tpu.cvimodel.no.last.resize ${dataset_dir}/cityscapes/image_val/ ${results_dir}/deeplab_result/