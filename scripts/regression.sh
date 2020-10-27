counter=0
#!/bin/bash
run() {
    result="PASSED"
    ./"$@" || result="FAILED"
    counter=$((counter+1))
    echo "TEST $counter $1: $result"
}
# For normal CI check
run sample_init
run sample_read_fr /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/bmface.cvimodel /mnt/data/image/ryan.png
run sample_read_fr_custom /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/bmface.cvimodel /mnt/data/image/ryan.png
run sample_read_dt /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/image/ryan.png
run reg_object_intersect

# For daily build tests
if [[ "$1" != "daily" ]]; then
  exit 0
fi

run reg_wider_face /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/dataset/wider_face /mnt/data/result/wider_face_result
run reg_lfw /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/bmface.cvimodel /mnt/data/dataset/lfw.txt /mnt/data/result/lfw_result.txt
run reg_face_attribute /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/bmface.cvimodel /mnt/data/cvimodel/fqnet-v5_shufflenetv2-softmax.cvimodel /mnt/data/face_zkt_3000 /mnt/data/face_attribute_feature/
run reg_face_quality /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/fqnet-v5_shufflenetv2-softmax.cvimodel /mnt/data/pic2 /mnt/data/neg_14_28
run reg_mask_classification /mnt/data/cvimodel/mask_classifier.cvimodel /mnt/data/dataset/mask /mnt/data/dataset/unmask

run reg_yolov3 /mnt/data/cvimodel/yolo_v3_416.cvimodel /mnt/data/dataset/val2017 /mnt/data/dataset/instances_val2017.json /mnt/data/yolov3_result.json
run reg_mobiledetv2 /mnt/data/cvimodel/mobiledetv2_d0.cvimodel /mnt/data/dataset/val2017 /mnt/data/dataset/instances_val2017.json /mnt/data/mobiledetv2_result.json
run reg_thermal /mnt/data/cvimodel/thermalfd-v1_resnet18-bifpn-sh.cvimodel /mnt/data/dataset/thermal_val /mnt/data/dataset/thermal_val/valid.json /mnt/data/result/thermal_result.json
run reg_rgbir_liveness /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/liveness_batch9.cvimodel /mnt/data/face_spoof_RGBIR/ /mnt/data/face_spoof_RGBIR/list_wo_backlight.txt /mnt/data/rgbir_liveness_result.txt
run reg_mask_fr /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/cvimodel/masked_fr_r50.cvimodel /mnt/data/mask_fr_images/images /mnt/data/mask_fr_images/pair_list.txt /mnt/data/mask_fr_result.txt
run reg_reid /mnt/data/cvimodel/reid_mobilenetv2_x1_0.cvimodel /mnt/data/Market-1501-v15.09.15/
run reg_face_align /mnt/data/cvimodel/retina_face.cvimodel /mnt/data/WFLW/test_data
run reg_es_classification /mnt/data/cvimodel/es_classification.cvimodel /mnt/data/ESC50/
