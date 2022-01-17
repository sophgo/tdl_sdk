#!/bin/sh

CHIPSET="${CHIP:=182x}"
CHIPSET=$(echo ${CHIP} | tr '[:upper:]' '[:lower:]')

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

# FIXME: There is a bug when you run --gtest_filter=*
# test_suite="*"

test_suite="CoreTestSuite.*"
test_suite="${test_suite} MobileDetV2TestSuite.*"
test_suite="${test_suite} FaceRecognitionTestSuite.*"
test_suite="${test_suite} MaskClassification.*"
test_suite="${test_suite} FaceQualityTestSuite.*"
test_suite="${test_suite} LicensePlateDetectionTestSuite.*"
test_suite="${test_suite} LicensePlateRecognitionTestSuite.*"
test_suite="${test_suite} MultiObjectTrackingTestSuite.*"
test_suite="${test_suite} ReIdentificationTestSuite.*"
test_suite="${test_suite} TamperDetectionTestSuite.*"
test_suite="${test_suite} ThermalFaceDetectionTestSuite.*"
test_suite="${test_suite} ThermalPersonDetectionTestSuite.*"
test_suite="${test_suite} LivenessTestSuite.*"
test_suite="${test_suite} AlphaposeTestSuite.*"
test_suite="${test_suite} FallTestSuite.*"
test_suite="${test_suite} RetinafaceTestSuite.*"
test_suite="${test_suite} RetinafaceIRTestSuite.*"
test_suite="${test_suite} RetinafaceHardhatTestSuite.*"
test_suite="${test_suite} IncarTestSuite.*"
test_suite="${test_suite} ESCTestSuite.*"
test_suite="${test_suite} EyeCTestSuite.*"
test_suite="${test_suite} YawnCTestSuite.*"
test_suite="${test_suite} SoundCTestSuite.*"
test_suite="${test_suite} FLTestSuite.*"
test_suite="${test_suite} FeatureMatchingTestSuite.*"

echo "----------------------"
echo -e "regression setting:"
echo -e "model dir: \t\t${model_dir}"
echo -e "dataset dir: \t\t${dataset_dir}"
echo -e "asset dir: \t\t${asset_dir}"
echo -e "CHIPSET=${CHIPSET}"

echo "Test Suites:"
for suite_name in ${test_suite}
do
    echo -e "\t${suite_name}"
done
echo "----------------------"

for suite_name in ${test_suite}
do
    regression/test_main ${model_dir} ${dataset_dir} ${asset_dir} --gtest_filter=$suite_name
done