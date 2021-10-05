#!/bin/bash

CHIPSET="${CHIP:=183x}"
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

if [[ "$CHIPSET" = "183x" ]]; then
    test_suite="*"
elif [[ "$CHIPSET" = "182x" ]]; then
    test_suite="MobileDetV2TestSuite.*"
    test_suite+=":FaceQualityTestSuite.*"
fi

echo "----------------------"
echo -e "regression setting:"
echo -e "model dir: \t\t${model_dir}"
echo -e "dataset dir: \t\t${dataset_dir}"
echo -e "asset dir: \t\t${asset_dir}"
echo -e "CHIPSET=${CHIPSET}"
IFS=':' read -a strarr <<<"${test_suite}" #reading str as an array as tokens separated by IFS  
echo "Test Suites:"
for suite_name in "${strarr[@]}"
do
    echo -e "\t${suite_name}"
done
echo "----------------------"

./test_main ${model_dir} ${dataset_dir} ${asset_dir} --gtest_filter=${test_suite}