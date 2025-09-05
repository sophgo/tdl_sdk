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

./test_runner config/test_config.json $model_dir $dataset_dir