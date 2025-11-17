#!/bin/sh

print_usage() {
  echo ""
  echo "Usage: daily_regression.sh [-j] [-m] [-d] [-f] [-a] [-h]"
  echo ""
  echo "Options:"
  echo -e "\t-j, regression json directory"
  echo -e "\t-m, cvimodel directory (default: /mnt/data/cvimodel)"
  echo -e "\t-d, dataset directory (default: /mnt/data/dataset)"
  echo -e "\t-f, flag"
  echo -e "\t-a, json data directory (default: /mnt/data/asset)"
  echo -e "\t-h, help"
}

while getopts "j:m:d:f:a:h?" opt; do
  case ${opt} in
    j)
      json_dir=$OPTARG
      ;;
    m)
      model_dir=$OPTARG
      ;;
    d)
      dataset_dir=$OPTARG
      ;;
    f)
      flag=$OPTARG
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

# model_dir=${model_dir:-/mnt/data/cvimodel}
# dataset_dir=${dataset_dir:-/mnt/data/dataset}
json_dir=${json_dir:-config/test_config.json}
model_dir=${model_dir:-/mnt/data/sdk_package}
dataset_dir=${dataset_dir:-/mnt/data/sdk_package/aisdk_daily_regression}
flag=${flag:-"func"}
asset_dir=${asset_dir:-/mnt/data/asset}

# 若要进行功能测试，option传入generate_perf，默认是func
./test_runner $json_dir $model_dir $dataset_dir $flag
