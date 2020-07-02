#!/bin/bash

basedir=`pwd -P`

CORE_NUM=`cat /proc/cpuinfo | grep "processor"| wc -l`

if [[ -z "$TOOLCHAIN_PATH" ]]; then
  echo "TOOLCHAIN_PATH $TOOLCHAIN_PATH does not exist"
  echo "  Please export TOOLCHAIN_PATH "
  echo "  e.g. export  TOOLCHAIN_PATH=/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/  "
  return 1
fi
if [ ! -e $TOOLCHAIN_PATH ]; then
  echo "TOOLCHAIN_PATH $TOOLCHAIN_PATH does not exist"
  return 1
fi


if [[ -z "$SDK_PATH" ]]; then
  echo "SDK_PATH $SDK_PATH does not exist"
  echo "  Please export SDK_PATH "
  echo "  e.g. export  SDK_PATH=/home/xxx/workspace_183x/cvitek_tpu_sdk/  "
  return 1
fi
if [ ! -e $SDK_PATH ]; then
  echo "SDK_PATH $SDK_PATH does not exist"
  return 1
fi

if [[ -z "$MW_PATH" ]]; then
  echo "MW_PATH $MW_PATH does not exist"
  echo "  Please export MW_PATH "
  echo "  e.g. export  MW_PATH=/home/xxx/workspace_183x/mmf/  "
  return 1
fi
if [ ! -e $MW_PATH ]; then
  echo "MW_PATH $MW_PATH does not exist"
  return 1
fi

if [[ -z "$TRACER_PATH" ]]; then
  echo "TRACER_PATH $TRACER_PATH does not exist"
  echo "  Please export TRACER_PATH "
  echo "  e.g. export  TRACER_PATH=/home/xxx/workspace_183x/tracer/  "
  return 1
fi
if [ ! -e $TRACER_PATH ]; then
  echo "TRACER_PATH $TRACER_PATH does not exist"
  return 1
fi


function print_error()
{
  printf "\e[1;31;47m\tERROR\t     \e[0m\n"
}

function print_notice()
{
  printf "\e[1;34;47m $1 \e[0m\n"
}



function build_cvi_ai_lib()
{
  make -j $CORE_NUM
  test $? -ne 0 && popd && return 1
}

function clean_cvi_ai_lib()
{
  make clean
  test $? -ne 0 &&  popd && return 1
}


function build_all()
{
  build_cvi_ai_lib 
  test $? -ne 0 && print_notice "make cvi_ai_lib failed !!" && return 1
}


function clean_all()
{
  clean_cvi_ai_lib || ( print_error ; return 1 )
}

