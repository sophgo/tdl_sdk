#!/bin/bash

export LD_LIBRARY_PATH=/mnt/data/sdk_package/tdl_sdk/install/CV181X/lib:/mnt/data/sdk_package/tdl_sdk/install/CV181X/sample/utils/lib:/mnt/data/sdk_package/cvi_mpi/lib:/mnt/data/sdk_package/cvi_mpi/lib/3rd:/mnt/data/sdk_package/tdl_sdk/install/CV181X/sample/3rd/rtsp/lib:/mnt/data/sdk_package/install/soc_cv1811h_wevb_0007a_spinor/tpu_musl_riscv64/cvitek_tpu_sdk/lib:/mnt/data/sdk_package/tdl_sdk/install/CV181X/sample/3rd/opencv/lib:/mnt/data/sdk_package/cvi_rtsp/install/lib

export CVIMODEL_TOOL_PATH=/mnt/data/sdk_package/cviruntime/build_sdk/build_cviruntime/tool/cvimodel_tool

export PERF_EVAL=1

export TPU_REL=1

echo "[OK] Environment variables configured."