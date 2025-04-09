
#先挂载到BM1684X盒子上
sudo mount 10.80.39.3:/data/algo.public/nfsuser/yibo.feng/sdk_package /data/sdk_package
#激活python环境
source /data/LLM-TPU/llmtpu_env/bin/activate
#进入Qwen2_VL目录
cd /data/sdk_package/tdl_sdk/sample/python/Qwen2_VL
#设置环境变量
export LD\_PRELOAD="/opt/sophon/libsophon-current/lib/libbmlib.so:/opt/sophon/libsophon-current/lib/libbmrt.so"
#设置lib路径
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib/:/opt/sophon/sophon-opencv-latest/lib/:/data/sdk_package/tdl_sdk/install/BM1684X/lib
#设置python路径
export PYTHONPATH=/data/sdk_package/tdl_sdk/install/BM1684X/lib:$PYTHONPATH
#运行pipeline.py
python pipeline.py -m /data/LLM-TPU/models/Qwen2\_VL/python\_demo\_video/python\_demo/qwen2-vl-7b\_int4\_seq2048\_1dev\_20250317\_191843\_bf16\_hp\_full.bmodel -t /data/LLM-TPU/models/Qwen2\_VL/python\_demo\_video/python\_demo/qwen2-vl/qwen/qwen2-vl-7b-instruct/ -c /data/LLM-TPU/models/Qwen2\_VL/python\_demo\_video/compile/files/Qwen2-VL-7B-Instruct/config.json