安装依赖：
sudo apt-get update
pip3 install transformers_stream_generator einops tiktoken accelerate gradio transformers==4.45.2 
pip3 install pybind11[global]

模型下载:

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b_int4_seq512_1dev.bmodel

运行sample

python sdk_package/tdl_sdk/sample/python/sample_qwen.py -m /path/to/bmodel -t ./token_config