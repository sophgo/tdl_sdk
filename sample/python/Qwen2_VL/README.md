# Qwen2_VL 运行指南

## 环境准备
* 确保已安装 tdl_sdk 并配置好 BM1684X 环境
* 准备 Python 虚拟环境

## 操作步骤

1. **挂载 NFS 共享目录**
   ```bash
   sudo mount <NFS_SERVER_IP>:/path/to/sdk_package /data/sdk_package
   ```
   *请将 `<NFS_SERVER_IP>` 替换为实际的 NFS 服务器地址*
   *将 `/path/to/sdk_package` 替换为实际的共享目录路径*

2. **激活 Python 环境**
   ```bash
   source /data/LLM-TPU/llmtpu_env/bin/activate
   ```

3. **进入项目目录**
   ```bash
   cd /data/sdk_package/tdl_sdk/sample/python/Qwen2_VL
   ```

4. **设置环境变量**
   ```bash
   export LD_PRELOAD="/opt/sophon/libsophon-current/lib/libbmlib.so:/opt/sophon/libsophon-current/lib/libbmrt.so"
   export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib/:/opt/sophon/sophon-opencv-latest/lib/:/data/sdk_package/tdl_sdk/install/BM1684X/lib
   export PYTHONPATH=/data/sdk_package/tdl_sdk/install/BM1684X/lib:$PYTHONPATH
   ```

5. **运行 pipeline 脚本**
   ```bash
   python pipeline.py -m /path/to/model.bmodel -t /path/to/tokenizer/ -c /path/to/config.json -v $video_path
   ```
   *请替换上述命令中的路径为实际模型、tokenizer、配置文件的路径和输入视频的路径*

6. **参数说明**
   * `-m`: 指定 BMODEL 模型路径
   * `-t`: 指定 tokenizer 路径
   * `-c`: 指定配置文件路径
   * `-v`: 指定视频文件路径