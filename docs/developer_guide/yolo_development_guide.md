# Yolo通用推理接口使用文档

## 目的
提供便捷的C++接口，集成YOLO系列算法，以加快外部开发者的模型部署速度。

## 介绍
TDL_SDK集成了YOLO算法的前后处理与推理过程，支持以下算法：
- YOLOv5
- YOLOv6
- YOLOv7
- YOLOv8
- YOLOX
- PP-YOLOE
- YOLOv10

---

## 通用Yolov5模型部署

### 引言
部署流程包含三个主要步骤：
1. pt模型转换为onnx
2. onnx转换为bmodel/cvimodel
3. TDL_SDK接口调用

### pt模型转onnx

* 下载yolov5官方仓库代码

  ```
  git clone https://github.com/ultralytics/yolov5.git
  ```

* 获取yolov5的.pt格式的模型

  * 以yolov5s.pt为例, 下载地址：`yolov5s <https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt>`_

* 获取yolov5 onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/yolov5_export.py 复制到yolov5仓库目录下

* 导出onnx模型

  ```
  python yolov5_export.py --weights path/to/yolov5s.pt --img-size 640 640
  ```
  参数说明：
  * --weights pt文件的相对路径
  * --img-size 输出尺寸
  生成的onnx模型在当前目录下 
### onnx转换环境配置

onnx转成 bmodel/cvimodel 需要 TPU-MLIR 的发布包，其中TPU-MLIR 指算能TDL处理器的TPU编译器工程

* TPU-MLIR工具包下载

  * 代码地址:  `TPU-MLIR <https://github.com/sophgo/tpu-mlir>`_ 

  * 根据 `TPU-MLIR论坛 <https://developer.sophgo.com/thread/473.html>`_，下载对应的工具包

* Docker配置

  转换模型需要在指定的docker执行

  * 可以直接下载docker镜像(速度比较慢)：

    ```
    docker pull sophgo/tpuc_dev:latest
    ```

  * 或者直接加载TPU-MLIR工具包中下载的docker镜像(速度比较快)：

    ```
    docker load -i  docker_tpuc_dev_v2.2.tar.gz
    ``` 

  如果是首次使用Docker, 需执行下述命令进行安装和配置（仅首次执行）：

  ```
  sudo apt install docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  sudo groupadd docker
  sudo usermod -aG docker $USER
  newgrp docker
  ```
* 进入docker环境

  确保安装包在当前目录, 然后在当前目录创建容器如下：

  ```
  docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v2.2
  ```

以下操作需在Docker容器执行，后续步骤假定用户当前处在docker里面的/workspace目录

* 解压tpu_mlir工具包

  新建一个文件夹tpu_mlir, 将新工具链解压到tpu_mlir目录下, 并设置环境变量：

  ```

  mkdir tpu_mlir && cd tpu_mlir
  cp tpu-mlir_xxx.tar.gz ./
  tar zxf tpu-mlir_xxx.tar.gz
  source tpu_mli_xxx/envsetup.sh
  ```
  其中tpu-mlir_xxx.tar.gz的xxx是版本号, 根据对应的文件名而决定

* 拷贝onnx模型和图片

  创建一个文件夹, 以yolov5s举例, 创建一个文件夹yolov5s, 并将onnx模型放在yolov5s/onnx/路径下, 将一张训练图片放在yolov5s/images/路径下

  ```
  mkdir yolov5s && cd yolov5s && mkdir onnx && mkdir images
  cp path/to/yolov5s.onnx ./onnx/
  cp path/to/train_image.jpg ./images/
  ```

上述准备工作完成之后, 就可以开始转换模型

### onnx转MLIR

如果模型是图片输入, 在转模型之前我们需要了解模型的预处理

如果模型用预处理后的 npz 文件做输入, 则不需要考虑预处理

本例子中yolov5的图片是rgb, mean和scale对应为:

* mean:  0.0, 0.0, 0.0
* scale: 0.0039216, 0.0039216, 0.0039216

模型转换的命令如下：

  ```
  cd yolov5s
  model_transform.py \
  --model_name yolov5s \
  --model_def ./onnx/yolov5s.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input ./images/train_image.jpg \
  --test_result yolov5s_top_outputs.npz \
  --mlir yolov5s.mlir
  ```
其中参数详情请参考  `TPU-MLIR快速入门手册 <https://tpumlir.org/docs/quick_start/index.html>`_

### MLIR转INT8模型

我们以部署在cv180x, cv181x, cv182x, cv186x平台为例, 导出这些平台所需的cvimodel

部署在bm1684x, bm1688平台生成bmodel是类似的, 具体请参考 `TPU-MLIR快速入门手册 <https://tpumlir.org/docs/quick_start/index.html>`_

* 生成校准表

  * 准备100~1000张图片, 尽可能和训练数据分布相似, 放于yolov5s/images/路径下, 这里使用100张图片

  * 执行以下命令, 得到校准表

      ```
      run_calibration.py yolov5s.mlir \
      --dataset ./images \
      --input_num 100 \
      -o yolov5s_cali_table
      ```
    其中参数详情请参考  `TPU-MLIR快速入门手册 <https://tpumlir.org/docs/quick_start/index.html>`_

* 生成cvimodel

    ```
    model_deploy.py \
    --mlir yolov5s.mlir \
    --quant_input \
    --quant_output \
    --quantize INT8 \
    --calibration_table yolov5s_cali_table \
    --processor cv181x \
    --test_input yolov5s_in_f32.npz \
    --test_reference yolov5s_top_outputs.npz \
    --tolerance 0.85,0.45 \
    --model yolov5_cv181x_int8_sym.cvimodel
    ```
  其中参数详情请参考  `TPU-MLIR快速入门手册 <https://tpumlir.org/docs/quick_start/index.html>`_

编译完成后, 会生成名为yolov5_cv181x_int8_sym.cvimodel的文件

之后可以使用TDL_SDK接口调用该文件进行推理。

**注意运行的对应平台要一一对应！**

### TDL_SDK接口调用

可参考 sample/cpp/sample_img_objdet.cpp,  主要接口如下: 

* 图像读取

    ```
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    ```
* 模型实例化
    ```
    std::shared_ptr<BaseModel> model = model_factory.getModel(model_id, model_path);
    ```

* 设置模型阈值

    ```
    model->setModelThreshold(model_threshold);
    ```

* 推理执行

    ```
    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    model->inference(input_images, out_datas);
    ```

##  通用Yolov6模型部署

###  引言

本文档介绍了如何将yolov6架构的模型部署在cv181x开发板的操作流程, 主要的操作步骤包括：

* pt模型转换为onnx

* onnx转换成bmodel/cvimodel

* TDL_SDK接口调用

###  pt模型转换为onnx

* 下载yolov6官方仓库代码

    ```
    git clone https://github.com/meituan/YOLOv6.git
    ```
* 获取yolov6的.pt格式的模型

* 获取yolov6 onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/yolov6_export.py 复制到yolov6仓库目录下

* 导出onnx模型

    ```
    python yolov6_export.py --weights path/to/yolov6n.pt --img-size 640 640
    ```
  参数说明：

  * --weights pt文件的相对路径
  * --img-size 输出尺寸

  生成的onnx模型在当前目录下


###  onnx转换环境配置

此部分可以参考通用Yolov5模型部署章节的onnx转换环境配置部分。


###  TDL_SDK接口调用

此部分可以参考通用Yolov5模型部署章节的TDL_SDK接口调用部分。


##  通用yolov7模型部署

###  引言

本文档介绍了如何将yolov7架构的模型部署在cv181x开发板的操作流程, 主要的操作步骤包括：

* pt模型转换为onnx

* onnx转换成bmodel/cvimodel

* TDL_SDK接口调用

###  pt模型转换为onnx

* 下载yolov7官方仓库代码

    ```
    git clone https://github.com/WongKinYiu/yolov7.git
    ```
* 获取yolov7的.pt格式的模型

* 获取yolov7 onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/yolov7_export.py 复制到yolov7仓库目录下

* 导出onnx模型

    ```
    python yolov7_export.py --weights path/to/yolov7-tiny.pt --img-size 640 640
    ```
  参数说明：

  * --weights pt文件的相对路径
  * --img-size 输出尺寸

  生成的onnx模型在当前目录下

###  onnx转换环境配置

此部分可以参考通用Yolov5模型部署章节的onnx转换环境配置部分。


###  TDL_SDK接口调用

此部分可以参考通用Yolov5模型部署章节的TDL_SDK接口调用部分。


##  通用yolov8模型部署

###  引言

本文档介绍了如何将yolov8架构的模型部署在cv181x开发板的操作流程, 主要的操作步骤包括：

* pt模型转换为onnx

* onnx转换成bmodel/cvimodel

* TDL_SDK接口调用

###  pt模型转换为onnx

* 下载yolov8官方仓库代码

    ```
    git clone https://github.com/ultralytics/ultralytics.git
    ```
* 获取yolov8的.pt格式的模型

  调整yolov8输出分支, 去掉forward函数的解码部分, 并将三个不同的feature map的box以及cls分开, 得到6个分支, 这一步可以直接使用下一步的yolo_export的脚本完成

* 获取yolov8 onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/yolov8_export.py 复制到yolov8仓库目录下

* 导出onnx模型

    ```
    python yolov8_export.py --weights path/to/yolov8n.pt --img-size 640 640
    ```
  参数说明：

  * --weights pt文件的相对路径
  * --img-size 输出尺寸

  生成的onnx模型在当前目录下


###  onnx转换环境配置

此部分可以参考通用Yolov5模型部署章节的onnx转换环境配置部分。

###  TDL_SDK接口调用

此部分可以参考通用Yolov5模型部署章节的TDL_SDK接口调用部分。


##  通用yolox模型部署

###  引言

本文档介绍了如何将yolox架构的模型部署在cv181x开发板的操作流程, 主要的操作步骤包括：

* pt模型转换为onnx

* onnx转换成bmodel/cvimodel

* TDL_SDK接口调用

###  pt模型转换为onnx

* 下载yolox官方仓库代码, 使用以下命令从源代码安装yolox

    ```
    git clone git@github.com:Megvii-BaseDetection/YOLOX.git
    cd YOLOX
    pip3 install -v -e .  # or  python3 setup.py develop
    ```
* 获取yolox的.pt格式的模型

* 获取yolox onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/yolox_export.py 复制到yolox仓库目录下

* 导出onnx模型

  为了保证量化的精度, 需要将YOLOX解码的head分为三个不同的branch输出, 而不是官方版本的合并输出

  将yolo_export/yolox_export.py复制到YOLOX/tools目录下, 然后使用以下命令导出三个不同branch的输出的onnx模型：

    ```
    python \
    yolox_export.py \
    --output-name ../weights/yolox_s_9_branch_640_640.onnx \
    -n yolox-s \
    -c ../weights/yolox_s.pt \
    --img-size 640 640
    ```
  参数说明：

  * --output-name 输出的onnx文件路径
  * -n 模型名称
  * -c pt文件路径
  * --img-size 输出尺寸


###  onnx转换环境配置

此部分可以参考通用Yolov5模型部署章节的onnx转换环境配置部分。

###  TDL_SDK接口调用

此部分可以参考通用Yolov5模型部署章节的TDL_SDK接口调用部分。


##  通用pp-yoloe模型部署

###  引言

本文档介绍了如何将ppyoloe架构的模型部署在cv181x开发板的操作流程, 主要的操作步骤包括：

* pt模型转换为onnx

* onnx转换成bmodel/cvimodel

* TDL_SDK接口调用

###  pt模型转换为onnx

* 下载ppyoloe官方仓库代码

    ```
    git clone https://github.com/PaddlePaddle/PaddleDetection.git
    # CUDA10.2
    python -m pip install paddlepaddle-gpu==2.3.2 -i https://mirror.baidu.com/pypi/simple
    ```
  其他版本参照官方安装文档 `paddlepaddle安装 <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html>`_

* 获取ppyoloe的.pt格式的模型

  * 以ppyoloe_crn_s_300e_coco.pdparams为例, 下载地址：`ppyoloe_crn_s_300e_coco <https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams>`_

* 获取pp-yoloe onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/pp_yolo_export.py 复制到PaddleDetection/tools目录下


* 导出onnx模型

  为了更好地进行模型量化, 需要将检测头解码的部分去掉, 再导出onnx模型, 使用以下方式导出不解码的onnx模型

  将yolo_export/pp_yolo_export.py复制到tools/目录下, 然后使用如下命令导出不解码的pp-yoloe的onnx模型

    ```
    python \
    tools/export_model_no_decode.py \
    -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
    -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams

    paddle2onnx \
    --model_dir output_inference/ppyoloe_crn_s_300e_coco \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output_inference/ppyoloe_crn_s_300e_coco/ppyoloe_crn_s_300e_coco.onnx
    ```
  参数说明：

  * -c 模型配置文件
  * -o paddle 模型权重
  * --model_dir 模型导出目录
  * --model_filename paddle模型的名称
  * --params_filename paddle模型配置
  * --opset_version opset版本配置
  * --save_file 导出onnx模型的相对路径



  **如果需要修改模型的输入尺寸, 可以在上述导出的onnx模型进行修改, 例如改为384x640的输入尺寸, 使用以下命令进行修改:**

    ```
    python -m paddle2onnx.optimize \
    --input_model ./output_inference/ppyoloe_crn_s_300e_coco/ppyoloe_crn_s_300e_coco.onnx \
    --output_model ./output_inference/ppyoloe_crn_s_300e_coco/ppyoloe_384.onnx \
    --input_shape_dict "{'x':[1,3,384,640]}"
    ```
  参数说明：
  * --input_model 输入的onnx模型
  * --output_model 输出的onnx模型
  * --input_shape_dict 输入的shape

###  onnx转换环境配置

此部分可以参考通用Yolov5模型部署章节的onnx转换环境配置部分。

###  TDL_SDK接口调用

此部分可以参考通用Yolov5模型部署章节的TDL_SDK接口调用部分。
  

##  通用yolov10模型部署

###  引言

本文档介绍了如何将yolov10架构的模型部署在cv181x开发板的操作流程, 主要的操作步骤包括：

* pt模型转换为onnx

* onnx转换成bmodel/cvimodel

* TDL_SDK接口调用

##  pt模型转换为onnx

* 下载yolov10官方仓库代码

    ```
    git clone https://github.com/THU-MIG/yolov10.git
    ```
* 获取yolov10的.pt格式的模型

  调整yolov10输出分支, 去掉forward函数的解码部分, 并将三个不同的feature map的box以及cls分开, 得到6个分支, 这一步可以直接使用下一步yolo_export的脚本完成

* 获取yolov10 onnx导出脚本：

  * 官方导出方式的模型中解码部分不适合量化, 因此需要使用TDL_SDK提供的导出方式

  * 将此仓库tool/yolo_export/yolov10_export.py 复制到yolov10仓库目录下

* 导出onnx模型

    ```
    python yolov10_export.py --weights path/to/yolov10n.pt --img-size 640 640
    ```
  参数说明：

  * --weights pt文件的相对路径
  * --img-size 输出尺寸

  生成的onnx模型在当前目录下


###  onnx转换环境配置

此部分可以参考通用Yolov5模型部署章节的onnx转换环境配置部分。

###  TDL_SDK接口调用

此部分可以参考通用Yolov5模型部署章节的TDL_SDK接口调用部分。