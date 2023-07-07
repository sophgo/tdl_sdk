# 通用yolov5模型部署

## 引言

本文档介绍了如何将yolov5架构的模型部署在cv181x开发板的操作流程，主要的操作步骤包括：

* yolov5模型pytorch版本转换为onnx模型
* onnx模型转换为cvimodel格式
* 最后编写调用接口获取推理结果

以下是各个步骤的详细讲解：

## pt模型转换为onnx

首先可以下载yolov5官方仓库代码，地址如下

[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

```shell
git clone https://github.com/ultralytics/yolov5.git
```

然后获取yolov5的`.pt`格式的模型，例如下载[yolov5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)，在yolov5文件夹下创建一个文件夹`weights`，并将下载的`yolov5s.pt`文件移动至此

转换onnx前，需要修改`yolov5/models/yolo.py`文件中`Detect`类中的`forward`函数：

* 首先为了避免数值不统一，注释下列`forward`函数中的19和20行代码，保证模型最后一层输出的数值范围在`[0-1]`之间，以免后续模型量化失败
* 然后修改`forward`函数的返回结果，不用`cat`操作，而是返回三个不同的下采样结果，直接`return z`

```python
 def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return z
        # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
```

> 这样修改后模型的输出分别三个不同的branch:
>
> * (batch_size, 1200, 85)
> * (batch_size, 4800, 85)
> * (batch_size, 19200, 85)

然后使用官方的`export.py`导出onnx模型

```shell
python export.py --weights ./weights/yolov5s.pt --include onnx
```

## onnx模型转换cvimodel

导出onnx模型之后，需要将onnx模型转换为cvimodel，才能实现cv181x开发板的c++推理。cvimodel的转换需要借助量化工具。

* 首先获取`cvitek_mlir_ubuntu-18.04_tpu_rel_v1.5.0-xxxxxx.tar.gz`

* 然后创建一个文件夹名为`mlir_cvi`，并在该文件夹下创建`cvitek_mlir`，将`tpu-mlir_vxxxxxx.tar.gz`解压在`cvitek_mlir`文件夹下
* 另外，在`mlir_cvi`文件夹下创建一个路径`yolov5s/onnx`，将上一步得到的`yolov5s.onnx`移动至此

### docker环境配置

创建docker环境

```shell
docker run \
-itd \
-v /etc/localtime:/etc/localtime:ro \
-v /path/to/cache/.cache:/root/.cache \
-v /path/to/workspace/codes:/workspace \
--name="cvi_docker" \
--rm \
cvitek/cvitek_dev:1.7-ubuntu-18.04
```

使用之前创建的docker

```shell
docker exec -it mipeng_cvi bash
```

此时用户已经处理docker里面的`/workspace`目录下

声明环境变量

```shell
source cvitek_mlir/envsetup.sh
```

然后使用以下脚本转换模型

```bash
# cv182x | cv183x
chip="cv181x"
# model_name="mobiledetv2-pedestrian-d0-448-p10"  | mv2_448_256

############################################################################################################
model_dir="yolov5m"
root="/workspace/mlir_cvi/${model_dir}"
model_name="yolov5m"
version_name="yolov5m"
img_dir="/path/to/img_folder/"
img="/path/to/single_image/"

intpu_size=640,640                                       ########## h,w

mlir="${root}/mlir/${version_name}_fp32.mlir"
table="${root}/calibration_table/${version_name}.threshold_table"
bf16="${root}/bf16/${version_name}_${chip}_bf16.cvimodel"
int8="${root}/int8/${version_name}_${chip}.cvimodel"
model_onnx="${root}/onnx/${model_name}.onnx"

# -------------------------------------------------------------------------------------------------------- #
############################################################################################################


mkdir "${root}/mlir"

model_transform.py \
--model_type onnx \
--model_name ${model_name} \
--model_def ${model_onnx} \
--image ${img} \
--image_resize_dims ${intpu_size}  \
--keep_aspect_ratio 1 \
--net_input_dims ${intpu_size} \
--raw_scale 255.0 \
--mean 0.0,0.0,0.0 \
--std 255.0,255.0,255.0 \
--input_scale 1.0 \
--model_channel_order "rgb" \
--tolerance 0.99,0.99,0.99 \
--mlir ${mlir}



# gen calibration_table

mkdir "${root}/calibration_table"
run_calibration.py \
${mlir} \
--dataset=${img_dir} \
--input_num 100 \
-o ${table} \
--tune_num 20 \
--tune_thread_num 10 \
--forward_thread_num 15 \
--buffer_size=20G \
--calibration_table ${table}


mkdir "${root}/int8"

model_deploy.py \
--model_name ${model_name} \
--mlir ${mlir} \
--calibration_table ${table} \
--quantize INT8 \
--chip ${chip} \
--tg_op_divide=true \
--image ${img} \
--pixel_format BGR_PLANAR \
--tolerance 0.8,0.8,0.25 \
--correctness 0.95,0.95,0.95 \
--cvimodel ${int8}
```

运行完成之后，可以在`mlir_cvi/yolov5s/int8/`目录获取到转换的cvimodel

## AISDK接口说明

集成的yolov5接口开放了预处理的设置，yolov5模型算法的anchor，conf置信度以及nms置信度设置

预处理设置的结构体为`Yolov5PreParam`

```c++
/** @struct Yolov5PreParam
 *  @ingroup core_cviaicore
 *  @brief Config the yolov5 detection preprocess.
 *  @var Yolov5PreParam::factor
 *  Preprocess factor, one dimension matrix, r g b channel
 *  @var Yolov5PreParam::mean
 *  Preprocess mean, one dimension matrix, r g b channel
 *  @var Yolov5PreParam::rescale_type
 *  Preprocess config, vpss rescale type config
 *  @var Yolov5PreParam::pad_reverse
 *  Preprocess padding config
 *  @var Yolov5PreParam::keep_aspect_ratio
 *  Preprocess config quantize scale
 *  @var Yolov5PreParam::use_crop
 *  Preprocess config, config crop
 *  @var Yolov5PreParam:: resize_method
 *  Preprocess resize method config
 *  @var Yolov5PreParam::format
 *  Preprocess pixcel format config
 */
typedef struct {
  float factor[3];
  float mean[3];
  meta_rescale_type_e rescale_type;
  bool pad_reverse;
  bool keep_aspect_ratio;
  bool use_quantize_scale;
  bool use_crop;
  VPSS_SCALE_COEF_E resize_method;
  PIXEL_FORMAT_E format;
}Yolov5PreParam;

/** @struct YOLOV5AlgParam
 *  @ingroup core_cviaicore
 *  @brief Config the yolov5 detection algorithm parameters.
 *  @var YOLOV5AlgParam::anchors
 *  Configure yolov5 model anchors
 *  @var YOLOV5AlgParam::conf_thresh
 *  Configure yolov5 model conf threshold val
 *  @var YOLOV5AlgParam::nms_thresh
 *  Configure yolov5 model nms threshold val
 */
typedef struct {
  uint32_t anchors[3][3][2];
  float conf_thresh;
  float nms_thresh;
} YOLOV5AlgParam;
```

yolov5算法中设置的结构体为`YOLOV5AlgParam`

```c++
/** @struct Yolov5PreParam
 *  @ingroup core_cviaicore
 *  @brief Config the yolov5 detection preprocess.
 *  @var Yolov5PreParam::factor
 *  Preprocess factor, one dimension matrix, r g b channel
 *  @var Yolov5PreParam::mean
 *  Preprocess mean, one dimension matrix, r g b channel
 *  @var Yolov5PreParam::rescale_type
 *  Preprocess config, vpss rescale type config
 *  @var Yolov5PreParam::pad_reverse
 *  Preprocess padding config
 *  @var Yolov5PreParam::keep_aspect_ratio
 *  Preprocess config quantize scale
 *  @var Yolov5PreParam::use_crop
 *  Preprocess config, config crop
 *  @var Yolov5PreParam:: resize_method
 *  Preprocess resize method config
 *  @var Yolov5PreParam::format
 *  Preprocess pixcel format config
 */
typedef struct {
  float factor[3];
  float mean[3];
  meta_rescale_type_e rescale_type;
  bool pad_reverse;
  bool keep_aspect_ratio;
  bool use_quantize_scale;
  bool use_crop;
  VPSS_SCALE_COEF_E resize_method;
  PIXEL_FORMAT_E format;
}Yolov5PreParam;

/** @struct YOLOV5AlgParam
 *  @ingroup core_cviaicore
 *  @brief Config the yolov5 detection algorithm parameters.
 *  @var YOLOV5AlgParam::anchors
 *  Configure yolov5 model anchors
 *  @var YOLOV5AlgParam::conf_thresh
 *  Configure yolov5 model conf threshold val
 *  @var YOLOV5AlgParam::nms_thresh
 *  Configure yolov5 model nms threshold val
 */
typedef struct {
  uint32_t anchors[3][3][2];
  float conf_thresh;
  float nms_thresh;
} YOLOV5AlgParam;
```

以下是一个简单的设置案例，初始化预处理设置`Yolov5PreParam`以及yolov5模型设置`YOLOV5AlgParam`，使用`CVI_AI_Set_YOLOV5_Param`传入设置的参数

```c++
// setup preprocess
Yolov5PreParam p_preprocess_cfg;

for (int i = 0; i < 3; i++) {
printf("asign val %d \n", i);
p_preprocess_cfg.factor[i] = 0.003922;
p_preprocess_cfg.mean[i] = 0.0;
}
p_preprocess_cfg.use_quantize_scale = true;
p_preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;

printf("start yolov algorithm config \n");
// setup yolov5 param
YOLOV5AlgParam p_yolov5_param;
uint32_t p_anchors[3][3][2] =  {{{10, 13}, {16, 30}, {33, 23}},
                        {{30, 61}, {62, 45}, {59, 119}},
                        {{116, 90}, {156, 198}, {373, 326}}};
memcpy(p_yolov5_param.anchors, p_anchors, sizeof(*p_anchors)*12);
p_yolov5_param.conf_thresh = 0.5;
p_yolov5_param.nms_thresh = 0.5;

printf("setup yolov5 param \n");
ret = CVI_AI_Set_YOLOV5_Param(ai_handle, &p_preprocess_cfg, &p_yolov5_param);
```

## 测试结果

* conf_thresh: 0.001 
* nms_thresh: 0.65

| model   | map50-official | map50-cvimodel | map-official | map-cvimodel |
| ------- | -------------- | -------------- | ------------ | ------------ |
| yolov5s | 56.8           | 53.7           | 37.4         | 33.6         |
| yolov5m | 64.1           | 61.3           | 45.4         | 39.6         |

ion:

* yolov5s: 8.15 MB
* yolov5m: 23.79MB

运行时间：

* yolov5s: 107.897 ms
* yolov5m: 265.91 ms













