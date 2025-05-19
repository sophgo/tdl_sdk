# 运行自有模型

当前tdl_sdk内已经支持如下类别的模型：

* 目标检测模型
  * YOLOV6
  * YOLOV8
  * YOLOV10

* 人脸检测模型
  * SCRFD
* 分类模型
  * 单标签分类（RgbImageClassification）
  * 多标签分类
  
* 语义分割模型
  * TOP_FORMER
* 实例分割模型
  * YOLOV8_SEG
* 关键点检测模型
  * YOLOV8_KEYPOINT
* 特征提取模型
  * 图像特征提取
  * CLIP文本特征提取
* 文本识别
  * 车牌识别
  
假如需要运行的自有模型与上述任务匹配，且属于同一类别，即可以无需修改代码，直接调用接口进行模型推理。

## 操作流程

* 编译模型
  * YOLO系列模型编译，参考[yolo_development_guide](../developer_guide/yolo_development_guide.md)
  * 其他类型模型，直接按照标准流程进行编译
* 配置模型，可以有三种配置方式
  * 在[model_factory.json](../../configs/model/model_factory.json)文件中替换现有模型的参数，后续基于现有模型的ID进行调用
  * 参考[model_factory.json](../../configs/model/model_factory.json)创建新的配置文件，后续基于新的配置文件进行调用
  * 在代码中使用json字符串提供配置信息
* 调用举例

  ```cpp
  //方式1:内部会使用lib_tdl.so所在目录下的configs/model/model_config.json文件
  TDLModelFactory::getInstance().setModelDir("/path/to/tdl_models");
  TDLModelFactory::getInstance().loadModelConfig();
  //方式2:使用新的配置文件
  TDLModelFactory::getInstance().setModelDir("/path/to/tdl_models");
  TDLModelFactory::getInstance().loadModelConfig("new_model_config.json");
  //方式3:使用json字符串
  static const std::string cviface_config = R"JSON(
  {
    "_comment": "cviface 256-dimensional feature",
    "file_name": "recognition_cviface_112_112_INT8",
    "rgb_order": "rgb",
    "mean": [127.5, 127.5, 127.5],
    "std": [128, 128, 128]
  }
  )JSON";
  std::shared_ptr<BaseModel> model =
      TDLModelFactory::getInstance().getModel(ModelType::FEATURE_IMG, model_path, cviface_config);
  
  ```

* 配置信息加载说明
  * 为了方便使用tdl_models仓库中的所有模型，建议使用方式1加载配置信息
  * 后续的同类型的新增模型文件，可以使用方式3补充使用
  * 大部分模型都无需提供额外的配置信息
  * 少数的通用模型ID如特征提取，需要提供mean、std、rgb_order等参数

### 检测模型调用举例

检测类任务，对于特定类型的模型，它的预处理参数都是一样的，所以无需额外配置预处理参数

```cpp

  
  std::string model_path = "/path/to/my_own_yolov8_model.bmodel";
  ModelType model_id = ModelType::YOLOV8;
  std::shared_ptr<BaseModel> model =
      TDLModelFactory::getInstance().getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

```

对于自有模型的model_id选取，请参考[tdl_model_defs.hpp](../../include/nn/tdl_model_defs.hpp)

### 分类任务调用举例

分类任务，包含两类：

* 单标签分类
* 多标签分类（待补充）

对于单标签分类，使用如下调用方式：

```cpp

  std::string model_path = "/path/to/my_own_cls_model.bmodel";
  ModelType model_id = ModelType::CLS_IMG;
  

  // 配置预处理参数
  ModelConfig config;
  config.mean = {0, 0, 0};
  config.std = {255.0, 255.0, 255.0};
  config.dst_image_format = ImageFormat::RGB_PLANAR;//BGR_PLANAR
  config.keep_aspect_ratio = false;

  std::shared_ptr<BaseModel> model =
      TDLModelFactory::getInstance().getModel(model_id, model_path, config);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

```

### 特征提取任务调用举例

特征提取任务，包含两类：

* 图像特征提取
* CLIP文本特征提取

图像特征提取，使用如下调用方式：

```cpp

  ModelConfig config;
  config.mean = {127.5, 127.5, 127.5};
  config.std = {128, 128, 128};
  config.dst_image_format = ImageFormat::RGB_PLANAR;//BGR_PLANAR
  config.keep_aspect_ratio = false;
  std::string model_path = "/path/to/my_own_feature_model.bmodel";
  ModelType model_id = ModelType::FEATURE_IMG;
  std::shared_ptr<BaseModel> model =
      TDLModelFactory::getInstance().getModel(model_id, model_path, config);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

```
