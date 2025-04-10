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
* 使用现有接口进行调用
  * 获取模型
  * 配置预处理参数，只需要配置mean、scale、dst_image_format、keep_aspect_ratio

### 检测模型调用举例

检测类任务，对于特定类型的模型，它的预处理参数都是一样的，所以无需额外配置预处理参数

```cpp

  TDLModelFactory model_factory;
  std::string model_path = "/path/to/my_own_yolov8_model.bmodel";
  ModelType model_id = ModelType::YOLOV8;
  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
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

  TDLModelFactory model_factory;
  std::string model_path = "/path/to/my_own_cls_model.bmodel";
  ModelType model_id = ModelType::CLS_IMG;
  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  // 配置预处理参数
  PreprocessParams pre_param;
  pre_param.mean[0] =  2.1179;
  pre_param.mean[1] =  2.0357;
  pre_param.mean[2] =  1.8044;
  pre_param.scale[0] = 0.017126;
  pre_param.scale[1] = 0.017509;
  pre_param.scale[2] = 0.017431;
  pre_param.dst_image_format = ImageFormat::RGB_PLANAR;//BGR_PLANAR
  pre_param.keep_aspect_ratio = true;

  model_hc->setPreprocessParameters(pre_param);
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

  TDLModelFactory model_factory;
  std::string model_path = "/path/to/my_own_feature_model.bmodel";
  ModelType model_id = ModelType::FEATURE_IMG;
  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }
  // 配置预处理参数
  PreprocessParams pre_param;
  pre_param.mean[0] = 0;
  pre_param.mean[1] = 0;
  pre_param.mean[2] = 0;
  pre_param.scale[0] = 1.0/255;
  pre_param.scale[1] = 1.0/255;
  pre_param.scale[2] = 1.0/255;
  pre_param.dst_image_format = ImageFormat::RGB_PLANAR;//BGR_PLANAR
  pre_param.keep_aspect_ratio = true;

  model_hc->setPreprocessParameters(pre_param);
  
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

```
