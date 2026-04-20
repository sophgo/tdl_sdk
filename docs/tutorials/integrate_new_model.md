# 集成新模型

借助tdl_sdk框架，可以使新模型集成变得更加简单，框架可以帮助完成如下工作：

* 图像预处理
  * 用户只需配置预处理参数，框架内会自动调用预处理设备实现预处理
* 模型推理，无需编写任何推理相关代码
* 内存管理，框架内会自动管理内存，用户无需担心内存泄漏问题

还具有如下优势：

* 支持多平台，用户无需关心平台差异
* 性能、资源优化
* 硬件资源充分利用

## 现有模型类添加新模型文件

现有的模型已经在tdl_sdk中实现，只是需要在模型工厂中添加对应的新模型ID方便调用，操作步骤如下：

* 在[tdl_model_list.h](../../include/nn/tdl_model_list.h)中添加新模型的model_id
* 在[tdl_model_factory.cpp](../../src/components/nn/tdl_model_factory.cpp)中添加新模型的创建函数
* 在[model_factory.json](../../configs/model/model_factory.json)中添加新模型的配置信息

## 集成新的模型类型

### 代码添加

* include下仅放对外接口，不必暴露的接口放到src下
* 考虑当前模块如何适用于所有平台，抽象出基类模块
* 代码复用
  * 功能一样的代码，使用同一份代码，如图像分类：[rgb_image_classification.cpp](../../src/components/nn/image_classification/rgb_image_classification.cpp)。
  * 功能类似的代码，放在相同目录，如检测类的统一放到[object_detection](../../src/components/nn/object_detection)。
* 代码实现（以人脸检测模型为例）
  * 人脸检测为一般深度学习模型，代码位于[face_detection](../../src/components/nn/face_detection)，头文件为：

    ```cpp
    #ifndef SCRFD_HPP
    #define SCRFD_HPP

    #include "image/base_image.hpp"
    #include "model/base_model.hpp"
    class SCRFD : public BaseModel {
    public:
      SCRFD();
      ~SCRFD();

      virtual int32_t outputParse(
          const std::vector<std::shared_ptr<BaseImage>>& images,
          std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) override;
      virtual int onModelOpened() override;

    private:
      std::vector<int> m_feat_stride_fpn;
      std::map<int, std::vector<std::vector<float>>> fpn_anchors_;
      std::map<int, std::map<std::string, std::string>>
          fpn_out_nodes_;  //{stride:{"box":"xxxx","score":"xxx","landmark":"xxxx"}}
      std::map<int, int> fpn_grid_anchor_num_;
      float iou_threshold_ = 0.5;
    };
    #endif
    ```

  * 其中SCRFD继承至BaseModel，主要包含初始化、onModelOpened、outputParse函数。
  * 人脸检测cpp主要包含以下内容：
    * 初始化函数，完成模型mean、scale的设置，其值与pytorch代码一致：

      ```cpp
      SCRFD::SCRFD() {
        net_param_.model_config.mean = {127.5, 127.5, 127.5};
        net_param_.model_config.std = {128, 128, 128};
        net_param_.model_config.rgb_order = "rgb";
        keep_aspect_ratio_ = true;
      }
      ```

    * onModelOpened函数，完成模型解析。

      ```cpp
        int32_t SCRFD::onModelOpened() {
          struct anchor_cfg {
            std::vector<int> SCALES;
            int BASE_SIZE;
            std::vector<float> RATIOS;
            int ALLOWED_BORDER;
            int STRIDE;
          };
          std::vector<anchor_cfg> cfg;
          anchor_cfg tmp;

          m_feat_stride_fpn = {8, 16, 32};

          tmp.SCALES = {1, 2};
          tmp.BASE_SIZE = 16;
          tmp.RATIOS = {1.0};
          tmp.ALLOWED_BORDER = 9999;
          tmp.STRIDE = 8;
          cfg.push_back(tmp);
          // ...

        }

      ```

    * outputParse函数，完成模型后处理,将模型推理结果给out_datas。

      ```cpp
      int32_t SCRFD::outputParse(
          const std::vector<std::shared_ptr<BaseImage>> &images,
          std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
        std::string input_tensor_name = net_*>getInputNames()[0];
        TensorInfo input_tensor = net_*>getTensorInfo(input_tensor_name);

        LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
            input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
            input_tensor.shape[3]);

        const int FACE_LANDMARKS_NUM = 5;
        int total_face_num = 0;
        for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
          uint32_t image_width = images[b]*>getWidth();
          uint32_t image_height = images[b]*>getHeight();

          std::vector<ObjectBoxLandmarkInfo> vec_bbox;
          std::vector<ObjectBoxLandmarkInfo> vec_bbox_nms;
          std::vector<float> &rescale_params = batch_rescale_params_[b];

          //...
        }
      }

      ```

  * 如模型与一般的推理过程不同，还需重写inference函数，可以参考AudioClassification实现。
* 模型输出
  * 输出结构体定义在[model_output_types.hpp](../../include/framework/common/model_output_types.hpp)。
  * 如需新增，确保功能不重复。
* 其他
  * [utils](../../src/framework/utils)，工具函数，如图像对齐、一些公共函数等。
  * [model_output_types.cpp](../../src/framework/common/model_output_types.cpp) 析构函数的实现（如有）。

### 注册模型

* 模型ID
  * 模型ID写在[tdl_model_list.h](../../include/nn/tdl_model_list.h)中。

    ```cpp
    #define MODEL_TYPE_LIST                                                       \
      X(MBV2_DET_PERSON_256_448, "0:person")                                      \
      X(YOLOV8N_DET_HAND, "")                                                     \
      X(YOLOV8N_DET_PET_PERSON, "0:cat,1:dog,2:person")                           \
      X(YOLOV8N_DET_PERSON_VEHICLE,                                               \
        "0:car,1:bus,2:truck,3:rider with "                                       \
        "motorcycle,4:person,5:bike,6:motorcycle")       
      // ...
    #endif
    ```

  * 模型ID后面需注明检测类别及含义，如"0:cat,1:dog,2:person"。
  * 功能类似的模型id写到相同的分组中。
  * 对于yolo系列等常见网络结构，命名为"网络_任务_检测目标"，如YOLOV8N_DET_HAND为基于yolov8n的手部检测。
  * 对于其他类型，命名为"大类_任务"。如CLS_HAND_GESTURE为手势分类。

* 模型注册
  * 写在[tdl_model_factory.cpp](../../src/components/nn/tdl_model_factory.cpp)中。
  * 先找到模型大类，然后在createxxxModel中创建其实例：

  ```cpp
    std::shared_ptr<BaseModel> TDLModelFactory::createFaceDetectionModel(
        const ModelType model_type) {
      std::shared_ptr<BaseModel> model = nullptr;

      if (model_type == ModelType::SCRFD_DET_FACE) {
        model = std::make_shared<SCRFD>();
      }

      return model;
    }
    ```

  * 若为目标检测，需在model_type_mapping中构建检测类别。

### sample编译

* 参考[编译自有模型](run_private_model.md)
