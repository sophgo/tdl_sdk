#pragma once
#include <bitset>

#include "model/base_model.hpp"

class YoloV8Segmentation final : public BaseModel {
 public:
  YoloV8Segmentation();
  YoloV8Segmentation(std::tuple<int, int, int> yolov8_tuple);
  ~YoloV8Segmentation();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  void decodeBboxFeatureMap(int batch_idx, int stride, int anchor_idx,
                            std::vector<float> &decode_box);

  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> mask_out_names;
  std::map<int, std::string> proto_out_names;
  std::map<int, std::string> bbox_class_out_names;
  int num_box_channel_ = 64;
  int num_mask_channel_ = 32;
  int num_cls_ = 0;
  float nms_threshold_ = 0.5;
};
