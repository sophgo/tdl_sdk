#pragma once
#include <bitset>

#include "model/base_model.hpp"

class PPYoloEDetection final : public BaseModel {
 public:
  PPYoloEDetection();
  PPYoloEDetection(std::pair<int, int> yolov8_pair);
  ~PPYoloEDetection();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  void decodeBboxFeatureMap(int batch_idx, int stride, int anchor_idx,
                            std::vector<float> &decode_box);

  std::map<std::string, std::string> out_names_;

  // if output seperate featuremap
  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> bbox_class_out_names;
  int num_box_channel_ = 4;
  int num_cls_ = 0;  // would parse automatically,should not be equal with
                     // num_box_channel_
  float m_model_threshold = 0.5;
};
