#pragma once
#include <bitset>

#include "model/base_model.hpp"

class YoloV5Detection final : public BaseModel {
 public:
  YoloV5Detection();
  YoloV5Detection(std::pair<int, int> yolov5_pair);
  ~YoloV5Detection();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  void decodeBboxFeatureMap(int batch_idx, int stride, int basic_pos,
                            int grid_x, int grid_y, float pw, float ph,
                            std::vector<float> &decode_box);
  std::map<int, std::string> class_out_names_;
  std::map<int, std::string> object_out_names_;
  std::map<int, std::string> box_out_names_;
  std::vector<int> strides_;
  float nms_threshold_ = 0.5;

  uint32_t *initial_anchors = nullptr;
  int num_cls = 0;
};
