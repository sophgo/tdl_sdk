#pragma once
#include <bitset>

#include "model/base_model.hpp"

class YoloV10Detection final : public BaseModel {
 public:
  YoloV10Detection();
  YoloV10Detection(std::pair<int, int> yolov8_pair);
  ~YoloV10Detection();
  // int inference(VIDEO_FRAME_INFO_S *srcFrame, tdl_object_t *obj_meta)
  // override;
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
  int num_box_channel_ = 64;
  int num_cls_ = 0;  // would parse automatically,should not be equal with
                     // num_box_channel_
};
