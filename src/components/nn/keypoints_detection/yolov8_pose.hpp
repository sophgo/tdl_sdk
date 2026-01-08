#pragma once
#include <bitset>

#include "model/base_model.hpp"

class YoloV8Pose final : public BaseModel {
 public:
  YoloV8Pose();
  YoloV8Pose(std::tuple<int, int, int> pose_tuple);
  ~YoloV8Pose();
  // int inference(VIDEO_FRAME_INFO_S *srcFrame, TDLObject *obj_meta)
  // override;
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

 private:
  void decodeBboxFeatureMap(int batch_idx, int stride, int anchor_idx,
                            std::vector<float> &decode_box);
  void decodeKeypointsFeatureMap(int batch_idx, int stride, int anchor_idx,
                                 std::vector<float> &decode_kpts);

  // if output seperate featuremap
  std::vector<int> strides;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> keypoints_out_names;
  int num_box_channel_ = 64;
  int num_kpts_channel_ = 0;
  int keypoint_dimension_ = 3;  // 2 or 3
  int num_kpts_;
  int num_cls_ = 0;
  float nms_threshold_ = 0.5;
};
