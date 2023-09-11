#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"
#include "object_utils.hpp"

namespace cviai {

class YoloV8Pose final : public Core {
 public:
  YoloV8Pose();

  virtual ~YoloV8Pose();
  void setBranchChannel(int box_channel, int kpts_channel, int m_cls_channel);
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int onModelOpened() override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;

  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);

  void decode_bbox_feature_map(int stride, int anchor_idx, std::vector<float> &decode_box);
  void decode_keypoints_feature_map(int stride, int anchor_idx, std::vector<float> &decode_kpts);

  void postProcess(Detections &dets, int frame_width, int frame_height, cvai_object_t *obj,
                   std::vector<std::pair<int, int>> &valild_pairs);

  std::vector<int> strides;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> keypoints_out_names;
  int m_box_channel_ = 0;
  int m_kpts_channel_ = 0;
  int m_cls_channel_ = 0;
};
}  // namespace cviai
