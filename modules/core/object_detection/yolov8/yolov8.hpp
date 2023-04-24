#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class YoloV8Detection final : public Core {
 public:
  YoloV8Detection();

  void setBranchChannel(int box_channel, int cls_channel);
  virtual ~YoloV8Detection();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);

 private:
  virtual int onModelOpened() override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;

  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);

  void parseDecodeBranch(const int image_width, const int image_height, const int frame_width,
                         const int frame_height, cvai_object_t *obj_meta);

  void decode_bbox_feature_map(int stride, int anchor_idx, std::vector<float> &decode_box);
  void postProcess(Detections &dets, int frame_width, int frame_height, cvai_object_t *obj_meta);
  std::map<std::string, std::string> out_names_;

  // if output seperate featuremap
  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> bbox_class_out_names;
  int m_box_channel_ = 0;
  int m_cls_channel_ = 0;
};
}  // namespace cviai
