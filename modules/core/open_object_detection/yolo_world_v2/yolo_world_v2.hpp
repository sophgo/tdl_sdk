#pragma once
#include <bitset>
#include "core/object/cvtdl_object_types.h"
#include "obj_detection.hpp"

namespace cvitdl {

typedef std::pair<int, int> PAIR_INT;

class Yolo_World_V2 final : public DetectionBase {
 public:
  Yolo_World_V2();
  Yolo_World_V2(PAIR_INT yolo_world_v2_pair);
  ~Yolo_World_V2();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_clip_feature **clip_txt_feats,
                cvtdl_object_t *obj_meta);

 private:
  int onModelOpened() override;

  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvtdl_object_t *obj_meta);

  void decode_bbox_feature_map(int stride, int anchor_idx, std::vector<float> &decode_box);
  void postProcess(Detections &dets, int frame_width, int frame_height, cvtdl_object_t *obj_meta);
  std::map<std::string, std::string> out_names_;

  // if output seperate featuremap
  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> bbox_class_out_names;
  int m_box_channel_ = 64;
};
}  // namespace cvitdl
