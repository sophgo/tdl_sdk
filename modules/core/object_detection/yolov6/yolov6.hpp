#pragma once
#include <bitset>
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class Yolov6 final : public Core {
 public:
  Yolov6();
  virtual ~Yolov6();
  YoloPreParam get_preparam();
  void set_preparam(YoloPreParam pre_param);
  YoloAlgParam get_algparam();
  void set_algparam(YoloAlgParam alg_param);
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *obj_meta);

 private:
  virtual int onModelOpened() override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void decode_bbox_feature_map(int stride, int anchor_idx, std::vector<float> &decode_box);
  void clip_bbox(int frame_width, int frame_height, cvtdl_bbox_t *bbox);
  cvtdl_bbox_t boxRescale(int frame_width, int frame_height, int width, int height,
                          cvtdl_bbox_t bbox);
  void postProcess(Detections &dets, int frame_width, int frame_height, cvtdl_object_t *obj_meta);
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_hegiht, cvtdl_object_t *obj_meta);

  YoloPreParam p_preprocess_cfg_;
  YoloAlgParam p_alg_param_;

  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;
};
}  // namespace cvitdl