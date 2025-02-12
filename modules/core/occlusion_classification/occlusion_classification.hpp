#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"
#include "opencv2/core.hpp"

namespace cvitdl {
class OcclusionClassification final : public Core {
 public:
  OcclusionClassification();
  virtual ~OcclusionClassification();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_class_meta_t *occlusion_classification_meta);
  virtual bool allowExportChannelAttribute() const override { return true; }
  void set_algparam(OcclusionAlgParam occ_pre_param);

 private:
  int cv_method(VIDEO_FRAME_INFO_S *frame, cvtdl_class_meta_t *occlusion_classification_meta,
                float ai_occ_rato);
  float Lapulasi2(cv::Mat gray, bool flag);

  bool keyframe_flag = false;
  // bool auto_la_th = true;

  float occ_dev_ave = -1;
  float nml_dev_ave = -1;
  float auto_lap_dev_th = 20;
  // float _lap_dev_th = 20;
  float _ai_cv_th = 0.8;

  int frame_id = -1;

  cvtdl_bbox_t _crop_bbox;
  cv::Mat key_frame_gray;
  cv::Mat laplacianAbs;
};
}  // namespace cvitdl