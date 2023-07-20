#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class Yolo final : public Core {
 public:
  Yolo();
  virtual ~Yolo();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);
  void set_param(YoloPreParam *p_preprocess_cfg, YoloAlgParam *p_Yolo_param);

 private:
  virtual int onModelOpened() override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void clip_bbox(int frame_width, int frame_height, cvai_bbox_t *bbox);
  cvai_bbox_t Yolo_box_rescale(int frame_width, int frame_height, int width, int height,
                               cvai_bbox_t bbox);
  void getYoloDetections(int8_t *ptr_int8, float *ptr_float, int num_per_pixel, float qscale,
                         int det_num, int channel, Detections &vec_obj);
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);
  void YoloPostProcess(Detections &dets, int frame_width, int frame_height,
                       cvai_object_t *obj_meta);

  YoloPreParam *p_preprocess_cfg_;
  YoloAlgParam *p_Yolo_param_;
};
}  // namespace cviai
