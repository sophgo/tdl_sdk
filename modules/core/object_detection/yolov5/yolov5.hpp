#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class Yolov5 final : public Core {
 public:
  Yolov5();
  virtual ~Yolov5();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj_meta);
  void set_param(YoloPreParam *p_preprocess_cfg, YoloAlgParam *p_yolov5_param);

 private:
  virtual int onModelOpened() override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void clip_bbox(int frame_width, int frame_height, cvai_bbox_t *bbox);
  cvai_bbox_t yolov5_box_rescale(int frame_width, int frame_height, int width, int height,
                                 cvai_bbox_t bbox);
  void getYolov5Detections(int8_t *ptr_int8, float *ptr_float, int num_per_pixel, float qscale,
                           int stride_x, int stride_y, int grid_x_len, int grid_y_len,
                           uint32_t *anchor, Detections &vec_obj);
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_object_t *obj_meta);
  void Yolov5PostProcess(Detections &dets, int frame_width, int frame_height,
                         cvai_object_t *obj_meta);
  std::map<int, std::string> out_names_;
  std::map<int, int> stride_keys_;
  YoloPreParam *p_preprocess_cfg_;
  YoloAlgParam *p_yolov5_param_;
};
}  // namespace cviai
