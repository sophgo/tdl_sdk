#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"
#include "opencv2/opencv.hpp"
#include "yolov3_utils.h"

namespace cviai {

class Yolov3 final : public Core {
 public:
  Yolov3();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *obj, cvai_obj_det_type_t det_type);

 private:
  void outputParser(cvai_object_t *obj, cvai_obj_det_type_t det_type);
  void doYolo(YOLOLayer &l);
  void getYOLOResults(detection *dets, int num, float threshold, int ori_w, int ori_h,
                      std::vector<object_detect_rect_t> &results, cvai_obj_det_type_t det_type);

  YOLOParamter m_yolov3_param;
};
}  // namespace cviai
