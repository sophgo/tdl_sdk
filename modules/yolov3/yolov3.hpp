#pragma once
#include "core.hpp"
#include "opencv2/opencv.hpp"
#include "object/cvi_object_types.h"
#include "yolov3_utils.h"

namespace cviai {

class Yolov3 : public Core {
public:
  Yolov3();
  ~Yolov3();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvi_object_t *meta, cvi_obj_det_type_t det_type);

private:
  void outputParser(cvi_object_t *obj, cvi_obj_det_type_t det_type);
  void doYolo(YOLOLayer &l);
  void getYOLOResults(detection *dets, int num, float threshold, int ori_w, int ori_h,
                      std::vector<object_detect_rect_t> &results, cvi_obj_det_type_t det_type);

  YOLOParamter m_yolov3_param;
};
}  // namespace cviai
