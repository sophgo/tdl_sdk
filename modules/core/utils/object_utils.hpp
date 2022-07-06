#pragma once
#include <memory>
#include <vector>

namespace cviai {

struct object_detect_rect_t {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
};

typedef std::shared_ptr<object_detect_rect_t> PtrDectRect;
typedef std::vector<PtrDectRect> Detections;

Detections nms_multi_class(const Detections &dets, float iou_threshold);
}  // namespace cviai