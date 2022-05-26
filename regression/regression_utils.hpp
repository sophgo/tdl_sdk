#pragma once
#include <vector>
#include "cviai.h"

namespace cviai {
namespace unitest {

void init_face_meta(cvai_face_t *meta, uint32_t size);

void init_obj_meta(cvai_object_t *meta, uint32_t size, uint32_t height, uint32_t width,
                   int class_id);

void init_vehicle_meta(cvai_object_t *meta);

float iou(cvai_bbox_t &bbox1, cvai_bbox_t &bbox2);

template <typename Predections, typename ExpectedResult, typename Compare>
bool match_dets(Predections &preds, ExpectedResult &expected, Compare compare) {
  if (preds.size <= 0) return false;

  for (uint32_t actual_det_index = 0; actual_det_index < preds.size; actual_det_index++) {
    if (compare(preds.info[actual_det_index], expected)) {
      return true;
    }
  }
  return false;
}

}  // namespace unitest
}  // namespace cviai