#pragma once
#include <vector>
// #include "cvi_tdl.h"
#include "core/core/cvtdl_core_types.h"
#include "core/face/cvtdl_face_types.h"
#include "core/object/cvtdl_object_types.h"
namespace cvitdl {
namespace unitest {

void init_face_meta(cvtdl_face_t *meta, uint32_t size);

void init_obj_meta(cvtdl_object_t *meta, uint32_t size, uint32_t height,
                   uint32_t width, int class_id);

void init_vehicle_meta(cvtdl_object_t *meta);

float iou(cvtdl_bbox_t &bbox1, cvtdl_bbox_t &bbox2);

template <typename Predections, typename ExpectedResult, typename Compare>
bool match_dets(Predections &preds, ExpectedResult &expected, Compare compare) {
  if (preds.size <= 0) return false;

  for (uint32_t actual_det_index = 0; actual_det_index < preds.size;
       actual_det_index++) {
    if (compare(preds.info[actual_det_index], expected)) {
      return true;
    }
  }
  return false;
}

}  // namespace unitest
}  // namespace cvitdl