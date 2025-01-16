#pragma once
#include <map>
#include <vector>

#include "core/face/cvtdl_face_types.h"
#include "core/object/cvtdl_object_types.h"
class DetectionHelper {
 public:
  DetectionHelper();
  ~DetectionHelper();

  static std::vector<std::vector<float>> generateMmdetBaseAnchors(
      float base_size, float center_offset, const std::vector<float> &ratios,
      const std::vector<int> &scales);

  static std::vector<std::vector<float>> generateMmdetGridAnchors(
      int feat_w, int feat_h, int stride,
      std::vector<std::vector<float>> &base_anchors);

  static void nmsFaces(std::vector<cvtdl_face_info_t> &faces,
                       float iou_threshold);
  static void nmsObjects(std::vector<cvtdl_bbox_t> &bboxes,
                         float iou_threshold);

  static void nmsObjects(std::map<int, std::vector<cvtdl_bbox_t>> &bboxes,
                         float iou_threshold);
  static void rescaleBbox(cvtdl_bbox_t &bbox,
                          const std::vector<float> &scale_params);

  static void convertDetStruct(std::map<int, std::vector<cvtdl_bbox_t>> &dets,
                               cvtdl_object_t *obj, int im_height,
                               int im_width);
};
