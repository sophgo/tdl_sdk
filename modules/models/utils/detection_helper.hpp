#ifndef __DETECTION_HELPER_HPP__
#define __DETECTION_HELPER_HPP__
#include <map>
#include <vector>

#include "core/face/cvtdl_face_types.h"

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
};

#endif
