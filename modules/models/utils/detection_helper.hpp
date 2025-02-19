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

  static std::vector<std::vector<std::vector<float>>> generateRetinaNetAnchors(
      int min_level, int max_level, int num_scales,
      const std::vector<std::pair<float, float>> &aspect_ratios,
      float anchor_scale, int image_width, int image_height);

  static void nmsFaces(std::vector<cvtdl_face_info_t> &faces,
                       float iou_threshold);
  static void nmsObjects(std::vector<cvtdl_bbox_t> &bboxes,
                         float iou_threshold);

  static void nmsObjects(std::map<int, std::vector<cvtdl_bbox_t>> &bboxes,
                         float iou_threshold);
  static void rescaleBbox(cvtdl_bbox_t &bbox,
                          const std::vector<float> &scale_params,
                          const int crop_x = 0, const int crop_y = 0);

  static void convertDetStruct(std::map<int, std::vector<cvtdl_bbox_t>> &dets,
                               cvtdl_object_t *obj, int im_height,
                               int im_width);
};
