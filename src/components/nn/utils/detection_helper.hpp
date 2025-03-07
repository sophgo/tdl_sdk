#pragma once
#include <map>
#include <vector>

#include "common/model_output_types.hpp"
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

  static void nmsObjects(std::vector<ObjectBoxLandmarkInfo> &objects,
                         float iou_threshold);
  static void nmsObjects(std::vector<ObjectBoxInfo> &objects,
                         float iou_threshold);
  static void nmsObjects(std::map<int, std::vector<ObjectBoxInfo>> &bboxes,
                         float iou_threshold);
  static void nmsObjects(std::vector<ObjectBoxSegmentationInfo> &objects,
                         float iou_threshold, std::vector<std::pair<int, uint32_t>> &stride_index);
  static void rescaleBbox(ObjectBoxInfo &bbox,
                          const std::vector<float> &scale_params,
                          const int crop_x = 0, const int crop_y = 0);
  static void rescaleBbox(ObjectBoxSegmentationInfo &bbox,
                                    const std::vector<float> &scale_params,
                                    const int crop_x, const int crop_y);
  //   static void convertDetStruct(std::map<int, std::vector<cvtdl_bbox_t>>
  //   &dets,
  //                                cvtdl_object_t *obj, int im_height,
  //                                int im_width);
};
