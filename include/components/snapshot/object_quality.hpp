#ifndef OBJECT_QUALITY_HPP
#define OBJECT_QUALITY_HPP
#include "framework/common/model_output_types.hpp"
class ObjectQualityHelper {
 public:
  static float getFaceQuality(
      const ObjectBoxInfo& box, const std::vector<float>& landmark_xys,
      const int img_width, const int img_height,
      const std::map<std::string, float>& other_info = {});

  static float getFaceQuality(
      const ObjectBoxInfo& box, const std::vector<float>& landmark_xys,
      const int img_width, const int img_height, bool fl_model,
      const std::map<std::string, float>& other_info = {});

  static float getFaceQuality(
      const ObjectBoxLandmarkInfo& box_landmark, const int img_width,
      const int img_height,
      const std::map<std::string, float>& other_info = {});
  static void getFaceQuality(const std::vector<ObjectBoxInfo>& face_bbox,
                             const std::vector<ObjectBoxInfo>& head_bbox,
                             std::vector<float>& face_quality);
};

#endif
