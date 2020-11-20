#pragma once
#include "core/face/cvai_face_types.h"
#include "core/utils/vpss_helper.h"
#include "face_utils.hpp"

#include <string>
#include <vector>

namespace cviai {
namespace evaluation {

class cityscapesEval {
 public:
  cityscapesEval(const char *image_dir, const char *output_dir);
  void writeResult(VIDEO_FRAME_INFO_S *label_frame, const int index);
  void getImage(int index, std::string &image_path);
  void getImageNum(uint32_t *num);

 private:
  std::string m_output_dir;
  std::vector<std::string> m_images;
};
}  // namespace evaluation
}  // namespace cviai