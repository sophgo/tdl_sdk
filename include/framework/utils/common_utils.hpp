#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
#include "common/common_types.hpp"
#include "common/model_output_types.hpp"

class CommonUtils {
 public:
  static uint32_t getDataTypeSize(TDLDataType data_type);

  static InferencePlatform getPlatform();

  static bool readBinaryFile(const std::string& strf,
                             std::vector<uint8_t>& buffer);
  static std::string getLibraryPath();
  static std::string getLibraryDir();
  static std::string getParentDir(const std::string& path);
  static std::string getExecutableDir();
  static int32_t randomFill(uint8_t* data, uint32_t size);
  static float dot_product(const std::vector<float>& a,
                           const std::vector<float>& b);
  static void normalize(std::vector<float>& v);
  static std::vector<float> softmax(const std::vector<float>& logits);

  static void visualizeMask(std::shared_ptr<ModelBoxSegmentationInfo> obj_meta,
                            const std::string& str_img_name);

  static std::pair<int, int> findMaskOrSmallestBbox(
      std::shared_ptr<ModelBoxSegmentationInfo> obj_meta, uint32_t image_height,
      uint32_t image_width, cv::Point point);
};

#endif  // COMMON_UTILS_H