#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <vector>
#include "common/common_types.hpp"

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
};

#endif  // COMMON_UTILS_H