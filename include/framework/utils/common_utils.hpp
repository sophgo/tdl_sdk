#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include "common/common_types.hpp"

class CommonUtils {
 public:
  static uint32_t getDataTypeSize(TDLDataType data_type);

  static InferencePlatform getPlatform();

  static bool readBinaryFile(const std::string &strf,
                             std::vector<uint8_t> &buffer);
  static std::string getLibraryPath();
  static std::string getLibraryDir();
  static std::string getParentDir(const std::string &path);
  static std::string getExecutableDir();
  static int32_t randomFill(uint8_t *data, uint32_t size);
};

#endif  // COMMON_UTILS_H