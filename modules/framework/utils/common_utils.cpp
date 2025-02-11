#include "utils/common_utils.hpp"

uint32_t get_data_type_size(ImagePixDataType data_type) {
  switch (data_type) {
    case ImagePixDataType::FP32:
      return 4;
    case ImagePixDataType::INT32:
      return 4;
    case ImagePixDataType::UINT32:
      return 4;
    case ImagePixDataType::FP16:
      return 2;
    case ImagePixDataType::BF16:
      return 2;
    case ImagePixDataType::INT16:
      return 2;
    case ImagePixDataType::UINT16:
      return 2;
    case ImagePixDataType::INT8:
      return 1;
    case ImagePixDataType::UINT8:
      return 1;
    default:
      return 0;
  }
}

InferencePlatform get_platform() {
#if defined(__BM168X__)
  return InferencePlatform::BM168X;
#elif defined(__CV186X__)
  return InferencePlatform::CV186X;
#elif defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__)
  return InferencePlatform::CVITEK;
#elif defined(__CMODEL__)
  return InferencePlatform::CMODEL;
#else
  return InferencePlatform::UNKOWN;
#endif
}
