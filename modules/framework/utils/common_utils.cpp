#include "utils/common_utils.hpp"

uint32_t get_data_type_size(TDLDataType data_type) {
  switch (data_type) {
    case TDLDataType::FP32:
      return 4;
    case TDLDataType::INT32:
      return 4;
    case TDLDataType::UINT32:
      return 4;
    case TDLDataType::FP16:
      return 2;
    case TDLDataType::BF16:
      return 2;
    case TDLDataType::INT16:
      return 2;
    case TDLDataType::UINT16:
      return 2;
    case TDLDataType::INT8:
      return 1;
    case TDLDataType::UINT8:
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
