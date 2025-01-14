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
