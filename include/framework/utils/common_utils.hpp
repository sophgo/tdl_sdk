#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include "common/common_types.hpp"

uint32_t get_data_type_size(TDLDataType data_type);

InferencePlatform get_platform();

bool read_binary_file(const std::string &strf, void *p_buffer, int buffer_len);

#endif  // COMMON_UTILS_H