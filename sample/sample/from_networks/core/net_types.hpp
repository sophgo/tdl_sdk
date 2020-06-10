// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file net_types.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once

namespace qnn {
/**
 * @brief Net return error status.
 *
 */
typedef enum {
    RET_SUCCESS = 0,
    RET_UNSUPPORTED_SHAPE,
    RET_UNSUPPORTED_RESOLUTION,
    RET_INFERENCE_ERROR,
    RET_GET_OUTPUT_INFO_ERROR,
    RET_INVALID_HANDLE,
    RET_UNKNOW_ERROR,
} net_err_t;

}  // namespace qnn