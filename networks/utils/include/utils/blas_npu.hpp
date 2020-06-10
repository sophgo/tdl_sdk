// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file blas_npu.hpp
 * @ingroup NETWORKS_BLAS
 */
#pragma once

#include <cstdint>
#include <vector>

namespace qnn {
namespace math {

class NBlas {
    public:
    static int Dot(std::vector<int8_t>& a, std::vector<int8_t>& b);
};

}  // namespace math
}  // namespace qnn
