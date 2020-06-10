// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file blas_cpu.hpp
 * @ingroup NETWORKS_BLAS
 */
#pragma once

#include <vector>

namespace qnn {
namespace math {

class CBlas {
    public:
    static float Dot(std::vector<float>& a, std::vector<float>& b);
};

}  // namespace math
}  // namespace qnn
