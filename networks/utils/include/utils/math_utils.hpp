// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>
/**
 * @file math_utils.hpp
 * @ingroup NETWORKS_UTILS
 */
#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>

namespace qnn {
namespace math {

inline float FastExp(float x) {
    union {
        unsigned int i;
        float f;
    } v;

    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);

    return v.f;
}

void SoftMaxForBuffer(float *src, float *dst, size_t size);
int CRC16(unsigned char *Buf, int len);
void SoftMax(const char *src, float *dst, int num, int channel, int spatial_size, float scale);
void SoftMax(const float *src, float *dst, int num, int channel, int spatial_size,
             bool fast_exp = false);
float VectorDot(const std::vector<float> &vec1, const std::vector<float> &vec2);
float VectorNorm(const std::vector<float> &vec);
float VectorCosine(const std::vector<float> &vec1, const std::vector<float> &vec2);
float VectorEuclidean(const std::vector<float> &vec1, const std::vector<float> &vec2);

}  // namespace math
}  // namespace qnn
