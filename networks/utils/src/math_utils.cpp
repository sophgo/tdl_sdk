// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>

#include "math_utils.hpp"
#include <algorithm>
#include <cmath>

namespace qnn {
namespace math {

using std::vector;

void SoftMaxForBuffer(float *src, float *dst, size_t size) {
    float sum = 0;

    const float max = *std::max_element(src, src + size);

    for (size_t i = 0; i < size; i++) {
        dst[i] = std::exp(src[i] - max);
        sum += dst[i];
    }

    for (size_t i = 0; i < size; i++) {
        dst[i] /= sum;
    }
}

int CRC16(unsigned char *Buf, int len) {
    int CRC;
    int i, j;
    CRC = 0xffff;
    for (i = 0; i < len; i++) {
        CRC = CRC ^ (Buf[i] & 0xff);
        for (j = 0; j < 8; j++) {
            if ((CRC & 0x01) == 1)
                CRC = (CRC >> 1) ^ 0xA001;
            else
                CRC = CRC >> 1;
        }
    }
    return CRC;
}

void SoftMax(const char *src, float *dst, int num, int channel, int spatial_size, float scale) {
    int inner_num = spatial_size;
    int dim = channel * spatial_size;
    int totalsize = inner_num * channel;

    char max[totalsize];

    for (int o = 0; o < num; o++) {
        memset(max, 0, totalsize * sizeof(char));

        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < inner_num; k++)
                max[k] = std::max(max[k], src[o * dim + j * inner_num + k]);
        }

        for (int i = 0; i < inner_num; i++) {
            float denominator = 0;
            for (int j = 0; j < channel; j++) {
                dst[j * inner_num + i] =
                    std::exp((src[i + j * inner_num + o * dim] - max[i]) * scale);
                denominator += dst[i + j * inner_num];
            }

            for (int j = 0; j < channel; j++) {
                dst[i + j * inner_num] /= denominator;
            }
        }
    }
}

void SoftMax(const float *src, float *dst, int num, int channel, int spatial_size, bool fast_exp) {
    int inner_num = spatial_size;
    int dim = channel * spatial_size;
    int totalsize = inner_num * channel;

    float max[totalsize];

    for (int o = 0; o < num; o++) {
        memset(max, 0, totalsize * sizeof(float));

        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < inner_num; k++)
                max[k] = std::max(max[k], src[o * dim + j * inner_num + k]);
        }

        for (int i = 0; i < inner_num; i++) {
            float denominator = 0;
            for (int j = 0; j < channel; j++) {
                if (!fast_exp) {
                    dst[j * inner_num + i] = std::exp((src[i + j * inner_num + o * dim] - max[i]));
                } else {
                    dst[j * inner_num + i] = FastExp((src[i + j * inner_num + o * dim] - max[i]));
                }
                denominator += dst[i + j * inner_num];
            }

            for (int j = 0; j < channel; j++) {
                dst[i + j * inner_num] /= denominator;
            }
        }
    }
}

float VectorDot(const vector<float> &vec1, const vector<float> &vec2) {
    float a = 0;
    for (size_t i = 0; i < vec1.size(); i++) {
        a += vec1[i] * vec2[i];
    }
    return a;
}

// return the Euclidean norm of a vector
float VectorNorm(const vector<float> &vec) {
    float a = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        a += vec[i] * vec[i];
    }
    return sqrt(a);
}

// return the cosine(theta) of two vector
float VectorCosine(const vector<float> &vec1, const vector<float> &vec2) {
    float a = 0;
    for (size_t i = 0; i < vec1.size(); i++) {
        a += vec1[i] * vec2[i];
    }
    float b = VectorNorm(vec1) * VectorNorm(vec2);
    return a / (b + 0.0000001);
}

float VectorEuclidean(const vector<float> &vec1, const vector<float> &vec2) {
    float a = 0;
    for (size_t i = 0; i < vec1.size(); i++) {
        a += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return -sqrt(a);
}

}  // namespace math
}  // namespace qnn
