/**
 * @file generate_anchors.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once

#define RATE_SIZE 2
#define ANCHOR_NUM 4

namespace qnn {
namespace vision {

extern void whctrs(const float *anchor, float &w, float &h, float &x_ctr, float &y_ctr);
extern void mkanchors(float x_ctr, float y_ctr, const float *ws, const float *hs, float *anchors,
                      int size);
extern void ratio_enum(const float *anchor, const float *ratios, int ratio_size, float *output);
extern void scale_enum(const float *anchor, const float *scales, int scales_size, float *output);

extern void generate_anchors(const int base_size, const float *ratios, int ratio_size,
                             const float *scales, int scales_size, float *output);

}  // namespace vision
}  // namespace qnn