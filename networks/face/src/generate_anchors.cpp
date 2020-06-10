#include "generate_anchors.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdio.h>

namespace qnn {
namespace vision {

/* Return width, height, x center, and y center for an anchor (window).*/
void whctrs(const float *anchor, float &w, float &h, float &x_ctr, float &y_ctr) {
    w = anchor[2] - anchor[0] + 1;
    h = anchor[3] - anchor[1] + 1;
    x_ctr = anchor[0] + 0.5 * (w - 1);
    y_ctr = anchor[1] + 0.5 * (h - 1);

    return;
}

/* Given a vector of widths (ws) and heights (hs) around a center
   (x_ctr, y_ctr), output a set of anchors (windows).*/
void mkanchors(float x_ctr, float y_ctr, const float *ws, const float *hs, float *anchors,
               int size) {
    for (int i = 0; i < size; i++) {
        *(anchors + i * ANCHOR_NUM) = x_ctr - 0.5 * (ws[i] - 1);
        *(anchors + i * ANCHOR_NUM + 1) = y_ctr - 0.5 * (hs[i] - 1);
        *(anchors + i * ANCHOR_NUM + 2) = x_ctr + 0.5 * (ws[i] - 1);
        *(anchors + i * ANCHOR_NUM + 3) = y_ctr + 0.5 * (hs[i] - 1);
    }
    return;
}

/* Enumerate a set of anchors for each aspect ratio wrt an anchor. */
void ratio_enum(const float *anchor, const float *ratios, int ratio_size, float *output) {
    float w, h;
    float x_ctr, y_ctr;
    std::unique_ptr<float[]> ws(new float[ratio_size]);
    std::unique_ptr<float[]> hs(new float[ratio_size]);
    whctrs(anchor, w, h, x_ctr, y_ctr);

    int size = w * h;
    for (int i = 0; i < ratio_size; i++) {
        float size_ratios = size / ratios[i];
        ws[i] = std::round(std::sqrt(size_ratios));
        hs[i] = std::round(ws[i] * ratios[i]);
    }
    mkanchors(x_ctr, y_ctr, ws.get(), hs.get(), output, ratio_size);
    return;
}

/* Enumerate a set of anchors for each scale wrt an anchor. */
void scale_enum(const float *anchor, const float *scales, int scales_size, float *output) {
    float w, h;
    float x_ctr, y_ctr;
    std::unique_ptr<float[]> ws(new float[scales_size]);
    std::unique_ptr<float[]> hs(new float[scales_size]);
    whctrs(anchor, w, h, x_ctr, y_ctr);

    for (int i = 0; i < scales_size; i++) {
        ws[i] = w * scales[i];
        hs[i] = h * scales[i];
    }
    mkanchors(x_ctr, y_ctr, ws.get(), hs.get(), output, scales_size);
    return;
}

void generate_anchors(const int base_size, const float *ratios, int ratio_size, const float *scales,
                      int scales_size, float *output) {
    float base_anchor[ANCHOR_NUM] = {0.0, 0.0, (float)(base_size - 1), (float)(base_size - 1)};
    std::unique_ptr<float[]> rat_num(new float[ratio_size * ANCHOR_NUM]);
    ratio_enum(base_anchor, ratios, ratio_size, rat_num.get());

    for (int i = 0; i < ratio_size; i++) {
        float *out = output + i;
        const float *in = &rat_num[i * ANCHOR_NUM];
        scale_enum(in, scales, scales_size, out);
    }
    return;
}

}  // namespace vision
}  // namespace qnn
