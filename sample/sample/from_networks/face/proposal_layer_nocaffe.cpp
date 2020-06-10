#include "face/proposal_layer_nocaffe.hpp"
#include "face/generate_anchors.hpp"
#include "utils/log_common.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>

#define NUM_TOP_NUMBER 100

namespace qnn {
namespace vision {

using std::string;
using std::vector;

ProposalLayer::ProposalLayer() {
    _base_size = 16;
    _trans_anchors_enable = false;
    return;
}

ProposalLayer::ProposalLayer(std::string score_layer_name, std::string box_layer_name)
    : score_layer_name(score_layer_name), box_layer_name(box_layer_name) {
    _base_size = 16;
    _trans_anchors_enable = false;
    return;
}

// heap sort
void ProposalLayer::Swap(std::vector<std::pair<float, int>> &A, int i, int j) {
    float tmp_f = A[i].first;
    int tmp_i = A[i].second;
    A[i].first = A[j].first;
    A[i].second = A[j].second;
    A[j].first = tmp_f;
    A[j].second = tmp_i;
}

void ProposalLayer::Heapify(std::vector<std::pair<float, int>> &A, int i, int size) {
    int left_child = 2 * i + 1;
    int right_child = 2 * i + 2;
    int max = i;
    if (left_child < size && (A[left_child].first > A[max].first)) max = left_child;
    if (right_child < size && (A[right_child].first > A[max].first)) max = right_child;
    if (max != i) {
        Swap(A, i, max);
        Heapify(A, max, size);
    }
}

int ProposalLayer::BuildHeap(std::vector<std::pair<float, int>> &A, int n) {
    int heap_size = n;
    for (int i = heap_size / 2 - 1; i >= 0; i--) Heapify(A, i, heap_size);
    return heap_size;
}

void ProposalLayer::HeapSort(std::vector<std::pair<float, int>> &A, int n) {
    int cnt = 0;
    int heap_size = BuildHeap(A, n);
    _order.clear();
    while (heap_size > 1) {
        if (cnt >= _pre_nms_topN) break;
        _order.emplace_back(A[0]);
        Swap(A, 0, --heap_size);
        Heapify(A, 0, heap_size);
        cnt++;
    }
}

void ProposalLayer::forward(float *input_score, float *input_bbox, const int *shape) {
    // cfg
    _cfg_key = "TEST";
    _pre_nms_topN = std::min(NUM_TOP_NUMBER, shape[2] * shape[3]);
    //_pre_nms_topN = NUM_TOP_NUMBER;
    // int min_size = 0;//cfg[cfg_key].ANCHOR_MIN_SIZE;

    if (!_trans_anchors_enable) {
        LOGD << "error do not finish setup!";
        return;
    }

    if ((NULL == _mod_anchors_ptr) || (NULL == input_score) || (NULL == input_bbox) ||
        (NULL == shape)) {
        LOGD << "anchors is empty!";
        return;
    }

    int length = shape[6] * shape[7];
    int offset = _num_anchors * length;
    float *score_offset = input_score + offset;

    _datain = std::unique_ptr<float[]>(new float[8 * length]);
    float *datain_ptr = _datain.get();

    std::vector<std::pair<float, int>> score_v;
    /*
      bbox deltas :(1, H, W, 4 * A) => (1 * H * W * A, 4)
      scores : (1, H, W, A) => (1 * H * W * A, 1)
    */
    // change data
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < 8; j++) {
            *(datain_ptr + i * shape[5] + j) = *(input_bbox + j * length + i);
        }
    }

    for (int i = 0; i < length; i++) {
        score_v.emplace_back(std::make_pair(*(score_offset + 0 * length + i), 2 * i));
        score_v.emplace_back(std::make_pair(*(score_offset + 1 * length + i), 2 * i + 1));
    }

    // heap sort
    HeapSort(score_v, length * 2);
    // Convert anchors into proposals via bbox transformations
    bbox_transform_inv(_mod_anchors_ptr.get(), datain_ptr, length * 2);

    // clip predicted boxes to image
    clip_boxes(datain_ptr, length * 2);

    // remove predicted boxes with either height or width < threshold
    // filter_boxes(min_size*1);
    return;
}

void ProposalLayer::bbox_transform_inv(float *anchors, float *boxes, int length) {
    float *p_anchors = anchors;
    float *p_boxes = boxes;

    for (int i = 0; i < _pre_nms_topN; i++) {
        p_anchors = anchors + (_order[i].second) * 4;
        p_boxes = boxes + (_order[i].second) * 4;

        float widths = p_anchors[2] - p_anchors[0] + 1.0;
        float heights = p_anchors[3] - p_anchors[1] + 1.0;
        float ctr_x = p_anchors[0] + 0.5 * widths;
        float ctr_y = p_anchors[1] + 0.5 * heights;

        float dx = p_boxes[0];
        float dy = p_boxes[1];
        float dw = p_boxes[2];
        float dh = p_boxes[3];

        float pred_ctr_x = dx * widths + ctr_x;
        float pred_ctr_y = dy * heights + ctr_y;

        // exception
        if (dw > 50) {
            dw = 50;
        }
        if (dh > 50) {
            dh = 5;
        }
        float pred_w = std::exp(dw) * widths;
        float pred_h = std::exp(dh) * heights;
        p_boxes[0] = pred_ctr_x - 0.5 * pred_w;  // x1
        p_boxes[1] = pred_ctr_y - 0.5 * pred_h;  // y1
        p_boxes[2] = pred_ctr_x + 0.5 * pred_w;  // x2
        p_boxes[3] = pred_ctr_y + 0.5 * pred_h;  // y2
    }

    return;
}

void ProposalLayer::clip_boxes(float *datain, int length) {
    float *p = datain;
    for (int i = 0; i < _pre_nms_topN; i++) {
        p = datain + (_order[i].second) * 4;
        p[0] = std::max(std::min(p[0], (float)(_im_info[1] - 1)), (float)0.0);  // x1 >= 0
        p[1] = std::max(std::min(p[1], (float)(_im_info[0] - 1)), (float)0.0);  // y1 >= 0
        p[2] = std::max(std::min(p[2], (float)(_im_info[1] - 1)), (float)0.0);  // x2 < im_shape[1]
        p[3] = std::max(std::min(p[3], (float)(_im_info[0] - 1)), (float)0.0);  // y2 < im_shape[0]
    }
    return;
}

void ProposalLayer::setup(const param_pro_t &param_str, int image_width, int image_height) {
    _feat_stride = param_str.feat_stride;
    _anchor_ratios = param_str.ratios;
    _num_anchors = 2;
    _anchor_scales[0] = param_str.scales[0];
    _anchor_scales[1] = param_str.scales[1];
    _base_size = param_str.base_size;
    _im_info[0] = image_height;
    _im_info[1] = image_width;

    // alloc memory
    int height = std::ceil(float(_im_info[0]) / float(_feat_stride));
    int width = std::ceil(float(_im_info[1]) / float(_feat_stride));
    std::unique_ptr<float[]> anchors_ptr(new float[_num_anchors * 4]);
    std::unique_ptr<float[]> shift_data_x(new float[width]);
    std::unique_ptr<float[]> shift_data_y(new float[height]);
    std::unique_ptr<float[]> data_shift(new float[4 * height * width]);
    _mod_anchors_ptr = std::unique_ptr<float[]>(new float[height * width * _num_anchors * 4]);

    // ratios size : 1 , scales size : 2
    generate_anchors(_base_size, reinterpret_cast<const float *>(&_anchor_ratios), 1,
                     _anchor_scales, 2, anchors_ptr.get());

    // Generate proposals from bbox deltas and shifted anchors
    for (int i = 0; i < width; i++) {
        shift_data_x[i] = i * _feat_stride;
    }
    for (int i = 0; i < height; i++) {
        shift_data_y[i] = i * _feat_stride;
    }

    // meshgrid
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data_shift[4 * i * width + 4 * j] = shift_data_x[j];
            data_shift[4 * i * width + 4 * j + 1] = shift_data_y[i];
            data_shift[4 * i * width + 4 * j + 2] = shift_data_x[j];
            data_shift[4 * i * width + 4 * j + 3] = shift_data_y[i];
        }
    }

    // (1, A, 4) + (K, A, 4) => (K*A, 4)
    int leng_shift = height * width;
    for (int i = 0; i < leng_shift; i++) {
        for (int j = 0; j < _num_anchors; j++) {
            int mod_anchors_idx = (i * _num_anchors + j) * 4;
            int anchors_idx = j * 4;
            int data_idx = i * 4;
            _mod_anchors_ptr[mod_anchors_idx] = anchors_ptr[anchors_idx] + data_shift[data_idx];
            ++mod_anchors_idx;
            ++anchors_idx;
            ++data_idx;

            _mod_anchors_ptr[mod_anchors_idx] = anchors_ptr[anchors_idx] + data_shift[data_idx];
            ++mod_anchors_idx;
            ++anchors_idx;
            ++data_idx;

            _mod_anchors_ptr[mod_anchors_idx] = anchors_ptr[anchors_idx] + data_shift[data_idx];
            ++mod_anchors_idx;
            ++anchors_idx;
            ++data_idx;

            _mod_anchors_ptr[mod_anchors_idx] = anchors_ptr[anchors_idx] + data_shift[data_idx];
            ++mod_anchors_idx;
            ++anchors_idx;
            ++data_idx;
        }
    }
    _trans_anchors_enable = true;
    return;
}

}  // namespace vision
}  // namespace qnn
