// Copyright 2018 Bitmain Inc.
// License
// Author
#include "ssh.hpp"

#include "mtcnn_helperfunc.hpp"
#include "utils/math_utils.hpp"
#include "utils/function_tracer.h"
#include <cassert>
#include <cmath>

namespace qnn {
namespace vision {

#define SSH_B_MEAN (-102.9801)
#define SSH_G_MEAN (-115.9465)
#define SSH_R_MEAN (-122.7717)
#define SSH_C 3

#define SSH_SCALE_FACTOR float(128) / 152.094
#define MAX_BATCH (8)
#define DEFAULT_POSTSIZE (12800 * MAX_BATCH)

using namespace qnn::math;
using cv::Mat;
using std::string;
using std::vector;

static int output_90_90_shape[6][4] = {{1, 4, 3, 3}, {1, 8, 3, 3},   {1, 4, 6, 6},
                                       {1, 8, 6, 6}, {1, 4, 12, 12}, {1, 8, 12, 12}};
static int output_160_160_shape[6][4] = {{1, 4, 5, 5},   {1, 8, 5, 5},   {1, 4, 10, 10},
                                         {1, 8, 10, 10}, {1, 4, 20, 20}, {1, 8, 20, 20}};
static int output_320_320_shape[6][4] = {{1, 4, 10, 10}, {1, 8, 10, 10}, {1, 4, 20, 20},
                                         {1, 8, 20, 20}, {1, 4, 40, 40}, {1, 8, 40, 40}};

static const vector<param_pro_t> tinyssh_param = {
    {32, 4, {32, 64}, 1.0}, {16, 4, {8, 16}, 1.0}, {8, 4, {2, 4}, 1.0}};
static const vector<param_pro_t> ssh_param = {
    {32, 16, {16, 32}, 1.0}, {16, 16, {4, 8}, 1.0}, {8, 16, {1, 2}, 1.0}};

SSH::SSH(const string &model_path, const SSH_TYPE &type, int input_width, int input_height,
         PADDING_POLICY padding, QNNCtx *qnn_ctx)
    : ImageNet(model_path, {NetShape(1, 3, input_width, input_height)}, SSH_C, input_width,
               input_height, true, cv::INTER_LINEAR, qnn_ctx),
      post_probs_buf(new float[DEFAULT_POSTSIZE]),
      padding_policy(padding),
      m_threshold(0.5) {
    vector<param_pro_t> param_pro = (type == SSH_TYPE::ORIGIN) ? ssh_param : tinyssh_param;

    m_input_width = input_width;
    m_input_height = input_height;

    proposal_layers.reserve(3);
    proposal_layers.emplace_back(
        new ProposalLayer("m3@ssh_cls_score_output", "m3@ssh_bbox_pred_output"));
    proposal_layers.back()->setup(param_pro[0], input_width, input_height);
    proposal_layers.emplace_back(
        new ProposalLayer("m2@ssh_cls_score_output", "m2@ssh_bbox_pred_output"));
    proposal_layers.back()->setup(param_pro[1], input_width, input_height);
    proposal_layers.emplace_back(
        new ProposalLayer("m2@ssh_cls_score_output", "m1@ssh_bbox_pred_output"));
    proposal_layers.back()->setup(param_pro[2], input_width, input_height);
    SetQuanParams({std::vector<float>{SSH_B_MEAN, SSH_G_MEAN, SSH_R_MEAN},
                   std::vector<float>{SSH_SCALE_FACTOR}});
}

void SSH::PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) {
    ratio = 1.0;

    if (padding_policy == PADDING_POLICY::UP_LEFT) {
        ratio = ResizeImage(image, prepared, net_width, net_height, m_ibuf, m_resize_policy,
                            preserve_ratio);
    } else if (padding_policy == PADDING_POLICY::CENTER) {
        ratio = ResizeImageCenter(image, prepared, net_width, net_height, m_ibuf, m_resize_policy,
                                  preserve_ratio);
    }
}

void SSH::Detect(const Mat &image, vector<face_detect_rect_t> &results) {
    if (image.empty()) {
        assert(false);
        return;
    }
    vector<Mat> images = {image};
    vector<vector<face_detect_rect_t>> all_results;
    Detect(images, all_results);
    results.swap(all_results.at(0));
}

void SSH::Detect(const vector<Mat> &images, vector<vector<face_detect_rect_t>> &results) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (images.empty()) {
        assert(false);
        return;
    }

    vector<float> ratios;
    results.clear();

    vector<face_detect_rect_t> boxes;

    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= int(images.size()));
        for (int i = start; i < end; ++i) {
            do_proposal(out, boxes);
            vector<face_detect_rect_t> res;
            for (face_detect_rect_t bbox : boxes) {
                bbox.id = i;
                if (padding_policy == PADDING_POLICY::CENTER) {
                    // padding in the center, need to shift back the padding w/ scaling ratio
                    int pad_left = int((net_width - images[i].cols / ratios[i]) / 2);
                    int pad_top = int((net_height - images[i].rows / ratios[i]) / 2);
                    if (pad_left || pad_top) {
                        bbox.x1 -= pad_left;
                        bbox.x2 -= pad_left;
                        bbox.y1 -= pad_top;
                        bbox.y2 -= pad_top;
                    }
                }
                bbox.x1 *= ratios[i];
                bbox.y1 *= ratios[i];
                bbox.x2 *= ratios[i];
                bbox.y2 *= ratios[i];
                if (std::abs(bbox.x2 - bbox.x1) < MIN_FACE_WIDTH) {
                    continue;
                }
                if (std::abs(bbox.y2 - bbox.y1) < MIN_FACE_HEIGHT) {
                    continue;
                }
                res.emplace_back(bbox);
            }
            dump_faceinfo(res);
            results.emplace_back(res);
        }
    });
}

void SSH::SetThreshold(float threshold) { m_threshold = threshold; }

void SSH::do_proposal(OutTensors &out_tensors, vector<face_detect_rect_t> &boxes) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    float *scores[3];
    int *output_shape = get_output_shape();

    scores[0] = post_probs_buf.get();
    scores[1] = scores[0] + out_tensors["m3@ssh_cls_score_output"].count;
    scores[2] = scores[1] + out_tensors["m2@ssh_cls_score_output"].count;

    SoftMax(out_tensors["m3@ssh_cls_score_output"].data, scores[0], 1, 2,
            *(output_shape + 2) * 2 * *(output_shape + 3));
    SoftMax(out_tensors["m2@ssh_cls_score_output"].data, scores[1], 1, 2,
            *(output_shape + 10) * 2 * *(output_shape + 11));
    SoftMax(out_tensors["m1@ssh_cls_score_output"].data, scores[2], 1, 2,
            *(output_shape + 18) * 2 * *(output_shape + 19));

    vector<size_t> layer_sizes(1, 0);
    for (size_t i = 0; i < proposal_layers.size(); i++) {
        LOGD << "Handle proposal " << i;
        std::string bbox_layer_name = proposal_layers[i]->box_layer_name;
        proposal_layers[i]->forward(scores[i], out_tensors[bbox_layer_name].data,
                                    output_shape + i * 2 * 4);
        layer_sizes.emplace_back(
            layer_sizes.back() +
            std::min(*(output_shape + i * 2 * 4 + 2) * *(output_shape + i * 2 * 4 + 3), 100));
    }
    // handle proposal boxs && scores
    handle_proposal(boxes, layer_sizes);
}

// handle proposal boxs && scores
void SSH::handle_proposal(vector<face_detect_rect_t> &boxes, vector<size_t> &layer_sizes) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    assert(proposal_layers.size() == 3);

    float threshold = 0.05;
    std::vector<face_detect_rect_t> BBoxes;
    float *data_boxes = NULL;
    std::vector<std::pair<float, int>>::iterator iter_order;
    size_t proposal_idx = 0;
    for (size_t i = 0; i < layer_sizes.back(); i++) {
        if (layer_sizes[proposal_idx] == i) {
            iter_order = proposal_layers[proposal_idx]->_order.begin();
            data_boxes = proposal_layers[proposal_idx]->_datain.get();
            proposal_idx++;
        }

        float *p = data_boxes + (iter_order->second) * 4;
        if (iter_order->first > threshold) {
            face_detect_rect_t tmp;
            tmp.x1 = p[0];
            tmp.y1 = p[1];
            tmp.x2 = p[2];
            tmp.y2 = p[3];
            tmp.score = iter_order->first;
            tmp.id = 0;
            BBoxes.emplace_back(tmp);
        }
        ++iter_order;
    }
    proposal_layers[0]->_order.clear();
    proposal_layers[1]->_order.clear();
    proposal_layers[2]->_order.clear();

    // nms
    std::vector<face_detect_rect_t> bboxes_nms;
    NonMaximumSuppression(BBoxes, bboxes_nms, 0.3, 'u');

    // inds = np.where(cls_dets[:, -1] >= 0.5)[0]
    boxes.clear();
    for (size_t i = 0; i < bboxes_nms.size(); i++) {
        if (bboxes_nms[i].score > m_threshold) {
            boxes.emplace_back(bboxes_nms[i]);
        }
    }

    return;
}

int *SSH::get_output_shape() {
    if (m_input_width == 320 && m_input_height == 320) {
        return &output_320_320_shape[0][0];
    } else if (m_input_width == 160 && m_input_height == 160) {
        return &output_160_160_shape[0][0];
    } else if (m_input_width == 90 && m_input_height == 90) {
        return &output_90_90_shape[0][0];
    } else {
        assert(false);
        return NULL;
    }
}

}  // namespace vision
}  // namespace qnn
