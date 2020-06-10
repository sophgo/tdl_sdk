/**
 * @file proposal_layer_nocaffe.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

struct param_pro_t {
    int feat_stride;
    int base_size;
    int scales[2];
    float ratios;
};

class ProposalLayer final {
    public:
    // std::multimap<float,std::vector<float>,std::greater<float>> _result;
    std::vector<std::pair<float, int>> _order;
    std::unique_ptr<float[]> _datain;
    // std::vector<std::pair<float,std::vector<float>>> _result;
    private:
    int _feat_stride;
    float _anchor_ratios;
    float _anchor_scales[2];
    size_t _base_size;
    int _im_info[2];
    int _num_anchors;
    bool _trans_anchors_enable;
    std::unique_ptr<float[]> _mod_anchors_ptr;
    std::string _cfg_key;
    int _pre_nms_topN;

    public:
    ProposalLayer();
    ProposalLayer(std::string score_layer_name, std::string box_layer_name);
    ~ProposalLayer() {}
    void forward(float *input_score, float *input_bbox, const int *shape);
    void setup(const param_pro_t &param_str, int image_width, int image_height);

    std::string score_layer_name;
    std::string box_layer_name;

    private:
    void bbox_transform_inv(float *anchors, float *boxes, int length);
    void clip_boxes(float *datain, int length);
    void filter_boxes(int min_size);
    void Swap(std::vector<std::pair<float, int>> &A, int i, int j);
    void Heapify(std::vector<std::pair<float, int>> &A, int i, int size);
    int BuildHeap(std::vector<std::pair<float, int>> &A, int n);
    void HeapSort(std::vector<std::pair<float, int>> &A, int n);
    void softmax(const char *src, float *dst, int length, float scale);
};

}  // namespace vision
}  // namespace qnn
