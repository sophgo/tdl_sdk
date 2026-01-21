#include "object_detection/yolo26.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
inline void parse_cls_info(T *p_cls_ptr, int num_anchor, int num_cls,
                           int anchor_idx, int cls_offset, float qscale,
                           float *p_max_logit, int *p_max_cls) {
  int max_logit_c = -1;
  float max_logit = -1000;
  for (int c = 0; c < num_cls; c++) {
    float logit = p_cls_ptr[(c + cls_offset) * num_anchor + anchor_idx];
    if (logit > max_logit) {
      max_logit = logit;
      max_logit_c = c;
    }
  }
  *p_max_logit = max_logit * qscale;
  *p_max_cls = max_logit_c;
}

template <typename T>
inline void get_box_vals(T *p_box_ptr, int num_anchor, int anchor_idx,
                         int num_box_channel, float qscale, float *out_vals) {
  for (int c = 0; c < num_box_channel; c++) {
    out_vals[c] = p_box_ptr[c * num_anchor + anchor_idx] * qscale;
  }
}

template <typename T>
inline void get_cls_scores(T *p_cls_ptr, int num_anchor, int num_cls,
                           int anchor_idx, int cls_offset, float qscale,
                           float *out_scores) {
  for (int c = 0; c < num_cls; c++) {
    float logit =
        p_cls_ptr[(c + cls_offset) * num_anchor + anchor_idx] * qscale;
    // Sigmoid 归一化 - 对应 inferonnx.py: final_cls = 1 / (1 +
    // np.exp(-final_cls))
    out_scores[c] = 1.0f / (1.0f + std::exp(-logit));
  }
}

Yolo26Detection::Yolo26Detection(const int num_cls)
    : Yolo26Detection(std::make_pair(4, num_cls)) {}

Yolo26Detection::Yolo26Detection(std::pair<int, int> yolov8_pair) {
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {254.97195, 254.97195, 254.97195};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;

  num_box_channel_ = yolov8_pair.first;
  num_cls_ = yolov8_pair.second;
  if (num_box_channel_ == num_cls_) {
    LOGE("error,num_box_channel_(%d) == num_cls_(%d)", num_box_channel_,
         num_cls_);
    throw std::runtime_error("num_box_channel_ == num_cls_");
  }
}

Yolo26Detection::~Yolo26Detection() {}

void Yolo26Detection::makeAnchors(int input_h, int input_w) {
  anchor_points_x_.clear();
  anchor_points_y_.clear();
  stride_tensor_.clear();
  stride_anchor_start_.clear();
  stride_anchor_count_.clear();

  const float grid_cell_offset = 0.5f;
  int anchor_offset = 0;

  for (size_t i = 0; i < strides.size(); i++) {
    int stride = strides[i];
    int feat_h = input_h / stride;
    int feat_w = input_w / stride;

    stride_anchor_start_[stride] = anchor_offset;
    stride_anchor_count_[stride] = feat_h * feat_w;

    // 生成网格 - 对应 meshgrid(sy, sx, indexing='ij')
    for (int y = 0; y < feat_h; y++) {
      for (int x = 0; x < feat_w; x++) {
        anchor_points_x_.push_back(x + grid_cell_offset);
        anchor_points_y_.push_back(y + grid_cell_offset);
        stride_tensor_.push_back(static_cast<float>(stride));
      }
    }
    anchor_offset += feat_h * feat_w;
  }
  total_anchors_ = anchor_offset;
  LOGI("makeAnchors: total_anchors=%d, strides.size=%zu", total_anchors_,
       strides.size());
}

int32_t Yolo26Detection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  auto input_shape = net_->getTensorInfo(input_layer).shape;
  int input_h = input_shape[2];
  int input_w = input_shape[3];
  strides.clear();
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();

  LOGI("to parse output branch,box_channel:%d,num_cls:%d", num_box_channel_,
       num_cls_);
  if (num_cls_ == 0) {
    LOGI(
        "num_cls is 0,would take branches whose channel not equal to "
        "num_box_channel_ as cls branch");
  }

  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);
    int feat_h = oinfo.shape[2];
    int feat_w = oinfo.shape[3];
    int channel = oinfo.shape[1];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    if (stride_h != stride_w) {
      LOGE("stride not equal,stridew:%d,strideh:%d,featw:%d,feath:%d\n",
           stride_w, stride_h, feat_w, feat_h);
      return -1;
    }
    if (num_cls_ == 0) {
      if (channel == num_box_channel_) {
        bbox_out_names[stride_h] = output_layers[j];
        strides.push_back(stride_h);
        LOGI("parse box branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      } else {
        num_cls_ = channel;
        class_out_names[stride_h] = output_layers[j];
        LOGI("parse class branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      }
    } else {
      if (channel == num_box_channel_) {
        bbox_out_names[stride_h] = output_layers[j];
        strides.push_back(stride_h);
        LOGI("parse box branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      } else if (channel == num_cls_) {
        class_out_names[stride_h] = output_layers[j];
        LOGI("parse class branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      } else {
        LOGE("unexpected branch:%s,channel:%d\n", output_layers[j].c_str(),
             channel);
        return -1;
      }
    }
  }

  if (bbox_out_names.size() != class_out_names.size()) {
    LOGE(
        "Unsupported output type: mismatched number of box and class branches. "
        "bbox branches: %zu, class branches: %zu.",
        bbox_out_names.size(), class_out_names.size());
    return -1;
  }

  // 对 strides 排序，确保顺序一致 (小到大: 8, 16, 32)
  std::sort(strides.begin(), strides.end());

  // 生成锚点
  makeAnchors(input_h, input_w);

  return 0;
}

void Yolo26Detection::getAllClassScores(int batch_idx,
                                        std::vector<float> &all_scores) {
  all_scores.resize(total_anchors_ * num_cls_);

  for (size_t i = 0; i < strides.size(); i++) {
    int stride = strides[i];
    std::string cls_name;
    int cls_offset = 0;

    if (class_out_names.count(stride)) {
      cls_name = class_out_names[stride];
    } else {
      LOGE("getAllClassScores: no class branch for stride %d", stride);
    }

    TensorInfo classinfo = net_->getTensorInfo(cls_name);
    std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);

    int num_anchor = classinfo.shape[2] * classinfo.shape[3];
    int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;
    float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;

    int anchor_start = stride_anchor_start_[stride];

    // 遍历每个锚点，获取分类得分
    for (int j = 0; j < num_anchor; j++) {
      float *out_ptr = all_scores.data() + (anchor_start + j) * num_cls_;

      if (classinfo.data_type == TDLDataType::INT8) {
        get_cls_scores<int8_t>(cls_tensor->getBatchPtr<int8_t>(batch_idx),
                               num_anchor, num_cls_, j, cls_offset, cls_qscale,
                               out_ptr);
      } else if (classinfo.data_type == TDLDataType::UINT8) {
        get_cls_scores<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(batch_idx),
                                num_anchor, num_cls_, j, cls_offset, cls_qscale,
                                out_ptr);
      } else if (classinfo.data_type == TDLDataType::FP32) {
        get_cls_scores<float>(cls_tensor->getBatchPtr<float>(batch_idx),
                              num_anchor, num_cls_, j, cls_offset, cls_qscale,
                              out_ptr);
      }
    }
  }
}

void Yolo26Detection::getAllRegValues(int batch_idx,
                                      std::vector<float> &all_reg) {
  all_reg.resize(total_anchors_ * num_box_channel_);

  for (size_t i = 0; i < strides.size(); i++) {
    int stride = strides[i];
    std::string box_name;

    if (bbox_out_names.count(stride)) {
      box_name = bbox_out_names[stride];
    } else {
      LOGE("getAllRegValues: no box branch for stride %d", stride);
    }

    TensorInfo boxinfo = net_->getTensorInfo(box_name);
    std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);

    int num_anchor = boxinfo.shape[2] * boxinfo.shape[3];
    float qscale = boxinfo.qscale;

    int anchor_start = stride_anchor_start_[stride];

    // 遍历每个锚点，获取回归值
    for (int j = 0; j < num_anchor; j++) {
      float *out_ptr = all_reg.data() + (anchor_start + j) * num_box_channel_;

      if (boxinfo.data_type == TDLDataType::INT8) {
        get_box_vals<int8_t>(box_tensor->getBatchPtr<int8_t>(batch_idx),
                             num_anchor, j, num_box_channel_, qscale, out_ptr);
      } else if (boxinfo.data_type == TDLDataType::UINT8) {
        get_box_vals<uint8_t>(box_tensor->getBatchPtr<uint8_t>(batch_idx),
                              num_anchor, j, num_box_channel_, qscale, out_ptr);
      } else if (boxinfo.data_type == TDLDataType::FP32) {
        get_box_vals<float>(box_tensor->getBatchPtr<float>(batch_idx),
                            num_anchor, j, num_box_channel_, qscale, out_ptr);
      }
    }
  }
}

void Yolo26Detection::getTopkIndex(const std::vector<float> &scores,
                                   int num_anchors, int num_cls, int max_det,
                                   std::vector<float> &top_scores,
                                   std::vector<int> &top_cls_idx,
                                   std::vector<int> &top_anchor_idx) {
  // 1) 先用每个 anchor 的最大类别分数挑选 top-k anchors
  // 2) 再在这些 anchors 的全部类别分数 (k * num_cls) 上做 flatten top-k
  top_scores.clear();
  top_cls_idx.clear();
  top_anchor_idx.clear();

  if (num_anchors <= 0 || num_cls <= 0 || max_det <= 0) {
    return;
  }
  if (static_cast<size_t>(num_anchors * num_cls) > scores.size()) {
    LOGE("getTopkIndex: scores size mismatch. expect >= %d, got %zu",
         num_anchors * num_cls, scores.size());
    return;
  }

  const int k = std::min(max_det, num_anchors);

  // ---------- Stage 1: top-k anchors by max class score ----------
  std::vector<std::pair<float, int>> anchor_max_scores;
  anchor_max_scores.reserve(num_anchors);
  for (int i = 0; i < num_anchors; i++) {
    const float *p = scores.data() + i * num_cls;
    float max_score = p[0];
    for (int c = 1; c < num_cls; c++) {
      if (p[c] > max_score) {
        max_score = p[c];
      }
    }
    anchor_max_scores.emplace_back(max_score, i);
  }

  auto score_desc = [](const std::pair<float, int> &a,
                       const std::pair<float, int> &b) {
    return a.first > b.first;
  };

  if (k < num_anchors) {
    std::partial_sort(anchor_max_scores.begin(), anchor_max_scores.begin() + k,
                      anchor_max_scores.end(), score_desc);
    anchor_max_scores.resize(k);
  } else {
    std::sort(anchor_max_scores.begin(), anchor_max_scores.end(), score_desc);
  }

  std::vector<int> ori_index(k);
  for (int i = 0; i < k; i++) {
    ori_index[i] = anchor_max_scores[i].second;
  }

  // ---------- Stage 2: flatten (k * num_cls) then top-k ----------
  // flatten index = (anchor_pos_in_ori_index * num_cls + cls)
  std::vector<std::pair<float, int>> flat_scores;
  flat_scores.reserve(static_cast<size_t>(k) * static_cast<size_t>(num_cls));
  for (int ai = 0; ai < k; ai++) {
    const int anchor_id = ori_index[ai];
    const float *p = scores.data() + anchor_id * num_cls;
    for (int c = 0; c < num_cls; c++) {
      flat_scores.emplace_back(p[c], ai * num_cls + c);
    }
  }

  const int total = static_cast<int>(flat_scores.size());  // k * num_cls
  const int out_k = std::min(k, total);
  if (out_k <= 0) {
    return;
  }

  if (out_k < total) {
    std::partial_sort(flat_scores.begin(), flat_scores.begin() + out_k,
                      flat_scores.end(), score_desc);
    flat_scores.resize(out_k);
  } else {
    std::sort(flat_scores.begin(), flat_scores.end(), score_desc);
  }

  top_scores.resize(out_k);
  top_cls_idx.resize(out_k);
  top_anchor_idx.resize(out_k);

  for (int i = 0; i < out_k; i++) {
    const float s = flat_scores[i].first;
    const int flat_idx = flat_scores[i].second;
    const int anchor_pos = flat_idx / num_cls;  // index in ori_index
    const int cls_id = flat_idx % num_cls;

    top_scores[i] = s;
    top_cls_idx[i] = cls_id;
    top_anchor_idx[i] = ori_index[anchor_pos];  // original anchor index
  }
}

void Yolo26Detection::dist2bbox(const float *reg_vals, int anchor_idx,
                                std::vector<float> &bbox) {
  float anchor_x = anchor_points_x_[anchor_idx];
  float anchor_y = anchor_points_y_[anchor_idx];
  float stride = stride_tensor_[anchor_idx];

  bbox.resize(4);
  bbox[0] = (anchor_x - reg_vals[0]) * stride;
  bbox[1] = (anchor_y - reg_vals[1]) * stride;
  bbox[2] = (anchor_x + reg_vals[2]) * stride;
  bbox[3] = (anchor_y + reg_vals[3]) * stride;
}

int32_t Yolo26Detection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = static_cast<float>(input_width);
  float input_height_f = static_cast<float>(input_height);

  LOGI(
      "outputParse,batch size:%zu,input shape:%d,%d,%d,%d,model "
      "threshold:%f,max_det:%d,total_anchors:%d",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3], model_threshold_, max_det_,
      total_anchors_);

  for (uint32_t b = 0; b < static_cast<uint32_t>(input_tensor.shape[0]); b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    // 1. 收集所有分类得分 (sigmoid 归一化后)
    std::vector<float> all_scores;
    getAllClassScores(b, all_scores);

    // 2. 收集所有回归值
    std::vector<float> all_reg;
    getAllRegValues(b, all_reg);

    // 3. Top-k 筛选
    std::vector<float> top_scores;
    std::vector<int> top_cls_idx;
    std::vector<int> top_anchor_idx;
    getTopkIndex(all_scores, total_anchors_, num_cls_, max_det_, top_scores,
                 top_cls_idx, top_anchor_idx);

    // 4. 解码边界框并过滤
    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;
    for (size_t i = 0; i < top_scores.size(); i++) {
      float score = top_scores[i];

      // 置信度筛选
      if (score < model_threshold_) {
        continue;
      }

      int anchor_idx = top_anchor_idx[i];
      int cls_id = top_cls_idx[i];

      // 解码边界框
      const float *reg_ptr = all_reg.data() + anchor_idx * num_box_channel_;
      std::vector<float> bbox;
      dist2bbox(reg_ptr, anchor_idx, bbox);

      // 创建检测框对象
      ObjectBoxInfo box_info;
      box_info.score = score;
      box_info.x1 = std::max(0.0f, std::min(bbox[0], input_width_f));
      box_info.y1 = std::max(0.0f, std::min(bbox[1], input_height_f));
      box_info.x2 = std::max(0.0f, std::min(bbox[2], input_width_f));
      box_info.y2 = std::max(0.0f, std::min(bbox[3], input_height_f));
      box_info.class_id = cls_id;

      lb_boxes[cls_id].push_back(box_info);
    }

    // 5. 坐标映射回原图
    std::vector<float> scale_params =
        batch_rescale_params_[input_tensor_name][b];

    std::shared_ptr<ModelBoxInfo> obj = std::make_shared<ModelBoxInfo>();
    obj->image_width = image_width;
    obj->image_height = image_height;

    for (auto &bbox : lb_boxes) {
      for (auto &box : bbox.second) {
        DetectionHelper::rescaleBbox(box, scale_params);
        if (type_mapping_.count(box.class_id)) {
          box.object_type = type_mapping_[box.class_id];
        }
        obj->bboxes.push_back(box);
      }
    }
    out_datas.push_back(obj);
  }

  return 0;
}
