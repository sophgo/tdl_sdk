#include "face/face_cssd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <log/Logger.hpp>
#include <numeric>
#include <vector>

enum Decode_CodeType {
  PriorBoxParameter_CodeType_CORNER = 1,
  PriorBoxParameter_CodeType_CENTER_SIZE = 2,
  PriorBoxParameter_CodeType_CORNER_SIZE = 3
};
class BBox_l {
 public:
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float size;

  void CalcSize() {
    if (xmax < xmin || ymax < ymin) {
      size = 0;
    } else {
      float width = xmax - xmin;
      float height = ymax - ymin;
      size = width * height;
    }
  }
};

typedef Decode_CodeType CodeType;
typedef std::map<int, std::vector<BBox_l>> LabelBBox_l;

static bool SortScoreCmp0(const std::pair<float, int> &pair1,
                          const std::pair<float, int> &pair2) {
  return pair1.first > pair2.first;
}

std::vector<float> readBinaryFile(const std::string &filename) {
  const size_t dataSize = 19520;
  std::vector<float> data;
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    LOG(FATAL) << "can not open the file" << filename << std::endl;
    return data;
  }

  data.resize(dataSize);
  file.read(reinterpret_cast<char *>(data.data()), dataSize * sizeof(float));

  file.close();
  return data;
}
FaceCSSD::FaceCSSD(const stNetParam &param) { net_param_ = param; }

bmStatus_t FaceCSSD::setup() {
  setup_net(net_param_);
  std::string str_prior_data = model_dir_ + std::string("/cssd_prior_box.bin");
  prior_box_data_ = readBinaryFile(str_prior_data);
  return BM_COMMON_SUCCESS;
}
static void GetConfidenceScores_opt(
    const float *conf_data, const int num, const int num_preds_per_class,
    const int num_classes, const float score_threshold,
    std::vector<std::map<int, std::vector<std::pair<float, int>>>>
        *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    std::map<int, std::vector<std::pair<float, int>>> &label_scores =
        (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        // std::cout<<"--score_threshold--"<<score_threshold<<"==="<<conf_data[start_idx
        // + c] <<std::endl;
        if (conf_data[start_idx + c] > score_threshold) {
          label_scores[c].push_back(
              std::make_pair(conf_data[start_idx + c], p));
        }
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

static void GetConfidenceScores_int8_opt(
    const int8_t *conf_data, const float conf_scale, const int num,
    const int num_preds_per_class, const int num_classes,
    const float score_threshold,
    std::vector<std::map<int, std::vector<std::pair<float, int>>>>
        *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    std::map<int, std::vector<std::pair<float, int>>> &label_scores =
        (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        if (conf_data[start_idx + c] * conf_scale >
            std::log(score_threshold / (1.0f - score_threshold))) {
          // std::cout<<"--score_threshold--"<<conf_data[start_idx +
          // c]*conf_scale<<"==="<<std::log(score_threshold / (1.0f -
          // score_threshold)) <<std::endl;
          label_scores[c].push_back(
              std::make_pair(conf_data[start_idx + c] * conf_scale, p));
        }
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}
static void GetLocPredictions_opt(
    const bm_net_info_t *net_info, const float loc_scale, int loc_idx,
    const int8_t *loc_data_char, const float *loc_data, const int num,
    const int num_preds_per_class, const int num_loc_classes,
    const bool share_location, float *decode_index,
    std::vector<LabelBBox_l> *loc_preds) {
  loc_preds->clear();
  if (share_location) {
    assert(num_loc_classes == 1);
  }
  loc_preds->resize(num);
  float *decode_pos = decode_index;
  auto process_coord = [&](BBox_l &bbox, int start_idx, int c) {
    if (net_info->output_dtypes[loc_idx] == BM_FLOAT32) {
      bbox.xmin = loc_data[start_idx + c * 4];
      bbox.ymin = loc_data[start_idx + c * 4 + 1];
      bbox.xmax = loc_data[start_idx + c * 4 + 2];
      bbox.ymax = loc_data[start_idx + c * 4 + 3];
    } else if (net_info->output_dtypes[loc_idx] == BM_INT8) {
      bbox.xmin = loc_data_char[start_idx + c * 4] * loc_scale;
      bbox.ymin = loc_data_char[start_idx + c * 4 + 1] * loc_scale;
      bbox.xmax = loc_data_char[start_idx + c * 4 + 2] * loc_scale;
      bbox.ymax = loc_data_char[start_idx + c * 4 + 3] * loc_scale;
    }
  };

  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i * num_preds_per_class;
    }
    LabelBBox_l &label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        if (!share_location) {
          decode_pos = decode_index +
                       num_preds_per_class * num_loc_classes * i +
                       num_preds_per_class * c;
        }
        int label = share_location ? -1 : c;
        auto result = label_bbox.find(label);
        if (result == label_bbox.end()) {
          result = label_bbox
                       .insert(std::make_pair(
                           label, std::vector<BBox_l>(num_preds_per_class)))
                       .first;
        }
        BBox_l &bbox = result->second[p];

        process_coord(bbox, start_idx, c);
      }
    }

    if (net_info->output_dtypes[loc_idx] == BM_FLOAT32) {
      loc_data += num_preds_per_class * num_loc_classes * 4;
    } else if (net_info->output_dtypes[loc_idx] == BM_INT8) {
      loc_data_char += num_preds_per_class * num_loc_classes * 4;
    }
  }
}

static void DecodeBBoxesAll_opt(
    const std::vector<LabelBBox_l> &all_loc_preds, int num_priors,
    const float *prior_data, const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    float *decode_index, std::vector<LabelBBox_l> *all_decode_bboxes) {
  assert(all_loc_preds.size() == (size_t)num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  float *decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i * num_priors;
    }
    // Decode predictions into bboxes.
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        std::cout << "Could not find location predictions for label " << label;
      }
      const std::vector<BBox_l> &bboxes = all_loc_preds[i].find(label)->second;
      LabelBBox_l &decode_bboxes = (*all_decode_bboxes)[i];
      std::vector<BBox_l> *p = &(decode_bboxes[label]);
      p->clear();

      if (!share_location) {
        decode_pos =
            decode_index + num_priors * num_loc_classes * i + num_priors * c;
      }
      for (int k = 0; k < num_priors; ++k) {
        // NormalizedBBox decode_bbox;
        BBox_l decode_bbox;
        if (decode_pos[k] != 1) {
          p->push_back(decode_bbox);
          continue;
        }
        // opt CENTER_SIZE
        assert(code_type == PriorBoxParameter_CodeType_CENTER_SIZE);
        // prior_bboxes
        int start_idx = k * 4;
        const float *p0 = prior_data + start_idx;
        const float *p1 = prior_data + start_idx + 4 * num_priors;
        float prior_width = p0[2] - p0[0];
        assert(prior_width > 0);
        float prior_height = p0[3] - p0[1];
        assert(prior_height > 0);
        float prior_center_x = (p0[0] + p0[2]) * 0.5;
        float prior_center_y = (p0[1] + p0[3]) * 0.5;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
          // variance is encoded in target, we simply need to retore the offset
          // predictions.
          decode_bbox_center_x = bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y = bboxes[k].ymin * prior_height + prior_center_y;
          decode_bbox_width = exp(bboxes[k].xmax) * prior_width;
          decode_bbox_height = exp(bboxes[k].ymax) * prior_height;
        } else {
          // variance is encoded in bbox, we need to scale the offset
          // accordingly.
          decode_bbox_center_x =
              p1[0] * bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y =
              p1[1] * bboxes[k].ymin * prior_height + prior_center_y;
          decode_bbox_width = exp(p1[2] * bboxes[k].xmax) * prior_width;
          decode_bbox_height = exp(p1[3] * bboxes[k].ymax) * prior_height;
        }
        decode_bbox.xmin = decode_bbox_center_x - decode_bbox_width * 0.5;
        decode_bbox.ymin = decode_bbox_center_y - decode_bbox_height * 0.5;
        decode_bbox.xmax = decode_bbox_center_x + decode_bbox_width * 0.5;
        decode_bbox.ymax = decode_bbox_center_y + decode_bbox_height * 0.5;
        decode_bbox.CalcSize();
        p->push_back(decode_bbox);
      }
    }
  }
}
static void ApplyNMSFast_opt(
    const std::vector<BBox_l> &bboxes,
    const std::vector<std::pair<float, int>> &conf_score,
    const float score_threshold, const float nms_threshold, const float eta,
    int top_k, std::vector<std::pair<float, int>> *indices) {
  // Do nms.
  float adaptive_threshold = nms_threshold;
  int i = 0;
  int length = (top_k < (int)conf_score.size()) ? top_k : conf_score.size();
  while (length != i) {
    bool keep = true;
    for (int k = 0; k < (int)indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k].second;
        const BBox_l &b1 = bboxes[conf_score[i].second];
        const BBox_l &b2 = bboxes[kept_idx];
        if (b2.xmin > b1.xmax || b2.xmax < b1.xmin || b2.ymin > b1.ymax ||
            b2.ymax < b1.ymin) {
          keep = true;
        } else {
          const float inter_xmin = std::max(b1.xmin, b2.xmin);
          const float inter_ymin = std::max(b1.ymin, b2.ymin);
          const float inter_xmax = std::min(b1.xmax, b2.xmax);
          const float inter_ymax = std::min(b1.ymax, b2.ymax);
          const float inter_width = inter_xmax - inter_xmin;
          const float inter_height = inter_ymax - inter_ymin;
          const float inter_size = inter_width * inter_height;
          const float total_size = b1.size + b2.size;
          keep = (inter_size * (adaptive_threshold + 1) <=
                  total_size * adaptive_threshold)
                     ? true
                     : false;
        }
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(conf_score[i]);
    }
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
    i++;
  }
}
static bool SortScoreCmp1(const std::pair<float, std::pair<int, int>> &pair1,
                          const std::pair<float, std::pair<int, int>> &pair2) {
  return pair1.first > pair2.first;
}

void ApplyNMSPerImage(
    const LabelBBox_l &decode_bboxes,
    const std::map<int, std::vector<std::pair<float, int>>> &conf_scores,
    const int num_classes, const int background_label_id,
    const bool share_location, const float confidence_threshold,
    const float nms_threshold, const float eta, const int top_k,
    const int keep_top_k,
    std::map<int, std::vector<std::pair<float, int>>> &indices, int &num_det) {
  num_det = 0;
  for (int c = 0; c < num_classes; ++c) {
    if (c == background_label_id) {
      // Ignore background class.
      continue;
    }
    if (conf_scores.find(c) == conf_scores.end()) {
      continue;
    }
    int label = share_location ? -1 : c;
    if (decode_bboxes.find(label) == decode_bboxes.end()) {
      std::cout << "Could not find location predictions for label " << label
                << std::endl;
      continue;
    }
    const std::vector<BBox_l> &bboxes = decode_bboxes.find(label)->second;
    const std::vector<std::pair<float, int>> &scores =
        conf_scores.find(c)->second;
    ApplyNMSFast_opt(bboxes, scores, confidence_threshold, nms_threshold, eta,
                     top_k, &(indices[c]));
    num_det += indices[c].size();
  }

  // Keep top_k results if required.
  if (keep_top_k > -1 && num_det > keep_top_k) {
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto &it : indices) {
      for (const auto &score_index_pair : it.second) {
        score_index_pairs.emplace_back(
            score_index_pair.first,
            std::make_pair(it.first, score_index_pair.second));
      }
    }
    // Sort and keep top k results.
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
              SortScoreCmp1);
    if (score_index_pairs.size() > keep_top_k) {
      score_index_pairs.resize(keep_top_k);
    }
    // Reconstruct the indices with the top k results.
    std::map<int, std::vector<std::pair<float, int>>> new_indices;
    for (const auto &score_index_pair : score_index_pairs) {
      int label = score_index_pair.second.first;
      int idx = score_index_pair.second.second;
      float score = score_index_pair.first;
      new_indices[label].emplace_back(score, idx);
    }
    indices = std::move(new_indices);
    num_det = keep_top_k;
  }
}
void GenerateTopData(
    const std::vector<std::map<int, std::vector<std::pair<float, int>>>>
        &all_indices,
    const std::vector<LabelBBox_l> &all_decode_bboxes, const int num,
    const int num_kept, const bool share_location,
    std::unique_ptr<float[]> &top_data) {
  if (num_kept == 0) {
    for (int i = 0; i < num; ++i) {
      top_data[i * 7] = i;
    }
  } else {
    int count = 0;
    for (int i = 0; i < num; ++i) {
      const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
      for (auto it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
        int label = it->first;
        int loc_label = share_location ? -1 : label;
        auto bbox_it = decode_bboxes.find(loc_label);
        if (bbox_it == decode_bboxes.end()) {
          std::cout << "Could not find location predictions for " << loc_label;
          continue;
        }
        const std::vector<BBox_l> &bboxes = bbox_it->second;
        const std::vector<std::pair<float, int>> &indices = it->second;
        for (int j = 0; j < static_cast<int>(indices.size()); ++j) {
          int idx = indices[j].second;
          const BBox_l &bbox = bboxes[idx];

          int base_index = count * 7;
          float *base_ptr = &top_data[base_index];
          base_ptr[0] = i;
          base_ptr[1] = label;
          base_ptr[2] = indices[j].first;
          base_ptr[3] = bbox.xmin;
          base_ptr[4] = bbox.ymin;
          base_ptr[5] = bbox.xmax;
          base_ptr[6] = bbox.ymax;

          ++count;
        }
      }
    }
  }
}
void FilterAndScaleDetections(float *&out, const int width, const int height,
                              const float im_scale_w, const float im_scale_h,
                              const float pad_x, const float pad_y,
                              const float threshold,
                              std::vector<FaceRect> &last_result, int &cnt,
                              const cv::Size &input_geometry) {
  FaceRect bbox;
  bbox.x1 = std::min(
      std::max((out[3] * input_geometry.width - pad_x) * im_scale_w, 0.0f),
      (float)width);
  bbox.y1 = std::min(
      std::max((out[4] * input_geometry.height - pad_y) * im_scale_h, 0.0f),
      (float)height);
  bbox.x2 = std::min(
      std::max((out[5] * input_geometry.width - pad_x) * im_scale_w, 0.0f),
      (float)width);
  bbox.y2 = std::min(
      std::max((out[6] * input_geometry.height - pad_y) * im_scale_h, 0.0f),
      (float)height);
  bbox.score = out[2];
  if (bbox.score >= threshold && (bbox.x2 - bbox.x1 >= 1) &&
      (bbox.y2 - bbox.y1 >= 1)) {
    last_result.push_back(bbox);
    cnt++;
  }
  out += 7;
}

void PrepareDetectionResults(
    const std::vector<cv::Size> &frame_sizes,
    const std::vector<std::vector<float>> &frame_scale_params,
    const float threshold, std::vector<std::vector<FaceRect>> &results,
    float *out, const cv::Size &input_geometry, const size_t num_output) {
  size_t num_output_pass = 0;
  for (size_t i = 0; i < frame_sizes.size(); i++) {
    int cnt = 0;
    int width = frame_sizes[i].width;
    int height = frame_sizes[i].height;
    results.emplace_back();
    auto &last_result = results.back();

    const std::vector<float> &rescale_param = frame_scale_params[i];
    float im_scale_w = rescale_param[0];
    float im_scale_h = rescale_param[1];
    float pad_x = rescale_param[2];
    float pad_y = rescale_param[3];

    size_t num_output_batch = num_output - num_output_pass;
    for (int k = 0; k < num_output_batch; k++) {
      if (int(out[0]) != i) break;
      if (int(out[1]) != 1) {  // Only interested in objects of class 1
        out += 7;
        num_output_pass++;
        continue;
      }
      FilterAndScaleDetections(out, width, height, im_scale_w, im_scale_h,
                               pad_x, pad_y, threshold, last_result, cnt,
                               input_geometry);
      num_output_pass++;
    }
    LOG(INFO) << "Number of detections: " << cnt;
  }
}

bmStatus_t FaceCSSD::postprocess(
    const std::vector<cv::Size> &frame_sizes, const float threshold,
    const std::vector<std::vector<float>> &frame_scale_params,
    std::vector<std::vector<FaceRect>> &results) {
  // read bin file
  const float *prior_data = prior_box_data_.data();
  // end read bin file

  int batch_n = get_input_n();
  net_->update_output_tensors();
  nncompact::Tensor *mbox_loc_tensor =
      net_->get_output_tensor(net_->output_names_[0]).get();

  nncompact::Tensor *mbox_conf_flatten_tensor =
      net_->get_output_tensor(net_->output_names_[1]).get();

  int loc_idx = get_output_index(net_->output_names_[0]);
  int conf_idx = get_output_index(net_->output_names_[1]);
  const bm_net_info_t *net_info = (const bm_net_info_t *)net_->get_net_info();
  std::vector<int> mbox_loc_shape = mbox_loc_tensor->get_shape();
  std::vector<int> conf_shape = mbox_conf_flatten_tensor->get_shape();

  int onnx_nms = 0;
  int top_k = 100;
  std::vector<int> &locShape = mbox_loc_shape;
  int num = locShape[0];
  int num_priors = 2440;
  int num_loc_classes = 1;
  float eta = 1.0;
  double nms_threshold = 0.3;
  int64_t keep_top_k = 100;
  int num_classes = 2;
  float confidence_threshold = 0.05;
  bool share_location = 1;
  int background_label_id = 0;
  bool variance_encoded_in_target = false;

  std::vector<std::map<int, std::vector<std::pair<float, int>>>>
      all_conf_scores;

  if (net_info->output_dtypes[conf_idx] == BM_FLOAT32) {
    float *conf_data = (float *)mbox_conf_flatten_tensor->get_data();
    if (!onnx_nms) {
      GetConfidenceScores_opt(conf_data, num, num_priors, num_classes,
                              confidence_threshold, &all_conf_scores);
    }
  } else if (net_info->output_dtypes[conf_idx] == BM_INT8) {
    float conf_scale = net_info->output_scales[conf_idx];
    int8_t *conf_data_char = (int8_t *)(mbox_conf_flatten_tensor->get_data());
    if (!onnx_nms) {
      GetConfidenceScores_int8_opt(conf_data_char, conf_scale, num, num_priors,
                                   num_classes, confidence_threshold,
                                   &all_conf_scores);
    }
  }

  for (int i = 0; i < num; ++i) {
    for (auto &score_pair : all_conf_scores[i]) {
      std::vector<std::pair<float, int>> &scores = score_pair.second;

      if (top_k < (int)scores.size()) {
        std::nth_element(scores.begin(), scores.begin() + top_k, scores.end(),
                         SortScoreCmp0);
        std::sort(scores.begin(), scores.begin() + top_k, SortScoreCmp0);
      } else {
        std::sort(scores.begin(), scores.end(), SortScoreCmp0);
      }
    }
  }
  // build keep for decode ,recode vilad index
  // float *decode_keep_index;
  int buf_length = 0;
  if (share_location) {
    buf_length = num * num_priors;
  } else {
    buf_length = num * num_priors * num_classes;
  }
  // decode_keep_index = new float[buf_length];
  std::unique_ptr<float[]> decode_keep_index(new float[buf_length]);

  // memset(decode_keep_index, 0, buf_length * 4);
  // float *ptr = decode_keep_index;
  float *ptr = decode_keep_index.get();
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      ptr = decode_keep_index.get() + num_priors * i;
    }
    for (int c = 0; c < num_classes; ++c) {
      if (!share_location) {
        ptr = decode_keep_index.get() + num_priors * num_classes * i +
              num_priors * c;
      }
      if (c == background_label_id) {
        // Ignore background class.
        continue;
      }

      if (all_conf_scores[i].find(c) == all_conf_scores[i].end()) continue;
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;
      int length = top_k < (int)scores.size() ? top_k : scores.size();
      for (int k = 0; k < length; ++k) {
        ptr[scores[k].second] = 1;
      }
    }
  }
  // Retrieve all location predictions.
  std::vector<LabelBBox_l> all_loc_preds;
  float *loc_data = (float *)mbox_loc_tensor->get_data();
  int8_t *loc_data_char = (int8_t *)(loc_data);
  float loc_scale = net_info->output_scales[loc_idx];
  GetLocPredictions_opt(net_info, loc_scale, loc_idx, loc_data_char, loc_data,
                        num, num_priors, num_loc_classes, share_location,
                        decode_keep_index.get(), &all_loc_preds);
  // Decode all loc predictions to bboxes.
  std::vector<LabelBBox_l> all_decode_bboxes;
  const bool clip_bbox = false;
  Decode_CodeType code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  if (!onnx_nms) {
    DecodeBBoxesAll_opt(all_loc_preds, num_priors, prior_data, num,
                        share_location, num_loc_classes, background_label_id,
                        code_type, variance_encoded_in_target,
                        decode_keep_index.get(), &all_decode_bboxes);
  }

  // else {
  //   DecodeBBoxesAll_v2_opt(all_loc_preds, num_priors, prior_data, num,
  //                          param_.share_location, num_loc_classes,
  //                          param_.background_label_id, param_.code_type,
  //                          variance_encoded_in_target, clip_bbox,
  //                          decode_keep_index, &all_decode_bboxes);
  // }

  std::vector<std::map<int, std::vector<std::pair<float, int>>>> all_indices;
  int num_kept = 0;
  for (int i = 0; i < num; ++i) {
    const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
    const std::map<int, std::vector<std::pair<float, int>>> &conf_scores =
        all_conf_scores[i];
    std::map<int, std::vector<std::pair<float, int>>> indices;
    int num_det = 0;
    ApplyNMSPerImage(decode_bboxes, conf_scores, num_classes,
                     background_label_id, share_location, confidence_threshold,
                     nms_threshold, eta, top_k, keep_top_k, indices, num_det);

    all_indices.push_back(indices);
    num_kept +=
        (keep_top_k > -1 && num_det > keep_top_k) ? keep_top_k : num_det;
  }
  //
  int output_size = num * keep_top_k * 1 * 1 * 7;
  std::unique_ptr<float[]> top_data(new float[output_size]);
  // init output buf
  if (num_kept != 0) {
    std::fill_n(top_data.get(), output_size, -1);
  }
  GenerateTopData(all_indices, all_decode_bboxes, num, num_kept, share_location,
                  top_data);
  // finish decode output_tensor

  int num_det = keep_top_k;
  float *out = top_data.get();
  size_t num_output = batch_n * num_det;

  num_output = bmrt_tensor_bytesize(
                   &(((bm_tensor_t *)(net_->get_device_output_tensors()))[0])) /
               28;

  PrepareDetectionResults(frame_sizes, frame_scale_params, threshold, results,
                          out, input_geometry_, num_output);
  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceCSSD::detect(const std::vector<cv::Mat> &images,
                            const float threshold,
                            std::vector<std::vector<FaceRect>> &results) {
  auto left_size = images.size();
  auto img_iter = images.cbegin();

#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
#endif
  while (left_size > 0) {
    int batch_size = get_fit_n(left_size);

#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
#endif

    BM_CHECK_STATUS(preprocess_opencv(img_iter, batch_size));
#ifdef TIME_PRINT
    timer_.store_timestamp("preprocess");
    timer_.store_timestamp("forward");
#endif

    BM_CHECK_STATUS(forward());

#ifdef TIME_PRINT
    timer_.store_timestamp("forward");
    timer_.store_timestamp("postprocess");
#endif
    std::vector<cv::Size> frame_sizes;
    for (int i = 0; i < batch_size; i++) {
      frame_sizes.push_back((img_iter + i)->size());
    }
    BM_CHECK_STATUS(
        postprocess(frame_sizes, threshold, batch_rescale_params_, results));
    img_iter += batch_size;
    left_size -= batch_size;

#ifdef TIME_PRINT
    timer_.store_timestamp("postprocess");
#endif
  }

#ifdef TIME_PRINT
  timer_.store_timestamp("total detection");
  timer_.show();
  timer_.clear();
#endif

  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceCSSD::detect_direct(
    const std::vector<std::vector<cv::Mat>> &frame_bgrs, float threshold,
    const std::vector<std::vector<float>> &frame_rescale_params,
    const std::vector<cv::Size> &frame_sizes,
    std::vector<std::vector<FaceRect>> &results) {
  double freq = cv::getTickFrequency() / 1000;
  double ts_infer = 0, ts_post = 0;
  int left_size = frame_bgrs.size();
  int process_idx = 0;

  while (left_size > 0) {
    int n = get_fit_n(left_size);
    set_input_n(n);

    std::shared_ptr<nncompact::Tensor> input_tensor =
        net_->get_input_tensor(input_layer_);
    std::vector<std::vector<float>> batch_rescale_params;
    std::vector<cv::Size> batch_frame_sizes;

    for (int i = process_idx; i < process_idx + n; i++) {
      const std::vector<cv::Mat> &bgr = frame_bgrs[i];
      for (int j = 0; j < bgr.size(); j++) {
        // const cv::Mat &mat = bgr[j];
        input_tensor->from_mat(bgr[j], i - process_idx, j);
      }
      batch_rescale_params.push_back(frame_rescale_params[i]);
      batch_frame_sizes.push_back(frame_sizes[i]);
    }

    double tick0 = cv::getTickCount();
    net_->update_input_tensors();
    BM_CHECK_STATUS(forward());
    double tick1 = cv::getTickCount();

    std::vector<std::vector<FaceRect>> batch_res;
    BM_CHECK_STATUS(postprocess(batch_frame_sizes, threshold,
                                batch_rescale_params, batch_res));
    double tick2 = cv::getTickCount();

    results.insert(results.end(), batch_res.begin(), batch_res.end());
    left_size -= n;
    process_idx += n;

    ts_infer += (tick1 - tick0) / freq;
    ts_post += (tick2 - tick1) / freq;
  }
  return BM_COMMON_SUCCESS;
}
