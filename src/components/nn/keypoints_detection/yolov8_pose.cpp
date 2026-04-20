#include "keypoints_detection/yolov8_pose.hpp"

#include <cstdint>
#include <memory>
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
inline std::vector<float> get_box_vals(T *p_box_ptr, int num_anchor,
                                       int anchor_idx, int num_box_channel,
                                       float qscale) {
  std::vector<float> box_vals;
  for (int c = 0; c < num_box_channel; c++) {
    box_vals.push_back(p_box_ptr[c * num_anchor + anchor_idx] * qscale);
  }
  return box_vals;
}

YoloV8Pose::YoloV8Pose() : YoloV8Pose(std::make_tuple(64, 17, 1)) {}

YoloV8Pose::YoloV8Pose(std::tuple<int, int, int> pose_tuple) {
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {1.0 / 0.003922, 1.0 / 0.003922,
                                 1.0 / 0.003922};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;

  num_box_channel_ = std::get<0>(pose_tuple);
  num_kpts_ = std::get<1>(pose_tuple);
  num_cls_ = std::get<2>(pose_tuple);
}

// would parse 3 cases,1:box,cls seperate feature map,2 box+cls seperate
// featuremap,3 output decoded results
int32_t YoloV8Pose::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  auto input_shape = net_->getTensorInfo(input_layer).shape;
  int input_h = input_shape[2];
  int input_w = input_shape[3];
  strides.clear();

  auto &parameters = net_param_.model_config.custom_config_i;

  if (parameters.find("num_kpt") != parameters.end()) {
    num_kpts_ = static_cast<int>(parameters.at("num_kpt"));
    LOGI("num_kpts_channel_:%d", num_kpts_channel_);
  }
  if (parameters.find("num_cls") != parameters.end()) {
    num_cls_ = static_cast<int>(parameters.at("num_cls"));
    LOGI("num_cls_:%d", num_cls_);
  }

  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();
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

    if (channel == num_box_channel_) {
      bbox_out_names[stride_h] = output_layers[j];
      strides.push_back(stride_h);
      LOGI("parse box branch,name:%s,stride:%d\n", output_layers[j].c_str(),
           stride_h);
    } else if (channel == num_cls_) {
      class_out_names[stride_h] = output_layers[j];
      LOGI("parse class branch,name: %s,stride:%d\n", output_layers[j].c_str(),
           stride_h);
    } else {
      if (channel % num_kpts_ == 0 &&
          (channel / num_kpts_ == 2 || channel / num_kpts_ == 3)) {
        keypoints_out_names[stride_h] = output_layers[j];
        num_kpts_channel_ = channel;
        keypoint_dimension_ = channel / num_kpts_;
        LOGI("parse keypoints branch,name:%s,stride:%d\n",
             output_layers[j].c_str(), stride_h);
      } else {
        LOGE("unexpected branch:%s,channel:%d\n", output_layers[j].c_str(),
             channel);
        return -1;
      }
    }
  }
  return 0;
}

YoloV8Pose::~YoloV8Pose() {}

// the bbox featuremap shape is b x 4*regmax x h   x w
void YoloV8Pose::decodeBboxFeatureMap(int batch_idx, int stride, int anchor_idx,
                                      std::vector<float> &decode_box) {
  std::string box_name;
  box_name = bbox_out_names[stride];
  TensorInfo boxinfo = net_->getTensorInfo(box_name);
  std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);

  int num_channel = boxinfo.shape[1];
  int num_anchor = boxinfo.shape[2] * boxinfo.shape[3];
  int box_val_num = 4;
  int reg_max = 16;
  if (num_box_channel_ != box_val_num * reg_max) {
    LOGE("box channel size not ok,got:%d\n", num_channel);
  }

  int32_t feat_w = boxinfo.shape[3];

  int anchor_y = anchor_idx / feat_w;
  int anchor_x = anchor_idx % feat_w;

  float grid_y = anchor_y + 0.5;
  float grid_x = anchor_x + 0.5;

  std::vector<float> grid_logits;  // 4x16
  float qscale = boxinfo.qscale;
  if (boxinfo.data_type == TDLDataType::INT8) {
    int8_t *p_box_int8 = box_tensor->getBatchPtr<int8_t>(batch_idx);
    grid_logits = get_box_vals(p_box_int8, num_anchor, anchor_idx,
                               num_box_channel_, qscale);
  } else if (boxinfo.data_type == TDLDataType::UINT8) {
    uint8_t *p_box_uint8 = box_tensor->getBatchPtr<uint8_t>(batch_idx);
    grid_logits = get_box_vals(p_box_uint8, num_anchor, anchor_idx,
                               num_box_channel_, qscale);
  } else if (boxinfo.data_type == TDLDataType::FP32) {
    float *p_box_float = box_tensor->getBatchPtr<float>(batch_idx);
    grid_logits = get_box_vals(p_box_float, num_anchor, anchor_idx,
                               num_box_channel_, qscale);
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(boxinfo.data_type));
    return;
  }

  // compute softmax and accumulate val per 16
  std::vector<float> box_vals;
  for (int i = 0; i < box_val_num; i++) {
    float sum_softmax = 0;
    float sum_val = 0;
    for (int j = 0; j < reg_max; j++) {
      float expv = exp(grid_logits[i * reg_max + j]);
      sum_softmax += expv;
      sum_val += expv * j;
    }
    sum_softmax = sum_val / sum_softmax;
    box_vals.push_back(sum_softmax);
  }

  std::vector<float> box = {
      (grid_x - box_vals[0]) * stride, (grid_y - box_vals[1]) * stride,
      (grid_x + box_vals[2]) * stride, (grid_y + box_vals[3]) * stride};
  decode_box = box;
}
void YoloV8Pose::decodeKeypointsFeatureMap(int batch_idx, int stride,
                                           int anchor_idx,
                                           std::vector<float> &decode_kpts) {
  decode_kpts.clear();
  std::string kpts_name;
  kpts_name = keypoints_out_names[stride];
  TensorInfo kpts_info = net_->getTensorInfo(kpts_name);
  std::shared_ptr<BaseTensor> kpts_tensor = net_->getOutputTensor(kpts_name);
  int num_anchor = kpts_info.shape[2] * kpts_info.shape[3];

  int32_t feat_w = kpts_info.shape[3];
  int anchor_y = anchor_idx / feat_w;
  int anchor_x = anchor_idx % feat_w;
  float grid_x = anchor_x + 0.5;
  float grid_y = anchor_y + 0.5;
  float val;

  if (kpts_info.data_type == TDLDataType::INT8) {
    int8_t *p_kpts_int8 = kpts_tensor->getBatchPtr<int8_t>(batch_idx);
    for (int c = 0; c < num_kpts_channel_; c++) {
      if (c % keypoint_dimension_ == 0) {
        val =
            (p_kpts_int8[c * num_anchor + anchor_idx] * kpts_info.qscale * 2.0 +
             grid_x - 0.5) *
            (float)stride;
      } else if (c % keypoint_dimension_ == 1) {
        val =
            (p_kpts_int8[c * num_anchor + anchor_idx] * kpts_info.qscale * 2.0 +
             grid_y - 0.5) *
            (float)stride;
      } else {
        val = 1.0 / (1.0 + std::exp(-p_kpts_int8[c * num_anchor + anchor_idx] *
                                    kpts_info.qscale));
      }
      decode_kpts.push_back(val);
    }
  } else if (kpts_info.data_type == TDLDataType::UINT8) {
    uint8_t *p_kpts_uint8 = kpts_tensor->getBatchPtr<uint8_t>(batch_idx);
    for (int c = 0; c < num_kpts_channel_; c++) {
      if (c % keypoint_dimension_ == 0) {
        val = (p_kpts_uint8[c * num_anchor + anchor_idx] * kpts_info.qscale *
                   2.0 +
               grid_x - 0.5) *
              (float)stride;
      } else if (c % keypoint_dimension_ == 1) {
        val = (p_kpts_uint8[c * num_anchor + anchor_idx] * kpts_info.qscale *
                   2.0 +
               grid_y - 0.5) *
              (float)stride;
      } else {
        val = 1.0 / (1.0 + std::exp(-p_kpts_uint8[c * num_anchor + anchor_idx] *
                                    kpts_info.qscale));
      }
      decode_kpts.push_back(val);
    }
  } else if (kpts_info.data_type == TDLDataType::FP32) {
    float *p_kpts_float = kpts_tensor->getBatchPtr<float>(batch_idx);
    for (int c = 0; c < num_kpts_channel_; c++) {
      if (c % keypoint_dimension_ == 0) {
        val = (p_kpts_float[c * num_anchor + anchor_idx] * 2.0 + grid_x - 0.5) *
              (float)stride;
      } else if (c % keypoint_dimension_ == 1) {
        val = (p_kpts_float[c * num_anchor + anchor_idx] * 2.0 + grid_y - 0.5) *
              (float)stride;
      } else {
        val =
            1.0 / (1.0 + std::exp(-p_kpts_float[c * num_anchor + anchor_idx]));
      }
      decode_kpts.push_back(val);
    }
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(kpts_info.data_type));
    assert(0);
  }
}

int32_t YoloV8Pose::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  float inverse_th = std::log(model_threshold_ / (1 - model_threshold_));
  LOGI(
      "outputParse,batch size:%d,input shape:%d,%d,%d,%d,model "
      "threshold:%f,inverse th:%f",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3], model_threshold_,
      inverse_th);

  std::stringstream ss;
  const PreprocessParams &pre_param = preprocess_params_[input_tensor_name];
  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxLandmarkInfo>> lb_boxes;
    std::map<int, std::vector<std::pair<int, uint32_t>>> boxes_temp_info;

    for (size_t i = 0; i < strides.size(); i++) {
      int stride = strides[i];
      std::string cls_name;
      int cls_offset = 0;

      cls_name = class_out_names[stride];
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);

      int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;

      int num_anchor = classinfo.shape[2] * classinfo.shape[3];
      LOGI("stride:%d,featw:%d,feath:%d,numperpixel:%d,numcls:%d,qscale:%f\n",
           stride, classinfo.shape[3], classinfo.shape[2],
           classinfo.tensor_size / classinfo.tensor_elem, num_cls_,
           classinfo.qscale);
      float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;
      for (int j = 0; j < num_anchor; j++) {
        int max_logit_c = -1;
        float max_logit = -1000;
        if (classinfo.data_type == TDLDataType::INT8) {
          parse_cls_info<int8_t>(cls_tensor->getBatchPtr<int8_t>(b), num_anchor,
                                 num_cls_, j, cls_offset, cls_qscale,
                                 &max_logit, &max_logit_c);
        } else if (classinfo.data_type == TDLDataType::UINT8) {
          parse_cls_info<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                  num_anchor, num_cls_, j, cls_offset,
                                  cls_qscale, &max_logit, &max_logit_c);
        } else if (classinfo.data_type == TDLDataType::FP32) {
          parse_cls_info<float>(cls_tensor->getBatchPtr<float>(b), num_anchor,
                                num_cls_, j, cls_offset, cls_qscale, &max_logit,
                                &max_logit_c);
        } else {
          LOGE("unsupported data type:%d\n",
               static_cast<int>(classinfo.data_type));
          assert(0);
        }
        if (max_logit < inverse_th) {
          continue;
        }
        float score = 1 / (1 + exp(-max_logit));
        std::vector<float> box;
        decodeBboxFeatureMap(b, stride, j, box);
        ObjectBoxLandmarkInfo bbox;
        bbox.score = score;
        bbox.x1 = std::max(0.0f, std::min(box[0], input_width_f));
        bbox.y1 = std::max(0.0f, std::min(box[1], input_height_f));
        bbox.x2 = std::max(0.0f, std::min(box[2], input_width_f));
        bbox.y2 = std::max(0.0f, std::min(box[3], input_height_f));
        bbox.class_id = max_logit_c;
        LOGI("bbox:[%f,%f,%f,%f],score:%f,label:%d,logit:%f\n", bbox.x1,
             bbox.y1, bbox.x2, bbox.y2, bbox.score, max_logit_c, max_logit);
        if (type_mapping_.count(bbox.class_id)) {
          bbox.object_type = type_mapping_[bbox.class_id];
        }
        lb_boxes[max_logit_c].push_back(bbox);
        boxes_temp_info[max_logit_c].push_back(std::make_pair(stride, j));
      }
    }
    for (auto &bboxs : lb_boxes) {
      DetectionHelper::nmsObjects(bboxs.second, nms_threshold_,
                                  boxes_temp_info[bboxs.first]);
    }
    std::vector<float> scale_params =
        batch_rescale_params_[input_tensor_name][b];
    LOGI("scale_params:%f,%f,%f,%f", scale_params[0], scale_params[1],
         scale_params[2], scale_params[3]);
    ss << "batch:" << b << "\n";

    std::shared_ptr<ModelBoxLandmarkInfo> obj =
        std::make_shared<ModelBoxLandmarkInfo>();
    obj->image_width = image_width;
    obj->image_height = image_height;
    for (auto &bboxs : lb_boxes) {
      for (size_t i = 0; i < bboxs.second.size(); i++) {
        // auto &bbox_info = bboxs.second[i];
        std::vector<float> decode_kpts;
        decodeKeypointsFeatureMap(b, boxes_temp_info[bboxs.first][i].first,
                                  boxes_temp_info[bboxs.first][i].second,
                                  decode_kpts);
        int num_keypoints = num_kpts_channel_ / keypoint_dimension_;
        if (keypoint_dimension_ == 3) {
          for (int j = 0; j < num_keypoints; j++) {
            bboxs.second[i].landmarks_x.push_back(decode_kpts[j * 3]);
            bboxs.second[i].landmarks_y.push_back(decode_kpts[j * 3 + 1]);
            bboxs.second[i].landmarks_score.push_back(decode_kpts[j * 3 + 2]);
          }
        } else if (keypoint_dimension_ == 2) {
          for (int j = 0; j < num_keypoints; j++) {
            bboxs.second[i].landmarks_x.push_back(decode_kpts[j * 2]);
            bboxs.second[i].landmarks_y.push_back(decode_kpts[j * 2 + 1]);
            bboxs.second[i].landmarks_score.push_back(1.0);
          }
        } else {
          LOGE("unsupported keypoint_dimension_:%d\n", keypoint_dimension_);
          assert(0);
        }

        DetectionHelper::rescaleBbox(bboxs.second[i], scale_params);
        // ss << "bbox:[" << obj_seg->box_seg[i].x1 << "," <<
        // obj_seg->box_seg[i].y1 << "," <<obj_seg->box_seg[i].x2 << "," <<
        // obj_seg->box_seg[i].y2
        //    << "],score:" << obj_seg->box_seg[i].score << ",label:" <<
        //    obj_seg->box_seg[i].class_id << "\n";
        obj->box_landmarks.push_back(bboxs.second[i]);
      }
    }
    out_datas.push_back(obj);
  }
  LOGI("outputParse done,ss:%s", ss.str().c_str());
  return 0;
}
