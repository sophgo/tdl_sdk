#include "object_detection/ppyoloe.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
int yoloe_argmax(T *ptr, int basic_pos, int cls_len) {
  int max_idx = 0;
  for (int i = 0; i < cls_len; i++) {
    if (ptr[i + basic_pos] > ptr[max_idx + basic_pos]) {
      max_idx = i;
    }
  }
  return max_idx;
}

PPYoloEDetection::PPYoloEDetection()
    : PPYoloEDetection(std::make_pair(4, 80)) {}

PPYoloEDetection::PPYoloEDetection(std::pair<int, int> ppyoloe_pair) {
  // default param
  net_param_.model_config.mean = {123.675, 116.28, 103.52};
  net_param_.model_config.std = {58.395, 57.12, 57.375};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;

  num_box_channel_ = ppyoloe_pair.first;
  num_cls_ = ppyoloe_pair.second;
  if (num_box_channel_ == num_cls_) {
    LOGE("error,num_box_channel_(%d) == num_cls_(%d)", num_box_channel_,
         num_cls_);
    throw std::runtime_error("num_box_channel_ == num_cls_");
  }
}

PPYoloEDetection::~PPYoloEDetection() {}

int PPYoloEDetection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  auto input_shape = net_->getTensorInfo(input_layer).shape;
  int input_h = input_shape[2];
  int input_w = input_shape[3];
  strides.clear();
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();
  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);
    int feat_h = oinfo.shape[1];
    int feat_w = oinfo.shape[2];
    int channel = oinfo.shape[3];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    if (stride_h == 0 && num_output == 2) {
      if (channel == num_cls_) {
        class_out_names[stride_h] = output_layers[j];
        strides.push_back(stride_h);
        LOGI("parse class decode branch:%s,channel:%d\n",
             output_layers[j].c_str(), channel);
      } else {
        bbox_out_names[stride_h] = output_layers[j];
        LOGI("parse box decode branch:%s,channel:%d\n",
             output_layers[j].c_str(), channel);
      }
      continue;
    }

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
      LOGI("parse class branch,name:%s,stride:%d\n", output_layers[j].c_str(),
           stride_h);
    } else {
      LOGE("unexpected branch:%s,channel:%d\n", output_layers[j].c_str(),
           channel);
      return -1;
    }
  }

  return 0;
}

void PPYoloEDetection::decodeBboxFeatureMap(int batch_idx, int stride,
                                            int anchor_idx,
                                            std::vector<float> &decode_box) {
  std::string box_name;
  if (bbox_out_names.count(stride)) {
    box_name = bbox_out_names[stride];
  } else if (bbox_class_out_names.count(stride)) {
    box_name = bbox_class_out_names[stride];
  }
  TensorInfo boxinfo = net_->getTensorInfo(box_name);

  std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);
  // int8_t *p_box_int8 = static_cast<int8_t *>(boxinfo.sys_mem) + batch_offset;
  // float *p_box_float = static_cast<float *>(boxinfo.sys_mem) + batch_offset;
  // int num_channel = boxinfo.shape[3];
  // int num_anchor = boxinfo.shape[1] * boxinfo.shape[2];
  int box_val_num = 4;

  int32_t feat_w = boxinfo.shape[2];

  int anchor_y = anchor_idx / feat_w;
  int anchor_x = anchor_idx % feat_w;

  float grid_y = anchor_y + 0.5;
  float grid_x = anchor_x + 0.5;

  std::vector<float> box_vals;
  // float qscale = boxinfo.qscale;
  if (boxinfo.data_type == TDLDataType::INT8) {
    int8_t *p_box_int8 = box_tensor->getBatchPtr<int8_t>(batch_idx);
    for (int i = 0; i < box_val_num; i++) {
      box_vals.push_back(p_box_int8[anchor_idx * 4 + i] * boxinfo.qscale);
    }
  } else if (boxinfo.data_type == TDLDataType::UINT8) {
    uint8_t *p_box_uint8 = box_tensor->getBatchPtr<uint8_t>(batch_idx);
    for (int i = 0; i < box_val_num; i++) {
      box_vals.push_back(p_box_uint8[anchor_idx * 4 + i] * boxinfo.qscale);
    }
  } else if (boxinfo.data_type == TDLDataType::FP32) {
    float *p_box_float = box_tensor->getBatchPtr<float>(batch_idx);
    for (int i = 0; i < box_val_num; i++) {
      box_vals.push_back(p_box_float[anchor_idx * 4 + i] * boxinfo.qscale);
    }
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(boxinfo.data_type));
    return;
  }

  std::vector<float> box = {
      (grid_x - box_vals[0]) * stride, (grid_y - box_vals[1]) * stride,
      (grid_x + box_vals[2]) * stride, (grid_y + box_vals[3]) * stride};
  decode_box = box;
}

int32_t PPYoloEDetection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor_info = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor_info.shape[3];
  uint32_t input_height = input_tensor_info.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  // float inverse_th = std::log(model_threshold_ / (1 - model_threshold_));
  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor_info.shape[0], input_tensor_info.shape[1],
       input_tensor_info.shape[2], input_tensor_info.shape[3]);

  for (uint32_t b = 0; b < (uint32_t)input_tensor_info.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;
    for (size_t i = 0; i < strides.size(); i++) {
      int stride = strides[i];
      std::string cls_name;
      // int cls_offset = 0;
      if (class_out_names.count(stride)) {
        cls_name = class_out_names[stride];
      } else if (bbox_class_out_names.count(stride)) {
        cls_name = bbox_class_out_names[stride];
        // cls_offset = num_box_channel_;
      }
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);

      int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;
      int num_anchor = classinfo.shape[2] * classinfo.shape[1];
      LOGI(
          "name:%s,stride:%d,featw:%d,feath:%d,numperpixel:%d,numcls:%d,qscale:"
          "%f\n",
          cls_name.c_str(), stride, classinfo.shape[3], classinfo.shape[2],
          num_per_pixel, num_cls_, classinfo.qscale);
      float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;

      int basic_pos_cls = 0;
      int basic_pos_box = 0;
      for (int j = 0; j < num_anchor; j++) {
        float class_score = 0.0f;
        int label = 0;
        if (classinfo.data_type == TDLDataType::INT8) {
          label = yoloe_argmax<int8_t>(cls_tensor->getBatchPtr<int8_t>(b),
                                       basic_pos_cls, num_cls_);
          class_score =
              cls_tensor->getBatchPtr<int8_t>(b)[label + basic_pos_cls] *
              cls_qscale;
        } else if (classinfo.data_type == TDLDataType::UINT8) {
          label = yoloe_argmax<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                        basic_pos_cls, num_cls_);
          class_score =
              cls_tensor->getBatchPtr<uint8_t>(b)[label + basic_pos_cls] *
              cls_qscale;
        } else if (classinfo.data_type == TDLDataType::FP32) {
          label = yoloe_argmax<float>(cls_tensor->getBatchPtr<float>(b),
                                      basic_pos_cls, num_cls_);
          class_score =
              cls_tensor->getBatchPtr<float>(b)[label + basic_pos_cls];
        } else {
          LOGE("unsupported data type:%d\n", static_cast<int>(classinfo.data_type));
          assert(0);
        }

        class_score = 1 / (1 + exp(-class_score));
        if (class_score < m_model_threshold) {
          basic_pos_cls += num_cls_;
          basic_pos_box += 4;
          continue;
        }
        std::vector<float> box;
        decodeBboxFeatureMap(b, stride, j, box);
        basic_pos_cls += num_cls_;
        basic_pos_box += 4;
        ObjectBoxInfo bbox;
        bbox.score = class_score;
        bbox.x1 = std::max(0.0f, std::min(box[0], input_width_f));
        bbox.y1 = std::max(0.0f, std::min(box[1], input_height_f));
        bbox.x2 = std::max(0.0f, std::min(box[2], input_width_f));
        bbox.y2 = std::max(0.0f, std::min(box[3], input_height_f));
        bbox.class_id = label;
        lb_boxes[label].push_back(bbox);
      }
    }
    DetectionHelper::nmsObjects(lb_boxes, 0.5);
    std::vector<float> scale_params = batch_rescale_params_[b];
    int num_obj = 0;
    std::shared_ptr<ModelBoxInfo> obj = std::make_shared<ModelBoxInfo>();
    obj->image_width = image_width;
    obj->image_height = image_height;
    for (auto &bbox : lb_boxes) {
      num_obj += bbox.second.size();
      for (auto &b : bbox.second) {
        DetectionHelper::rescaleBbox(b, scale_params);
        if (type_mapping_.count(b.class_id)) {
          b.object_type = type_mapping_[b.class_id];
        }
        obj->bboxes.push_back(b);
      }
    }

    out_datas.push_back(obj);
  }
  return 0;
}