#include "segmentation/yolov8_seg.hpp"

#include "Eigen/Core"
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>
#include <tuple>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"
template <typename T>
inline void parse_cls_info(T *p_cls_ptr, int num_anchor, int num_cls, int anchor_idx,
                           int cls_offset, float qscale,
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

YoloV8Segmentation::YoloV8Segmentation() : YoloV8Segmentation(std::make_tuple(64, 32, 80)) {}

YoloV8Segmentation::YoloV8Segmentation(std::tuple<int, int, int> yolov8_tuple) {
  for (int i = 0; i < 3; i++) {
    net_param_.pre_params.scale[i] = 0.003922;
    net_param_.pre_params.mean[i] = 0.0;
  }
  net_param_.pre_params.dst_image_format = ImageFormat::RGB_PLANAR;
  net_param_.pre_params.keep_aspect_ratio = true;

  num_box_channel_ = std::get<0>(yolov8_tuple);
  num_mask_channel_ = std::get<1>(yolov8_tuple);
  num_cls_ = std::get<2>(yolov8_tuple);
  if (num_box_channel_ == num_cls_) {
      LOGE("Error: num_box_channel_ (%d) is equal to num_cls_ (%d)", num_box_channel_, num_cls_);
      throw std::runtime_error("Number of box channels cannot be equal to the number of classes.");
  }

  if (num_mask_channel_ == num_cls_) {
      LOGE("Error: num_mask_channel_ (%d) is equal to num_cls_ (%d)", num_mask_channel_, num_cls_);
      throw std::runtime_error("Number of mask channels cannot be equal to the number of classes.");
  }
}
YoloV8Segmentation::~YoloV8Segmentation(){}
// identity output branches
int YoloV8Segmentation::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  auto input_shape = net_->getTensorInfo(input_layer).shape;
  int input_h = input_shape[2];
  int input_w = input_shape[3];
  strides.clear();
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();
  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);
    int feat_h = oinfo.shape[2];
    int feat_w = oinfo.shape[3];
    int channel = oinfo.shape[1];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;
    if (stride_h == 0 && num_output == 2) {
      if (channel == num_cls_) {
        class_out_names[stride_h] =output_layers[j];
        strides.push_back(stride_h);
        LOGI("parse class decode branch:%s,channel:%d\n",output_layers[j].c_str(), channel);
      } else {
        bbox_out_names[stride_h] =output_layers[j];
        LOGI("parse box decode branch:%s,channel:%d\n",output_layers[j].c_str(), channel);
      }
      continue;
    }

    if (stride_h != stride_w) {
      LOGE("stride not equal,stridew:%d,strideh:%d,featw:%d,feath:%d\n", stride_w, stride_h, feat_w,
           feat_h);
      return -1;
    }
    if (channel == num_box_channel_) {
      bbox_out_names[stride_h] =output_layers[j];
      strides.push_back(stride_h);
      LOGI("parse box branch,name:%s,stride:%d\n",output_layers[j].c_str(), stride_h);
    } else if (num_cls_ == 0 && num_output == 6) {
      num_cls_ = channel;
      class_out_names[stride_h] =output_layers[j];
      LOGI("parse class branch,name:%s,stride:%d,num_cls:%d\n",output_layers[j].c_str(), stride_h,
           channel);
    } else if (channel == num_cls_) {
      class_out_names[stride_h] =output_layers[j];
      LOGI("parse class branch,name:%s,stride:%d\n",output_layers[j].c_str(), stride_h);
    } else if (channel == num_mask_channel_) {
      mask_out_names[stride_h] =output_layers[j];
      LOGI("parse mask branch,name:%s,stride:%d\n",output_layers[j].c_str(), stride_h);
    } else if (channel == (num_box_channel_ + num_cls_)) {
      strides.push_back(stride_h);
      bbox_class_out_names[stride_h] = output_layers[j];
      LOGI("parse box+class branch,name: %s,stride:%d\n",output_layers[j].c_str(), stride_h);
    } else {
      LOGE("unexpected branch:%s,channel:%d\n",output_layers[j].c_str(), channel);
      return -1;
    }
  }
  if (!mask_out_names.empty()) {
    auto min_it = mask_out_names.begin();
    for (auto it = mask_out_names.begin(); it != mask_out_names.end(); ++it) {
      if (it->first < min_it->first) {
        min_it = it;
      }
    }
    // copy the entry of the minimum key to proto_out_name
    proto_out_names[min_it->first] = min_it->second;
    // remove the entry with the smallest key from mask_out_name
    mask_out_names.erase(min_it);
  }

  return 0;
}

// the bbox featuremap shape is b x 4*regmax x h   x w
void YoloV8Segmentation::decodeBboxFeatureMap(int batch_idx, int stride,
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
    LOGE("unsupported data type:%d\n", boxinfo.data_type);
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

int32_t YoloV8Segmentation::outputParse(
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
  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxSegmentationInfo>> lb_boxes;
    std::map<int, std::vector<std::pair<int, uint32_t>>> boxes_temp_info;

    for (size_t i = 0; i < strides.size(); i++) {
      int stride = strides[i];
      std::string cls_name;
      int cls_offset = 0;
      if (class_out_names.count(stride)) {
        cls_name = class_out_names[stride];
      } else if (bbox_class_out_names.count(stride)) {
        cls_name = bbox_class_out_names[stride];
        cls_offset = num_box_channel_;
      }
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);

      int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;

      int num_cls = num_cls_;
      int num_anchor = classinfo.shape[2] * classinfo.shape[3];
      LOGI("stride:%d,featw:%d,feath:%d,numperpixel:%d,numcls:%d,qscale:%f\n",
           stride, classinfo.shape[3], classinfo.shape[2],
           classinfo.tensor_size / classinfo.tensor_elem, num_cls,
           classinfo.qscale);
      float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;
      for (int j = 0; j < num_anchor; j++) {
        int max_logit_c = -1;
        float max_logit = -1000;
        if (classinfo.data_type == TDLDataType::INT8) {
          parse_cls_info<int8_t>(cls_tensor->getBatchPtr<int8_t>(b), num_anchor,
                                 num_cls, j, cls_offset, cls_qscale, &max_logit,
                                 &max_logit_c);
        } else if (classinfo.data_type == TDLDataType::UINT8) {
          parse_cls_info<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                  num_anchor, num_cls, j, cls_offset,
                                  cls_qscale, &max_logit, &max_logit_c);
        } else if (classinfo.data_type == TDLDataType::FP32) {
          parse_cls_info<float>(cls_tensor->getBatchPtr<float>(b), num_anchor,
                                num_cls, j, cls_offset, cls_qscale, &max_logit,
                                &max_logit_c);
        } else {
          LOGE("unsupported data type:%d\n", classinfo.data_type);
          assert(0);
        }
        if (max_logit < inverse_th) {
          continue;
        }
        float score = 1 / (1 + exp(-max_logit));
        std::vector<float> box;
        decodeBboxFeatureMap(b, stride, j, box);
        ObjectBoxSegmentationInfo bbox;
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
    for (auto &bboxs : lb_boxes){
        DetectionHelper::nmsObjects(bboxs.second, nms_threshold_, boxes_temp_info[bboxs.first]);
    }

    std::vector<float> scale_params = batch_rescale_params_[b];
    LOGI("scale_params:%f,%f,%f,%f", scale_params[0], scale_params[1],
         scale_params[2], scale_params[3]);
    ss << "batch:" << b << "\n";

    int num_obj = 0;
    for (auto &bbox : lb_boxes) {
      num_obj += bbox.second.size();
    }

    std::shared_ptr<ModelBoxSegmentationInfo> obj_seg = std::make_shared<ModelBoxSegmentationInfo>();
    obj_seg->image_width = image_width;
    obj_seg->image_height = image_height;

    Eigen::MatrixXf mask_map(num_obj, num_mask_channel_);
    int row = 0;
    for (auto &bboxs : lb_boxes) {
      for (size_t i = 0; i < bboxs.second.size(); i++) {

        auto &bbox_info = bboxs.second[i];
        obj_seg->box_seg.push_back(bboxs.second[i]);
        std::string mask_name;
        mask_name = mask_out_names[boxes_temp_info[bboxs.first][i].first];
        TensorInfo maskinfo = net_->getTensorInfo(mask_name);
        std::shared_ptr<BaseTensor> mask_tensor = net_->getOutputTensor(mask_name);
        int num_map = maskinfo.shape[2] * maskinfo.shape[3];
        if (maskinfo.data_type == TDLDataType::INT8) {
          int8_t *p_mask_int8 = mask_tensor->getBatchPtr<int8_t>(b);
          for (int c = 0; c < num_mask_channel_; c++) {
            mask_map(row, c) = p_mask_int8[c * num_map + boxes_temp_info[bboxs.first][i].second] * maskinfo.qscale;
          }
        } else if (maskinfo.data_type == TDLDataType::UINT8) {
          uint8_t *p_mask_uint8 = mask_tensor->getBatchPtr<uint8_t>(b);
          for (int c = 0; c < num_mask_channel_; c++) {
            mask_map(row, c) = p_mask_uint8[c * num_map + boxes_temp_info[bboxs.first][i].second] * maskinfo.qscale;
          }
        } else if (maskinfo.data_type == TDLDataType::FP32) {
          float *p_mask_float = mask_tensor->getBatchPtr<float>(b);
          for (int c = 0; c < num_mask_channel_; c++) {
            mask_map(row, c) = p_mask_float[c * num_map + boxes_temp_info[bboxs.first][i].second];
          }
        } else {
          LOGE("unsupported data type:%d\n", maskinfo.data_type);
          assert(0);
        }
        row++;
      }
    }

    // obtain prototype branch data
    auto firstElement = proto_out_names.begin();
    int proto_stride = firstElement->first;
    std::string proto_output_name = firstElement->second;
    TensorInfo protoinfo = net_->getTensorInfo(proto_output_name);
    std::shared_ptr<BaseTensor> proto_tensor = net_->getOutputTensor(proto_output_name);
    int proto_c = protoinfo.shape[1];
    int proto_h = protoinfo.shape[2];
    int proto_w = protoinfo.shape[3];
    int proto_hw = proto_h * proto_w;

    Eigen::MatrixXf proto_output(proto_c, proto_hw);  // (32,96*160)
    if (protoinfo.data_type == TDLDataType::INT8) {
        int8_t *p_proto_int8 = proto_tensor->getBatchPtr<int8_t>(b);
        for (int i = 0; i < proto_c; i++) {
          for (int j = 0; j < proto_hw; j++) {
            proto_output(i, j) = p_proto_int8[i * proto_hw + j] * protoinfo.qscale;
          }
        }
    } else if (protoinfo.data_type == TDLDataType::UINT8) {
        uint8_t *p_proto_uint8 = proto_tensor->getBatchPtr<uint8_t>(b);
        for (int i = 0; i < proto_c; i++) {
          for (int j = 0; j < proto_hw; j++) {
            proto_output(i, j) = p_proto_uint8[i * proto_hw + j] * protoinfo.qscale;
          }
        }
    } else if (protoinfo.data_type == TDLDataType::FP32) {
        float *p_proto_float = proto_tensor->getBatchPtr<float>(b);
        for (int i = 0; i < proto_c; i++) {
          for (int j = 0; j < proto_hw; j++) {
            proto_output(i, j) = p_proto_float[i * proto_hw + j];
          }
        }
    } else {
        LOGE("unsupported data type:%d\n", protoinfo.data_type);
        assert(0);
    }
    Eigen::MatrixXf masks_output = mask_map * proto_output;

    obj_seg->mask_height = proto_h;
    obj_seg->mask_width = proto_w;
    for (uint32_t i = 0; i < obj_seg->box_seg.size(); i++) {
      int x1 = static_cast<int>(round(obj_seg->box_seg[i].x1 / proto_stride));
      int x2 = static_cast<int>(round(obj_seg->box_seg[i].x2 / proto_stride));
      int y1 = static_cast<int>(round(obj_seg->box_seg[i].y1 / proto_stride));
      int y2 = static_cast<int>(round(obj_seg->box_seg[i].y2 / proto_stride));
      if (obj_seg->box_seg[i].mask != nullptr) {
          free(obj_seg->box_seg[i].mask); 
      }
      obj_seg->box_seg[i].mask = (uint8_t*)malloc(proto_hw * sizeof(uint8_t)); 
      if (obj_seg->box_seg[i].mask == nullptr) {
          LOGE("Failed to allocate memory for mask_property\n");
      }
      memset(obj_seg->box_seg[i].mask, 0, proto_hw * sizeof(uint8_t));
      for (int j = y1; j < y2; ++j) {
        for (int k = x1; k < x2; ++k) {
          if (1 / (1 + exp(-masks_output(i, j * proto_w + k))) >= 0.5) {
            obj_seg->box_seg[i].mask[j * proto_w + k] = 255;
          } else {
            obj_seg->box_seg[i].mask[j * proto_w + k] = 0;
          }
        }
      }
      DetectionHelper::rescaleBbox(obj_seg->box_seg[i], scale_params,
                                     net_param_.pre_params.crop_x,
                                     net_param_.pre_params.crop_y);
      ss << "bbox:[" << obj_seg->box_seg[i].x1 << "," << obj_seg->box_seg[i].y1 << "," <<obj_seg->box_seg[i].x2 << "," << obj_seg->box_seg[i].y2
           << "],score:" << obj_seg->box_seg[i].score << ",label:" << obj_seg->box_seg[i].class_id << "\n";
    }
    out_datas.push_back(obj_seg);
  }
  LOGI("outputParse done,ss:%s", ss.str().c_str());
  return 0;
}
