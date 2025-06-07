#include "object_detection/yolox.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

float yolox_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

template <typename T>
void get_box_vals(T *ptr, float qscale, int basic_pos, int grid0, int grid1,
                  int stride, std::vector<float> &decode_box) {
  // 计算中心点坐标和宽高
  float x_center = (ptr[basic_pos + 0] * qscale + grid0) * stride;
  float y_center = (ptr[basic_pos + 1] * qscale + grid1) * stride;
  float w = std::exp(ptr[basic_pos + 2] * qscale) * stride;
  float h = std::exp(ptr[basic_pos + 3] * qscale) * stride;

  float x0 = x_center - w * 0.5f;
  float y0 = y_center - h * 0.5f;
  float x1 = x0 + w;
  float y1 = y0 + h;

  // 清空并写入 decode_box
  decode_box.clear();
  decode_box.push_back(x0);
  decode_box.push_back(y0);
  decode_box.push_back(x1);
  decode_box.push_back(y1);
}

void YoloXDetection::decodeBboxFeatureMap(int batch_idx, int stride,
                                          int basic_pos, int grid0, int grid1,
                                          std::vector<float> &decode_box) {
  std::string box_name;
  if (box_out_names_.count(stride)) {
    box_name = box_out_names_[stride];
  } else {
    LOGE("No box name found for stride %d\n", stride);
    return;
  }

  TensorInfo boxinfo = net_->getTensorInfo(box_name);
  std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);

  float qscale = boxinfo.qscale;

  if (boxinfo.data_type == TDLDataType::INT8) {
    int8_t *p_box_int8 = box_tensor->getBatchPtr<int8_t>(batch_idx);
    get_box_vals(p_box_int8, qscale, basic_pos, grid0, grid1, stride,
                 decode_box);
  } else if (boxinfo.data_type == TDLDataType::UINT8) {
    uint8_t *p_box_uint8 = box_tensor->getBatchPtr<uint8_t>(batch_idx);
    get_box_vals(p_box_uint8, qscale, basic_pos, grid0, grid1, stride,
                 decode_box);
  } else if (boxinfo.data_type == TDLDataType::FP32) {
    float *p_box_float = box_tensor->getBatchPtr<float>(batch_idx);
    get_box_vals(p_box_float, qscale, basic_pos, grid0, grid1, stride,
                 decode_box);
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(boxinfo.data_type));
    return;
  }
}

template <typename T>
int yolox_argmax(T *ptr, int basic_pos, int cls_len) {
  int max_idx = 0;
  for (int i = 0; i < cls_len; i++) {
    if (ptr[i + basic_pos] > ptr[max_idx + basic_pos]) {
      max_idx = i;
    }
  }
  return max_idx;
}

int32_t YoloXDetection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  LOGI(
      "outputParse,batch size:%d,input shape:%d,%d,%d,%d,model "
      "threshold:%f",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3], model_threshold_);

  std::stringstream ss;
  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;
    for (size_t i = 0; i < strides.size(); i++) {
      int stride = strides[i];
      std::string cls_name = class_out_names_[stride];
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);
      int num_cls = classinfo.shape[3];

      std::string obj_name = object_out_names_[stride];
      TensorInfo objectinfo = net_->getTensorInfo(obj_name);
      std::shared_ptr<BaseTensor> obj_tensor = net_->getOutputTensor(obj_name);

      int num_grid_w = input_width / stride;
      int num_grid_h = input_height / stride;

      int basic_pos_class = 0;
      int basic_pos_object = 0;
      int basic_pos_box = 0;

      for (int g1 = 0; g1 < num_grid_h; g1++) {
        for (int g0 = 0; g0 < num_grid_w; g0++) {
          float class_score = 0.0f;
          float box_objectness = 0.0f;
          int label = 0;

          if (objectinfo.data_type == TDLDataType::INT8) {
            box_objectness =
                obj_tensor->getBatchPtr<int8_t>(b)[basic_pos_object] *
                objectinfo.qscale;
          } else if (objectinfo.data_type == TDLDataType::UINT8) {
            box_objectness =
                obj_tensor->getBatchPtr<uint8_t>(b)[basic_pos_object] *
                objectinfo.qscale;
          } else if (objectinfo.data_type == TDLDataType::FP32) {
            box_objectness =
                obj_tensor->getBatchPtr<float>(b)[basic_pos_object] *
                objectinfo.qscale;
          } else {
            LOGE("unsupported data type:%d\n",
                 static_cast<int>(objectinfo.data_type));
            assert(0);
          }

          if (classinfo.data_type == TDLDataType::INT8) {
            label = yolox_argmax<int8_t>(cls_tensor->getBatchPtr<int8_t>(b),
                                         basic_pos_class, num_cls);
            class_score =
                cls_tensor->getBatchPtr<int8_t>(b)[basic_pos_class + label] *
                classinfo.qscale;
          } else if (classinfo.data_type == TDLDataType::UINT8) {
            label = yolox_argmax<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                          basic_pos_class, num_cls);
            class_score =
                cls_tensor->getBatchPtr<uint8_t>(b)[basic_pos_class + label] *
                classinfo.qscale;
          } else if (classinfo.data_type == TDLDataType::FP32) {
            label = yolox_argmax<float>(cls_tensor->getBatchPtr<float>(b),
                                        basic_pos_class, num_cls);
            class_score =
                cls_tensor->getBatchPtr<float>(b)[basic_pos_class + label] *
                classinfo.qscale;
          } else {
            LOGE("unsupported data type:%d\n",
                 static_cast<int>(classinfo.data_type));
            assert(0);
          }

          box_objectness = yolox_sigmoid(box_objectness);
          class_score = yolox_sigmoid(class_score);
          float box_prob = box_objectness * class_score;
          // std::cout<< box_prob<<std::endl;
          if (box_prob < model_threshold_) {
            basic_pos_class += num_cls;
            basic_pos_box += 4;
            basic_pos_object += 1;
            continue;
          }
          std::vector<float> box;
          decodeBboxFeatureMap(b, stride, basic_pos_box, g0, g1, box);
          ObjectBoxInfo bbox;
          bbox.score = class_score;
          bbox.x1 = std::max(0.0f, std::min(box[0], input_width_f));
          bbox.y1 = std::max(0.0f, std::min(box[1], input_height_f));
          bbox.x2 = std::max(0.0f, std::min(box[2], input_width_f));
          bbox.y2 = std::max(0.0f, std::min(box[3], input_height_f));
          bbox.class_id = label;
          LOGI("bbox:[%f,%f,%f,%f],score:%f,label:%d:%f\n", bbox.x1, bbox.y1,
               bbox.x2, bbox.y2, bbox.score, label);

          lb_boxes[label].push_back(bbox);

          basic_pos_class += num_cls;
          basic_pos_box += 4;
          basic_pos_object += 1;
        }
      }
    }
    DetectionHelper::nmsObjects(lb_boxes, nms_threshold_);
    std::vector<float> scale_params = batch_rescale_params_[b];
    LOGI("scale_params:%f,%f,%f,%f", scale_params[0], scale_params[1],
         scale_params[2], scale_params[3]);
    ss << "batch:" << b << "\n";

    std::shared_ptr<ModelBoxInfo> obj = std::make_shared<ModelBoxInfo>();
    obj->image_width = image_width;
    obj->image_height = image_height;
    for (auto &bbox : lb_boxes) {
      for (auto &b : bbox.second) {
        DetectionHelper::rescaleBbox(b, scale_params);
        if (type_mapping_.count(b.class_id)) {
          b.object_type = type_mapping_[b.class_id];
        }
        obj->bboxes.push_back(b);
        ss << "bbox:[" << b.x1 << "," << b.y1 << "," << b.x2 << "," << b.y2
           << "],score:" << b.score << ",label:" << bbox.first << "\n";
      }
    }
    out_datas.push_back(obj);
  }
  LOGI("outputParse done,ss:%s", ss.str().c_str());
  return 0;
}

YoloXDetection::YoloXDetection() {
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {1.0, 1.0, 1.0};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;
}

int YoloXDetection::onModelOpened() {
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

    if (j % 3 == 2) {
      class_out_names_[stride_h] = output_layers[j];
      LOGI("class feature %s: (%d %d %d %d)\n", output_layers[j].c_str(),
           oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
      strides.push_back(stride_h);
    } else if (j % 3 == 0) {
      box_out_names_[stride_h] = output_layers[j];
      LOGI("box feature %s: (%d %d %d %d)\n", output_layers[j].c_str(),
           oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
    } else {
      object_out_names_[stride_h] = output_layers[j];
      LOGI("object feature %s: (%d %d %d %d)\n", output_layers[j].c_str(),
           oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
    }
  }
  for (size_t i = 0; i < strides.size(); i++) {
    if (!class_out_names_.count(strides[i]) ||
        !box_out_names_.count(strides[i]) ||
        !object_out_names_.count(strides[i])) {
      return -1;
    }
  }

  return 0;
}

YoloXDetection::~YoloXDetection() {}
