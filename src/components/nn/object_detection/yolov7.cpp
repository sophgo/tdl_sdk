#include "object_detection/yolov7.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
int yolov7_argmax(T *ptr, int start_idx, int arr_len) {
  int max_idx = start_idx;
  for (int i = start_idx + 1; i < start_idx + arr_len; i++) {
    if (ptr[i] > ptr[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx - start_idx;
}

float Sigmoid(float x) { return 1.0 / (1 + exp(-x)); }

template <typename T>
void parseDet(T *ptr, float qscale, int start_idx, int grid_x, int grid_y,
              float pw, float ph, int stride, std::vector<float> &decode_box) {
  float sigmoid_x = Sigmoid(ptr[start_idx] * qscale);
  float sigmoid_y = Sigmoid(ptr[start_idx + 1] * qscale);
  float sigmoid_w = Sigmoid(ptr[start_idx + 2] * qscale);
  float sigmoid_h = Sigmoid(ptr[start_idx + 3] * qscale);

  // decode predicted bounding box of each grid to whole image
  float x = (2 * sigmoid_x - 0.5 + (float)grid_x) * (float)stride;
  float y = (2 * sigmoid_y - 0.5 + (float)grid_y) * (float)stride;
  float w = pow((sigmoid_w * 2), 2) * pw;
  float h = pow((sigmoid_h * 2), 2) * ph;

  decode_box.clear();
  decode_box.push_back(x - w / 2);
  decode_box.push_back(y - h / 2);
  decode_box.push_back(x + w / 2);
  decode_box.push_back(y + h / 2);
}

YoloV7Detection::YoloV7Detection() : YoloV7Detection(std::make_pair(4, 80)) {}

YoloV7Detection::YoloV7Detection(std::pair<int, int> yolov7_pair) {
  // default param
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {255.0, 255.0, 255.0};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;

  initial_anchors = new uint32_t[18]{12, 16, 19,  36,  40,  28,  36,  75,  76,
                                     55, 72, 146, 142, 110, 192, 243, 459, 401};
  num_cls = yolov7_pair.second;
}

int YoloV7Detection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  auto input_shape = net_->getTensorInfo(input_layer).shape;
  int input_h = input_shape[2];
  int input_w = input_shape[3];

  strides_.clear();
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();

  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);
    int feat_h = oinfo.shape[1];
    int feat_w = oinfo.shape[2];
    int channel = oinfo.shape[3];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    if (j % 3 == 1) {
      object_out_names_[stride_h] = output_layers[j];
      strides_.push_back(stride_h);
    } else if (j % 3 == 0) {
      box_out_names_[stride_h] = output_layers[j];
    } else {
      class_out_names_[stride_h] = output_layers[j];
    }
  }

  for (size_t i = 0; i < strides_.size(); i++) {
    if (object_out_names_.count(strides_[i]) == 0 ||
        box_out_names_.count(strides_[i]) == 0 ||
        class_out_names_.count(strides_[i]) == 0) {
      return -1;
    }
  }

  return 0;
}

YoloV7Detection::~YoloV7Detection() {
  if (initial_anchors) {
    delete[] initial_anchors;
    initial_anchors = nullptr;
  }
}
void YoloV7Detection::decodeBboxFeatureMap(int batch_idx, int stride,
                                           int basic_pos, int grid_x,
                                           int grid_y, float pw, float ph,
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
    parseDet(p_box_int8, qscale, basic_pos, grid_x, grid_y, pw, ph, stride,
             decode_box);
  } else if (boxinfo.data_type == TDLDataType::UINT8) {
    uint8_t *p_box_uint8 = box_tensor->getBatchPtr<uint8_t>(batch_idx);
    parseDet(p_box_uint8, qscale, basic_pos, grid_x, grid_y, pw, ph, stride,
             decode_box);
  } else if (boxinfo.data_type == TDLDataType::FP32) {
    float *p_box_float = box_tensor->getBatchPtr<float>(batch_idx);
    parseDet(p_box_float, qscale, basic_pos, grid_x, grid_y, pw, ph, stride,
             decode_box);
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(boxinfo.data_type));
    return;
  }
}

int32_t YoloV7Detection::outputParse(
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
    uint32_t anchor_pos = 0;

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;
    for (size_t i = 0; i < strides_.size(); i++) {
      int stride = strides_[i];
      std::string cls_name = class_out_names_[stride];
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);
      int num_cls = classinfo.shape[3];

      std::string obj_name = object_out_names_[stride];
      TensorInfo objectinfo = net_->getTensorInfo(obj_name);
      std::shared_ptr<BaseTensor> obj_tensor = net_->getOutputTensor(obj_name);

      uint32_t anchor_len = objectinfo.shape[0];

      int num_grid_w = input_width / stride;
      int num_grid_h = input_height / stride;

      int basic_pos_class = 0;
      int basic_pos_object = 0;
      int basic_pos_box = 0;

      for (uint32_t anchor_idx = 0; anchor_idx < anchor_len; anchor_idx++) {
        uint32_t *anchors = initial_anchors + anchor_pos;
        float pw = anchors[0];
        float ph = anchors[1];

        for (int grid_y = 0; grid_y < num_grid_h; grid_y++) {
          for (int grid_x = 0; grid_x < num_grid_w; grid_x++) {
            float class_score = 0.0f;
            float box_objectness = 0.0f;
            int label = 0;

            // parse object conf
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

            // parse class score
            if (classinfo.data_type == TDLDataType::INT8) {
              label = yolov7_argmax<int8_t>(cls_tensor->getBatchPtr<int8_t>(b),
                                            basic_pos_class, num_cls);
              class_score =
                  cls_tensor->getBatchPtr<int8_t>(b)[basic_pos_class + label] *
                  classinfo.qscale;
            } else if (classinfo.data_type == TDLDataType::UINT8) {
              label =
                  yolov7_argmax<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                         basic_pos_class, num_cls);
              class_score =
                  cls_tensor->getBatchPtr<uint8_t>(b)[basic_pos_class + label] *
                  classinfo.qscale;
            } else if (classinfo.data_type == TDLDataType::FP32) {
              label = yolov7_argmax<float>(cls_tensor->getBatchPtr<float>(b),
                                           basic_pos_class, num_cls);
              class_score =
                  cls_tensor->getBatchPtr<float>(b)[basic_pos_class + label] *
                  classinfo.qscale;
            } else {
              LOGE("unsupported data type:%d\n",
                   static_cast<int>(classinfo.data_type));
              assert(0);
            }

            box_objectness = Sigmoid(box_objectness);
            class_score = Sigmoid(class_score);
            float box_prob = box_objectness * class_score;
            if (box_prob < model_threshold_) {
              basic_pos_class += num_cls;
              basic_pos_box += 4;
              basic_pos_object += 1;
              continue;
            }

            std::vector<float> box;
            decodeBboxFeatureMap(b, stride, basic_pos_box, grid_x, grid_y, pw,
                                 ph, box);
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
        anchor_pos += 2;
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