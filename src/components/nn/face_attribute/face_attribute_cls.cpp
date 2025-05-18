#include "face_attribute/face_attribute_cls.hpp"

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

FaceAttribute_CLS::FaceAttribute_CLS() {
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {1.0 / 0.003922, 1.0 / 0.003922,
                                 1.0 / 0.003922};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = false;
}

FaceAttribute_CLS::~FaceAttribute_CLS() {}

int32_t FaceAttribute_CLS::onModelOpened() {
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();
  for (size_t j = 0; j < num_output; j++) {
    if (gender_name.empty() &&
        output_layers[j].find("gender") != std::string::npos) {
      gender_name = output_layers[j];
    } else if (age_name.empty() &&
               output_layers[j].find("age") != std::string::npos) {
      age_name = output_layers[j];
    } else if (glass_name.empty() &&
               output_layers[j].find("glass") != std::string::npos) {
      glass_name = output_layers[j];
    } else if (mask_name.empty() &&
               output_layers[j].find("mask") != std::string::npos) {
      mask_name = output_layers[j];
    }
  }

  return 0;
}
int32_t FaceAttribute_CLS::inference(
    const std::shared_ptr<BaseImage> &image,
    const std::shared_ptr<ModelOutputInfo> &model_object_infos,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  std::vector<ObjectBoxInfo> crop_boxes;
  if (model_object_infos->getType() == ModelOutputType::OBJECT_DETECTION) {
    std::shared_ptr<ModelBoxInfo> model_box_infos =
        std::static_pointer_cast<ModelBoxInfo>(model_object_infos);
    for (uint32_t i = 0; i < model_box_infos->bboxes.size(); i++) {
      crop_boxes.push_back(model_box_infos->bboxes[i]);
    }
  } else if (model_object_infos->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    std::shared_ptr<ModelBoxLandmarkInfo> model_box_infos =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(model_object_infos);
    // TODO(fuquan.ke):use landmarks to adjust crop boxes
    for (uint32_t i = 0; i < model_box_infos->box_landmarks.size(); i++) {
      ObjectBoxInfo box_info;
      box_info.x1 = model_box_infos->box_landmarks[i].x1;
      box_info.y1 = model_box_infos->box_landmarks[i].y1;
      box_info.x2 = model_box_infos->box_landmarks[i].x2;
      box_info.y2 = model_box_infos->box_landmarks[i].y2;
      crop_boxes.push_back(box_info);
    }
  } else {
    LOGE("not supported model output type: %d",
         (int)model_object_infos->getType());
    return -1;
  }
  std::string input_layer_name = net_->getInputNames()[0];
  PreprocessParams &preprocess_params = preprocess_params_[input_layer_name];
  std::vector<std::shared_ptr<BaseImage>> batch_images{image};
  bool keep_aspect_ratio = preprocess_params.keep_aspect_ratio;
  for (uint32_t i = 0; i < crop_boxes.size(); i++) {
    preprocess_params.crop_x = (uint32_t)crop_boxes[i].x1;
    preprocess_params.crop_y = (uint32_t)crop_boxes[i].y1;
    preprocess_params.crop_width =
        (uint32_t)(crop_boxes[i].x2 - crop_boxes[i].x1);
    preprocess_params.crop_height =
        (uint32_t)(crop_boxes[i].y2 - crop_boxes[i].y1);
    preprocess_params.keep_aspect_ratio = false;

    std::vector<std::shared_ptr<ModelOutputInfo>> batch_out_datas;
    int ret = BaseModel::inference(batch_images, batch_out_datas);
    if (ret != 0) {
      LOGE("inference failed");
      return ret;
    }
    out_datas.push_back(batch_out_datas[0]);

    // reset preprocess params
    preprocess_params.crop_x = 0;
    preprocess_params.crop_y = 0;
    preprocess_params.crop_width = 0;
    preprocess_params.crop_height = 0;
    preprocess_params.keep_aspect_ratio = keep_aspect_ratio;
  }

  return 0;
}
int32_t FaceAttribute_CLS::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  TensorInfo oinfo_gender = net_->getTensorInfo(gender_name);
  std::shared_ptr<BaseTensor> gender_tensor =
      net_->getOutputTensor(gender_name);

  TensorInfo oinfo_age = net_->getTensorInfo(age_name);
  std::shared_ptr<BaseTensor> age_tensor = net_->getOutputTensor(age_name);

  TensorInfo oinfo_glass = net_->getTensorInfo(glass_name);
  std::shared_ptr<BaseTensor> glass_tensor = net_->getOutputTensor(glass_name);

  TensorInfo oinfo_mask = net_->getTensorInfo(mask_name);
  std::shared_ptr<BaseTensor> mask_tensor = net_->getOutputTensor(mask_name);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    std::vector<float> feature;
    if (oinfo_gender.data_type != TDLDataType::FP32 ||
        oinfo_age.data_type != TDLDataType::FP32 ||
        oinfo_glass.data_type != TDLDataType::FP32 ||
        oinfo_mask.data_type != TDLDataType::FP32) {
      LOGE("not supported data type: gender=%d, age=%d, glass=%d, mask=%d",
           (int)oinfo_gender.data_type, (int)oinfo_age.data_type,
           (int)oinfo_glass.data_type, (int)oinfo_mask.data_type);
      return -1;
    }
    float *gender_score = gender_tensor->getBatchPtr<float>(b);
    float *age_score = age_tensor->getBatchPtr<float>(b);
    float *glass_score = glass_tensor->getBatchPtr<float>(b);
    float *mask_score = mask_tensor->getBatchPtr<float>(b);

    std::shared_ptr<ModelAttributeInfo> attr_info =
        std::make_shared<ModelAttributeInfo>();
    attr_info
        ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER] =
        gender_score[0];
    attr_info->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE] =
        age_score[0];
    attr_info
        ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES] =
        glass_score[0];
    attr_info->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK] =
        mask_score[0];

    out_datas.push_back(attr_info);
  }
  return 0;
}
