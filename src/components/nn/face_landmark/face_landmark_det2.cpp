#include "face_landmark/face_landmark_det2.hpp"

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
void parse_point_info(T *p_cls_ptr, int num_point, float qscale,
                      std::vector<float> &points) {
  for (int i = 0; i < num_point; i++) {
    float val = p_cls_ptr[i] * qscale;
    points.push_back(val);
  }
}

template <typename T>
void parse_score_info(T *p_cls_ptr, int num_data, float qscale,
                      std::vector<float> &scores) {
  for (int i = 0; i < num_data; i++) {
    float val = p_cls_ptr[i] * qscale;
    float score = 1.0 / (1.0 + exp(-val));
    scores.push_back(score);
  }
}

void parse_point_info_anytype(BaseTensor *tensor, TDLDataType data_type,
                              int batch_idx, int num_point, float qscale,
                              std::vector<float> &points) {
  switch (data_type) {
    case TDLDataType::FP32:
      parse_point_info<float>(tensor->getBatchPtr<float>(batch_idx), num_point,
                              qscale, points);
      break;
    case TDLDataType::INT8:
      parse_point_info<int8_t>(tensor->getBatchPtr<int8_t>(batch_idx),
                               num_point, qscale, points);
      break;
    case TDLDataType::UINT8:
      parse_point_info<uint8_t>(tensor->getBatchPtr<uint8_t>(batch_idx),
                                num_point, qscale, points);
      break;
    default:
      assert(0);
  }
}
FaceLandmarkerDet2::FaceLandmarkerDet2() {
  std::vector<float> means = {1.0, 1.0, 1.0};
  std::vector<float> scales = {1 / 127.5, 1 / 127.5, 1 / 127.5};

  for (int i = 0; i < 3; i++) {
    net_param_.pre_params.scale[i] = scales[i];
    net_param_.pre_params.mean[i] = means[i];
  }
  net_param_.pre_params.dst_image_format = ImageFormat::RGB_PLANAR;
  net_param_.pre_params.keep_aspect_ratio = false;

  //   preprocess_params_[0].rescale_type = RESCALE_NOASPECT;
}

int32_t FaceLandmarkerDet2::onModelOpened() {
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();

  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);

    if (oinfo.shape[1] != 1) {
      out_names_["score"] = output_layers[j];
    } else if (out_names_.count("point_x") == 0) {
      out_names_["point_x"] = output_layers[j];
    } else {
      out_names_["point_y"] = output_layers[j];
    }
  }
  if (out_names_.count("score") == 0 || out_names_.count("point_x") == 0 ||
      out_names_.count("point_y") == 0) {
    return -1;
  }
  //   preprocess_params_[0].rescale_type = RESCALE_NOASPECT;
  //   preprocess_params_[0].keep_aspect_ratio = false;
  return 0;
}

FaceLandmarkerDet2::~FaceLandmarkerDet2() {}

int32_t FaceLandmarkerDet2::inference(
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
    preprocess_params.crop_y= (uint32_t)crop_boxes[i].y1;
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
int32_t FaceLandmarkerDet2::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  TensorInfo oinfo_x = net_->getTensorInfo(out_names_["point_x"]);
  std::shared_ptr<BaseTensor> x_tensor =
      net_->getOutputTensor(out_names_["point_x"]);

  TensorInfo oinfo_y = net_->getTensorInfo(out_names_["point_y"]);
  std::shared_ptr<BaseTensor> y_tensor =
      net_->getOutputTensor(out_names_["point_y"]);

  TensorInfo oinfo_cls = net_->getTensorInfo(out_names_["score"]);
  std::shared_ptr<BaseTensor> cls_tensor =
      net_->getOutputTensor(out_names_["score"]);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::vector<float> output_point_x;
    std::vector<float> output_point_y;
    std::vector<float> output_score;

    parse_point_info_anytype(x_tensor.get(), oinfo_x.data_type, b,
                             oinfo_x.shape[3], oinfo_x.qscale, output_point_x);
    parse_point_info_anytype(y_tensor.get(), oinfo_y.data_type, b,
                             oinfo_y.shape[3], oinfo_y.qscale, output_point_y);

    if (oinfo_cls.data_type == TDLDataType::FP32) {
      parse_score_info<float>(cls_tensor->getBatchPtr<float>(b),
                              oinfo_cls.shape[1], oinfo_cls.qscale,
                              output_score);
    } else if (oinfo_cls.data_type == TDLDataType::INT8) {
      parse_score_info<int8_t>(cls_tensor->getBatchPtr<int8_t>(b),
                               oinfo_cls.shape[1], oinfo_cls.qscale,
                               output_score);
    } else if (oinfo_cls.data_type == TDLDataType::UINT8) {
      parse_score_info<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                oinfo_cls.shape[1], oinfo_cls.qscale,
                                output_score);
    }

    std::shared_ptr<ModelLandmarksInfo> facemeta =
        std::make_shared<ModelLandmarksInfo>();
    facemeta->image_width = image_width;
    facemeta->image_height = image_height;
    facemeta->landmarks_x = output_point_x;
    facemeta->landmarks_y = output_point_y;
    facemeta->landmarks_score = output_score;

    for (int i = 0; i < 5; i++) {
      facemeta->landmarks_x[i] = output_point_x[i] * image_width;
      facemeta->landmarks_y[i] = output_point_y[i] * image_height;
    }

    // blur model
    if (oinfo_cls.shape[1] == 2) {
      facemeta
          ->attributes[TDLObjectAttributeType::OBJECT_CLS_ATTRIBUTE_FACE_BLURNESS] =
          output_score[1];
    }

    out_datas.push_back(facemeta);
  }
  return 0;
}
