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
