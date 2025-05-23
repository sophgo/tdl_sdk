#include "keypoints_detection/simcc_pose.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

#define NUM_KEYPOINTS 17
#define EXPAND_RATIO 2.0f
#define MAX_NUM 5

SimccPose::SimccPose() {
  net_param_.model_config.std = {255.0 * 0.229, 255.0 * 0.224, 255.0 * 0.225};
  net_param_.model_config.mean = {0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = false;
}

SimccPose::~SimccPose() {}

int32_t SimccPose::inference(
    const std::shared_ptr<BaseImage> &image,
    const std::shared_ptr<ModelOutputInfo> &model_object_infos,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
    const std::map<std::string, float> &parameters) {
  std::vector<ObjectBoxInfo> crop_boxes;
  std::shared_ptr<ModelBoxInfo> model_box_infos =
      std::static_pointer_cast<ModelBoxInfo>(model_object_infos);
  for (uint32_t i = 0; i < model_box_infos->bboxes.size(); i++) {
    crop_boxes.push_back(model_box_infos->bboxes[i]);
  }
  std::vector<std::shared_ptr<BaseImage>> batch_images{};
  for (uint32_t i = 0; i < crop_boxes.size(); i++) {
    int width = (int)crop_boxes[i].x2 - (int)crop_boxes[i].x1;
    int height = (int)crop_boxes[i].y2 - (int)crop_boxes[i].y1;
    int crop_x = (int)crop_boxes[i].x1;
    int crop_y = (int)crop_boxes[i].y1;
    std::shared_ptr<BaseImage> human_crop =
        preprocessor_->crop(image, crop_x, crop_y, width, height);
    batch_images.clear();
    batch_images.push_back(human_crop);
    std::vector<std::shared_ptr<ModelOutputInfo>> batch_out_datas;
    int ret = BaseModel::inference(batch_images, batch_out_datas);
    if (ret != 0) {
      LOGE("inference failed");
      return ret;
    }
    out_datas.push_back(batch_out_datas[0]);
  }

  return 0;
}

int32_t SimccPose::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  // uint32_t input_width = input_tensor.shape[3];
  // uint32_t input_height = input_tensor.shape[2];
  // float input_width_f = float(input_width);
  // float input_height_f = float(input_height);
  float inverse_th = std::log(model_threshold_ / (1 - model_threshold_));
  LOGI(
      "outputParse,batch size:%d,input shape:%d,%d,%d,%d,model "
      "threshold:%f,inverse th:%f",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3], model_threshold_,
      inverse_th);

  std::string x_feature_name = net_->getOutputNames()[0];
  std::string y_feature_name = net_->getOutputNames()[1];

  TensorInfo x_feature_info = net_->getTensorInfo(x_feature_name);
  std::shared_ptr<BaseTensor> x_feature_tensor =
      net_->getOutputTensor(x_feature_name);

  TensorInfo y_feature_info = net_->getTensorInfo(y_feature_name);
  std::shared_ptr<BaseTensor> y_feature_tensor =
      net_->getOutputTensor(y_feature_name);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    std::vector<float> scale_params = batch_rescale_params_[b];
    std::shared_ptr<ModelLandmarksInfo> obj =
        std::make_shared<ModelLandmarksInfo>();

    obj->image_width = image_width;
    obj->image_height = image_height;

    float *data_x = x_feature_tensor->getBatchPtr<float>(b);
    float *data_y = y_feature_tensor->getBatchPtr<float>(b);

    for (int i = 0; i < NUM_KEYPOINTS; i++) {
      float *score_start_x = data_x + i * x_feature_info.shape[3];
      float *score_end_x = data_x + (i + 1) * x_feature_info.shape[3];
      auto iter_x = std::max_element(score_start_x, score_end_x);
      uint32_t pos_x = iter_x - score_start_x;
      float score_x = *iter_x;
      float *score_start_y = data_y + i * y_feature_info.shape[3];
      float *score_end_y = data_y + (i + 1) * y_feature_info.shape[3];
      auto iter_y = std::max_element(score_start_y, score_end_y);
      uint32_t pos_y = iter_y - score_start_y;
      float score_y = *iter_y;

      float x = (float)pos_x / EXPAND_RATIO;
      float y = (float)pos_y / EXPAND_RATIO;

      obj->landmarks_x.push_back((x - scale_params[2]) / scale_params[0]);
      obj->landmarks_y.push_back((y - scale_params[3]) / scale_params[1]);
      obj->landmarks_score.push_back(std::min(score_x, score_y));
    }

    out_datas.push_back(obj);
  }

  return 0;
}
