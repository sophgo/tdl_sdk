#include "keypoints_detection/hand_keypoint.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

#define R_SCALE (0.003922 / 0.229)
#define G_SCALE (0.003922 / 0.224)
#define B_SCALE (0.003922 / 0.225)
#define R_MEAN (0.485 / 0.229)
#define G_MEAN (0.456 / 0.224)
#define B_MEAN (0.406 / 0.225)

HandKeypoint::HandKeypoint() {
  net_param_.model_config.mean = {R_MEAN / R_SCALE, G_MEAN / G_SCALE,
                                  B_MEAN / B_SCALE};
  net_param_.model_config.std = {1.0 / R_SCALE, 1.0 / G_SCALE, 1.0 / B_SCALE};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = false;
}

HandKeypoint::~HandKeypoint() {}

int32_t HandKeypoint::inference(
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
    float expansion_factor = 1.25f;
    int new_width = static_cast<int>(width * expansion_factor);
    int new_height = static_cast<int>(height * expansion_factor);
    int crop_x = (int)crop_boxes[i].x1 - (new_width - width) / 2;
    int crop_y = (int)crop_boxes[i].y1 - (new_height - height) / 2;
    std::shared_ptr<BaseImage> human_crop =
        preprocessor_->crop(image, crop_x, crop_y, new_width, new_height);
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

int32_t HandKeypoint::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  // uint32_t input_width = input_tensor.shape[3];
  // uint32_t input_height = input_tensor.shape[2];
  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  std::string out_data_name = net_->getOutputNames()[0];

  TensorInfo out_data_info = net_->getTensorInfo(out_data_name);
  std::shared_ptr<BaseTensor> out_data_tensor =
      net_->getOutputTensor(out_data_name);

  LOGI("output shape:%d,%d,%d,%d\n", out_data_info.shape[0],
       out_data_info.shape[1], out_data_info.shape[2], out_data_info.shape[3]);
  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    std::shared_ptr<ModelLandmarksInfo> obj =
        std::make_shared<ModelLandmarksInfo>();

    obj->image_width = image_width;
    obj->image_height = image_height;

    float *out_data = out_data_tensor->getBatchPtr<float>(b);

    for (int k = 0; k < 42; k++) {
      if (k % 2 == 0) {
        obj->landmarks_x.push_back(out_data[k]);
      } else {
        obj->landmarks_y.push_back(out_data[k]);
      }
    }
    // TODO:should restore to original image coordinate
    out_datas.push_back(obj);
  }
  return 0;
}
