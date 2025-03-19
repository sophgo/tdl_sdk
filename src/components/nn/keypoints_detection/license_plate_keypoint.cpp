#include "keypoints_detection/license_plate_keypoint.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

#define SCALE (1 / 127.5)
#define MEAN (1)

LicensePlateKeypoint::LicensePlateKeypoint() {
  for (int i = 0; i < 3; i++) {
    net_param_.pre_params.scale[i] = SCALE;
    net_param_.pre_params.mean[i] = MEAN;
  }
  net_param_.pre_params.keepAspectRatio = false;
  net_param_.pre_params.dstImageFormat = ImageFormat::RGB_PLANAR;
}

LicensePlateKeypoint::~LicensePlateKeypoint() {}

int32_t LicensePlateKeypoint::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  LOGI(
      "outputParse,batch size:%d,input shape:%d,%d,%d,%d",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3]);

  std::string out_data_name = net_->getOutputNames()[0];
  
  TensorInfo out_data_info = net_->getTensorInfo(out_data_name);
  std::shared_ptr<BaseTensor> out_data_tensor =
      net_->getOutputTensor(out_data_name);

  LOGI(
      "output shape:%d,%d,%d,%d\n", out_data_info.shape[0], out_data_info.shape[1],
      out_data_info.shape[2], out_data_info.shape[3]);
  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    std::shared_ptr<ModelLandmarksInfo> obj = std::make_shared<ModelLandmarksInfo>();

    obj->image_width = image_width;
    obj->image_height = image_height;
 
    float *out_data = out_data_tensor->getBatchPtr<float>(b);

    for (int k = 0; k < 4; k++) {
      obj->landmarks_x.push_back(out_data[2 * k] * image_width);
      obj->landmarks_y.push_back(out_data[2 * k + 1] * image_height);
    }
    out_datas.push_back(obj);
  }
  return 0;
}



