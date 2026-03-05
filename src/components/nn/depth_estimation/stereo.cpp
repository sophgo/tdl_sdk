#include "depth_estimation/stereo.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "utils/tdl_log.hpp"

Stereo::Stereo() {
  net_param_.model_config.mean = {0, 0, 0};
  net_param_.model_config.std = {1.0, 1.0, 1.0};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = false;

  w_ = 0;
  h_ = 0;
}

Stereo::~Stereo() {}
int32_t Stereo::onModelOpened() {
  // 获取输入输出层信息
  const auto& input_layers = net_->getInputNames();
  const auto& output_layers = net_->getOutputNames();

  if (input_layers.size() != 2 || output_layers.size() != 1) {
    LOGE("模型输入输出层数量不符合预期，输入层：%zu，输出层：%zu",
         input_layers.size(), output_layers.size());
    return -1;
  }
  return 0;
}
int32_t Stereo::outputParse(
    const std::vector<std::vector<std::shared_ptr<BaseImage>>>& images,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);

  LOGI("outputParse, batch size:%d, input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  std::string output_tensor_name = net_->getOutputNames()[0];
  TensorInfo output_tensor = net_->getTensorInfo(output_tensor_name);
  std::shared_ptr<BaseTensor> output_tensor_ptr =
      net_->getOutputTensor(output_tensor_name);

  h_ = output_tensor.shape[1];
  w_ = output_tensor.shape[2];

  int byte_per_pixel = output_tensor.tensor_size / output_tensor.tensor_elem;
  float qscale_output = byte_per_pixel == 1 ? output_tensor.qscale : 1;

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b][0]->getWidth();
    uint32_t image_height = images[b][0]->getHeight();

    std::shared_ptr<ModelDepthInfo> depth_info =
        std::make_shared<ModelDepthInfo>();

    depth_info->w = w_;
    depth_info->h = h_;

    int pix_size = w_ * h_;
    depth_info->logits = (float*)malloc(pix_size * sizeof(float));

    if (byte_per_pixel == 1) {
      int8_t* int8_out_data = output_tensor_ptr->getBatchPtr<int8_t>(b);
      for (int i = 0; i < pix_size; i++) {
        depth_info->logits[i] =
            static_cast<float>(int8_out_data[i] * qscale_output);
      }
    } else {
      float* float_out_data = output_tensor_ptr->getBatchPtr<float>(b);
      for (int i = 0; i < pix_size; i++) {
        depth_info->logits[i] = float_out_data[i] * qscale_output;
      }
    }

    out_datas.push_back(depth_info);
  }

  model_timer_.TicToc("post");
  return 0;
}
