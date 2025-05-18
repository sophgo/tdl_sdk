#include "segmentation/topformer_seg.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

#define R_SCALE 0.01712475
#define G_SCALE 0.0175070
#define B_SCALE 0.0174291
#define NAME_SCORE 0

template <typename T, typename U>
void nearestNeighborInterpolation(
    std::shared_ptr<ModelSegmentationInfo>& filter, T* out_id, U* out_conf,
    int outH, int outW, int preH, int preW, int outShapeH, int outShapeW) {
  filter->class_id = (uint8_t*)malloc(outH * outW * sizeof(uint8_t));
  filter->class_conf = (uint8_t*)malloc(outH * outW * sizeof(uint8_t));

  uint8_t* class_id = filter->class_id;
  uint8_t* class_conf = filter->class_conf;
  filter->output_height = outH;
  filter->output_width = outW;
  float scale_H = static_cast<float>(preH - 1) / static_cast<float>(outH - 1);
  float scale_W = static_cast<float>(preW - 1) / static_cast<float>(outW - 1);
  int* srcX_s = new int[outH];
  int* srcY_s = new int[outW];

  for (int x = 0; x < outH; ++x) {
    srcX_s[x] = static_cast<int>((x + 0.5) * scale_H);
  }
  for (int y = 0; y < outW; ++y) {
    srcY_s[y] = static_cast<int>((y + 0.5) * scale_W);
  }

  for (int x = 0; x < outH; ++x) {
    int srcX = srcX_s[x];
    for (int y = 0; y < outW; ++y) {
      int srcY = srcY_s[y];
      class_id[x * outW + y] =
          static_cast<uint8_t>(out_id[srcX * outShapeW + srcY]);
      class_conf[x * outW + y] =
          static_cast<uint8_t>(out_conf[srcX * outShapeW + srcY]);
    }
  }

  delete[] srcX_s;
  delete[] srcY_s;
}

TopformerSeg::TopformerSeg() : TopformerSeg(16) {}

TopformerSeg::TopformerSeg(int down_rato) {
  net_param_.model_config.mean = {2.117903 / R_SCALE, 2.035714 / G_SCALE,
                                  1.804444 / B_SCALE};
  net_param_.model_config.std = {1.0 / R_SCALE, 1.0 / G_SCALE, 1.0 / B_SCALE};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;

  oriW = 0;
  oriH = 0;
  outW = 0;
  outH = 0;
  preW = 0;
  preH = 0;
  outShapeH = 0;
  outShapeW = 0;
  downRato = down_rato;
}
TopformerSeg::~TopformerSeg() {}

int32_t TopformerSeg::outputParse(
    const std::vector<std::shared_ptr<BaseImage>>& images,
    std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);

  LOGI("outputParse,batch size:%d,input shape:%d,%d,%d,%d", images.size(),
       input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
       input_tensor.shape[3]);

  std::string out_id_name = net_->getOutputNames()[0];
  std::string out_conf_name = net_->getOutputNames()[1];

  TensorInfo out_id_info = net_->getTensorInfo(out_id_name);
  std::shared_ptr<BaseTensor> out_id_tensor =
      net_->getOutputTensor(out_id_name);

  TensorInfo out_conf_info = net_->getTensorInfo(out_conf_name);
  std::shared_ptr<BaseTensor> out_conf_tensor =
      net_->getOutputTensor(out_conf_name);

#if defined(__BM168X__) || defined(__CV186X__)
  outShapeH = out_id_info.shape[2];
  outShapeW = out_id_info.shape[3];
#else
  outShapeH = out_id_info.shape[1];
  outShapeW = out_id_info.shape[2];
#endif

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    oriW = image_width;
    oriH = image_height;

    float* out_conf = out_conf_tensor->getBatchPtr<float>(b);

    std::shared_ptr<ModelSegmentationInfo> seg =
        std::make_shared<ModelSegmentationInfo>();

    float oriWHRato = static_cast<float>(oriW) / static_cast<float>(oriH);
    float preWHRato =
        static_cast<float>(outShapeW) / static_cast<float>(outShapeH);

    if (preWHRato > oriWHRato) {
      preW = std::ceil(outShapeH * oriWHRato);
      preH = outShapeH;
    } else {
      preW = outShapeW;
      preH = std::ceil(outShapeW / oriWHRato);
    }

    outW = std::ceil(static_cast<float>(image_width) / downRato);
    outH = std::ceil(static_cast<float>(image_height) / downRato);

    if (out_id_info.data_type == TDLDataType::FP32) {
      nearestNeighborInterpolation(seg, out_id_tensor->getBatchPtr<float>(b),
                                   out_conf, outH, outW, preH, preW, outShapeH,
                                   outShapeW);
    } else if (out_id_info.data_type == TDLDataType::INT32) {
      nearestNeighborInterpolation(seg, out_id_tensor->getBatchPtr<int32_t>(b),
                                   out_conf, outH, outW, preH, preW, outShapeH,
                                   outShapeW);
    } else {
      LOGE("unsupported data type:%d\n", out_id_info.data_type);
      return -1;
    }
    out_datas.push_back(seg);
  }

  return 0;
}
