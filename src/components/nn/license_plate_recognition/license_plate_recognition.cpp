
#include "license_plate_recognition.hpp"
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

#define SCALE (1 / 128.)
#define MEAN (127.5 / 128.)



std::vector<std::string> CHARS = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "学",
    "警", "港", "澳", "挂", "使", "领", "民", "深", "危", "险", "空", "0",  "1",  "2",  "3",  "4",
    "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",  "H",  "J",  "K",  "L",
    "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-"};

LicensePlateRecognition::LicensePlateRecognition(){
  for (int i = 0; i < 3; i++) {
    net_param_.pre_params.scale[i] = SCALE;
    net_param_.pre_params.mean[i] = MEAN;
  }

  net_param_.pre_params.dst_image_format = ImageFormat::BGR_PLANAR;
  net_param_.pre_params.keep_aspect_ratio = false;
}

std::string LicensePlateRecognition::greedy_decode(float *prebs) {
  auto shape = net_->getTensorInfo(net_->getOutputNames()[0]).shape;

  const auto &output_layer = net_->getOutputNames()[0];
  auto output_shape = net_->getTensorInfo(output_layer).shape;

  // 80，18
  int rows = output_shape[1];
  int cols = output_shape[2];

  int index[cols];
  // argmax index
  for (int i = 0; i < cols; i++) {
    float max = prebs[i];
    int maxIndex = 0;
    for (int j = 0; j < rows; j++) {
      if (prebs[i + j * cols] > max) {
        max = prebs[i + j * cols];
        maxIndex = j;
      }
    }
    index[i] = maxIndex;
  }
  std::vector<int> no_repeat_blank_label;
  uint32_t pre_c = index[0];
  if (pre_c != CHARS.size() - 1) {
    no_repeat_blank_label.push_back(pre_c);
  }
  for (int i = 0; i < cols; i++) {
    uint32_t c = index[i];
    if ((pre_c == c) || (c == CHARS.size() - 1)) {
      if (c == CHARS.size() - 1) {
        pre_c = c;
      }
      continue;
    }
    no_repeat_blank_label.push_back(c);
    pre_c = c;
  }
  std::string lb;
  for (int k : no_repeat_blank_label) {
    lb += CHARS[k];
  }

  return lb;
}

int32_t LicensePlateRecognition::outputParse(
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

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();
    std::shared_ptr<ModelOcrInfo> ocr_info = std::make_shared<ModelOcrInfo>();

    float *out_data = out_data_tensor->getBatchPtr<float>(b);

    std::string license = greedy_decode(out_data);

    ocr_info->length = license.length();
    ocr_info->text_info = new char[license.length()];
    strcpy(ocr_info->text_info, license.c_str());

    out_datas.push_back(ocr_info);
  }
  return 0;
}
