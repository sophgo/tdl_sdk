#include <time.h>
#include "audio_classification/fsmn_vad.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_id_name> <model_dir> <bin_data_path> ", argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];

  std::string model_dir = argv[2];
  std::string strf = argv[3];

  int frame_size = 0;

  std::vector<uint8_t> buffer;
  if (!CommonUtils::readBinaryFile(strf, buffer)) {
    printf("read file failed\n");
    return -1;
  }
  frame_size = buffer.size();
  std::cout << "frame_size:" << frame_size << std::endl;

  const int sample_rate = 16000;
  int64_t speech_sample_count = frame_size / sizeof(int16_t);

  std::cout << "音频采样点数量：" << speech_sample_count << std::endl;
  double audio_duration =
      static_cast<double>(speech_sample_count) / sample_rate;  // 音频时长（秒）
  std::cout << "音频时长（秒）：" << audio_duration << std::endl;

  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  uint8_t *data_buffer = bin_data->getVirtualAddress()[0];
  memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_sound =
      model_factory.getModel(model_id_name);
  if (!model_sound) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::shared_ptr<ModelOutputInfo> output_info;

  model_sound->inference(bin_data, output_info);
  // 评测耗时
  // for (int i = 0; i < 1001; i++) {
  //   output_info.reset();
  //   model_sound->inference(bin_data, output_info);
  // }
  std::shared_ptr<ModelVADInfo> vad_meta =
      std::static_pointer_cast<ModelVADInfo>(output_info);
  std::cout << "segments = [";
  for (size_t i = 0; i < vad_meta->segments.size(); ++i) {
    std::cout << "[";
    const auto &outer = vad_meta->segments[i];
    for (size_t j = 0; j < outer.size(); ++j) {
      if (j > 0) std::cout << ", ";
      std::cout << "[";
      const auto &inner = outer[j];
      for (size_t k = 0; k < inner.size(); ++k) {
        if (k > 0) std::cout << ", ";
        std::cout << inner[k];
      }
      std::cout << "]";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;

  return 0;
}
