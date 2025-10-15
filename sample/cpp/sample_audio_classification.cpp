#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

#define AUDIOFORMATSIZE 2

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_id_name> <model_dir> <bin_data_path> ", argv[0]);
    return -1;
  }

  std::string model_id_name = argv[1];

  std::string model_dir = argv[2];
  std::string strf = argv[3];
  // int sample_rate = atoi(argv[4]);
  // int seconds = atoi(argv[5]);
  int frame_size = 0;

  std::vector<uint8_t> buffer;
  if (!CommonUtils::readBinaryFile(strf, buffer)) {
    printf("read file failed\n");
    return -1;
  }
  frame_size = buffer.size();
  std::vector<std::shared_ptr<BaseImage>> input_datas;
  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  uint8_t *data_buffer = bin_data->getVirtualAddress()[0];
  memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));
  input_datas.push_back(bin_data);

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_sound =
      model_factory.getModel(model_id_name);
  if (!model_sound) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> output_info;
  model_sound->inference(input_datas, output_info);

  for (size_t i = 0; i < output_info.size(); i++) {
    std::shared_ptr<ModelClassificationInfo> output_cls =
        std::static_pointer_cast<ModelClassificationInfo>(output_info[i]);
    printf("data[%ld]:\n", i);
    printf("scores: %f\n", output_cls->topk_scores[0]);
    printf("class_ids: %d\n", output_cls->topk_class_ids[0]);
  }

  return 0;
}

// ./sample_bin_audio CLS_SOUND_COMMAND
// nihaoshiyun_cnn10_126_40_INT8_cv181x.cvimodel xxx.bin 8000 2
// ./sample_bin_audio CLS_SOUND_BABAY_CRY
// baby_cry_cnn10_188_40_INT8_cv181x.cvimodel xxx.bin 16000 3