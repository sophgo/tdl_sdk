#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

#define AUDIOFORMATSIZE 2


void construct_model_id_mapping(
    std::map<std::string, ModelType> &model_id_mapping) {
  model_id_mapping["CLS_SOUND_COMMAND"] = ModelType::CLS_SOUND_COMMAND;
  model_id_mapping["CLS_SOUND_BABAY_CRY"] = ModelType::CLS_SOUND_BABAY_CRY;
}


int main(int argc, char** argv) {

  std::map<std::string, ModelType> model_id_mapping;
  construct_model_id_mapping(model_id_mapping);

  if (argc != 6) {
    printf("Usage: %s <model_id_name> <sound_model_path> <bin_data_path> <sample_rate> <seconds>\n", argv[0]);
    printf("model_id_name:\n");
    for (auto &item : model_id_mapping) {
      printf("%s\n", item.first.c_str());
    }
    return -1;
  }

std::string model_id_name = argv[1];
  std::string model_path = argv[2];
  std::string strf = argv[3];
  int sample_rate = atoi(argv[4]);
  int seconds = atoi(argv[5]);
  int frame_size = sample_rate * AUDIOFORMATSIZE * seconds;

  unsigned char buffer[frame_size];
  if (!read_binary_file(strf, buffer, frame_size)) {
    printf("read file failed\n");
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> input_datas;
  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  uint8_t* data_buffer = bin_data->getVirtualAddress()[0];
  memcpy(data_buffer, buffer, frame_size * sizeof(uint8_t));
  input_datas.push_back(bin_data);


  TDLModelFactory model_factory;

  ModelType model_id = model_id_mapping[model_id_name];
  std::shared_ptr<BaseModel> model_sound = model_factory.getModel(
      model_id, model_path);
  if (!model_sound) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> output_info;
  model_sound->inference(input_datas, output_info);

  for (size_t i = 0; i < output_info.size(); i++) {

    std::shared_ptr<ModelClassificationInfo> output_cls =
        std::static_pointer_cast<ModelClassificationInfo>(output_info[i]);  
    printf("data[%d]:\n", i);
    printf("scores: %f\n", output_cls->topk_scores[0]);
    printf("class_ids: %d\n", output_cls->topk_class_ids[0]);

  }

  return 0;
}



// ./sample_bin_audio CLS_SOUND_COMMAND nihaoshiyun_cnn10_126_40_INT8_cv181x.cvimodel xxx.bin 8000 2
// ./sample_bin_audio CLS_SOUND_BABAY_CRY baby_cry_cnn10_188_40_INT8_cv181x.cvimodel xxx.bin 16000 3