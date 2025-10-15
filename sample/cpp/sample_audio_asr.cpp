#include <time.h>
#include "speech_recognition/zipformer_encoder.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_dir> <tokens_path> <bin_data_path> ",
           argv[0]);  // bin data with 16000 sr
    return -1;
  }

  std::string model_dir = argv[1];
  std::string tokens_path = argv[2];
  std::string strf = argv[3];

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model_zipformer_encoder =
      model_factory.getModel("RECOGNITION_SPEECH_ZIPFORMER_ENCODER");
  if (!model_zipformer_encoder) {
    printf("Failed to create model_zipformer_encoder\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_zipformer_decoder =
      model_factory.getModel("RECOGNITION_SPEECH_ZIPFORMER_DECODER");
  if (!model_zipformer_decoder) {
    printf("Failed to create model_zipformer_decoder\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_zipformer_joiner =
      model_factory.getModel("RECOGNITION_SPEECH_ZIPFORMER_JOINER");
  if (!model_zipformer_joiner) {
    printf("Failed to create model_zipformer_joiner\n");
    return -1;
  }

  auto zipformer =
      std::dynamic_pointer_cast<ZipformerEncoder>(model_zipformer_encoder);

  zipformer->setTokensPath(tokens_path);
  zipformer->setModel(model_zipformer_decoder, model_zipformer_joiner);

  std::vector<uint8_t> buffer;
  if (!CommonUtils::readBinaryFile(strf, buffer)) {
    printf("read file failed\n");
    return -1;
  }

  for (int i = 0; i < 16000 * 0.5 * 2;
       i++) {  // pad 0.5s, *2 for uint8_t to int16
    buffer.push_back(0);
  }

  int frame_size = buffer.size();

  std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
      frame_size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  uint8_t *data_buffer = bin_data->getVirtualAddress()[0];
  memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));

  std::shared_ptr<ModelASRInfo> asr_meta = std::make_shared<ModelASRInfo>();
  std::shared_ptr<ModelOutputInfo> output_data =
      std::static_pointer_cast<ModelOutputInfo>(asr_meta);

  zipformer->inference(bin_data, output_data);

  std::shared_ptr<ModelASRInfo> tmp_meta =
      std::static_pointer_cast<ModelASRInfo>(output_data);

  printf("ASR result: \n");
  if (tmp_meta->text_info) {
    printf("%s\n", tmp_meta->text_info);
  } else {
    printf("(no speech detected)\n");
  }

  return 0;
}
