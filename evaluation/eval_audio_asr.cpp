#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "speech_recognition/zipformer_encoder.hpp"
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"

int bench_mark_all(const std::string& txt_file_list_path,
                   const std::string& bin_data_dir, const std::string& res_path,
                   std::shared_ptr<BaseModel> model) {
  std::ifstream bench_fstream(txt_file_list_path);
  if (!bench_fstream.is_open()) {
    std::cerr << "打开 benchmark 文件失败: " << txt_file_list_path << "\n";
    return -1;
  }

  std::ofstream file(res_path);
  if (!file.is_open()) {
    std::cerr << "打开结果文件失败: " << res_path << "\n";
    return -1;
  }

  std::string bin_data_name;
  int count = 0;
  while (bench_fstream >> bin_data_name) {
    const std::string bin_data_path = bin_data_dir + "/" + bin_data_name;
    count += 1;
    std::cout << "Process: " << count << " " << bin_data_path << "\n";
    std::vector<uint8_t> buffer;
    if (!CommonUtils::readBinaryFile(bin_data_path, buffer)) {
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
    uint8_t* data_buffer = bin_data->getVirtualAddress()[0];
    memcpy(data_buffer, buffer.data(), frame_size * sizeof(uint8_t));

    std::shared_ptr<ModelASRInfo> tmp_meta = std::make_shared<ModelASRInfo>();
    tmp_meta->input_finished = true;
    std::shared_ptr<ModelOutputInfo> output_data =
        std::static_pointer_cast<ModelOutputInfo>(tmp_meta);

    auto zipformer = std::dynamic_pointer_cast<ZipformerEncoder>(model);
    zipformer->inference(bin_data, output_data);

    std::shared_ptr<ModelASRInfo> asr_meta =
        std::static_pointer_cast<ModelASRInfo>(output_data);

    if (asr_meta->text_info) {
      printf("text_info: %s\n\n", asr_meta->text_info);
      file << asr_meta->text_info << "\n";
    } else {
      file << "\n";
    }
  }

  file.close();

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf(
        "Usage: %s <model_dir> <tokens_path> <bin_data_dir> "
        "<txt_file_list_path> <txt_result_path>",
        argv[0]);  // bin data with 16000 sr
    return -1;
  }

  std::string model_dir = argv[1];
  std::string tokens_path = argv[2];
  std::string bin_data_dir = argv[3];
  std::string txt_file_list_path = argv[4];
  std::string txt_result_path = argv[5];

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
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

  bench_mark_all(txt_file_list_path, bin_data_dir, txt_result_path, zipformer);

  return 0;
}
