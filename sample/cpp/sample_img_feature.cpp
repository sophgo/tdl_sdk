#include "tdl_model_factory.hpp"
#include "utils/tokenizer_bpe.hpp"

void checkModelIdName(const std::string& model_id_name) {
  if (model_id_name != "CLIP_FEATURE_IMG" &&
      model_id_name != "CLIP_FEATURE_TEXT" &&
      model_id_name != "RESNET_FEATURE_BMFACE_R34" &&
      model_id_name != "RESNET_FEATURE_BMFACE_R50" &&
      model_id_name != "RECOGNITION_CVIFACE") {
    std::cerr << "model_id_name: " << model_id_name << " not supported"
              << std::endl;
    exit(1);
  }
}

template <typename InT, typename OutT>
void embeddingToVec(void* embedding, size_t num,
                    std::vector<OutT>& feature_vec) {
  InT* feature = reinterpret_cast<InT*>(embedding);
  for (size_t i = 0; i < num; ++i) {
    feature_vec[i] = static_cast<OutT>(feature[i]);
  }
  return;
}

void getInputData(const std::string& model_id_name, const std::string& argv_3,
                  std::vector<std::shared_ptr<BaseImage>>& input_datas) {
  if (model_id_name == "CLIP_FEATURE_TEXT") {
    std::string txt_dir = argv_3;
    if (!txt_dir.empty() && txt_dir.back() == '/') {
      txt_dir.pop_back();
    }
    std::string encoder_file = txt_dir + "/encoder.txt";
    std::string bpe_file = txt_dir + "/vocab.txt";
    std::string input_file = txt_dir + "/input.txt";
    std::vector<std::vector<int32_t>> tokens;
    BytePairEncoder bpe(encoder_file, bpe_file);
    bpe.tokenizerBPE(input_file, tokens);
    // 打印 tokens 的内容
    for (size_t i = 0; i < tokens.size(); ++i) {
      for (size_t j = 0; j < tokens[i].size(); ++j) {
        std::cout << tokens[i][j] << " ";
      }
      std::cout << "Current token index i: " << i << std::endl;

      std::shared_ptr<BaseImage> text = ImageFactory::createImage(
          77, 1, ImageFormat::GRAY, TDLDataType::INT32, true);
      uint8_t* txt_buffer = text->getVirtualAddress()[0];
      memcpy(txt_buffer, tokens[i].data(), 77 * sizeof(int32_t));

      input_datas.push_back(text);
    }
  } else {
    std::string image_path = argv_3;
    auto image = ImageFactory::readImage(image_path);
    if (!image) {
      std::cerr << "Failed to load images" << std::endl;
    }
    input_datas.push_back(image);
  }
}

template <typename T>
struct FeatureVec {
  std::vector<T> data;
};

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <model_id_name> <model_dir> <image_path>\n", argv[0]);
    printf("Usage: %s <model_id_name> <model_dir> <txt_dir>\n", argv[0]);
    return -1;
  }
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::vector<std::shared_ptr<BaseImage>> input_datas;

  checkModelIdName(model_id_name);
  getInputData(model_id_name, argv[3], input_datas);

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id_name);
  if (!model) {
    printf("Failed to load model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_features;
  model->inference(input_datas, out_features);

  std::vector<std::shared_ptr<void>> features;
  std::shared_ptr<ModelFeatureInfo> feature_meta;
  for (size_t i = 0; i < out_features.size(); i++) {
    feature_meta = std::static_pointer_cast<ModelFeatureInfo>(out_features[i]);
    printf("feature size: %d\n", feature_meta->embedding_num);

    switch (feature_meta->embedding_type) {
      case TDLDataType::INT8: {
        auto vec = std::make_shared<FeatureVec<int8_t>>();
        vec->data.resize(feature_meta->embedding_num);
        embeddingToVec<int8_t, int8_t>(feature_meta->embedding,
                                       feature_meta->embedding_num, vec->data);
        features.push_back(vec);
        FILE* file =
            fopen(("feature_int8_" + std::to_string(i) + ".bin").c_str(), "wb");
        fwrite(vec->data.data(), sizeof(int8_t), feature_meta->embedding_num,
               file);
        fclose(file);
        printf("The feature file have been saved as feature_int8_%d.bin\n", i);
        break;
      }
      case TDLDataType::UINT8: {
        auto vec = std::make_shared<FeatureVec<uint8_t>>();
        vec->data.resize(feature_meta->embedding_num);
        embeddingToVec<uint8_t, uint8_t>(
            feature_meta->embedding, feature_meta->embedding_num, vec->data);
        features.push_back(vec);
        FILE* file = fopen(
            ("feature_uint8_" + std::to_string(i) + ".bin").c_str(), "wb");
        fwrite(vec->data.data(), sizeof(uint8_t), feature_meta->embedding_num,
               file);
        fclose(file);
        printf("The feature file have been saved as feature_uint8_%d.bin\n", i);
        break;
      }
      case TDLDataType::FP32: {
        auto vec = std::make_shared<FeatureVec<float>>();
        vec->data.resize(feature_meta->embedding_num);
        embeddingToVec<float, float>(feature_meta->embedding,
                                     feature_meta->embedding_num, vec->data);
        features.push_back(vec);
        FILE* file =
            fopen(("feature_fp32_" + std::to_string(i) + ".bin").c_str(), "wb");
        fwrite(vec->data.data(), sizeof(float), feature_meta->embedding_num,
               file);
        fclose(file);
        printf("The feature file have been saved as feature_fp32_%d.bin\n", i);
        break;
      }
      default:
        assert(false && "Unsupported embedding_type");
    }
  }

  return 0;
}