#include "tdl_model_factory.hpp"
#include "utils/tokenizer_bpe.hpp"

int main(int argc, char** argv) {
  std::vector<std::shared_ptr<BaseImage>> input_texts;
  std::string encoderFile = "./encoder.txt";
  std::string bpeFile = "./bpe_simple_vocab_16e6.txt";
  std::string textFile = "./text.txt";
  std::vector<std::vector<int32_t>> tokens;
  BytePairEncoder bpe(encoderFile, bpeFile);
  int result = bpe.tokenizerBPE(textFile, tokens);
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

    input_texts.push_back(text);
  }

  std::string model_dir = argv[1];
  TDLModelFactory model_factory(model_dir);

  std::shared_ptr<BaseModel> model_clip_text =
      model_factory.getModel(ModelType::TEXT_FEATURE_CLIP, argv[1]);

  if (!model_clip_text) {
    printf("Failed to load clip text model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_fe;
  model_clip_text->inference(input_texts, out_fe);
  std::vector<std::vector<float>> features;
  for (size_t i = 0; i < out_fe.size(); i++) {
    std::shared_ptr<ModelClipFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelClipFeatureInfo>(out_fe[i]);
    printf("feature size: %d\n", feature_meta->embedding_num);
    std::vector<float> feature_vec(feature_meta->embedding_num);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      feature_vec[j] = feature_meta->embedding[j];
      std::cout << feature_vec[j] << " ";
    }
    std::cout << std::endl;
    features.push_back(feature_vec);
  }

  return 0;
}