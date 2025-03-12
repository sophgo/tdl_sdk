
#include "tdl_model_factory.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <model_dir> <image_path>\n", argv[0]);
    return -1;
  }

  std::string model_dir = argv[1];
  std::string image_path = argv[2];

  auto image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to load images\n");
    return -1;
  }

  TDLModelFactory model_factory(model_dir);

  std::shared_ptr<BaseModel> model_clip_image =
      model_factory.getModel(TDL_MODEL_TYPE_CLIP_IMAGE, argv[1]);

  if (!model_clip_image) {
    printf("Failed to load clip image model\n");
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> input_images = {image};

  std::vector<std::shared_ptr<ModelOutputInfo>> out_fe;
  model_clip_image->inference(input_images, out_fe);
  std::vector<std::vector<float>> features;
  for (size_t i = 0; i < out_fe.size(); i++) {
    std::shared_ptr<ModelClipFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelClipFeatureInfo>(out_fe[i]);
    printf("feature size: %d\n", feature_meta->embedding_num);
    std::vector<float> feature_vec(feature_meta->embedding_num);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      feature_vec[j] = feature_meta->embedding[j];
      std::cout<<feature_vec[j]<<" ";
    }
    std::cout<<std::endl;
    features.push_back(feature_vec);
  }

  return 0;
}