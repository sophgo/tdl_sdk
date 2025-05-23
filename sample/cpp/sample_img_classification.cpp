
#include "tdl_model_factory.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <model_file> <image_path>\n", argv[0]);
    return -1;
  }

  std::string model_file = argv[1];
  std::string image_path = argv[2];

  auto image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to load images\n");
    return -1;
  }

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();

  std::shared_ptr<BaseModel> model_cls =
      model_factory.getModel(ModelType::CLS_RGBLIVENESS, model_file);

  if (!model_cls) {
    printf("Failed to load classification model\n");
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> input_images = {image};

  std::vector<std::shared_ptr<ModelOutputInfo>> out_cls;
  model_cls->inference(input_images, out_cls);
  for (size_t i = 0; i < input_images.size(); i++) {
    if (out_cls[i]->getType() != ModelOutputType::CLASSIFICATION) {
      printf("out_cls[%ld] is not ModelOutputType::CLASSIFICATION\n", i);
      continue;
    }
    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_cls[i]);
    printf("pred_label: %d\n", cls_meta->topk_class_ids[0]);
  }

  return 0;
}