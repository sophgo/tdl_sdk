
#include "tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_dir> <image_path> \n", argv[0]);
    return -1;
  }
  std::string model_dir = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model_seg =
      model_factory.getModel(ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE);
  if (!model_seg) {
    printf("Failed to create model_seg\n");
    return -1;
  }
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  model_seg->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelSegmentationInfo> seg_meta =
        std::static_pointer_cast<ModelSegmentationInfo>(out_datas[i]);

    uint32_t image_width = input_images[i]->getWidth();
    uint32_t image_height = input_images[i]->getHeight();
    for (int x = 0; x < seg_meta->output_height; ++x) {
      for (int y = 0; y < seg_meta->output_width; ++y) {
        printf("%d ", (int)seg_meta->class_id[x * seg_meta->output_width + y]);
      }
      printf("\n");
    }
  }

  return 0;
}
