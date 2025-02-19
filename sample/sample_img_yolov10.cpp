#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "face/cvtdl_face_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_path> <image_path>\n", argv[0]);
    return -1;
  }
  std::string model_path = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }
  TDLModelFactory model_factory;
  std::shared_ptr<BaseModel> model = model_factory.getModel(
      TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV10, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    cvtdl_object_t *obj_meta = (cvtdl_object_t *)out_datas[i];
    for (int i = 0; i < obj_meta->size; i++) {
      std::cout << "obj_meta_index: " << i << "  "
                << "class: " << obj_meta->info[i].classes << "  "
                << "score: " << obj_meta->info[i].bbox.score << "  "
                << "bbox: " << obj_meta->info[i].bbox.x1 << " "
                << obj_meta->info[i].bbox.y1 << " " << obj_meta->info[i].bbox.x2
                << " " << obj_meta->info[i].bbox.y2 << std::endl;
    }
  }
  model_factory.releaseOutput(TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV10,
                              out_datas);

  return 0;
}
