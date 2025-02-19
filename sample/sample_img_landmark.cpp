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
  std::shared_ptr<BaseModel> model = TDLModelFactory::createModel(
      TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    cvtdl_face_info_t *face_meta = (cvtdl_face_info_t *)out_datas[i];
    printf("face_meta size: %d,score:%f,blurness:%f\n", face_meta->pts.size,
           face_meta->pts.score, face_meta->blurness);
    for (size_t j = 0; j < face_meta->pts.size; j++) {
      printf("face_meta pts: %f, %f\n", face_meta->pts.x[j],
             face_meta->pts.y[j]);
    }
    CVI_TDL_FreeCpp(face_meta);
    free(face_meta);
  }

  return 0;
}