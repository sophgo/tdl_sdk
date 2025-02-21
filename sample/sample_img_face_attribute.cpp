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
  TDL_MODEL_TYPE model_id = TDL_MODEL_TYPE_FACE_ATTRIBUTE_CLS;

  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<void *> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    cvtdl_face_t *face_meta = (cvtdl_face_t *)out_datas[i];
    printf("gender score:%f,age score:%f,glass score:%f,mask score:%f\n", face_meta->info->gender_score,
           face_meta->info->age, face_meta->info->glass, face_meta->info->mask_score);
    if(face_meta->info->gender_score>0.5){
      printf("Gender:Male\n");
    }else{
      printf("Gender:Female\n");
    }
    printf("Age:%d\n",int(face_meta->info->age*100));
    if(face_meta->info->glass>0.5){
      printf("Glass:Yes\n");
    }else{
      printf("Glass:No\n");
    }
    if(face_meta->info->mask_score>0.5){
      printf("Mask:Yes\n");
    }else{
      printf("Mask:No\n");
    }

    CVI_TDL_FreeCpp(face_meta);
    free(face_meta);
  }

  return 0;
}