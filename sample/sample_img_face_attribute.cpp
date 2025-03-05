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

  std::shared_ptr<BaseModel> model =
      model_factory.getModel(model_id, model_path);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);

  for (size_t i = 0; i < out_datas.size(); i++) {
    if (out_datas[i]->getType() != ModelOutputType::ATTRIBUTE) {
      printf("out_datas[%d] is not ModelOutputType::ATTRIBUTE\n", i);
      continue;
    }
    std::shared_ptr<ModelAttributeInfo> face_meta =
        std::static_pointer_cast<ModelAttributeInfo>(out_datas[i]);
    printf(
        "gender score:%f,age score:%f,glass score:%f,mask score:%f\n",
        face_meta
            ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER],
        face_meta
            ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE],
        face_meta->attributes
            [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES],
        face_meta
            ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK]);
    if (face_meta->attributes
            [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER] > 0.5) {
      printf("Gender:Male\n");
    } else {
      printf("Gender:Female\n");
    }
    printf("Age:%d\n",
           int(face_meta->attributes
                   [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE] *
               100));
    if (face_meta->attributes
            [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES] > 0.5) {
      printf("Glass:Yes\n");
    } else {
      printf("Glass:No\n");
    }
    if (face_meta
            ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK] >
        0.5) {
      printf("Mask:Yes\n");
    } else {
      printf("Mask:No\n");
    }
  }

  return 0;
}