
#include "tdl_model_factory.hpp"

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_id_name> <model_dir> <image_path>\n", argv[0]);
    return -1;
  }
  std::string model_id_name = argv[1];
  std::string model_dir = argv[2];
  std::string image_path = argv[3];

  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to create image\n");
    return -1;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);
  std::shared_ptr<BaseModel> model = model_factory.getModel(model_id_name);
  if (!model) {
    printf("Failed to create model\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  model->inference(input_images, out_datas);
  const std::array<std::string, 7> id_to_emotion = {
      "Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"};
  for (size_t i = 0; i < out_datas.size(); i++) {
    if (out_datas[i]->getType() != ModelOutputType::CLS_ATTRIBUTE) {
      printf("out_datas[%ld] is not ModelOutputType::CLS_ATTRIBUTE\n", i);
      continue;
    }
    std::shared_ptr<ModelAttributeInfo> face_meta =
        std::static_pointer_cast<ModelAttributeInfo>(out_datas[i]);
    if (model_id_name == "CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK") {
      printf(
          "gender score:%f,age score:%f,glass score:%f,mask score:%f\n",
          face_meta->attributes
              [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER],
          face_meta
              ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE],
          face_meta->attributes
              [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES],
          face_meta->attributes
              [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK]);
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
      if (face_meta->attributes
              [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK] > 0.5) {
        printf("Mask:Yes\n");
      } else {
        printf("Mask:No\n");
      }
    } else if (model_id_name == "CLS_ATTRIBUTE_GENDER_AGE_GLASS") {
      printf(
          "gender score:%f,age score:%f,glass score:%f\n",
          face_meta->attributes
              [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER],
          face_meta
              ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE],
          face_meta->attributes
              [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES]);
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
    } else if (model_id_name == "CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION") {
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
      printf(
          "Emotion:%s\n",
          id_to_emotion
              [int(face_meta->attributes[TDLObjectAttributeType::
                                             OBJECT_ATTRIBUTE_HUMAN_EMOTION])]
                  .c_str());
    } else {
      printf("Not supported model id: %s\n", argv[1]);
      return -1;
    }
  }

  return 0;
}