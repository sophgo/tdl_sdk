#include "tdl_model_factory.hpp"

void set_preprocess_parameters(std::shared_ptr<BaseModel> model_hc) {
  PreprocessParams pre_param;

  model_hc->getPreprocessParameters(pre_param);

  pre_param.mean[0] = 2.1179;
  pre_param.mean[1] = 2.0357;
  pre_param.mean[2] = 1.8044;

  pre_param.scale[0] = 0.017126;
  pre_param.scale[1] = 0.017509;
  pre_param.scale[2] = 0.017431;

  model_hc->setPreprocessParameters(pre_param);
}

std::vector<std::shared_ptr<ModelOutputInfo>> extract_crop_hand_landmark(
    std::shared_ptr<BaseModel> model_hc,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<BaseImage>> &hand_crops,
    std::vector<std::shared_ptr<ModelOutputInfo>> &hand_metas) {
  std::shared_ptr<BasePreprocessor> preprocessor = model_hc->getPreprocessor();

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  for (size_t b = 0; b < images.size(); b++) {
    std::shared_ptr<ModelBoxInfo> hand_meta =
        std::static_pointer_cast<ModelBoxInfo>(hand_metas[b]);

    for (int i = 0; i < hand_meta->bboxes.size(); i++) {
      int x1 = hand_meta->bboxes[i].x1;
      int y1 = hand_meta->bboxes[i].y1;
      int x2 = hand_meta->bboxes[i].x2;
      int y2 = hand_meta->bboxes[i].y2;

      int width = x2 - x1;
      int height = y2 - y1;

      float expansion_factor = 1.125f;
      int new_width = static_cast<int>(width * expansion_factor);
      int new_height = static_cast<int>(height * expansion_factor);

      int crop_x1 = x1 - (new_width - width) / 2;
      int crop_y1 = y1 - (new_height - height) / 2;
      int crop_x2 = crop_x1 + new_width;
      int crop_y2 = crop_y1 + new_height;

      std::shared_ptr<BaseImage> hand_crop = preprocessor->crop(
          images[b], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

      hand_crops.push_back(hand_crop);
    }
  }
  model_hc->inference(hand_crops, out_datas);

  return out_datas;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <hand_det_model_path> <hand_classification_model_path>\n",
           argv[0]);
    return -1;
  }
  std::string hd_model_path = argv[1];
  std::string hc_model_path = argv[2];
  std::string image_path = argv[3];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  TDLModelFactory model_factory;

  std::shared_ptr<BaseModel> model_hd =
      model_factory.getModel(ModelType::YOLOV8N_DET_HAND, hd_model_path);
  if (!model_hd) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hc =
      model_factory.getModel(ModelType::CLS_HAND_GESTURE, hc_model_path);
  if (!model_hc) {
    printf("Failed to create model_hc\n");
    return -1;
  }

  // set preprocess parameters
  set_preprocess_parameters(model_hc);

  std::vector<std::shared_ptr<ModelOutputInfo>> out_hd;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  std::vector<std::shared_ptr<BaseImage>> hand_crops;
  model_hd->inference(input_images, out_hd);
  std::vector<std::shared_ptr<ModelOutputInfo>> out_hc =
      extract_crop_hand_landmark(model_hc, input_images, hand_crops, out_hd);

  char sz_img_name[128];
  for (size_t i = 0; i < out_hc.size(); i++) {
    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_hc[i]);

    printf("hand_crops[%d], label:%d, score:%.2f\n", i,
           cls_meta->topk_class_ids[0], cls_meta->topk_scores[0]);

    sprintf(sz_img_name, "hand_crop_%d_label_%d.jpg", i,
            cls_meta->topk_class_ids[0]);
    ImageFactory::writeImage(sz_img_name, hand_crops[i]);
  }

  return 0;
}
