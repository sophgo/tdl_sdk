#include "tdl_model_factory.hpp"
std::vector<cv::Scalar> color = {
    cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76), cv::Scalar(255, 215, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(0, 255, 0)};

void visualize_keypoints_detection(std::shared_ptr<BaseImage> image,
                                   std::shared_ptr<ModelLandmarksInfo> obj_meta,
                                   float score, const std::string &save_path) {
  cv::Mat mat;
  bool is_rgb;
  int32_t ret = ImageFactory::convertToMat(image, mat, is_rgb);
  if (ret != 0) {
    std::cout << "Failed to convert to mat" << std::endl;
    return;
  }

  if (is_rgb) {
    std::cout << "convert to bgr" << std::endl;
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  }
  uint32_t image_width = obj_meta->image_width;
  uint32_t image_height = obj_meta->image_height;

  for (uint32_t j = 0; j < 21; j++) {
    int x = static_cast<int>(obj_meta->landmarks_x[j] * image_width);
    int y = static_cast<int>(obj_meta->landmarks_y[j] * image_height);
    cv::circle(mat, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
  }

  cv::imwrite(save_path, mat);
}

std::vector<std::shared_ptr<ModelOutputInfo>> extract_crop_hand_landmark(
    std::shared_ptr<BaseModel> model_hk,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<BaseImage>> &hand_crops,
    std::vector<std::shared_ptr<ModelOutputInfo>> &hand_metas) {
  std::shared_ptr<BasePreprocessor> preprocessor = model_hk->getPreprocessor();

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  char sz_img_name[128];
  for (size_t b = 0; b < images.size(); b++) {
    std::shared_ptr<ModelBoxInfo> hand_meta =
        std::static_pointer_cast<ModelBoxInfo>(hand_metas[b]);

    for (size_t i = 0; i < hand_meta->bboxes.size(); i++) {
      int x1 = hand_meta->bboxes[i].x1;
      int y1 = hand_meta->bboxes[i].y1;
      int x2 = hand_meta->bboxes[i].x2;
      int y2 = hand_meta->bboxes[i].y2;

      int width = x2 - x1;
      int height = y2 - y1;

      float expansion_factor = 1.25f;
      int new_width = static_cast<int>(width * expansion_factor);
      int new_height = static_cast<int>(height * expansion_factor);

      int crop_x1 = x1 - (new_width - width) / 2;
      int crop_y1 = y1 - (new_height - height) / 2;
      int crop_x2 = crop_x1 + new_width;
      int crop_y2 = crop_y1 + new_height;

      std::shared_ptr<BaseImage> hand_crop = preprocessor->crop(
          images[b], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

      sprintf(sz_img_name, "hand_crop_%ld.jpg", i);
      ImageFactory::writeImage(sz_img_name, hand_crop);
      hand_crops.push_back(hand_crop);
    }
  }
  model_hk->inference(hand_crops, out_datas);

  return out_datas;
}

std::vector<std::shared_ptr<ModelOutputInfo>> hand_keypoint_classification(
    std::shared_ptr<BaseModel> model_hc,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_hk) {
  std::vector<std::shared_ptr<BaseImage>> input_datas;
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  for (size_t i = 0; i < out_hk.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> obj_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);

    std::vector<float> keypoints;

    for (uint32_t k = 0; k < 21; k++) {
      keypoints.push_back(obj_meta->landmarks_x[k]);
      keypoints.push_back(obj_meta->landmarks_y[k]);
    }

    std::shared_ptr<BaseImage> bin_data = ImageFactory::createImage(
        42, 1, ImageFormat::GRAY, TDLDataType::FP32, true);

    float *data_buffer =
        reinterpret_cast<float *>(bin_data->getVirtualAddress()[0]);

    memcpy(data_buffer, &keypoints[0], 42 * sizeof(float));
    input_datas.push_back(bin_data);
  }

  model_hc->inference(input_datas, out_datas);
  return out_datas;
}

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

  std::shared_ptr<BaseModel> model_hd =
      model_factory.getModel(ModelType::YOLOV8N_DET_HAND);
  if (!model_hd) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hk =
      model_factory.getModel(ModelType::KEYPOINT_HAND);
  if (!model_hk) {
    printf("Failed to create model_hk\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hc =
      model_factory.getModel(ModelType::CLS_KEYPOINT_HAND_GESTURE);
  if (!model_hc) {
    printf("Failed to create model_hr\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_hd;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  std::vector<std::shared_ptr<BaseImage>> hand_crops;
  model_hd->inference(input_images, out_hd);
  std::vector<std::shared_ptr<ModelOutputInfo>> out_hk =
      extract_crop_hand_landmark(model_hk, input_images, hand_crops, out_hd);

  std::vector<std::shared_ptr<ModelOutputInfo>> out_hc =
      hand_keypoint_classification(model_hc, out_hk);

  for (size_t i = 0; i < out_hk.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> obj_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);

    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_hc[i]);

    for (int k = 0; k < 21; k++) {
      printf("%d: %f %f\n", k, obj_meta->landmarks_x[k] * obj_meta->image_width,
             obj_meta->landmarks_y[k] * obj_meta->image_height);
    }
    printf("hand[%ld]: label: %d, score: %.2f\n", i, cls_meta->topk_class_ids[0],
           cls_meta->topk_scores[0]);

    char img_name[128];
    sprintf(img_name, "hand_keypoints_%ld_label_%d.jpg", i,
            cls_meta->topk_class_ids[0]);
    visualize_keypoints_detection(hand_crops[i], obj_meta, 0.5,
                                  std::string(img_name));
  }

  return 0;
}
