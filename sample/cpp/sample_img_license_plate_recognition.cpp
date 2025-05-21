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

  for (uint32_t j = 0; j < 4; j++) {
    int x = static_cast<int>(obj_meta->landmarks_x[j]);
    int y = static_cast<int>(obj_meta->landmarks_y[j]);
    cv::circle(mat, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
  }

  cv::imwrite(save_path, mat);
}

std::vector<std::shared_ptr<ModelOutputInfo>>
extract_crop_license_plate_landmark(
    std::shared_ptr<BaseModel> model_hk,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<BaseImage>> &license_plate_crops,
    std::vector<std::shared_ptr<ModelOutputInfo>> &license_plate_metas) {
  std::shared_ptr<BasePreprocessor> preprocessor = model_hk->getPreprocessor();

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  for (size_t i = 0; i < images.size(); i++) {
    std::shared_ptr<ModelBoxInfo> license_plate_meta =
        std::static_pointer_cast<ModelBoxInfo>(license_plate_metas[i]);
    for (size_t j = 0; j < license_plate_meta->bboxes.size(); j++) {
      int x1 = license_plate_meta->bboxes[j].x1;
      int y1 = license_plate_meta->bboxes[j].y1;
      int x2 = license_plate_meta->bboxes[j].x2;
      int y2 = license_plate_meta->bboxes[j].y2;

      int width = x2 - x1;
      int height = y2 - y1;

      float expansion_factor = 1.25f;
      int new_width = static_cast<int>(width * expansion_factor);
      int new_height = static_cast<int>(height * expansion_factor);

      int crop_x1 = x1 - (new_width - width) / 2;
      int crop_y1 = y1 - (new_height - height) / 2;
      int crop_x2 = crop_x1 + new_width;
      int crop_y2 = crop_y1 + new_height;

      std::shared_ptr<BaseImage> license_plate_crop = preprocessor->crop(
          images[i], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

      license_plate_crops.push_back(license_plate_crop);
    }
  }
  model_hk->inference(license_plate_crops, out_datas);

  return out_datas;
}

std::vector<std::shared_ptr<ModelOutputInfo>> license_plate_recognition(
    std::shared_ptr<BaseModel> model_hr,
    std::vector<std::shared_ptr<BaseImage>> &license_plate_crops,
    std::vector<std::shared_ptr<BaseImage>> &license_plate_aligns,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_hk) {
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;

  for (size_t i = 0; i < out_hk.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);

    float landmarks[8];

    for (int k = 0; k < 4; k++) {
      landmarks[2 * k] = landmarks_meta->landmarks_x[k];
      landmarks[2 * k + 1] = landmarks_meta->landmarks_y[k];
    }
    std::shared_ptr<BaseImage> license_plate_align =
        ImageFactory::alignLicensePlate(license_plate_crops[i], landmarks,
                                        nullptr, 4, nullptr);

    license_plate_aligns.push_back(license_plate_align);
  }

  model_hr->inference(license_plate_aligns, out_datas);

  char sz_img_name[128];
  for (size_t i = 0; i < out_datas.size(); i++) {
    std::shared_ptr<ModelOcrInfo> text_meta =
        std::static_pointer_cast<ModelOcrInfo>(out_datas[i]);

    sprintf(sz_img_name, "%s.jpg", text_meta->text_info);

    ImageFactory::writeImage(sz_img_name, license_plate_aligns[i]);
  }

  return out_datas;
}

int main(int argc, char **argv) {
  if (argc != 5) {
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
      model_factory.getModel(ModelType::YOLOV8N_DET_LICENSE_PLATE);
  if (!model_hd) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hk =
      model_factory.getModel(ModelType::KEYPOINT_LICENSE_PLATE);
  if (!model_hk) {
    printf("Failed to create model_hk\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hr =
      model_factory.getModel(ModelType::RECOGNITION_LICENSE_PLATE);
  if (!model_hr) {
    printf("Failed to create model_hr\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_hd;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
  std::vector<std::shared_ptr<BaseImage>> license_plate_crops;
  std::vector<std::shared_ptr<BaseImage>> license_plate_align;

  model_hd->inference(input_images, out_hd);
  std::vector<std::shared_ptr<ModelOutputInfo>> out_hk =
      extract_crop_license_plate_landmark(model_hk, input_images,
                                          license_plate_crops, out_hd);

  std::vector<std::shared_ptr<ModelOutputInfo>> out_hr =
      license_plate_recognition(model_hr, license_plate_crops,
                                license_plate_align, out_hk);

  for (size_t i = 0; i < out_hk.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> obj_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);
    std::shared_ptr<ModelOcrInfo> text_meta =
        std::static_pointer_cast<ModelOcrInfo>(out_hr[i]);
    printf("keypoints:\n");
    for (int k = 0; k < 4; k++) {
      printf("%d: %.2f %.2f\n", k, obj_meta->landmarks_x[k],
             obj_meta->landmarks_y[k]);
    }
    printf("license_plate:\n");
    printf("%s\n", text_meta->text_info);
    // 创建文件名并添加索引
    std::ostringstream filename;
    filename << "license_plate_keypoints_" << i << ".jpg";  // 生成新的文件名

    // 使用动态生成的文件名
    visualize_keypoints_detection(license_plate_crops[i], obj_meta, 0.5,
                                  filename.str());
  }

  return 0;
}
