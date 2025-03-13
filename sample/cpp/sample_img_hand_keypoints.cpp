#include "tdl_model_factory.hpp"
std::vector<cv::Scalar> color = {cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76),
                                 cv::Scalar(255, 215, 0), cv::Scalar(255, 128, 0),
                                 cv::Scalar(0, 255, 0)};

void visualize_keypoints_detection(std::shared_ptr<BaseImage> image, std::shared_ptr<ModelLandmarksInfo> obj_meta, float score,
                              const std::string &save_path) {
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
        int x = static_cast<int>(obj_meta->landmarks_x[j]*image_width);
        int y = static_cast<int>(obj_meta->landmarks_y[j]*image_height);
        cv::circle(mat, cv::Point(x, y), 7, cv::Scalar(0, 0, 255), -1);
    }

    cv::imwrite(save_path, mat);
}

std::vector<std::shared_ptr<ModelOutputInfo>>
extract_crop_human_landmark(
    std::shared_ptr<BaseModel> model_hk,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<BaseImage>> &human_crops,
    std::vector<std::shared_ptr<ModelOutputInfo>> &human_metas) {

  std::shared_ptr<BasePreprocessor> preprocessor = model_hk->getPreprocessor();

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  char sz_img_name[128];
  for (size_t i = 0; i < images.size(); i++) {
    std::shared_ptr<ModelBoxInfo> human_meta =
        std::static_pointer_cast<ModelBoxInfo>(human_metas[i]);

    int x1 = human_meta->bboxes[0].x1;
    int y1 = human_meta->bboxes[0].y1;
    int x2 = human_meta->bboxes[0].x2;
    int y2 = human_meta->bboxes[0].y2;

    int width = x2 - x1;
    int height = y2 - y1;

    float expansion_factor = 1.25f;
    int new_width = static_cast<int>(width * expansion_factor);
    int new_height = static_cast<int>(height * expansion_factor);

    int crop_x1 = x1 - (new_width - width) / 2; 
    int crop_y1 = y1 - (new_height - height) / 2; 
    int crop_x2 = crop_x1 + new_width; 
    int crop_y2 = crop_y1 + new_height; 

    std::shared_ptr<BaseImage> human_crop = preprocessor->crop(
          images[i], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

    sprintf(sz_img_name, "human_crop_%d.jpg", i);
    ImageFactory::writeImage(sz_img_name, human_crop);
    human_crops.push_back(human_crop);
  }
  model_hk->inference(human_crops, out_datas);

  return out_datas;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <model_path> <image_path> \n", argv[0]);
    return -1;
  }
  std::string hd_model_path = argv[1];
  std::string hk_model_path = argv[2];
  std::string image_path = argv[3];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  TDLModelFactory model_factory;

  std::shared_ptr<BaseModel> model_hd = model_factory.getModel(
      ModelType::YOLOV8N_HAND, hd_model_path);
  if (!model_hd) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hk = model_factory.getModel(
      ModelType::KEYPOINT_HAND, hk_model_path);
  if (!model_hk) {
    printf("Failed to create model_hk\n");
    return -1;
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_hd;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
    std::vector<std::shared_ptr<BaseImage>> human_crops;
  model_hd->inference(input_images, out_hd);
  std::vector<std::shared_ptr<ModelOutputInfo>>
      out_hk =
          extract_crop_human_landmark(model_hk, input_images, human_crops, out_hd);

  for (size_t i = 0; i < out_hk.size(); i++) {
    std::shared_ptr<ModelLandmarksInfo> obj_meta =
        std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);
    for (int k = 0; k < 21; k++) {
        printf("%d: %f %f\n", k, obj_meta->landmarks_x[k]*obj_meta->image_width,
              obj_meta->landmarks_y[k]*obj_meta->image_height);
    }
    visualize_keypoints_detection(human_crops[0],obj_meta, 0.5 ,"hand_keypoints.jpg");
  }

  return 0;
}
