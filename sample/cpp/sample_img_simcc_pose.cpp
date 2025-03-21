#include "tdl_model_factory.hpp"
std::vector<cv::Scalar> color = {cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76),
                                 cv::Scalar(255, 215, 0), cv::Scalar(255, 128, 0),
                                 cv::Scalar(0, 255, 0)};

int line_map[19] = {4, 4, 3, 3, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0};
int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                       {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
                       {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};
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

    for (uint32_t j = 0; j < 17; j++) {
        if (obj_meta->landmarks_score[j] < score) {
            continue;
        }
        int x = static_cast<int>(obj_meta->landmarks_x[j]);
        int y = static_cast<int>(obj_meta->landmarks_y[j]);
        cv::circle(mat, cv::Point(x, y), 7, color[j], -1);
    }

    for (uint32_t k = 0; k < 19; k++) {
        int kps1 = skeleton[k][0];
        int kps2 = skeleton[k][1];
        if (obj_meta->landmarks_score[kps1] < score ||
            obj_meta->landmarks_score[kps2] < score) {
            continue;
        }

        int x1 = static_cast<int>(obj_meta->landmarks_x[kps1]);
        int y1 = static_cast<int>(obj_meta->landmarks_y[kps1]);

        int x2 = static_cast<int>(obj_meta->landmarks_x[kps2]);
        int y2 = static_cast<int>(obj_meta->landmarks_y[kps2]);

        cv::line(mat, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]], 2);
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

    int crop_x1 = x1;
    int crop_y1 = y1;
    int crop_x2 = x2;
    int crop_y2 = y2;

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
      ModelType::MBV2_DET_PERSON, hd_model_path);
  if (!model_hd) {
    printf("Failed to create model_hd\n");
    return -1;
  }

  std::shared_ptr<BaseModel> model_hk = model_factory.getModel(
      ModelType::KEYPOINT_SIMCC_PERSON17, hk_model_path);
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
    for (int k = 0; k < 17; k++) {
        printf("%d: %f %f %f\n", k, obj_meta->landmarks_x[k],
              obj_meta->landmarks_y[k],
              obj_meta->landmarks_score[k]);
    }
    visualize_keypoints_detection(human_crops[0],obj_meta, 0.5 ,"simcc_keypoints.jpg");
  }

  return 0;
}
