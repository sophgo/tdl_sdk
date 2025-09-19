#include "tdl_model_factory.hpp"

std::vector<cv::Scalar> color = {
    cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76), cv::Scalar(255, 215, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(0, 255, 0)};
int line_map[19] = {4, 4, 3, 3, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0};
int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                       {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
                       {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
                       {1, 3},   {2, 4},   {3, 5},   {4, 6}};

void visualize_keypoints_detection(std::shared_ptr<BaseImage> image,
                                   std::shared_ptr<ModelOutputInfo> meta,
                                   float score, const std::string &save_path) {
  cv::Mat mat;
  bool is_rgb;
  int32_t ret = ImageFactory::convertToMat(image, mat, is_rgb);
  if (ret != 0) {
    std::cout << "Failed to convert to mat" << std::endl;
    return;
  }
  if (is_rgb) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

  // 1. simcc
  if (auto simcc = std::dynamic_pointer_cast<ModelLandmarksInfo>(meta)) {
    for (uint32_t j = 0; j < 17; j++) {
      if (simcc->landmarks_score[j] < score) continue;
      int x = static_cast<int>(simcc->landmarks_x[j]);
      int y = static_cast<int>(simcc->landmarks_y[j]);
      cv::circle(mat, cv::Point(x, y), 7, color[j], -1);
    }
    for (uint32_t k = 0; k < 19; k++) {
      int kps1 = skeleton[k][0];
      int kps2 = skeleton[k][1];
      if (simcc->landmarks_score[kps1] < score ||
          simcc->landmarks_score[kps2] < score)
        continue;
      int x1 = static_cast<int>(simcc->landmarks_x[kps1]);
      int y1 = static_cast<int>(simcc->landmarks_y[kps1]);
      int x2 = static_cast<int>(simcc->landmarks_x[kps2]);
      int y2 = static_cast<int>(simcc->landmarks_y[kps2]);
      cv::line(mat, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]],
               2);
    }
  }
  // 2. yolov8
  else if (auto yolov8 =
               std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(meta)) {
    for (uint32_t i = 0; i < yolov8->box_landmarks.size(); i++) {
      for (uint32_t j = 0; j < 17; j++) {
        if (yolov8->box_landmarks[i].landmarks_score[j] < score) continue;
        int x = static_cast<int>(yolov8->box_landmarks[i].landmarks_x[j]);
        int y = static_cast<int>(yolov8->box_landmarks[i].landmarks_y[j]);
        cv::circle(mat, cv::Point(x, y), 7, color[j], -1);
      }
      for (uint32_t k = 0; k < 19; k++) {
        int kps1 = skeleton[k][0];
        int kps2 = skeleton[k][1];
        if (yolov8->box_landmarks[i].landmarks_score[kps1] < score ||
            yolov8->box_landmarks[i].landmarks_score[kps2] < score)
          continue;
        int x1 = static_cast<int>(yolov8->box_landmarks[i].landmarks_x[kps1]);
        int y1 = static_cast<int>(yolov8->box_landmarks[i].landmarks_y[kps1]);
        int x2 = static_cast<int>(yolov8->box_landmarks[i].landmarks_x[kps2]);
        int y2 = static_cast<int>(yolov8->box_landmarks[i].landmarks_y[kps2]);
        cv::line(mat, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]],
                 2);
      }
    }
  } else {
    std::cout << "Unknown meta type for keypoint visualization!" << std::endl;
    return;
  }
  cv::imwrite(save_path, mat);
}

std::vector<std::shared_ptr<ModelOutputInfo>> extract_crop_human_landmark(
    std::shared_ptr<BaseModel> model_hk,
    std::vector<std::shared_ptr<BaseImage>> images,
    std::vector<std::shared_ptr<BaseImage>> &human_crops,
    std::vector<std::shared_ptr<ModelOutputInfo>> &human_metas) {
  std::shared_ptr<BasePreprocessor> preprocessor = model_hk->getPreprocessor();
  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  char sz_img_name[128];
  for (size_t b = 0; b < images.size(); b++) {
    std::shared_ptr<ModelBoxInfo> human_meta =
        std::static_pointer_cast<ModelBoxInfo>(human_metas[b]);

    for (size_t i = 0; i < human_meta->bboxes.size(); i++) {
      int x1 = human_meta->bboxes[i].x1;
      int y1 = human_meta->bboxes[i].y1;
      int x2 = human_meta->bboxes[i].x2;
      int y2 = human_meta->bboxes[i].y2;

      int width = x2 - x1;
      int height = y2 - y1;

      float expansion_factor = 1.25f;
      int new_width = static_cast<int>(width * expansion_factor);
      int new_height = static_cast<int>(height * expansion_factor);

      int crop_x1 = std::max(x1 - (new_width - width) / 2, 0);
      int crop_y1 = std::max(y1 - (new_height - height) / 2, 0);
      int crop_x2 = std::min(crop_x1 + new_width, (int)images[b]->getWidth());
      int crop_y2 = std::min(crop_y1 + new_height, (int)images[b]->getHeight());

      std::shared_ptr<BaseImage> human_crop = preprocessor->crop(
          images[b], crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
      sprintf(sz_img_name, "human_crop_%d.jpg", int(i));
      ImageFactory::writeImage(sz_img_name, human_crop);
      human_crops.push_back(human_crop);
    }
  }
  model_hk->inference(human_crops, out_datas);
  return out_datas;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: %s <mode:simcc|yolov8> <model_dir> <image_path>\n", argv[0]);
    return -1;
  }
  std::string mode = argv[1];
  std::string model_dir = argv[2];
  std::string image_path = argv[3];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  TDLModelFactory &model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  if (mode == "simcc") {
    std::shared_ptr<BaseModel> model_hd =
        model_factory.getModel(ModelType::MBV2_DET_PERSON);
    if (!model_hd) {
      printf("Failed to create model_hd\n");
      return -1;
    }
    std::shared_ptr<BaseModel> model_hk =
        model_factory.getModel(ModelType::KEYPOINT_SIMCC_PERSON17);
    if (!model_hk) {
      printf("Failed to create model_hk\n");
      return -1;
    }
    std::vector<std::shared_ptr<ModelOutputInfo>> out_hd;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
    std::vector<std::shared_ptr<BaseImage>> human_crops;
    model_hd->inference(input_images, out_hd);
    std::vector<std::shared_ptr<ModelOutputInfo>> out_hk =
        extract_crop_human_landmark(model_hk, input_images, human_crops,
                                    out_hd);
    for (size_t i = 0; i < out_hk.size(); i++) {
      std::shared_ptr<ModelLandmarksInfo> obj_meta =
          std::static_pointer_cast<ModelLandmarksInfo>(out_hk[i]);
      for (int k = 0; k < 17; k++) {
        printf("%d: %f %f %f\n", k, obj_meta->landmarks_x[k],
               obj_meta->landmarks_y[k], obj_meta->landmarks_score[k]);
      }
      char sz_img_name[128];
      sprintf(sz_img_name, "simcc_keypoints_%d.jpg", i);
      visualize_keypoints_detection(human_crops[i], out_hk[i],
                                    3,  // simcc threshold greater than 1
                                    sz_img_name);
    }
  } else if (mode == "yolov8") {
    std::shared_ptr<BaseModel> model_od =
        model_factory.getModel(ModelType::KEYPOINT_YOLOV8POSE_PERSON17);
    if (!model_od) {
      printf("Failed to create model_od\n");
      return -1;
    }
    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
    std::vector<std::shared_ptr<BaseImage>> input_images = {image1};
    model_od->inference(input_images, out_datas);
    for (size_t i = 0; i < out_datas.size(); i++) {
      std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
          std::static_pointer_cast<ModelBoxLandmarkInfo>(out_datas[i]);
      uint32_t image_width = input_images[i]->getWidth();
      uint32_t image_height = input_images[i]->getHeight();
      printf("Sample Image dimensions - height: %d, width: %d\n", image_height,
             image_width);
      if (obj_meta->box_landmarks.size() == 0) {
        printf("No object detected\n");
      } else {
        for (size_t j = 0; j < obj_meta->box_landmarks.size(); j++) {
          std::cout << "obj_meta_index: " << j << "  "
                    << "class: " << obj_meta->box_landmarks[j].class_id << "  "
                    << "score: " << obj_meta->box_landmarks[j].score << "  "
                    << "bbox: " << obj_meta->box_landmarks[j].x1 << " "
                    << obj_meta->box_landmarks[j].y1 << " "
                    << obj_meta->box_landmarks[j].x2 << " "
                    << obj_meta->box_landmarks[j].y2 << std::endl;
          for (int k = 0; k < 17; k++) {
            printf("%d: %f %f %f\n", k,
                   obj_meta->box_landmarks[j].landmarks_x[k],
                   obj_meta->box_landmarks[j].landmarks_y[k],
                   obj_meta->box_landmarks[j].landmarks_score[k]);
          }
        }
      }
      visualize_keypoints_detection(image1, out_datas[i], 0.5,
                                    "yolov8_keypoints.jpg");
    }
  } else {
    printf("Unknown mode: %s. Use simcc or yolov8\n", mode.c_str());
    return -1;
  }
  return 0;
}
