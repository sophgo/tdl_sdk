
#include "tdl_model_factory.hpp"
std::vector<cv::Scalar> color = {
    cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76), cv::Scalar(255, 215, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(0, 255, 0)};

int line_map[19] = {4, 4, 3, 3, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0};
int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                       {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
                       {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
                       {1, 3},   {2, 4},   {3, 5},   {4, 6}};
void visualize_keypoints_detection(
    std::shared_ptr<BaseImage> image,
    std::shared_ptr<ModelBoxLandmarkInfo> obj_meta, float score,
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

  for (uint32_t i = 0; i < obj_meta->box_landmarks.size(); i++) {
    for (uint32_t j = 0; j < 17; j++) {
      if (obj_meta->box_landmarks[i].landmarks_score[j] < score) {
        continue;
      }
      int x = static_cast<int>(obj_meta->box_landmarks[i].landmarks_x[j]);
      int y = static_cast<int>(obj_meta->box_landmarks[i].landmarks_y[j]);
      cv::circle(mat, cv::Point(x, y), 7, color[j], -1);
    }

    for (uint32_t k = 0; k < 19; k++) {
      int kps1 = skeleton[k][0];
      int kps2 = skeleton[k][1];
      if (obj_meta->box_landmarks[i].landmarks_score[kps1] < score ||
          obj_meta->box_landmarks[i].landmarks_score[kps2] < score) {
        continue;
      }

      int x1 = static_cast<int>(obj_meta->box_landmarks[i].landmarks_x[kps1]);
      int y1 = static_cast<int>(obj_meta->box_landmarks[i].landmarks_y[kps1]);

      int x2 = static_cast<int>(obj_meta->box_landmarks[i].landmarks_x[kps2]);
      int y2 = static_cast<int>(obj_meta->box_landmarks[i].landmarks_y[kps2]);

      cv::line(mat, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]],
               2);
    }
  }

  cv::imwrite(save_path, mat);
}

void visualize_object_detection(std::shared_ptr<BaseImage> image,
                                std::shared_ptr<ModelBoxLandmarkInfo> obj_meta,
                                const std::string &str_img_name) {
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

  for (size_t i = 0; i < obj_meta->box_landmarks.size(); i++) {
    cv::rectangle(
        mat,
        cv::Rect(
            int(obj_meta->box_landmarks[i].x1),
            int(obj_meta->box_landmarks[i].y1),
            int(obj_meta->box_landmarks[i].x2 - obj_meta->box_landmarks[i].x1),
            int(obj_meta->box_landmarks[i].y2 - obj_meta->box_landmarks[i].y1)),
        cv::Scalar(0, 0, 255), 2);
    char sz_text[128];
    sprintf(sz_text, "%d,%.2f", obj_meta->box_landmarks[i].class_id,
            obj_meta->box_landmarks[i].score);
    cv::putText(mat, sz_text,
                cv::Point(int(obj_meta->box_landmarks[i].x1),
                          int(obj_meta->box_landmarks[i].y1)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
  }
  std::cout << "save image to " << str_img_name << std::endl;

  cv::imwrite(str_img_name, mat);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <model_path> <image_path> \n", argv[0]);
    return -1;
  }
  std::string model_path = argv[1];
  std::string image_path = argv[2];

  std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path);
  if (!image1) {
    printf("Failed to create image1\n");
    return -1;
  }

  TDLModelFactory model_factory;

  std::shared_ptr<BaseModel> model_od =
      model_factory.getModel(ModelType::YOLOV8_POSE_PERSON17, model_path);
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
      cv::Mat image = cv::imread(argv[2]);
      for (size_t j = 0; j < obj_meta->box_landmarks.size(); j++) {
        std::cout << "obj_meta_index: " << j << "  "
                  << "class: " << obj_meta->box_landmarks[j].class_id << "  "
                  << "score: " << obj_meta->box_landmarks[j].score << "  "
                  << "bbox: " << obj_meta->box_landmarks[j].x1 << " "
                  << obj_meta->box_landmarks[j].y1 << " "
                  << obj_meta->box_landmarks[j].x2 << " "
                  << obj_meta->box_landmarks[j].y2 << std::endl;
        for (int k = 0; k < 17; k++) {
          printf("%d: %f %f %f\n", k, obj_meta->box_landmarks[j].landmarks_x[k],
                 obj_meta->box_landmarks[j].landmarks_y[k],
                 obj_meta->box_landmarks[j].landmarks_score[k]);
        }
      }
    }
    std::string str_img_name = "object_detection_" + std::to_string(i) + ".jpg";
    // visualize_object_detection(image1, obj_meta, "yolov8_pose.jpg");
    visualize_keypoints_detection(image1, obj_meta, 0.5,
                                  "yolov8_keypoints.jpg");
  }

  return 0;
}
