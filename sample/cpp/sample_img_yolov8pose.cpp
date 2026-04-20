#include <iostream>
#include <opencv2/opencv.hpp>
#include "tdl_model_factory.hpp"

std::vector<cv::Scalar> color = {
    cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76), cv::Scalar(255, 215, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(0, 255, 0)};
cv::Scalar box_color = cv::Scalar(0, 0, 255);

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

  if (auto yolov8 = std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(meta)) {
    for (uint32_t i = 0; i < yolov8->box_landmarks.size(); i++) {
      int x1 = static_cast<int>(yolov8->box_landmarks[i].x1);
      int y1 = static_cast<int>(yolov8->box_landmarks[i].y1);
      int x2 = static_cast<int>(yolov8->box_landmarks[i].x2);
      int y2 = static_cast<int>(yolov8->box_landmarks[i].y2);

      cv::rectangle(mat, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);

      std::string label =
          "Class: " + std::to_string(yolov8->box_landmarks[i].class_id) +
          " Score: " + std::to_string(yolov8->box_landmarks[i].score);

      cv::rectangle(mat, cv::Point(x1, y1 - 20),
                    cv::Point(x1 + label.length() * 10, y1), box_color, -1);
      cv::putText(mat, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(255, 255, 255), 1);
      for (uint32_t j = 0; j < yolov8->box_landmarks[i].landmarks_score.size();
           j++) {
        if (yolov8->box_landmarks[i].landmarks_score[j] < score) continue;
        int x = static_cast<int>(yolov8->box_landmarks[i].landmarks_x[j]);
        int y = static_cast<int>(yolov8->box_landmarks[i].landmarks_y[j]);
        cv::circle(mat, cv::Point(x, y), 7, color[j], -1);
      }
    }
  } else {
    std::cout << "Unknown meta type for keypoint visualization!" << std::endl;
    return;
  }
  cv::imwrite(save_path, mat);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: %s <moded_id> <model_dir> <image_path>\n", argv[0]);
    return -1;
  }
  std::string moded_id = argv[1];
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

  std::shared_ptr<BaseModel> model_od = model_factory.getModel(moded_id);
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
        for (int k = 0; k < obj_meta->box_landmarks[j].landmarks_score.size();
             k++) {
          printf("%d: %f %f %f\n", k, obj_meta->box_landmarks[j].landmarks_x[k],
                 obj_meta->box_landmarks[j].landmarks_y[k],
                 obj_meta->box_landmarks[j].landmarks_score[k]);
        }
      }
    }
    visualize_keypoints_detection(image1, out_datas[i], 0.5,
                                  "yolov8_keypoints.jpg");
  }

  return 0;
}