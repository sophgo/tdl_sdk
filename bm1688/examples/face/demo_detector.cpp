#include <iostream>
#include <log/Logger.hpp>

#include "factory/model.hpp"

void test_sync_api(const std::string &model_path, std::vector<cv::Mat> &images,
                   const float &threshold) {
  NNFactory factory(model_path);
  auto detector = (FaceSCRFD *)(factory.get_model(SCRFD));
  // auto detector = (FaceCSSD *)(factory.get_model(CSSD));
  std::vector<std::vector<FaceRect>> detections;
  detector->detect(images, threshold, detections);
  for (int k = 0; k < images.size(); k++) {
    std::vector<FaceRect> detection = detections[k];
    for (const auto &rect : detection) {
      printf("(x1, y1, x2, y2): %f, %f, %f, %f\n", rect.x1, rect.y1, rect.x2,
             rect.y2);
      auto x1 = static_cast<int>(rect.x1);
      auto y1 = static_cast<int>(rect.y1);
      auto x2 = static_cast<int>(rect.x2);
      auto y2 = static_cast<int>(rect.y2);
      cv::rectangle(images[k], cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite(std::to_string(k) + std::string("sync_.jpg"), images[k]);
  }

  delete detector;
}

int main(int argc, char **argv) {
  LogStream::logger.setLogLevel(INFO);  // 设置日志级别
  // CHECK_EQ(argc, 4) << "Example usage: ./bin/demo_detector ../models 0.5 "
  //                      "../sample/test_2.jpg";

  const std::string &model_path = argv[1];
  const float &threshold = strtof(argv[2], nullptr);
  const std::string &img_path = argv[3];
  std::vector<cv::Mat> images;
  for (int i = 3; i < argc; i++) {
    images.push_back(cv::imread(argv[i]));
  }

  test_sync_api(model_path, images, threshold);

  return 0;
}
