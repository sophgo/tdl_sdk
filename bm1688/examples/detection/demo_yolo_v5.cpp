#include <iostream>
#include <log/Logger.hpp>

#include "factory/model.hpp"
cv::Scalar clrs[] = {
    cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
    cv::Scalar(0, 0, 255),   cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),
    cv::Scalar(100, 0, 200), cv::Scalar(100, 200, 0), cv::Scalar(200, 0, 100),
    cv::Scalar(200, 100, 0), cv::Scalar(0, 100, 200), cv::Scalar(0, 0, 0)};
void test_sync_api(const std::string &model_path, std::vector<cv::Mat> &images,
                   const float &threshold, int device_id) {
  NNFactory factory(model_path);
  auto detector =
      (YOLOV5 *)(factory.get_model(NNBaseModel::YOLO_V5_VEHICLE, device_id));
  std::vector<std::vector<ObjectBox>> detections;
  LOG(INFO) << "----------------start to do sync detect--------------\n";
  detector->detect(images, threshold, detections);
  int num_clr = sizeof(clrs) / sizeof(clrs[0]);
  for (int k = 0; k < images.size(); k++) {
    auto &detection = detections[k];
    for (const auto &rect : detection) {
      auto x1 = static_cast<int>(rect.x1);
      auto y1 = static_cast<int>(rect.y1);
      auto x2 = static_cast<int>(rect.x2);
      auto y2 = static_cast<int>(rect.y2);
      cv::Scalar clr = clrs[rect.label % num_clr];
      cv::rectangle(images[k], cv::Point(x1, y1), cv::Point(x2, y2), clr, 2);
      std::string str_type = std::to_string((rect.label));

      cv::putText(images[k], str_type, cv::Point(x1, y1),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, clr, 1);
    }
    cv::imwrite(std::to_string(k) + std::string("sync_.jpg"), images[k]);
  }
  LOG(INFO) << "----------------finish do sync detect--------------\n";
  delete detector;
}

void test_async_api(const std::string &model_path, std::vector<cv::Mat> &images,
                    const float &threshold, int device_id) {
  NNFactory factory(model_path);
  auto detector =
      (YOLOV5 *)(factory.get_model(NNBaseModel::YOLO_V5_VEHICLE, device_id));

  std::vector<std::vector<float>> frame_scale_params;
  std::vector<cv::Size> frame_sizes;
  std::vector<std::vector<cv::Mat>> frame_bgrs;
  std::vector<cv::Mat> tmp_bgr;
  std::vector<std::vector<ObjectBox>> detections;
  cv::Mat tmp_resized;
  int num_clr = sizeof(clrs) / sizeof(clrs[0]);
  for (auto &img : images) {
    std::vector<cv::Mat> bgr;
    std::vector<float> scale_params;

    detector->preprocess_opencv_async(img, tmp_resized, tmp_bgr, bgr,
                                      scale_params);
    // LOG(INFO) << scale_params[0] << "\t" << scale_params[1] << "\t"
    //           << scale_params[2] << "\t" << scale_params[3];
    frame_bgrs.push_back(bgr);
    frame_scale_params.push_back(scale_params);
    frame_sizes.push_back(img.size());
  }

  detector->detect_direct(frame_bgrs, threshold, frame_scale_params,
                          frame_sizes, detections);
  for (int k = 0; k < images.size(); k++) {
    auto &detection = detections[k];
    for (const auto &rect : detection) {
      printf("(x1, y1, x2, y2): %f, %f, %f, %f\n", rect.x1, rect.y1, rect.x2,
             rect.y2);
      auto x1 = static_cast<int>(rect.x1);
      auto y1 = static_cast<int>(rect.y1);
      auto x2 = static_cast<int>(rect.x2);
      auto y2 = static_cast<int>(rect.y2);
      cv::Scalar clr = clrs[rect.label % num_clr];
      cv::rectangle(images[k], cv::Point(x1, y1), cv::Point(x2, y2), clr, 2);
      std::string str_type = std::to_string((rect.label));
      cv::putText(images[k], str_type, cv::Point(x1, y1),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, clr, 1);
    }
    cv::imwrite(std::to_string(k) + std::string("async_.jpg"), images[k]);
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

  test_sync_api(model_path, images, threshold, 0);
  // std::vector<cv::Mat> images1;
  // for (int i = 3; i < argc; i++) {
  //   images1.push_back(cv::imread(argv[i]));
  // }
  // test_async_api(model_path, images1, threshold, 0);

  return 0;
}
