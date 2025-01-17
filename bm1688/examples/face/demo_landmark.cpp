#include <face/face_detect_landmark.hpp>
#include <factory/model.hpp>
#include <log/Logger.hpp>
void test_sync_api(FaceCSSD *detector, FaceLandmark *landmark,
                   const std::string &img_path, const float &threshold) {
  std::vector<cv::Mat> images{cv::imread(img_path)};

  std::vector<std::vector<FaceRect>> detections;
  detector->detect(images, threshold, detections);
  for (int k = 0; k < images.size(); k++) {
    std::vector<FaceRect> detection = detections[k];
    std::vector<cv::Rect> crop_boxes;
    std::vector<FacePts> facePts;
    std::vector<cv::Mat> crop_mats;
    for (const auto &rect : detection) {
      // printf("(x1, y1, x2, y2): %f, %f, %f, %f\n", rect.x1, rect.y1, rect.x2,
      // rect.y2);
      int x1 = rect.x1;
      int x2 = rect.x2;
      int y1 = rect.y1;
      int y2 = rect.y2;
      int boxw = x2 - x1 + 1;
      int boxh = y2 - y1 + 1;
      int maxwh = std::max(boxw, boxh);
      int square_x = (x1 + x2) / 2 - maxwh / 2;
      int square_y = (y1 + y2) / 2 - maxwh / 2;
      cv::Rect crop_rect(square_x, square_y, maxwh, maxwh);
      cv::Rect inside_rect =
          crop_rect & cv::Rect(0, 0, images[k].cols, images[k].rows);
      cv::Mat crop_face = images[k](inside_rect);
      crop_mats.push_back(crop_face);
    }
    landmark->detect(crop_mats, threshold, facePts);
    // print points
    int n = 0;
    for (const auto &face_point : facePts) {
      printf("For face-%02d\n", n++);
      for (size_t i = 0; i < face_point.x.size(); i++) {
        printf("\t x: %f \t y: %f\n", face_point.x[i], face_point.y[i]);
      }
    }  // end print
  }
  delete landmark;
  delete detector;
}

void test_async_api(FaceCSSD *detector, FaceLandmark *landmark,
                    const std::string &img_path, const float &threshold) {
  std::vector<cv::Mat> images{cv::imread(img_path)};

  std::vector<std::vector<FaceRect>> detections;
  detector->detect(images, threshold, detections);

  std::vector<cv::Mat> tmp_bgr;
  cv::Mat tmp_resized, tmp_transposed;
  for (int k = 0; k < images.size(); k++) {
    std::vector<FaceRect> detection = detections[k];
    std::vector<std::vector<cv::Mat>> frame_bgrs;
    std::vector<cv::Size> frame_sizes;
    std::vector<cv::Rect> crop_boxes;
    std::vector<FacePts> facePts;
    std::vector<cv::Mat> crop_mats;
    LOG(INFO) << "Image-" << k << "\tNum of BBox: " << detection.size()
              << std::endl;
    for (const auto &rect : detection) {
      int x1 = static_cast<int>(rect.x1);
      int x2 = static_cast<int>(rect.x2);
      int y1 = static_cast<int>(rect.y1);
      int y2 = static_cast<int>(rect.y2);
      int boxw = x2 - x1 + 1;
      int boxh = y2 - y1 + 1;
      int maxwh = std::max(boxw, boxh);
      int square_x = (x1 + x2) / 2 - maxwh / 2;
      int square_y = (y1 + y2) / 2 - maxwh / 2;
      cv::Rect crop_rect(square_x, square_y, maxwh, maxwh);
      cv::Rect inside_rect =
          crop_rect & cv::Rect(0, 0, images[k].cols, images[k].rows);
      cv::Mat crop_face = images[k](inside_rect);

      crop_boxes.push_back(inside_rect);
      crop_mats.push_back(crop_face);
      std::vector<cv::Mat> bgr;
      landmark->preprocess(crop_face, tmp_resized, tmp_transposed, tmp_bgr,
                           bgr);
      frame_bgrs.push_back(bgr);
      frame_sizes.push_back(inside_rect.size());
      std::cout << "box:" << inside_rect << ",img_size:" << crop_face.size()
                << std::endl;
    }
    landmark->detect_direct(frame_bgrs, threshold, frame_sizes, facePts);
    for (int i = 0; i < facePts.size(); i++) {
      cv::rectangle(images[k], crop_boxes[i], cv::Scalar(255, 0, 0), 2);
      if (facePts[i].x.empty()) {
        std::cout << std::endl;
        continue;
      }
      for (int j = 0; j < 5; ++j) {
        facePts[i].x[j] += crop_boxes[i].x;
        facePts[i].y[j] += crop_boxes[i].y;
        std::cout << facePts[i].x[j] << "," << facePts[i].y[j] << std::endl;
        cv::circle(images[k], cv::Point(facePts[i].x[j], facePts[i].y[j]), 1,
                   cv::Scalar(255, 255, 0), 2);
      }
      std::cout << "start to estimate pose" << std::endl;
      std::vector<float> ret = face_pose_estimate(facePts[i], images[k].size());
      char sz_info[128];
      sprintf(sz_info, "%d,%d,%d,%.2f,%.2f", int(ret[0]), int(ret[1]),
              int(ret[2]), ret[3], ret[4]);
      cv::putText(images[k], sz_info, crop_boxes[i].tl(),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    static int counter = 0;
    cv::imwrite("detect" + std::to_string(counter++) + ".jpg", images[k]);
  }

  delete landmark;
  delete detector;
}

int main(int argc, char **argv) {
  LogStream::logger.setLogLevel(INFO);  // 设置日志级别
  // CHECK_EQ(argc, 4) << "Example usage: ./bin/demo_landmark ../models 0.5 "
  //                      "../sample/test_2.jpg";

  const std::string &model_path = argv[1];
  const float &threshold = strtof(argv[2], nullptr);
  const std::string &img_path = argv[3];

  NNFactory factory(model_path);
  auto detector = (FaceCSSD *)(factory.get_model(CSSD));
  auto landmark = (FaceLandmark *)(factory.get_model(DET3));
  test_sync_api(detector, landmark, img_path, threshold);

  return 0;
}
