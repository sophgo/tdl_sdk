#include "face/face_detect_landmark.hpp"
#include <log/Logger.hpp>
bmStatus_t FaceDetectLandmark::setup() {
  if (!init_with_ptr) {
    face_detector_ = std::make_shared<FaceCSSD>(detect_param_);
    face_detector_->setup();
    face_landmark_ = std::make_shared<FaceLandmark>(landmark_param_);
    face_landmark_->setup();
  }
  return BM_COMMON_SUCCESS;
}

bmStatus_t FaceDetectLandmark::detectlandmark(
    const std::vector<cv::Mat>& images,
    const float det_threshold,
    std::vector<std::vector<FaceRect>>& faceRects) {
  return detectlandmark(images, det_threshold, 0.5, faceRects);
}

bmStatus_t FaceDetectLandmark::detectlandmark(
    const std::vector<cv::Mat>& images,
    const float det_threshold,
    const float lm_threshold,
    std::vector<std::vector<FaceRect>>& faceRects) {
  faceRects.resize(images.size());
  std::vector<std::vector<FaceRect>> detects;
  face_detector_->detect(images, det_threshold, detects);
  std::vector<cv::Mat> crop_faces;

  for (int i = 0; i < images.size(); i++) {
    for (auto& rect : detects[i]) {
      int x1 = rect.x1;
      int x2 = rect.x2;
      int y1 = rect.y1;
      int y2 = rect.y2;
      //crop using square box
      int boxw = x2 - x1 + 1;
      int boxh = y2 - y1 + 1;
      int maxwh = std::max(boxw, boxh);
      int square_x = (x1 + x2) / 2 - maxwh / 2;
      int square_y = (y1 + y2) / 2 - maxwh / 2;
      cv::Rect crop_rect(square_x, square_y, maxwh, maxwh);
      cv::Rect inside_rect = crop_rect & cv::Rect(0, 0, images[i].cols, images[i].rows);

      cv::Mat crop_face;
      rect.temp_x1 = rect.x1;
      rect.temp_y1 = rect.y1;
      rect.temp_x2 = rect.x2;
      rect.temp_y2 = rect.y2;
      if (inside_rect.area() == crop_rect.area()) {//total inside the image
        crop_face = images[i](inside_rect);
        //modify face rect because the landmark points would be updated use face rect
        rect.x1 = square_x;
        rect.y1 = square_y;
        rect.x2 = square_x + maxwh;
        rect.y2 = square_y + maxwh;
      } else {
        crop_face = images[i](cv::Range(y1, y2), cv::Range(x1, x2));
      }
      crop_faces.push_back(crop_face);
    }
  }

  std::vector<FacePts> facePts;
  face_landmark_->detect(crop_faces, lm_threshold, facePts);

  size_t lm_idx = 0;
  for (int i = 0; i < images.size(); i++) {
    for (auto& rect : detects[i]) {
      if (facePts[lm_idx].valid()) {
        for (auto& pt_x : facePts[lm_idx].x) {
          pt_x += rect.x1;
        }
        for (auto& pt_y : facePts[lm_idx].y) {
          pt_y += rect.y1;
        }

        faceRects[i].push_back(rect);
        faceRects[i].back().facepts = facePts[lm_idx];
      }
      ++lm_idx;
    }
  }

  return BM_COMMON_SUCCESS;
}
