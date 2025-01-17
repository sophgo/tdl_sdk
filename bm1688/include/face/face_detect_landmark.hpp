#ifndef FACE_DETECT_LANDMARK_HPP_
#define FACE_DETECT_LANDMARK_HPP_
#include "face_cssd.hpp"
#include "face_landmark.hpp"

class FaceDetectLandmark {
 public:
  FaceDetectLandmark(const stNetParam detect_param, const stNetParam landmark_param)
      : detect_param_(detect_param),
        landmark_param_(landmark_param) { init_with_ptr = false; }

  FaceDetectLandmark(const std::shared_ptr<FaceCSSD> detector, const std::shared_ptr<FaceLandmark> landmark)
      : face_detector_(detector),
        face_landmark_(landmark) { init_with_ptr = true; }

  FaceDetectLandmark(FaceCSSD* detector, FaceLandmark* landmark)
      : face_detector_(std::shared_ptr<FaceCSSD>(detector)),
        face_landmark_(std::shared_ptr<FaceLandmark>(landmark)) { init_with_ptr = true; }

  bmStatus_t setup();
  bmStatus_t detectlandmark(const std::vector<cv::Mat>& images,
                            const float det_threshold,
                            std::vector<std::vector<FaceRect>>& faceRects);
  bmStatus_t detectlandmark(const std::vector<cv::Mat>& images,
                            const float det_threshold,
                            const float lm_threshold,
                            std::vector<std::vector<FaceRect>>& faceRects);

 private:
  stNetParam detect_param_;
  stNetParam landmark_param_;
  std::shared_ptr<FaceCSSD> face_detector_ = NULL;
  std::shared_ptr<FaceLandmark> face_landmark_ = NULL;
  bool init_with_ptr = false;
  TimeRecorder timer_;
};

#endif
