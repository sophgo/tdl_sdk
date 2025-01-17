
#include <face/face_detect_landmark.hpp>
#include <face/face_util.hpp>
#include <factory/model.hpp>
#include <log/Logger.hpp>
void test_sync_api(FaceCSSD *detector, FaceLandmark *landmark,
                   FeatureExtract *extractor,
                   const std::vector<cv::Mat> &images, const float &threshold) {
  // auto extractor = (FeatureExtract*) (factory.get_model(BMFACER18));

  const float landmark_threshold = 0.9;
  std::vector<std::vector<FaceRect>> detections;
  FaceDetectLandmark fdl(detector, landmark);
  fdl.setup();
  fdl.detectlandmark(images, threshold, landmark_threshold, detections);

  std::vector<cv::Mat> align_faces;
  for (int k = 0; k < images.size(); k++) {
    std::vector<FaceRect> detection = detections[k];
    auto &rect = detection[0];
    cv::Mat aligned_face = align_face(images[k], rect.facepts, 112, 112);
    align_faces.push_back(aligned_face);
  }

  std::vector<std::vector<float>> features;
  extractor->extract(align_faces, features);
  float sim = calc_cosine(features[0], features[1]);
  std::cout << "opencv sync Face similarity is: " << sim << std::endl;

  delete extractor;
}

int main(int argc, char **argv) {
  LogStream::logger.setLogLevel(INFO);  // 设置日志级别
  // CHECK_EQ(argc, 5)
  //     << "Example usage: ./bin/demo_recog ../models 0.5 "
  //        "../sample/Abel_Pacheco_0001.jpg ../sample/Abel_Pacheco_0004.jpg";

  const std::string &model_path = argv[1];
  const float &threshold = strtof(argv[2], nullptr);
  const std::string &img_path_a = argv[3];
  const std::string &img_path_b = argv[4];
  std::vector<cv::Mat> images;
  images.push_back(cv::imread(img_path_a));
  images.push_back(cv::imread(img_path_b));

  NNFactory factory(model_path);
  auto detector = (FaceCSSD *)(factory.get_model(CSSD));
  auto landmark = (FaceLandmark *)(factory.get_model(DET3));
  auto extractor = (FeatureExtract *)(factory.get_model(BMFACER34_V2));
  test_sync_api(detector, landmark, extractor, images, threshold);
}
