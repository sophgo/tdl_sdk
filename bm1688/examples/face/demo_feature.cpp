#include <face/face_util.hpp>
#include <factory/model.hpp>
#include <fstream>
#include <log/Logger.hpp>

int main(int argc, char **argv) {
  std::vector<cv::Mat> images;
  // FLAGS_alsologtostderr = 1;
  LogStream::logger.setLogLevel(INFO);  // 设置日志级别
  const std::string &model_dir = argv[1];
  for (int i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    images.push_back(img);
  }
  NNFactory factory(model_dir);
  auto *reid = (FeatureExtract *)(factory.get_model(BMFACER34));

  std::vector<std::vector<float>> features;
  reid->extract(images, features);
  float sim = calc_cosine(features[0], features[1]);
  LOG(INFO) << "face similarity is " << sim
            << ",featsize:" << features[0].size()
            << ",feat1:" << features[1].size() << std::endl;

  return 1;
}
