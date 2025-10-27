#include <iostream>
#include "utils/qwen_vl_helper.hpp"
int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    std::cout << "Usage: " << argv[0]
              << " <video_path>  [desired_fps] [desired_nframes]" << std::endl;
    std::cout << "Usage: " << argv[0]
              << " <video_path> [desired_fps] [desired_nframes] "
                 "[max_video_sec]"
              << std::endl;
    return 1;
  }
  std::string video_path = argv[1];

  float desired_fps = std::stof(argv[2]);
  int desired_nframes = std::stoi(argv[3]);
  int max_video_sec = 0;
  if (argc == 5) {
    max_video_sec = std::stoi(argv[4]);
  }
  std::map<std::string, float> perf_info = QwenVLHelper::testFetchVideoTs(
      video_path, desired_fps, desired_nframes, max_video_sec);
  for (auto &item : perf_info) {
    std::cout << item.first << ": " << item.second << std::endl;
  }
  return 0;
}
