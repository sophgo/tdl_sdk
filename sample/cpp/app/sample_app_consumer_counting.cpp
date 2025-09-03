#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include "app/app_data_types.hpp"
#include "app/app_task.hpp"
#include "framework/utils/tdl_log.hpp"
#include "opencv2/opencv.hpp"
bool make_dir(const char *path, mode_t mode = 0755) {
  if (mkdir(path, mode) == 0) {
    return true;  // 创建成功
  }
  if (errno == EEXIST) {
    return false;  // 已经存在
  }
  // 其他错误
  std::cerr << "mkdir failed: " << std::strerror(errno) << ",dir:" << path
            << "\n";
  return false;
}

void visualizeDetections(const std::string &dst_dir, uint32_t frame_id,
                         const std::vector<TrackerInfo> &track_results,
                         std::shared_ptr<BaseImage> image,
                         const std::vector<ObjectBoxInfo> &det_boxes,
                         std::vector<int> &counting_line) {
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
  cv::line(mat, cv::Point(counting_line[0], counting_line[1]),
           cv::Point(counting_line[2], counting_line[3]), cv::Scalar(0, 0, 255),
           2);

  char szinfo[128];
  for (int i = 0; i < track_results.size(); i++) {
    TrackerInfo track_info = track_results[i];

    if (track_info.obj_idx_ != -1) {
      uint64_t track_id = track_info.track_id_;

      ObjectBoxInfo box = det_boxes[track_info.obj_idx_];
      cv::rectangle(mat,
                    cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1),
                    cv::Scalar(0, 255, 0), 2);
      sprintf(szinfo, "id:%ld", track_id);
      int ctx = (box.x1 + box.x2) / 2;
      int cty = (box.y1 + box.y2) / 2;
      cv::putText(mat, szinfo, cv::Point(ctx, cty), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 255, 0), 2);
    }
  }

  char sz_frame_name[1024];
  sprintf(sz_frame_name, "%s/%08d.jpg", dst_dir.c_str(), frame_id);
  cv::imwrite(sz_frame_name, mat);
}

int main(int argc, char **argv) {
  const std::string config_file = argv[1];
  const std::string output_folder_path = argv[2];

  std::shared_ptr<AppTask> app_task =
      AppFactory::createAppTask("consumer_counting", config_file);

  int32_t ret = app_task->init();
  if (ret != 0) {
    std::cout << "app_task init failed" << std::endl;
    return -1;
  }
  std::vector<std::string> channel_names = app_task->getChannelNames();
  std::map<std::string, std::string> channel_output_dirs;
  char sz_frame_name[1024];
  std::map<std::string, int> channel_counter;
  for (auto &channel_name : channel_names) {
    std::string output_dir = output_folder_path + "/" + channel_name;
    make_dir(output_dir.c_str());
    channel_output_dirs[channel_name] = output_dir;
    channel_counter[channel_name] = 0;
  }

  while (true) {
    int processing_channel_num = app_task->getProcessingChannelNum();
    if (processing_channel_num == 0) {
      std::cout << "no processing channel, break" << std::endl;
      break;
    }
    for (const auto &channel_name : channel_names) {
      Packet result;
      std::cout << "to get result from channel:" << channel_name << std::endl;
      int ret = app_task->getResult(channel_name, result);
      if (ret != 0) {
        std::cout << "get result failed" << std::endl;
        app_task->removeChannel(channel_name);
        continue;
      }
      std::shared_ptr<ConsumerCountingResult> consumer_counting_result =
          result.get<std::shared_ptr<ConsumerCountingResult>>();
      if (consumer_counting_result == nullptr) {
        std::cout << "fd_result is nullptr" << std::endl;
        continue;
      }
      std::string output_dir = channel_output_dirs[channel_name];

      printf("enter:%d, miss:%d\n", consumer_counting_result->enter_num,
             consumer_counting_result->miss_num);

      std::vector<int> counting_line(4, 0);
      for (int i = 0; i < consumer_counting_result->counting_line.size(); i++) {
        counting_line[i] = consumer_counting_result->counting_line[i];
      }
      visualizeDetections(output_dir, consumer_counting_result->frame_id,
                          consumer_counting_result->track_results,
                          consumer_counting_result->image,
                          consumer_counting_result->object_boxes,
                          counting_line);

      channel_counter[channel_name]++;
      std::cout << "export consumer counting result done" << std::endl;
    }
  }
  app_task->release();
}
