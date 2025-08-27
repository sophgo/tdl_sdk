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
                         std::shared_ptr<BaseImage> image,
                         const std::vector<ObjectBoxInfo> &det_boxes) {
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

  char szinfo[128];
  int obj_idx = 0;
  for (auto &box : det_boxes) {
    cv::rectangle(mat,
                  cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1),
                  cv::Scalar(0, 255, 0), 2);
    sprintf(szinfo, "%d-%.1f", obj_idx, box.x1);
    int ctx = (box.x1 + box.x2) / 2;
    int cty = (box.y1 + box.y2) / 2;
    cv::putText(mat, szinfo, cv::Point(ctx, cty), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 2);
    obj_idx++;
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

      // visualizeDetections(output_folder_path,
      // consumer_counting_result->frame_id,
      //                     consumer_counting_result->image,
      //                     consumer_counting_result->object_boxes);

      channel_counter[channel_name]++;
      std::cout << "export consumer counting result done" << std::endl;
    }
  }
  app_task->release();
}
