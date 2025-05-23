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
std::string packOutput(const std::vector<TrackerInfo> &track_results,
                       uint32_t img_width, uint32_t img_height,
                       const std::map<uint64_t, int> &det_results) {
  std::string str_content;
  for (auto &track_result : track_results) {
    printf(
        "track_result: "
        "%d,box:[%.2f,%.2f,%.2f,%.2f],score:%.2f,image_width:%d,image_height:%"
        "d\n",
        int(track_result.track_id_), track_result.box_info_.x1,
        track_result.box_info_.y1, track_result.box_info_.x2,
        track_result.box_info_.y2, track_result.box_info_.score, img_width,
        img_height);
    if (track_result.obj_idx_ == -1) {
      printf("track_result.obj_idx_ == -1, continue\n");
      continue;
    }

    int falling = det_results.at(track_result.track_id_);

    float ctx = (track_result.box_info_.x1 + track_result.box_info_.x2) / 2;
    float cty = (track_result.box_info_.y1 + track_result.box_info_.y2) / 2;
    float w = track_result.box_info_.x2 - track_result.box_info_.x1;
    float h = track_result.box_info_.y2 - track_result.box_info_.y1;
    ctx = ctx / img_width;
    cty = cty / img_height;
    w = w / img_width;
    h = h / img_height;
    // printf("ctx:%.2f,cty:%.2f,w:%.2f,h:%.2f,imgw:%d,imgh:%d\n", ctx, cty, w,
    // h,
    //        img_width, img_height);
    char sz_content[1024];
    sprintf(sz_content, "%d %.2f %.2f %.2f %.2f %d %.2f falling:%d\n",
            int(track_result.box_info_.object_type), ctx, cty, w, h,
            int(track_result.track_id_), track_result.box_info_.score, falling);

    str_content += std::string(sz_content);
  }
  return str_content;
}

void visualizeDetections(
    const std::string &dst_dir, uint32_t frame_id,
    std::shared_ptr<BaseImage> image,
    const std::vector<ObjectBoxLandmarkInfo> &person_boxes_keypoints,
    const std::vector<TrackerInfo> &track_results,
    const std::map<uint64_t, int> &det_results) {
  cv::Mat mat = *(cv::Mat *)image->getInternalData();
  // int obj_idx = 0;
  char szinfo[128];

  for (auto &t : track_results) {
    if (t.obj_idx_ != -1) {
      ObjectBoxLandmarkInfo ferson_info = person_boxes_keypoints[t.obj_idx_];
      cv::rectangle(mat,
                    cv::Rect(ferson_info.x1, ferson_info.y1,
                             ferson_info.x2 - ferson_info.x1,
                             ferson_info.y2 - ferson_info.y1),
                    cv::Scalar(0, 255, 0), 2);
      sprintf(szinfo, "falling: %d", det_results.at(t.track_id_));
      int ctx = (ferson_info.x1 + ferson_info.x2) / 2;
      int cty = (ferson_info.y1 + ferson_info.y2) / 2;
      cv::putText(mat, szinfo, cv::Point(ctx, cty), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 0, 255), 2);
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
      AppFactory::createAppTask("fall_detection", config_file);

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
      std::shared_ptr<FallDetectionResult> fd_result =
          result.get<std::shared_ptr<FallDetectionResult>>();
      if (fd_result == nullptr) {
        std::cout << "fd_result is nullptr" << std::endl;
        continue;
      }
      std::string output_dir = channel_output_dirs[channel_name];
      // visualizeDetections(output_folder_path, fd_result->frame_id,
      //                     fd_result->image,
      //                     fd_result->person_boxes_keypoints,
      //                     fd_result->track_results, fd_result->det_results);
      std::cout << "fd_result->frame_id:" << fd_result->frame_id << std::endl;
      std::string str_content =
          packOutput(fd_result->track_results, fd_result->frame_width,
                     fd_result->frame_height, fd_result->det_results);

      sprintf(sz_frame_name, "%s/%08lu.txt", output_dir.c_str(),
              fd_result->frame_id - 1);

      std::ofstream outf(sz_frame_name);
      outf << str_content;
      outf.close();
      channel_counter[channel_name]++;
      std::cout << "export fall detection result done" << std::endl;
    }
  }
  app_task->release();
}
