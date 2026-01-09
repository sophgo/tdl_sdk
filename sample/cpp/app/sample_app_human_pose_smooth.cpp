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

std::vector<cv::Scalar> color = {
    cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76), cv::Scalar(255, 215, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(0, 255, 0)};
int color_map[17] = {0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 4, 3, 4, 3, 4, 3};
int line_map[19] = {4, 4, 3, 3, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0};
int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                       {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
                       {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
                       {1, 3},   {2, 4},   {3, 5},   {4, 6}};
// #ifdef __BM168X__
// static std::map<std::string, cv::VideoWriter> video_writers;
// #endif

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

void visualizeDetections(
    const std::string &dst_dir, uint32_t frame_id,
    std::shared_ptr<BaseImage> image,
    const std::vector<ObjectBoxLandmarkInfo> &person_boxes_keypoints,
    const std::vector<TrackerInfo> &track_results, float fps) {
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

      for (uint32_t j = 0; j < 17; j++) {
        if (ferson_info.landmarks_score[j] < 0.5) continue;
        int x = static_cast<int>(ferson_info.landmarks_x[j]);
        int y = static_cast<int>(ferson_info.landmarks_y[j]);
        cv::circle(mat, cv::Point(x, y), 7, color[color_map[j]], -1);
      }
      for (uint32_t k = 0; k < 19; k++) {
        int kps1 = skeleton[k][0];
        int kps2 = skeleton[k][1];
        if (ferson_info.landmarks_score[kps1] < 0.5 ||
            ferson_info.landmarks_score[kps2] < 0.5)
          continue;
        int x1 = static_cast<int>(ferson_info.landmarks_x[kps1]);
        int y1 = static_cast<int>(ferson_info.landmarks_y[kps1]);
        int x2 = static_cast<int>(ferson_info.landmarks_x[kps2]);
        int y2 = static_cast<int>(ferson_info.landmarks_y[kps2]);
        cv::line(mat, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]],
                 2);
      }
      sprintf(szinfo, "id: %d", int(t.track_id_));
      int ctx = (ferson_info.x1 + ferson_info.x2) / 2;
      int cty = (ferson_info.y1 + ferson_info.y2) / 2;
      cv::putText(mat, szinfo, cv::Point(ctx, cty), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 0, 255), 2);
    }
  }

#ifdef __BM168X__
  // BM168X平台: 合成视频
  static std::map<std::string, cv::VideoWriter> video_writers;

  if (video_writers.find(dst_dir) == video_writers.end()) {
    // 首次创建视频写入器
    // 修改1: 将文件扩展名改为.mp4
    std::string video_path = dst_dir + "/output.mp4";
    // 修改2: 使用H.264编码器(X264)替代MJPG
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::Size frame_size(mat.cols, mat.rows);

    video_writers[dst_dir].open(video_path, fourcc, fps, frame_size);
    if (!video_writers[dst_dir].isOpened()) {
      std::cerr << "Failed to create video writer for: " << video_path
                << std::endl;
      return;
    }
  }

  // 写入当前帧
  video_writers[dst_dir].write(mat);
#else
  // 其他平台: 保存单张图片
  char sz_frame_name[1024];
  sprintf(sz_frame_name, "%s/%08d.jpg", dst_dir.c_str(), frame_id);
  cv::imwrite(sz_frame_name, mat);
#endif
}

int main(int argc, char **argv) {
  const std::string config_file = argv[1];
  const std::string output_folder_path = argv[2];

  float fps =
      30.0;  //给BM168X平台合成画框视频用，由于缓冲机制，跑完程序要等一会合成的视频才能播放
  if (argc == 4) {
    fps = atof(argv[3]);
    printf("fps:%.2f\n", fps);
  }

  std::shared_ptr<AppTask> app_task =
      AppFactory::createAppTask("human_pose_smooth", config_file);

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
      std::shared_ptr<HumanPoseResult> hp_result =
          result.get<std::shared_ptr<HumanPoseResult>>();
      if (hp_result == nullptr) {
        std::cout << "hp_result is nullptr" << std::endl;
        continue;
      }
      std::string output_dir = channel_output_dirs[channel_name];
      visualizeDetections(output_dir, hp_result->frame_id, hp_result->image,
                          hp_result->person_boxes_keypoints,
                          hp_result->track_results, fps);
      std::cout << "hp_result->frame_id:" << hp_result->frame_id << std::endl;
      channel_counter[channel_name]++;
      std::cout << "export fall detection result done" << std::endl;
    }
  }
  app_task->release();
}
