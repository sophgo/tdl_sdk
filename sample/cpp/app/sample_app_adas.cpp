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
#include "vehicle_adas/vehicle_adas.hpp"

static const char *enumStr[] = {"NORMAL", "START", "WARNING"};

bool make_dir(const char *path, mode_t mode = 0755) {
  if (mkdir(path, mode) == 0) {
    return true;
  }
  if (errno == EEXIST) {
    return false;
  }
  return false;
}

static cv::Scalar genRandomColor(uint64_t seed, int min = 64) {
  float scale = (256.0f - (float)min) / 256.0f;
  srand((uint32_t)seed);
  int r = (int)((floor(((float)rand() / (RAND_MAX)) * 256.0f)) * scale) + min;
  int g = (int)((floor(((float)rand() / (RAND_MAX)) * 256.0f)) * scale) + min;
  int b = (int)((floor(((float)rand() / (RAND_MAX)) * 256.0f)) * scale) + min;
  return cv::Scalar(b, g, r);
}

std::string packOutput(const VehicleAdasResult &adas_result) {
  std::string str_content;
  uint32_t img_width = adas_result.frame_width;
  uint32_t img_height = adas_result.frame_height;

  for (auto &obj : adas_result.objects) {
    float ctx = (obj.info.x1 + obj.info.x2) / 2;
    float cty = (obj.info.y1 + obj.info.y2) / 2;
    float w = obj.info.x2 - obj.info.x1;
    float h = obj.info.y2 - obj.info.y1;
    ctx = ctx / img_width;
    cty = cty / img_height;
    w = w / img_width;
    h = h / img_height;

    char sz_content[1024];
    sprintf(sz_content,
            "cls:%d bbox:%.2f %.2f %.2f %.2f dis:%.1f m speed:%.1f m/s "
            "state:%s score:%.2f\n",
            obj.info.class_id, ctx, cty, w, h, obj.distance, obj.speed,
            enumStr[static_cast<int>(obj.state)], obj.info.score);
    str_content += std::string(sz_content);
  }

  if (adas_result.lane_state.lane_state == 1) {
    str_content += "LANE DEPARTURE WARNING!\n";
  }

  return str_content;
}

void visualizeResult(const std::string &dst_dir, uint32_t frame_id,
                     VehicleAdasResult &adas_result, float fps = 15.0f) {
  cv::Mat mat = *(cv::Mat *)adas_result.image->getInternalData();

  uint32_t img_width = adas_result.frame_width;
  uint32_t img_height = adas_result.frame_height;

  // Determine font scale based on image size (like dev_v1)
  int base_size = img_width >= 1080 ? 3 : 2;
  double font_scale = img_width >= 1080 ? 0.9 : 0.5;
  int thickness = img_width >= 1080 ? 2 : 1;

  for (auto &obj : adas_result.objects) {
    cv::Scalar box_color = genRandomColor(obj.track_id, 64);

    cv::Point top_left((int)obj.info.x1, (int)obj.info.y1);
    cv::Point bottom_right((int)obj.info.x2, (int)obj.info.y2);
    cv::rectangle(mat, top_left, bottom_right, box_color, 2);

    char txt_info[256];
    snprintf(txt_info, sizeof(txt_info), "[%d][%s]S:%.1f V:%.1f",
             obj.info.class_id, enumStr[static_cast<int>(obj.state)],
             obj.distance, obj.speed);

    // Red for non-normal states (START / COLLISION_WARNING)
    cv::Scalar text_color = box_color;
    if (obj.state != AdasState::NORMAL) {
      box_color = cv::Scalar(0, 0, 255);
      text_color = cv::Scalar(0, 0, 255);
      // Redraw the box in red
      cv::rectangle(mat, top_left, bottom_right, box_color, 2);
    }

    cv::putText(mat, txt_info,
                cv::Point(top_left.x,
                          top_left.y > 10 ? top_left.y - 8 : top_left.y + 20),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
  }

  // Draw lane lines
  if (!adas_result.lane_lines.empty()) {
    int lane_thickness = img_width >= 1080 ? 4 : 2;

    for (auto &lane : adas_result.lane_lines) {
      cv::line(mat, cv::Point((int)lane.x1, (int)lane.y1),
               cv::Point((int)lane.x2, (int)lane.y2), cv::Scalar(0, 255, 0),
               lane_thickness);
    }

    // Lane state text
    char lane_info[64];
    cv::Scalar lane_text_color;
    if (adas_result.lane_state.lane_state == 0) {
      strcpy(lane_info, "NORMAL");
      lane_text_color = cv::Scalar(0, 255, 0);
    } else {
      strcpy(lane_info, "LANE DEPARTURE WARNING!");
      lane_text_color = cv::Scalar(0, 0, 255);
    }

    double lane_font_scale = img_width >= 1080 ? 1.5 : 0.8;
    cv::putText(mat, lane_info,
                cv::Point((int)(0.3 * img_width), (int)(0.8 * img_height)),
                cv::FONT_HERSHEY_SIMPLEX, lane_font_scale, lane_text_color,
                lane_thickness);
  }

#ifdef __BM168X__
  // BM168X平台: 合成视频
  static std::map<std::string, cv::VideoWriter> video_writers;

  if (video_writers.find(dst_dir) == video_writers.end()) {
    std::string video_path = dst_dir + "/output.mp4";
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::Size frame_size(mat.cols, mat.rows);

    video_writers[dst_dir].open(video_path, fourcc, fps, frame_size);
    if (!video_writers[dst_dir].isOpened()) {
      std::cerr << "Failed to create video writer for: " << video_path
                << std::endl;
      return;
    }
  }

  video_writers[dst_dir].write(mat);
#else
  char sz_frame_name[1024];
  sprintf(sz_frame_name, "%s/%08d.jpg", dst_dir.c_str(), frame_id);
  cv::imwrite(sz_frame_name, mat);
#endif
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <config_file> <output_folder_path> [fps]\n", argv[0]);
    return -1;
  }

  const std::string config_file = argv[1];
  const std::string output_folder_path = argv[2];

  float fps = 25.0f;
  if (argc == 4) {
    fps = atof(argv[3]);
    printf("fps:%.2f\n", fps);
  }

  std::shared_ptr<AppTask> app_task =
      AppFactory::createAppTask("vehicle_adas", config_file);

  int32_t ret = app_task->init();
  if (ret != 0) {
    std::cout << "app_task init failed" << std::endl;
    return -1;
  }

  std::vector<std::string> channel_names = app_task->getChannelNames();
  std::map<std::string, std::string> channel_output_dirs;
  char sz_frame_name[1024];

  for (auto &channel_name : channel_names) {
    std::string output_dir = output_folder_path + "/" + channel_name;
    make_dir(output_dir.c_str());
    channel_output_dirs[channel_name] = output_dir;
    std::cout << "channel: " << channel_name << " -> " << output_dir
              << std::endl;
  }

  while (true) {
    int processing_channel_num = app_task->getProcessingChannelNum();
    if (processing_channel_num == 0) {
      std::cout << "no processing channel, break" << std::endl;
      break;
    }
    for (const auto &channel_name : channel_names) {
      Packet result;
      int ret = app_task->getResult(channel_name, result);

      if (ret != 0) {
        std::cout << "get result failed for channel:" << channel_name
                  << std::endl;
        app_task->removeChannel(channel_name);
        continue;
      }
      std::shared_ptr<VehicleAdasResult> adas_result =
          result.get<std::shared_ptr<VehicleAdasResult>>();
      if (adas_result == nullptr) {
        std::cout << "adas_result is nullptr" << std::endl;
        continue;
      }

      std::string output_dir = channel_output_dirs[channel_name];

      visualizeResult(output_dir, adas_result->frame_id, *adas_result, fps);

      std::string str_content = packOutput(*adas_result);
      sprintf(sz_frame_name, "%s/%08lu.txt", output_dir.c_str(),
              adas_result->frame_id);
      std::ofstream outf(sz_frame_name);
      outf << str_content;
      outf.close();

      std::cout << "frame_id:" << adas_result->frame_id
                << " objects:" << adas_result->objects.size()
                << " lane_state:" << adas_result->lane_state.lane_state
                << " lanes:" << adas_result->lane_lines.size() << std::endl;
    }
  }

  app_task->release();
  std::cout << "done" << std::endl;
  return 0;
}
