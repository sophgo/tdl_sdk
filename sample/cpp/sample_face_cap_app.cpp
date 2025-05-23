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
                       uint32_t img_width, uint32_t img_height) {
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
    sprintf(sz_content, "%d %.2f %.2f %.2f %.2f %d %.2f\n",
            int(track_result.box_info_.object_type), ctx, cty, w, h,
            int(track_result.track_id_), track_result.box_info_.score);
    str_content += std::string(sz_content);
  }
  return str_content;
}

void exportFaceSnapshots(
    const std::string &dst_dir, uint32_t frame_id,
    const std::vector<ObjectSnapshotInfo> &face_snapshots) {
  char sz_frame_name[1024];
  for (auto &face_snapshot : face_snapshots) {
    sprintf(sz_frame_name, "%s/track_%d_frame_%04d_quality_%.2f.jpg",
            dst_dir.c_str(), int(face_snapshot.track_id),
            int(face_snapshot.snapshot_frame_id), face_snapshot.quality);
    printf("write image %s\n", sz_frame_name);
    cv::Mat mat = *(cv::Mat *)face_snapshot.object_image->getInternalData();
    auto &box = face_snapshot.object_box_info;
    cv::rectangle(mat,
                  cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1),
                  cv::Scalar(0, 255, 0), 2);
    std::vector<float> landmarks =
        face_snapshot.other_info.at("landmarks").get<std::vector<float>>();
    for (size_t i = 0; i < landmarks.size(); i += 2) {
      cv::circle(mat, cv::Point(landmarks[i], landmarks[i + 1]), 2,
                 cv::Scalar(0, 0, 255), -1);
    }

    int32_t ret =
        ImageFactory::writeImage(sz_frame_name, face_snapshot.object_image);
    if (ret != 0) {
      std::cout << "write image " << sz_frame_name << " failed" << std::endl;
    }
  }
}
void visualizeDetections(const std::string &dst_dir, uint32_t frame_id,
                         std::shared_ptr<BaseImage> image,
                         const std::vector<ObjectBoxInfo> &person_boxes,
                         const std::vector<ObjectBoxLandmarkInfo> &face_boxes) {
  cv::Mat mat = *(cv::Mat *)image->getInternalData();
  int obj_idx = 0;
  char szinfo[128];
  for (auto &box : face_boxes) {
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
  for (auto &box : person_boxes) {
    cv::rectangle(mat,
                  cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1),
                  cv::Scalar(0, 0, 255), 2);
    sprintf(szinfo, "%d-%.1f", obj_idx, box.x1);
    int ctx = (box.x1 + box.x2) / 2;
    int cty = (box.y1 + box.y2) / 2;
    cv::putText(mat, szinfo, cv::Point(ctx, cty), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 255), 2);
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
      AppFactory::createAppTask("face_capture", config_file);

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
      std::shared_ptr<FaceCaptureResult> cap_result =
          result.get<std::shared_ptr<FaceCaptureResult>>();
      if (cap_result == nullptr) {
        std::cout << "cap_result is nullptr" << std::endl;
        continue;
      }
      std::string output_dir = channel_output_dirs[channel_name];
      // visualizeDetections(output_folder_path, cap_result->frame_id,
      //                     cap_result->image, cap_result->person_boxes,
      //                     cap_result->face_boxes);
      std::cout << "cap_result->frame_id:" << cap_result->frame_id << std::endl;
      std::string str_content =
          packOutput(cap_result->track_results, cap_result->frame_width,
                     cap_result->frame_height);

      sprintf(sz_frame_name, "%s/%08lu.txt", output_dir.c_str(),
              cap_result->frame_id - 1);

      std::ofstream outf(sz_frame_name);
      outf << str_content;
      outf.close();
      channel_counter[channel_name]++;
      exportFaceSnapshots(output_dir, cap_result->frame_id,
                          cap_result->face_snapshots);
      std::cout << "export face snapshots done" << std::endl;
    }
  }
  app_task->release();
}
