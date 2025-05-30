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

#define FACE_FEAT_SIZE 256

bool make_dir(const char *path, mode_t mode = 0755) {
  if (mkdir(path, mode) == 0) {
    return true;  // 创建成功
  }
  if (errno == EEXIST) {
    return false;  // 已经存在
  }
  return false;
}

int get_gallery_feature(const char *sz_feat_file,
                        std::vector<float> &g_feature) {
  FILE *fp = fopen(sz_feat_file, "rb");
  if (fp == NULL) {
    printf("read %s failed\n", sz_feat_file);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  int8_t *ptr_feat = (int8_t *)malloc(len);
  fread(ptr_feat, 1, len, fp);
  fclose(fp);
  printf("read %s done,len:%d\n", sz_feat_file, len);
  if (len != FACE_FEAT_SIZE) {
    free(ptr_feat);
    return -1;
  }

  for (size_t i = 0; i < FACE_FEAT_SIZE; i++) {
    g_feature[i] = (float)ptr_feat[i];
  }

  printf("register feature sucessfully\n");
  return 0;
}

float doFaceRecognition(const std::vector<std::vector<float>> &g_features,
                        const std::vector<float> &feature, uint8_t &top_index) {
  float max_sim = 0;
  for (size_t i = 0; i < g_features.size(); i++) {
    float sim = 0;
    float norm1 = 0;
    float norm2 = 0;
    for (size_t j = 0; j < FACE_FEAT_SIZE; j++) {
      sim += g_features[i][j] * feature[j];
      norm1 += g_features[i][j] * g_features[i][j];
      norm2 += feature[j] * feature[j];
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    sim = sim / (norm1 * norm2);
    if (sim > max_sim) {
      max_sim = sim;
      top_index = i;
    }
  }
  return max_sim;
}

std::string packOutput(const std::vector<TrackerInfo> &track_results,
                       uint32_t img_width, uint32_t img_height) {
  std::string str_content;
  for (auto &track_result : track_results) {
    // printf(
    //     "track_result: "
    //     "%d,box:[%.2f,%.2f,%.2f,%.2f],score:%.2f,image_width:%d,image_height:%"
    //     "d\n",
    //     int(track_result.track_id_), track_result.box_info_.x1,
    //     track_result.box_info_.y1, track_result.box_info_.x2,
    //     track_result.box_info_.y2, track_result.box_info_.score, img_width,
    //     img_height);
    if (track_result.obj_idx_ == -1) {
      // printf("track_result.obj_idx_ == -1, continue\n");
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

void exportFaceSnapshots(const std::string &dst_dir, uint32_t frame_id,
                         std::vector<ObjectSnapshotInfo> &face_snapshots) {
  char sz_frame_name[1024];
  for (auto &face_snapshot : face_snapshots) {
    sprintf(sz_frame_name, "%s/track_%d_frame_%04d_quality_%.2f.jpg",
            dst_dir.c_str(), int(face_snapshot.track_id),
            int(face_snapshot.snapshot_frame_id), face_snapshot.quality);
    printf("write image %s\n", sz_frame_name);
    cv::Mat mat;
    bool is_rgb;
    int32_t ret =
        ImageFactory::convertToMat(face_snapshot.object_image, mat, is_rgb);
    if (ret != 0) {
      std::cout << "Failed to convert to mat" << std::endl;
      return;
    }
    if (is_rgb) {
      std::cout << "convert to bgr" << std::endl;
      cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    }
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

    ret = ImageFactory::writeImage(sz_frame_name, face_snapshot.object_image);
    if (ret != 0) {
      std::cout << "write image " << sz_frame_name << " failed" << std::endl;
    }
  }
}
void visualizeDetections(const std::string &dst_dir, uint32_t frame_id,
                         std::shared_ptr<BaseImage> image,
                         const std::vector<ObjectBoxInfo> &person_boxes,
                         const std::vector<ObjectBoxLandmarkInfo> &face_boxes,
                         const std::vector<ObjectBoxInfo> &pet_boxes) {
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

  for (auto &box : pet_boxes) {
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
  if (argc != 4) {
    printf("Usage: %s <config_file> <face_feature_dir> <output_folder_path>\n",
           argv[0]);
    return -1;
  }

  const std::string config_file = argv[1];
  const std::string gallery_dir = argv[2];
  const std::string output_folder_path = argv[3];

  std::vector<std::vector<float>> gallery_features = {};

  for (int i = 0; i < 10; i++) {
    char szbinf[128];
    sprintf(szbinf, "%s/%d.bin", gallery_dir.c_str(), i);

    std::vector<float> g_feature(FACE_FEAT_SIZE);
    if (get_gallery_feature(szbinf, g_feature) != 0) {
      printf("skip %d.bin\n", i);
      continue;
    }
    gallery_features.push_back(g_feature);
  }

  if (gallery_features.size() == 0) {
    printf("failed to register feature!\n");
    return -1;
  }

  std::shared_ptr<AppTask> app_task =
      AppFactory::createAppTask("face_pet_capture", config_file);

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
      std::shared_ptr<FacePetCaptureResult> cap_result =
          result.get<std::shared_ptr<FacePetCaptureResult>>();
      if (cap_result == nullptr) {
        std::cout << "cap_result is nullptr" << std::endl;
        continue;
      }

      std::string output_dir = channel_output_dirs[channel_name];
      // visualizeDetections(output_folder_path, cap_result->frame_id,
      //                     cap_result->image, cap_result->person_boxes,
      //                     cap_result->face_boxes, cap_result->pet_boxes);
      std::cout << "cap_result->frame_id:" << cap_result->frame_id << std::endl;
      std::string str_content =
          packOutput(cap_result->track_results, cap_result->frame_width,
                     cap_result->frame_height);

      sprintf(sz_frame_name, "%s/%08d.txt", output_dir.c_str(),
              cap_result->frame_id - 1);

      std::ofstream outf(sz_frame_name);
      outf << str_content;
      outf.close();
      channel_counter[channel_name]++;
      exportFaceSnapshots(output_dir, cap_result->frame_id,
                          cap_result->face_snapshots);
      std::cout << "export face snapshots done" << std::endl;

      const std::map<uint64_t, std::vector<float>> &face_features =
          cap_result->face_features;

      for (auto iter = face_features.begin(); iter != face_features.end();
           ++iter) {
        std::vector<float> tmp_fea = iter->second;

        uint8_t top_index;
        float max_similarity =
            doFaceRecognition(gallery_features, iter->second, top_index);

        if (max_similarity > 0.4) {
          printf("match feature %d.bin, track id: %ld, similarity: %.2f\n",
                 top_index, iter->first, max_similarity);
        }
      }
    }
  }
  app_task->release();
}
