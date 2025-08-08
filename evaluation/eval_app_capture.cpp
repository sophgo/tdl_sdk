#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
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

std::string pack_track_results(const std::vector<TrackerInfo> &track_results,
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
      // || track_result.box_info_.object_type !=
      // TDLObjectType::OBJECT_TYPE_FACE) {
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
    sprintf(sz_content, "11 %.6f %.6f %.6f %.6f %d %.4f\n", ctx, cty, w, h,
            int(track_result.track_id_), track_result.box_info_.score);
    str_content += std::string(sz_content);
  }
  return str_content;
}

std::string pack_det_results(
    const std::vector<TrackerInfo> &track_results,
    const std::vector<ObjectBoxLandmarkInfo> &face_boxes,
    const std::vector<ObjectBoxInfo> &person_boxes, uint32_t img_width,
    uint32_t img_height) {
  std::string str_content;
  // printf("face_boxes size: %d, person_boxes size:%d\n",face_boxes.size(),
  // person_boxes.size() ); getchar();

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
      // || track_result.box_info_.object_type !=
      // TDLObjectType::OBJECT_TYPE_FACE) {
      // printf("track_result.obj_idx_ == -1, continue\n");
      continue;
    }

    if (track_result.box_info_.object_type == TDLObjectType::OBJECT_TYPE_FACE) {
      ObjectBoxLandmarkInfo f_box = face_boxes[track_result.obj_idx_];
      // printf("f_box: %f, %f, %f, %f\n", f_box.x1, f_box.x2, f_box.y1,
      // f_box.y2);

      float ctx = (f_box.x1 + f_box.x2) / 2;
      float cty = (f_box.y1 + f_box.y2) / 2;
      float w = f_box.x2 - f_box.x1;
      float h = f_box.y2 - f_box.y1;
      ctx = ctx / img_width;
      cty = cty / img_height;
      w = w / img_width;
      h = h / img_height;
      char sz_content[1024];
      sprintf(sz_content, "11 %.6f %.6f %.6f %.6f %d %.4f\n", ctx, cty, w, h,
              int(track_result.track_id_), f_box.score);
      str_content += std::string(sz_content);
    } else {
      if (track_result.box_info_.object_type !=
          TDLObjectType::OBJECT_TYPE_PERSON) {
        assert(false);
      }
      ObjectBoxInfo p_box =
          person_boxes[track_result.obj_idx_ - face_boxes.size()];
      // printf("id:%d, p_box: %f, %f, %f, %f\n", track_result.obj_idx_,
      // p_box.x1, p_box.x2, p_box.y1, p_box.y2);

      float ctx = (p_box.x1 + p_box.x2) / 2;
      float cty = (p_box.y1 + p_box.y2) / 2;
      float w = p_box.x2 - p_box.x1;
      float h = p_box.y2 - p_box.y1;
      ctx = ctx / img_width;
      cty = cty / img_height;
      w = w / img_width;
      h = h / img_height;
      char sz_content[1024];
      sprintf(sz_content, "11 %.6f %.6f %.6f %.6f %d %.4f\n", ctx, cty, w, h,
              int(track_result.track_id_), p_box.score);
      str_content += std::string(sz_content);
    }
  }
  return str_content;
}

std::string pack_features_attributes(
    const std::map<uint64_t, std::vector<float>> &face_features,
    std::vector<ObjectSnapshotInfo> &face_snapshots,
    std::vector<std::map<TDLObjectAttributeType, float>> &face_attributes) {
  std::stringstream str_content;

  for (int i = 0; i < face_snapshots.size(); i++) {
    std::vector<float> fea = face_features.at(face_snapshots[i].track_id);
    str_content << face_snapshots[i].track_id << "#"
                << face_snapshots[i].snapshot_frame_id << "#";
    for (size_t i = 0; i < fea.size(); i++) {
      str_content << fea[i] << " ";
    }

    str_content << "#";

    auto face_attribute = face_attributes[i];
    int pred_gender =
        face_attribute[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER] >
                0.5
            ? 1
            : 0;
    int pred_age =
        (int)(face_attribute
                  [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE] *
              100);
    int pred_glass =
        face_attribute[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES] >
                0.5
            ? 1
            : 0;
    int pred_emotion = (int)
        face_attribute[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_EMOTION];

    str_content << "pred_gender:" << pred_gender << "|pred_age:" << pred_age
                << "|pred_glass:" << pred_glass
                << "|pred_emotion:" << pred_emotion << "\n";
  }

  return str_content.str();
}

std::string capobj_to_str(std::vector<ObjectSnapshotInfo> &face_snapshots,
                          float w, float h, int lb = 11) {
  std::stringstream ss;

  for (auto &face_snapshot : face_snapshots) {
    ObjectBoxLandmarkInfo ori_face_meta =
        face_snapshot.other_info.at("ori_face_meta")
            .get<ObjectBoxLandmarkInfo>();

    float ctx = (ori_face_meta.x1 + ori_face_meta.x2) / 2.0 / w;
    float cty = (ori_face_meta.y1 + ori_face_meta.y2) / 2.0 / h;
    float ww = (ori_face_meta.x2 - ori_face_meta.x1) / w;
    float hh = (ori_face_meta.y2 - ori_face_meta.y1) / h;

    ss << face_snapshot.snapshot_frame_id << "," << lb << "," << ctx << ","
       << cty << "," << ww << "," << hh << "," << face_snapshot.track_id << ","
       << ori_face_meta.score << ";";

    for (int i = 0; i < ori_face_meta.landmarks_x.size(); i++) {
      ss << ori_face_meta.landmarks_x[i] << "," << ori_face_meta.landmarks_y[i]
         << ",";
    }
    ss << "0"
       << "\n";
  }

  return ss.str();
}

void exportFaceSnapshots(const std::string &dst_dir,
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

    ret = ImageFactory::writeImage(sz_frame_name, face_snapshot.object_image);
    if (ret != 0) {
      std::cout << "write image " << sz_frame_name << " failed" << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <task_name> <config_file> <output_folder_path>\n",
           argv[0]);
    return -1;
  }

  const std::string task_name = argv[1];
  const std::string config_file = argv[2];
  const std::string output_folder_path = argv[3];

  std::shared_ptr<AppTask> app_task =
      AppFactory::createAppTask(task_name, config_file);

  int32_t ret = app_task->init();
  if (ret != 0) {
    std::cout << "app_task init failed" << std::endl;
    return -1;
  }
  std::vector<std::string> channel_names = app_task->getChannelNames();
  std::map<std::string, std::string> channel_output_dirs;
  std::map<std::string, std::ofstream> channel_cap_fp;
  char sz_frame_name[1024];
  char sz_cap_result[1024];
  for (auto &channel_name : channel_names) {
    std::string output_dir = output_folder_path + "/" + channel_name;
    make_dir(output_dir.c_str());
    sprintf(sz_cap_result, "%s/cap_result.log", output_dir.c_str());
    std::ofstream cap_fp(sz_cap_result);
    channel_cap_fp[channel_name] = std::move(cap_fp);

    channel_output_dirs[channel_name] = output_dir;
  }

  while (true) {
    int processing_channel_num = app_task->getProcessingChannelNum();
    if (processing_channel_num == 0) {
      std::cout << "no processing channel, break" << std::endl;
      for (const auto &channel_name : channel_names) {
        channel_cap_fp[channel_name].close();
      }
      break;
    }
    for (const auto &channel_name : channel_names) {
      Packet result;
      std::cout << "to get result from channel:" << channel_name << std::endl;

      // getchar();
      int ret = app_task->getResult(channel_name, result);

      std::string output_dir = channel_output_dirs[channel_name];
      std::string track_content;
      std::string cap_content;
      uint64_t frame_id;

      if (task_name == "face_capture") {
        std::shared_ptr<FaceCaptureResult> cap_result =
            result.get<std::shared_ptr<FaceCaptureResult>>();
        if (cap_result == nullptr) {
          std::cout << "cap_result is nullptr" << std::endl;
          continue;
        }

        frame_id = cap_result->frame_id;

        // track_content = pack_track_results(cap_result->track_results,
        //                                    cap_result->frame_width,
        //                                    cap_result->frame_height);
        track_content =
            pack_det_results(cap_result->track_results, cap_result->face_boxes,
                             cap_result->person_boxes, cap_result->frame_width,
                             cap_result->frame_height);

        cap_content =
            capobj_to_str(cap_result->face_snapshots, cap_result->frame_width,
                          cap_result->frame_height);
        exportFaceSnapshots(output_dir, cap_result->face_snapshots);

      } else if (task_name == "face_pet_capture") {
        std::shared_ptr<FacePetCaptureResult> cap_result =
            result.get<std::shared_ptr<FacePetCaptureResult>>();
        if (cap_result == nullptr) {
          std::cout << "cap_result is nullptr" << std::endl;
          continue;
        }

        frame_id = cap_result->frame_id;

        // track_content = pack_track_results(cap_result->track_results,
        //                                    cap_result->frame_width,
        //                                    cap_result->frame_height);
        track_content =
            pack_det_results(cap_result->track_results, cap_result->face_boxes,
                             cap_result->person_boxes, cap_result->frame_width,
                             cap_result->frame_height);

        cap_content =
            capobj_to_str(cap_result->face_snapshots, cap_result->frame_width,
                          cap_result->frame_height);

        if (cap_result->face_features.size() > 0) {
          std::string fea_content = pack_features_attributes(
              cap_result->face_features, cap_result->face_snapshots,
              cap_result->face_attributes);
          char sz_fea_name[1024];
          sprintf(sz_fea_name, "%s/feature_%08lu.txt", output_dir.c_str(),
                  frame_id);
          std::ofstream outfea(sz_fea_name);
          outfea << fea_content;
          outfea.close();
        }
        exportFaceSnapshots(output_dir, cap_result->face_snapshots);
      }

      channel_cap_fp[channel_name] << cap_content;

      if (ret == 0) {
        std::cout << "cap_result->frame_id:" << frame_id << std::endl;
        sprintf(sz_frame_name, "%s/%08lu.txt", output_dir.c_str(), frame_id);

        std::ofstream outf(sz_frame_name);
        outf << track_content;
        outf.close();
      } else {
        std::cout << "get result failed" << std::endl;
        app_task->removeChannel(channel_name);
        continue;
      }

      // std::cout << "export face snapshots done" << std::endl;
    }
  }
  app_task->release();
}
