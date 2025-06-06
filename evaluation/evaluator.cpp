#include "evaluator.hpp"
#include <fstream>
#include <sstream>
Evaluator::Evaluator() {}
Evaluator::~Evaluator() {}

int32_t Evaluator::writeResult(const std::string &result_file,
                               const std::string &str_content) {
  std::ofstream ofs(result_file);
  if (!ofs.is_open()) {
    printf("Failed to open file: %s\n", result_file.c_str());
    return -1;
  }
  ofs << str_content;
  ofs.close();
  printf("writeResult: %s\n %s\n", result_file.c_str(), str_content.c_str());
  return 0;
}

std::string Evaluator::packOutput(
    std::shared_ptr<ModelOutputInfo> model_output) {
  std::string str_content;
  if (model_output->getType() == ModelOutputType::OBJECT_DETECTION) {
    auto object_detection_info =
        std::static_pointer_cast<ModelBoxInfo>(model_output);
    for (auto &box_info : object_detection_info->bboxes) {
      str_content += std::to_string(box_info.class_id) + " " +
                     std::to_string(box_info.score) + " " +
                     std::to_string(box_info.x1) + " " +
                     std::to_string(box_info.y1) + " " +
                     std::to_string(box_info.x2) + " " +
                     std::to_string(box_info.y2) + "\n";
    }
  } else if (model_output->getType() ==
             ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    auto object_detection_with_landmarks_info =
        std::static_pointer_cast<ModelBoxLandmarkInfo>(model_output);
    for (auto &box_landmark_info :
         object_detection_with_landmarks_info->box_landmarks) {
      str_content += std::to_string(box_landmark_info.class_id) + " " +
                     std::to_string(box_landmark_info.score) + " " +
                     std::to_string(box_landmark_info.x1) + " " +
                     std::to_string(box_landmark_info.y1) + " " +
                     std::to_string(box_landmark_info.x2) + " " +
                     std::to_string(box_landmark_info.y2) + "\n";
    }
  }
  return str_content;
}

std::string Evaluator::packOutput(
    const std::vector<TrackerInfo> &track_results) {
  std::string str_content;
  for (auto &track_result : track_results) {
    printf("track_result: %d,box:[%.2f,%.2f,%.2f,%.2f],score:%.2f\n",
           int(track_result.track_id_), track_result.box_info_.x1,
           track_result.box_info_.y1, track_result.box_info_.x2,
           track_result.box_info_.y2, track_result.box_info_.score);
    float ctx = (track_result.box_info_.x1 + track_result.box_info_.x2) / 2;
    float cty = (track_result.box_info_.y1 + track_result.box_info_.y2) / 2;
    float w = track_result.box_info_.x2 - track_result.box_info_.x1;
    float h = track_result.box_info_.y2 - track_result.box_info_.y1;
    ctx = ctx / img_width_;
    cty = cty / img_height_;
    w = w / img_width_;
    h = h / img_height_;
    printf("ctx:%.2f,cty:%.2f,w:%.2f,h:%.2f,imgw:%d,imgh:%d\n", ctx, cty, w, h,
           img_width_, img_height_);
    char sz_content[1024];
    sprintf(sz_content, "%d %.2f %.2f %.2f %.2f %d %.2f\n",
            int(track_result.box_info_.object_type), ctx, cty, w, h,
            int(track_result.track_id_), track_result.box_info_.score);
    str_content += std::string(sz_content);
  }
  return str_content;
}
