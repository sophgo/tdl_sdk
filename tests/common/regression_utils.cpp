#include "regression_utils.hpp"
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

std::string gen_model_suffix() {
#if defined(__CV181X__) || defined(__CMODEL_CV181X__)
  return std::string("_cv181x.cvimodel");

#elif defined(__CV184X__)
  return std::string("_cv184x.bmodel");

#elif defined(__CV186X__)
  return std::string("_cv186x.bmodel");

#elif defined(__BM1684X__)
  return std::string("_bm1684x.bmodel");

#elif defined(__BM168X__)
  return std::string("_bm1688.bmodel");

#else
  printf("Unrecognized platform !\n");
  return std::string("");

#endif
}

std::string get_platform_str() {
#if defined(__CV181X__) || defined(__CMODEL_CV181X__)
  return std::string("CV181X");

#elif defined(__CV184X__)
  return std::string("CV184X");

#elif defined(__CV186X__)
  return std::string("CV186X");

#elif defined(__BM1684X__)
  return std::string("BM1684X");

#elif defined(__BM168X__)
  return std::string("BM1688");

#else
  printf("Unrecognized platform !\n");
  return std::string("");

#endif
}

std::vector<std::string> get_platform_list() {
  std::vector<std::string> platform_list;
  platform_list.push_back(std::string("CV181X"));
  platform_list.push_back(std::string("CV182X"));
  platform_list.push_back(std::string("CV183X"));
  platform_list.push_back(std::string("CV184X"));
  platform_list.push_back(std::string("CV186X"));
  platform_list.push_back(std::string("BM1684X"));
  platform_list.push_back(std::string("BM1688"));
  platform_list.push_back(std::string("BM1684"));
  return platform_list;
}

std::string gen_model_dir() {
#if defined(__CV181X__) || defined(__CMODEL_CV181X__)
  return std::string("cv181x");

#elif defined(__CV184X__)
  return std::string("cv184x");

#elif defined(__CV186X__)
  return std::string("cv186x");

#elif defined(__BM1684X__)
  return std::string("bm1684x");

#elif defined(__BM168X__)
  return std::string("bm1688");

#else
  printf("Unrecognized platform !\n");
  return std::string("");

#endif
}

// 从模型名称提取模型ID的函数（参考generate_regression_info.cpp的逻辑）
std::string extractModelIdFromName(const std::string &model_name) {
  // 特殊模型ID映射（从generate_regression_info.cpp复制）
  static const std::unordered_map<std::string, std::string>
      special_model_id_map = {
          {"yolov8n_det_ir_person_384_640_INT8", "YOLOV8N_DET_MONITOR_PERSON"},
          {"yolov8n_det_ir_person_mbv2_384_640_INT8",
           "YOLOV8N_DET_MONITOR_PERSON"},
          {"yolov8n_det_overlook_person_256_448_INT8",
           "YOLOV8N_DET_MONITOR_PERSON"},
          {"yolov8n_det_hand_mv3_384_640_INT8", "YOLOV8N_DET_HAND"},
          {"yolov8n_det_person_vehicle_mv2_035_384_640_INT8",
           "YOLOV8N_DET_PERSON_VEHICLE"},
          {"yolov8n_det_pet_person_035_384_640_INT8", "YOLOV8N_DET_PET_PERSON"},
          {"yolov8n_det_bicycle_motor_ebicycle_mbv2_384_640_INT8",
           "YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE"},
          {"cls_4_attribute_face_112_112_INT8", "CLS_ATTRIBUTE_FACE"},
          {"cls_sound_nihaoshiyun_126_40_INT8",
           "CLS_SOUND_COMMAND_NIHAOSHIYUN"},
          {"cls_sound_xiaoaixiaoai_126_40_INT8",
           "CLS_SOUND_COMMAND_XIAOAIXIAOAI"},
          {"yolov8n_seg_coco80_640_640_INT8", "YOLOV8_SEG_COCO80"}};

  auto it = special_model_id_map.find(model_name);
  if (it != special_model_id_map.end()) {
    return it->second;
  }

  // 通用COCO80检测模型处理
  if (model_name.substr(0, 4) == "yolo" &&
      model_name.find("det_coco80") != std::string::npos) {
    size_t pos = model_name.find('_');
    if (model_name.substr(0, 5) == "yolox" ||
        model_name.substr(0, 6) == "yolov7") {
      std::string result = model_name.substr(0, pos) + "_det_coco80";
      std::transform(result.begin(), result.end(), result.begin(), ::toupper);
      return result;
    } else {
      std::string result = model_name.substr(0, pos - 1) + "_det_coco80";
      std::transform(result.begin(), result.end(), result.begin(), ::toupper);
      return result;
    }
  }

  // 默认处理：提取到第一个数字前的部分并转大写
  std::regex re("_(\\d)");
  std::smatch match;
  if (std::regex_search(model_name, match, re)) {
    size_t pos = match.position(0);
    std::string result = model_name.substr(0, pos);
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
  }

  std::string result = model_name;
  std::transform(result.begin(), result.end(), result.begin(), ::toupper);
  return result;
}

std::map<std::string, float> getCustomRegressionConfig(
    const std::string &model_name) {
  std::map<std::string, float> custom_config;
  if (model_name.find("keypoint_") == 0 ||
      model_name.find("lstr_det_lane_") == 0) {
    custom_config["model_score_threshold"] = 0.5;
    custom_config["reg_score_diff_threshold"] = 0.1;
    custom_config["reg_position_diff_threshold"] = 0.1;
  } else if (model_name.find("cls_") == 0 || model_name.find("feature_") == 0) {
    custom_config["reg_score_diff_threshold"] = 0.1;
  } else if (model_name.find("det_") != std::string::npos ||
             model_name.find("tracking_") == 0) {
    custom_config["model_score_threshold"] = 0.5;
    custom_config["reg_nms_threshold"] = 0.5;
    custom_config["reg_score_diff_threshold"] = 0.1;
  } else if (model_name.find("topformer_seg_") == 0) {
    custom_config["reg_mask_threshold"] = 0.1;
  } else if (model_name.find("seg_") != std::string::npos) {
    custom_config["model_score_threshold"] = 0.5;
    custom_config["reg_nms_threshold"] = 0.5;
    custom_config["reg_score_diff_threshold"] = 0.1;
    custom_config["reg_mask_threshold"] = 0.1;
  } else {
    LOGI("Do not need custom regression config.");
  }
  return custom_config;
}

std::vector<std::string> getFileList(const std::string &dir_path,
                                     const std::string &extension) {
  std::vector<std::string> file_list;
  LOGIP("dir_path: %s, extension: %s", dir_path.c_str(), extension.c_str());
  try {
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
      return file_list;  // 返回空列表如果目录不存在或不是目录
    }

    for (const auto &entry : fs::directory_iterator(dir_path)) {
      if (fs::is_regular_file(entry.status())) {
        std::string file_path = entry.path().string();
        std::string file_extension = entry.path().extension().string();
        LOGI("file_path: %s, file_extension: %s", file_path.c_str(),
             file_extension.c_str());
        // 检查文件扩展名
        std::string relative_path = file_path.substr(dir_path.size() + 1);
        LOGI("relative_path: %s", relative_path.c_str());
        if (extension.empty() || file_extension == extension) {
          file_list.push_back(relative_path);
        }
      }
    }
  } catch (const fs::filesystem_error &ex) {
    // 处理文件系统错误，返回空列表
    return file_list;
  }
  std::sort(file_list.begin(), file_list.end());
  return file_list;
}

float iou(const std::vector<float> &gt_object,
          const std::vector<float> &pred_object) {
  // float iout = 0.0f;
  float gt_x1 = gt_object[0];
  float gt_y1 = gt_object[1];
  float gt_x2 = gt_object[2];
  float gt_y2 = gt_object[3];
  float pred_x1 = pred_object[0];
  float pred_y1 = pred_object[1];
  float pred_x2 = pred_object[2];
  float pred_y2 = pred_object[3];
  // invalid boxes produce non-positive areas, treat as no-overlap
  if (gt_x2 <= gt_x1 || gt_y2 <= gt_y1 || pred_x2 <= pred_x1 ||
      pred_y2 <= pred_y1) {
    return 0.0f;
  }
  float area1 = (gt_x2 - gt_x1) * (gt_y2 - gt_y1);
  float area2 = (pred_x2 - pred_x1) * (pred_y2 - pred_y1);
  float inter_x1 = std::max(gt_x1, pred_x1);
  float inter_y1 = std::max(gt_y1, pred_y1);
  float inter_x2 = std::min(gt_x2, pred_x2);
  float inter_y2 = std::min(gt_y2, pred_y2);
  if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) {
    return 0.0f;
  }
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  return area_inter / (area1 + area2 - area_inter);
}

// object info:[x1,y1,x2,y2,score,class_id]
bool matchObjects(const std::vector<std::vector<float>> &gt_objects,
                  const std::vector<std::vector<float>> &pred_objects,
                  const float iout_thresh, const float score_diff_thresh) {
  std::vector<int> gt_matched(gt_objects.size(), 0);
  std::vector<int> pred_matched(pred_objects.size(), 0);

  bool is_matched = true;
  for (size_t i = 0; i < gt_objects.size(); i++) {
    auto &gt_object = gt_objects[i];
    std::vector<float> matched_dets;
    float max_iout = -1.0f;
    int pred_matched_idx = -1;
    for (size_t j = 0; j < pred_objects.size(); j++) {
      const auto &pred_object = pred_objects[j];
      float iout = iou(gt_object, pred_object);
      if (iout > max_iout) {
        max_iout = iout;
        pred_matched_idx = j;
      }
    }
    matched_dets = pred_objects[pred_matched_idx];
    float score_diff = std::abs(matched_dets[4] - gt_object[4]);
    if (gt_object[5] == pred_objects[pred_matched_idx][5] &&
        max_iout > iout_thresh && score_diff < score_diff_thresh) {
      gt_matched[i] = 1;
      pred_matched[pred_matched_idx] = 1;
    } else {
      std::cout << "score diff: " << score_diff << ",gtbox:[" << gt_object[0]
                << "," << gt_object[1] << "," << gt_object[2] << ","
                << gt_object[3] << "]"
                << ",score:" << gt_object[4] << ",class_id:" << gt_object[5]
                << ",predbox:[" << matched_dets[0] << "," << matched_dets[1]
                << "," << matched_dets[2] << "," << matched_dets[3] << "]"
                << ",pred_score:" << matched_dets[4]
                << ",pred_class_id:" << matched_dets[5] << " iou: " << max_iout
                << std::endl;
      is_matched = false;
    }
  }

  for (size_t i = 0; i < gt_objects.size(); i++) {
    if (gt_matched[i] == 0) {
      float area = (gt_objects[i][2] - gt_objects[i][0]) *
                   (gt_objects[i][3] - gt_objects[i][1]);
      if (area == 0) {
        std::cout << "warning, box area is 0 ,gtbox:[" << gt_objects[i][0]
                  << "," << gt_objects[i][1] << "," << gt_objects[i][2] << ","
                  << gt_objects[i][3] << "]"
                  << ",score:" << gt_objects[i][4]
                  << ",class_id:" << gt_objects[i][5] << std::endl;
      } else if (gt_objects[i][4] <= 0.5f) {
        std::cout << "warning, low conf box not matched ,gtbox:["
                  << gt_objects[i][0] << "," << gt_objects[i][1] << ","
                  << gt_objects[i][2] << "," << gt_objects[i][3] << "]"
                  << ",score:" << gt_objects[i][4]
                  << ",class_id:" << gt_objects[i][5] << std::endl;
      } else {
        std::cout << "gt box not matched ,gtbox:[" << gt_objects[i][0] << ","
                  << gt_objects[i][1] << "," << gt_objects[i][2] << ","
                  << gt_objects[i][3] << "]"
                  << ",score:" << gt_objects[i][4]
                  << ",class_id:" << gt_objects[i][5] << std::endl;
        is_matched = false;
      }
    }
  }
  for (size_t i = 0; i < pred_objects.size(); i++) {
    float area = (pred_objects[i][2] - pred_objects[i][0]) *
                 (pred_objects[i][3] - pred_objects[i][1]);
    if (area == 0) {
      std::cout << "warning, box area is 0 ,predbox:[" << pred_objects[i][0]
                << "," << pred_objects[i][1] << "," << pred_objects[i][2] << ","
                << pred_objects[i][3] << "]"
                << ",score:" << pred_objects[i][4]
                << ",class_id:" << pred_objects[i][5] << std::endl;
    } else if (pred_matched[i] == 0) {
      std::cout << "pred box not matched ,predbox:[" << pred_objects[i][0]
                << "," << pred_objects[i][1] << "," << pred_objects[i][2] << ","
                << pred_objects[i][3] << "],score:" << pred_objects[i][4]
                << ",class_id:" << pred_objects[i][5] << std::endl;
      is_matched = false;
    }
  }
  return is_matched;
}

bool matchScore(const std::vector<float> &gt_info,
                const std::vector<float> &pred_info,
                const float score_diff_thresh) {
  if (gt_info.size() != pred_info.size()) {
    return false;
  }

  for (size_t i = 0; i < gt_info.size(); ++i) {
    if (std::abs(gt_info[i] - pred_info[i]) > score_diff_thresh) {
      return false;
    }
  }
  return true;
};

bool matchKeypoints(const std::vector<float> &gt_keypoints_x,
                    const std::vector<float> &gt_keypoints_y,
                    const std::vector<float> &gt_keypoints_score,
                    const std::vector<float> &pred_keypoints_x,
                    const std::vector<float> &pred_keypoints_y,
                    const std::vector<float> &pred_keypoints_score,
                    const float position_thresh,
                    const float score_diff_thresh) {
  if (gt_keypoints_x.size() != pred_keypoints_x.size() ||
      gt_keypoints_y.size() != pred_keypoints_y.size() ||
      gt_keypoints_x.size() != gt_keypoints_y.size() ||
      gt_keypoints_score.size() != pred_keypoints_score.size()) {
    return false;
  }

  for (size_t i = 0; i < gt_keypoints_score.size(); i++) {
    float score_diff =
        std::abs(gt_keypoints_score[i] - pred_keypoints_score[i]);

    if (score_diff > score_diff_thresh) {
      return false;
    }
  }

  float total_distance = 0.0f;
  int num_keypoints = gt_keypoints_x.size();
  float score_thresh_for_distance = 0.5f;
  std::vector<float> keypoints_index_for_distance;
  if (gt_keypoints_score.size() != gt_keypoints_x.size()) {
    for (int i = 0; i < num_keypoints; i++) {
      keypoints_index_for_distance.push_back(i);
    }
  } else {
    for (int i = 0; i < num_keypoints; i++) {
      if (gt_keypoints_score[i] > score_thresh_for_distance &&
          pred_keypoints_score[i] > score_thresh_for_distance) {
        keypoints_index_for_distance.push_back(i);
      }
    }
  }

  for (auto &i : keypoints_index_for_distance) {
    float dx = gt_keypoints_x[i] - pred_keypoints_x[i];
    float dy = gt_keypoints_y[i] - pred_keypoints_y[i];
    float distance = std::sqrt(dx * dx + dy * dy);
    total_distance += distance;
  }

  float avg_distance = total_distance / keypoints_index_for_distance.size();
  return avg_distance <= position_thresh;
};

bool matchSegmentation(const cv::Mat &mat1, const cv::Mat &mat2,
                       float mask_thresh) {
  cv::Mat diff = (mat1 != mat2);  // 元素不相等的位置为255，相等为0
  int diffCount = cv::countNonZero(diff);

  int height = mat1.rows;
  int width = mat1.cols;

  float diff_ratio = float(diffCount) / (height * width);

  bool is_matched = diff_ratio < mask_thresh;
  if (!is_matched) {
    LOGIP("diff_ratio: %f, mask_thresh: %f", diff_ratio, mask_thresh);
  }
  return is_matched;
}

}  // namespace unitest
}  // namespace cvitdl
