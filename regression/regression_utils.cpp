#include "regression_utils.hpp"
#include <regex>
#include <unordered_map>

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

std::string gen_platform() {
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
std::string extractModelIdFromName(const std::string& model_name) {
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
    const std::string& model_name) {
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

}  // namespace unitest
}  // namespace cvitdl
