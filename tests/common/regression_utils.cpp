#include "regression_utils.hpp"
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include "speech_recognition/zipformer_encoder.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

std::string gen_model_suffix() {
#if defined(__CV180X__)
  return std::string("_cv180x.cvimodel");

#elif defined(__CV181X__) || defined(__CMODEL_CV181X__)
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
#if defined(__CV180X__)
  return std::string("CV180X");

#elif defined(__CV181X__) || defined(__CMODEL_CV181X__)
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
  platform_list.push_back(std::string("CV180X"));
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
#if defined(__CV180X__)
  return std::string("cv180x");

#elif defined(__CV181X__) || defined(__CMODEL_CV181X__)
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

std::string gen_model_tool_dir() {
  std::string parent_dir;
  std::string so_dir = CommonUtils::getLibraryDir();
  std::string exe_dir = CommonUtils::getExecutableDir();
  if (so_dir != exe_dir) {
    parent_dir = CommonUtils::getParentDir(so_dir);
  } else {
    parent_dir = CommonUtils::getParentDir(CommonUtils::getParentDir(exe_dir));
  }
  // 使用规范化路径而不是相对路径拼接
  std::string install_dir = CommonUtils::getParentDir(parent_dir);
  std::string tdl_dir = CommonUtils::getParentDir(install_dir);
  std::string sdk_dir = CommonUtils::getParentDir(tdl_dir);

#if defined(__CV180X__) || defined(__CV181X__)
  return "/mnt/data/yzx/sdk_sync/v410/cviruntime/build_sdk/build_cviruntime/"
         "tool";
#elif defined(__CV184X__) || defined(__CV186X__)
  return sdk_dir + "/libsophon/install/libsophon-0.4.9/bin";
#elif defined(__BM168X__)
  return tdl_dir + "/dependency/BM1688/libsophon/bin";

#else
  LOGE("Unrecognized platform !\n");
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
    LOGIP("size not equal,return false");
    return false;
  }

  for (size_t i = 0; i < gt_keypoints_score.size(); i++) {
    float score_diff =
        std::abs(gt_keypoints_score[i] - pred_keypoints_score[i]);

    if (score_diff > score_diff_thresh) {
      LOGIP("score_diff:(%f-%f) > score_diff_thresh:%f,return false",
            gt_keypoints_score[i], pred_keypoints_score[i], score_diff_thresh);
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
  LOGIP("avg_distance:%f,position_thresh:%f", avg_distance, position_thresh);
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

// 通过rfind当前目录，确认Tpu信息文件的路径
bool confirm_path(std::string &tpu_usage_path) {
#if defined(__CV180X__) || defined(__CV181X__)
  fs::path tpu_dir("/proc/tpu");
  if (fs::exists(tpu_dir) && fs::is_directory(tpu_dir)) {
    for (auto &entry : fs::directory_iterator(tpu_dir)) {
      // filename 的作用是返回路径中的文件名部分
      auto name = entry.path().filename().string();
      if (name.rfind("usage_", 0) == 0) {
        tpu_usage_path = entry.path().string();
        return true;
      }
    }
  }
  if (tpu_usage_path.empty()) {
    fs::path p("/proc/tpu/usage");
    if (fs::exists(p) && fs::is_regular_file(p)) {
      tpu_usage_path = p.string();
      return true;
    }
  }
  return false;

#elif defined(__CV184X__)
  tpu_usage_path = "/tmp/bmcpu_app_usage";
  return true;

#elif defined(__CV186X__)
  std::string tool_path = gen_model_tool_dir();
  tpu_usage_path = tool_path + "/bm-smi -noloop --file=tmp.txt";
  return true;

#elif defined(__BM168X__)
  tpu_usage_path = "bm-smi -noloop --file=tmp.txt";
  return true;

#else
  LOGE("Unrecognized platform !");
  return false;

#endif
}
// 使能tpu的性能计算
void enable_tpu_usage(const std::string &tpu_usage_path) {
#if defined(__CV180X__) || defined(__CV181X__)
  std::ofstream en(tpu_usage_path);
  if (en) {
    en << "1\n";
    en.close();
  } else {
    LOGE("enable_tpu_usage: failed to open %s", tpu_usage_path.c_str());
  }
#endif
}

// 通过popen与shell命令获取tpu的使用率文本输出
bool read_tpu_usage(const std::string &tpu_usage_path,
                    std::vector<double> &tpu_samples, std::mutex &tpu_mu) {
  if (tpu_usage_path.empty()) return false;
  std::string output;
  std::string value_str;

#if !defined(__BM168X__) && !defined(__CV186X__)
  std::string cmd = "cat " + tpu_usage_path;
  FILE *fp = popen(cmd.c_str(), "r");
  if (!fp) {
    std::cerr << "[WARN] Failed to open TPU usage path: " << tpu_usage_path
              << std::endl;
    return false;
  }
  char buffer[256] = {0};
  while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
    output += buffer;
  }
  pclose(fp);
#if defined(__CV180X__) || defined(__CV181X__)
  std::size_t pos = output.find("usage=");
  if (pos == std::string::npos) {
    std::cerr << "[WARN] No 'usage=' found in output: " << output << std::endl;
    return false;
  }
  value_str = output.substr(pos + 6);
#else
  value_str = output;
#endif

#else

  system(tpu_usage_path.c_str());  // 使用popen得到的结果会乱码
  std::ifstream file("tmp.txt");
  if (file.is_open()) {
    std::string line;
    // 查找包含 TPU-Util 数据的行（特征匹配 "BM1688-SOC" 所在行）
    while (getline(file, line)) {
      if (line.find("BM1688-SOC") != std::string::npos &&
          line.find("TPU-Util") == std::string::npos) {
        // 从行尾查找百分号
        size_t percentPos = line.find_last_of('%');
        if (percentPos != std::string::npos) {
          // 向前提取数字部分
          size_t start = percentPos - 1;
          while (start > 0 &&
                 isdigit(static_cast<unsigned char>(line[start]))) {
            start--;
          }
          // 处理边界情况（确保从数字开始）
          if (!isdigit(static_cast<unsigned char>(line[start]))) {
            start++;
          }
          value_str = line.substr(start, percentPos - start);
          break;
        }
      }
    }
    file.close();
  }

  //删除文件
  if (remove("tmp.txt") != 0) {
    std::cerr << "Error: 无法删除文件 tmp.txt" << std::endl;
  }

#endif
  value_str.erase(std::remove_if(value_str.begin(), value_str.end(),
                                 [](unsigned char c) {
                                   return std::isspace(c) || c == '%';
                                 }),
                  value_str.end());
  double v = 0.0;
  try {
    v = std::stod(value_str);  // 转换为 double
  } catch (...) {
    std::cerr << "[WARN] Failed to parse usage value from: " << value_str
              << std::endl;
    return false;
  }
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(tpu_mu);
  if (v <= 0.0 || v > 100.0) return false;
  // std::cout << "tpu usage: " << v << "%" << std::endl;
  tpu_samples.push_back(v);
  return true;
}

// 获取cpu load，获得pid、读取文件、解析jiffies、计算load 都在这个函数里面实现
bool read_cpu_line(std::vector<double> &cpu_samples, std::mutex &cpu_mu) {
  // 获取进程号
  pid_t pid = getpid();
  pid_t tid = get_tid_by_name("inference_load");
  if (tid == -1) return false;
  std::string stat_path =
      "/proc/" + std::to_string(pid) + "/task/" + std::to_string(tid) + "/stat";
  std::ifstream pf(stat_path);
  if (!pf) return false;
  std::string stat_line;
  std::getline(pf, stat_line);
  pf.close();
  // find last ')' to skip comm (which may contain spaces)
  auto pos = stat_line.find_last_of(')');
  if (pos == std::string::npos || pos + 2 >= stat_line.size()) return false;
  std::istringstream ss(stat_line.substr(pos + 2));
  // after ')' the tokens start at field 3; we need field 14(utime) and
  // 15(stime) skip 11 tokens to reach utime
  std::string tmp;
  for (int i = 0; i < 11; ++i) {
    if (!(ss >> tmp)) return false;
  }
  uint64_t utime = 0, stime = 0;
  if (!(ss >> utime >> stime)) return false;
  uint64_t proc_time = utime + stime;

  // 获取总共的cpu时钟中断数量
  std::ifstream sf("/proc/stat");
  if (!sf) return false;
  std::string cpu_line;
  std::getline(sf, cpu_line);
  sf.close();
  if (cpu_line.rfind("cpu ", 0) != 0) return false;
  std::istringstream cs(cpu_line);
  std::string cpu_tag;
  cs >> cpu_tag;  // "cpu"
  uint64_t total = 0, val = 0;
  while (cs >> val) total += val;

  // persist previous values
  static uint64_t prev_proc_time = 0;
  static uint64_t prev_total = 0;
  static bool has_prev = false;
  if (!has_prev) {
    prev_proc_time = proc_time;
    prev_total = total;
    has_prev = true;
    return true;  // need two samples to compute delta
  }
  uint64_t proc_delta =
      (proc_time > prev_proc_time) ? (proc_time - prev_proc_time) : 0;
  uint64_t total_delta = (total > prev_total) ? (total - prev_total) : 0;
  prev_proc_time = proc_time;
  prev_total = total;
  if (total_delta == 0) return false;

  double usage = 100.0 * (double(proc_delta) / double(total_delta));
  if (!(usage > 0.0 && usage <= 100.0)) return false;
  // 不可以用const，因为线程需要持有锁？
  std::lock_guard<std::mutex> lk(cpu_mu);
  // LOGIP("CPU usage: %.2f%%", usage);
  cpu_samples.push_back(usage);
  return true;
}

pid_t get_tid_by_name(const std::string &thread_name) {
  const std::string task_dir = "/proc/self/task";
  DIR *dir = opendir(task_dir.c_str());
  if (!dir) {
    perror("opendir failed");
    return -1;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    // 跳过 . 和 ..
    if (entry->d_name[0] == '.') continue;

    std::string tid_str = entry->d_name;

    std::string comm_path = task_dir + "/" + tid_str + "/comm";

    std::ifstream comm_file(comm_path);
    if (!comm_file.is_open()) continue;

    std::string name;
    std::getline(comm_file, name);
    // std::cout << name << std::endl;
    comm_file.close();

    // 去掉可能存在的换行符
    if (!name.empty() && name.back() == '\n') name.pop_back();

    if (name == thread_name) {
      closedir(dir);
      // std::cout << thread_name << "\" found." << std::endl;
      return std::stoi(tid_str);
    }
  }

  closedir(dir);
  // std::cerr << "Thread name \"" << thread_name << "\" not found." <<
  // std::endl;
  return -1;
}

// 输出 CPU 与 TPU 的总体统计
void load_show(const std::vector<double> &cpu_samples,
               const std::vector<double> &tpu_samples) {
  if (!cpu_samples.empty()) {
    double avg = std::accumulate(cpu_samples.begin(), cpu_samples.end(), 0.0) /
                 cpu_samples.size();
    double maxv = *std::max_element(cpu_samples.begin(), cpu_samples.end());
    LOGIP("CPU usage: avg=%.2f%%  max=%.2f%%", avg, maxv);
  } else {
    LOGIP("CPU usage: no samples collected");
  }
  if (!tpu_samples.empty()) {
    double avg = std::accumulate(tpu_samples.begin(), tpu_samples.end(), 0.0) /
                 tpu_samples.size();
    double maxv = *std::max_element(tpu_samples.begin(), tpu_samples.end());
    LOGIP("TPU usage: avg=%.2f%%  max=%.2f%%\n", avg, maxv);
  } else {
    LOGIP("TPU usage: no samples collected");
  }
}

// 展示推理时间消耗
bool time_consume_show(
    const std::unordered_map<std::string, std::vector<double>> &img_durations) {
  int image_number = 0;
  double avg_duration_sum = 0.0;
  for (auto &kv : img_durations) {
    const auto &name = kv.first;
    const auto &vec = kv.second;
    if (vec.empty()) continue;
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double avg = sum / vec.size();
    double minv = *std::min_element(vec.begin(), vec.end());
    double maxv = *std::max_element(vec.begin(), vec.end());
    ++image_number;
    avg_duration_sum += avg;
    LOGIP(
        "performance:image=%s samples=%zu avg=%.3f ms min=%.3f ms max=%.3f ms",
        name.c_str(), vec.size(), avg, minv, maxv);
  }
  if (image_number > 0) {
    LOGIP("ocr Average Performance duration: %.2f ms",
          avg_duration_sum / image_number);
    return true;
  } else {
    LOGIP("ocr Average Performance duration: N/A (no valid samples)");
    return false;
  }
}

std::string fileSizeInMB(const std::string &path, bool useMiB) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    throw std::runtime_error("Failed to stat file: " + path);
  }
  // st.st_size 是字节数（通常为 off_t，可能是 64 位）
  const double base = useMiB ? (1024.0 * 1024.0) : 1000000.0;
  double mb = static_cast<double>(st.st_size) / base;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << mb;

  std::cout << "file size: " << oss.str() << " MB" << std::endl;
  return oss.str();
}

int parse_cmd_result(std::string &cmd, const std::regex &re,
                     std::string &result) {
  // 打开子进程并读取其输出
  std::array<char, 512> buffer{};
  std::string info;
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::cerr << "popen() failed!" << std::endl;
    return -1;
  }
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    info += buffer.data();
  }
  int ret = pclose(pipe);
  if (ret != 0) {
    return -1;
  }

  std::smatch match;
  if (std::regex_search(info, match, re)) {
    result = match[1].str();
  } else {
    LOGE("regex_search failed! info: %s", info.c_str());
    return -1;
  }
  return 0;
}

int get_model_info(const std::string &model_path, float &infer_time_ms) {
  fileSizeInMB(model_path);
  // 构造命令
  std::string tool_path = gen_model_tool_dir();
  std::string parent_dir;
  std::string cmd;
  std::string result;
  std::regex re;

  LOGIP("get model memory from %s", tool_path.c_str());

#if defined(__CV180X__) || defined(__CV181X__)

  cmd = tool_path + "/cvimodel_tool -a dump -i " + model_path + " 2>&1";

  // 使用正则匹配内存信息
  re.assign(R"(CviModel\s+Need\s+ION\s+Memory\s+Size:\s*\(([\d\.]+)\s*MB\))");

  if (parse_cmd_result(cmd, re, result) != 0) {
    std::cerr << "memory requirements not found" << std::endl;
    return -1;
  } else {
    std::cout << "Model memory requirements: " << std::stof(result) << " MB"
              << std::endl;
  }

  cmd = tool_path + "/model_runner --model " + model_path +
        " --enable-timer 2>&1";
  // 使用正则匹配推理时间
  re.assign(R"(1\s+runs\s+take\s+([+-]?\d+(?:.\d+)?)(?:\s*ms)?)");
  if (parse_cmd_result(cmd, re, result) != 0) {
    std::cerr << "infer time not found" << std::endl;
    return -1;
  } else {
    infer_time_ms = std::stof(result);
    std::cout << "Model infer time: " << infer_time_ms << " ms" << std::endl;
  }
#else

  cmd = "cd " + tool_path + " && " + " ./model_tool --info " + model_path +
        " 2>&1";

  re.assign(R"(device\s+mem\s+size:\s*(\d+)\s*\()");

  if (parse_cmd_result(cmd, re, result) != 0) {
    std::cerr << "memory requirements not found" << std::endl;
    return -1;
  } else {
    double mem_mb = std::stod(result);
    mem_mb /= (1024.0 * 1024.0);
    std::cout << "Model memory requirements: " << mem_mb << " MB" << std::endl;
  }

  cmd = "export SHOW_TPU_USAGE=1 && cd " + tool_path + " && " +
        " ./bmrt_test --bmodel " + model_path + " 2>&1";

  re.assign(R"(INFO:calculate\s+time\(s\):\s+([\d]+\.[\d]+))");
  if (parse_cmd_result(cmd, re, result) != 0) {
    std::cerr << "infer time not found" << std::endl;
    return -1;
  } else {
    infer_time_ms = std::stof(result) * 1000.0;
    std::cout << "Model infer time: " << infer_time_ms << " ms" << std::endl;
  }
#endif

  return 0;
}

// tpu采样对象，使用指令循环。

void *run_tdl_thread_tpu(void *args) {
  // args 可以是 nullptr 或者传入 model_id 等参数
  // 注意：这里不能访问非静态成员（如 m_json_object），因为没有 this
  RUN_TDL_THREAD_TPU_ARG_S *tpu_args =
      static_cast<RUN_TDL_THREAD_TPU_ARG_S *>(args);
  int count = tpu_args->count;
  std::string tool_path = gen_model_tool_dir();
  std::string cmd;

  LOGIP("tpu run count %d", count);

#if defined(__CV180X__) || defined(__CV181X__)
  cmd = tool_path + "/model_runner --model " + tpu_args->model_path +
        " --count " + std::to_string(count);
#else
  cmd = "export SHOW_TPU_USAGE=1 && cd " + tool_path + " && " +
        " ./bmrt_test --bmodel " + tpu_args->model_path +
        " --calculate_times " + std::to_string(count);
#endif

  int ret = system(cmd.c_str());
  if (ret != 0) std::cerr << "命令执行失败: " << ret << std::endl;
  return nullptr;
}

void *run_tdl_thread_cpu(void *args) {
  RUN_TDL_THREAD_CPU_ARG_S *pstArgs = (RUN_TDL_THREAD_CPU_ARG_S *)args;
  std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
  float fps_period = pstArgs->fps_period;
  prctl(PR_SET_NAME, "inference_load", 0, 0, 0);
  for (int e = 0; e < 101; ++e) {
    auto t0 = std::chrono::steady_clock::now();

    if (pstArgs->model_type ==
        ModelType::RECOGNITION_SPEECH_ZIPFORMER_ENCODER) {
      auto zipformer =
          std::dynamic_pointer_cast<ZipformerEncoder>(pstArgs->model_);
      zipformer->inference(pstArgs->input_images, out_data[0]);
    } else {
      std::vector<std::shared_ptr<BaseImage>> input_images{
          pstArgs->input_images};
      pstArgs->model_->inference(input_images, out_data);
    }
    auto t1 = std::chrono::steady_clock::now();
    float time_consume =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            t1 - t0)
            .count();
    if (time_consume <= fps_period) {
      int sleep_time = (int)(fps_period - time_consume);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
  }
  return nullptr;
}

void run_performance(const std::string &model_path,
                     std::shared_ptr<BaseImage> &input_images,
                     std::shared_ptr<BaseModel> model, float fps_period) {
  const auto sample_period = std::chrono::milliseconds(100);
  const auto sample_period_cpu = std::chrono::milliseconds(800);
  const auto enable_period = std::chrono::milliseconds(200);

  float infer_time_ms;
  if (get_model_info(model_path, infer_time_ms) != 0) {
    LOGE("get model info failed");
    return;
  }

  int tpu_count = (int)(5000.0f / infer_time_ms);  // 5s 对应运行次数
  tpu_count = std::max(tpu_count, 1);

  std::vector<double> tpu_samples;
  std::mutex tpu_mu;
  std::vector<double> cpu_samples;
  std::mutex cpu_mu;
  std::atomic<bool> sampling{false};
  pid_t tid;

  std::string tpu_usage_path;
  if (!confirm_path(tpu_usage_path)) {
    std::cerr << "Failed to confirm TPU usage path: " << tpu_usage_path
              << std::endl;
    tpu_usage_path.clear();
  }
  auto tpu_sampler = [&]() {
    enable_tpu_usage(tpu_usage_path);
    while (!sampling.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    while (sampling.load()) {
      read_tpu_usage(tpu_usage_path, tpu_samples, tpu_mu);
      std::this_thread::sleep_for(sample_period);
    }
  };
  auto cpu_sampler = [&]() {
    while (!sampling.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    while (sampling.load()) {
      std::this_thread::sleep_for(sample_period_cpu);
      read_cpu_line(cpu_samples, cpu_mu);
    }
  };

  std::thread th_cpu(cpu_sampler);
  RUN_TDL_THREAD_CPU_ARG_S tdl_args = {.input_images = input_images,
                                       .model_ = model,
                                       .fps_period = fps_period,
                                       .model_type = model->getModelType()};

  {
    ScopedSampler_cpu sampler(sampling, th_cpu);
    pthread_t inferenceTDLThread;
    pthread_create(&inferenceTDLThread, nullptr, run_tdl_thread_cpu, &tdl_args);
    pid_t tid = get_tid_by_name("inference_load");
    sampling = true;
    pthread_join(inferenceTDLThread, NULL);
  }
  std::thread th_tpu(tpu_sampler);
  std::this_thread::sleep_for(enable_period);

  RUN_TDL_THREAD_TPU_ARG_S tdl_args_tpu = {.model_path = model_path,
                                           .count = tpu_count};
  {
    ScopedSampler_tpu sampler(sampling, th_tpu);
    pthread_t stTDLThread;
    pthread_create(&stTDLThread, nullptr, run_tdl_thread_tpu, &tdl_args_tpu);
    pthread_join(stTDLThread, NULL);
  }

  load_show(cpu_samples, tpu_samples);
}

}  // namespace unitest
}  // namespace cvitdl
