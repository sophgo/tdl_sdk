#include "cvi_tdl_test.hpp"
#include <inttypes.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "regression_utils.hpp"
#include "tdl_model_defs.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
// #include "core/utils/vpss_helper.h"

namespace fs = std::experimental::filesystem;

namespace cvitdl {
namespace unitest {

CVI_TDLTestEnvironment::CVI_TDLTestEnvironment(
    const std::string &model_dir, const std::string &regress_asset_dir,
    const std::string &json_file_name) {
  CVI_TDLTestContext::getInstance().init(model_dir, regress_asset_dir,
                                         json_file_name);
}

CVI_TDLTestContext &CVI_TDLTestContext::getInstance() {
  static CVI_TDLTestContext instance;
  return instance;
}

fs::path CVI_TDLTestContext::getImageBaseDir() { return image_root_; }

fs::path CVI_TDLTestContext::getModelBaseDir() { return model_dir_; }

fs::path CVI_TDLTestContext::getJsonFilePath() { return json_file_path_; }

CVI_TDLTestContext::CVI_TDLTestContext() {}

void CVI_TDLTestContext::init(const std::string &model_dir,
                              const std::string &regress_asset_dir,
                              const std::string &json_file_name) {
  model_dir_ = fs::path(model_dir);
  image_root_ = fs::path(regress_asset_dir);  /// fs::path("input");
  json_root_ = fs::path(regress_asset_dir) / fs::path("json");
  json_file_path_ = json_root_ / fs::path(json_file_name);
}

int64_t CVI_TDLTestSuite::get_ion_memory_size() {
#ifdef __CV186X__
  const char ION_SUMMARY_PATH[255] =
      "/sys/kernel/debug/ion/cvi_npu_heap_dump/total_mem";
#else
  const char ION_SUMMARY_PATH[255] =
      "/sys/kernel/debug/ion/cvi_carveout_heap_dump/total_mem";
#endif
  std::ifstream ifs(ION_SUMMARY_PATH);
  std::string line;
  if (std::getline(ifs, line)) {
    return std::stoll(line);
  }
  return -1;
}

std::vector<std::string> CVI_TDLTestSuite::getFileList(
    const std::string &dir_path, const std::string &extension) {
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
void CVI_TDLTestSuite::SetUpTestCase() {
  // int64_t ion_size = get_ion_memory_size();

  // const CVI_S32 vpssgrp_width = DEFAULT_IMG_WIDTH;
  // const CVI_S32 vpssgrp_height = DEFAULT_IMG_HEIGHT;
  // const uint32_t num_buffer = 1;

  // // check if ION is enough to use.
  // int64_t used_size = vpssgrp_width * vpssgrp_height * num_buffer * 2;
  // ASSERT_LT(used_size, ion_size) << "insufficient ion size";

  // COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;

  // // Init SYS and Common VB,
  // VB_CONFIG_S stVbConf;
  // memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  // stVbConf.u32MaxPoolCnt = 1;
  // CVI_U32 u32BlkSize;
  // u32BlkSize = COMMON_GetPicBufferSize(
  //     vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR,
  //     DATA_BITWIDTH_8, enCompressMode, DEFAULT_ALIGN);
  // stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  // stVbConf.astCommPool[0].u32BlkCnt = num_buffer;

  // CVI_SYS_Exit();
  // CVI_VB_Exit();

  // ASSERT_EQ(CVI_VB_SetConfig(&stVbConf), CVI_SUCCESS);

  // ASSERT_EQ(CVI_VB_Init(), CVI_SUCCESS);

  // ASSERT_EQ(CVI_SYS_Init(), CVI_SUCCESS);
  std::cout << "CVI_TDLTestSuite::SetUpTestCase done" << std::endl;
}

void CVI_TDLTestSuite::TearDownTestCase() {
  // CVI_SYS_Exit();
  // CVI_VB_Exit();
}

CVI_TDLModelTestSuite::CVI_TDLModelTestSuite() {
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();
  fs::path json_file_path = context.getJsonFilePath();
  std::cout << "json_file_path: " << json_file_path << std::endl;
  m_image_dir = context.getImageBaseDir();
  m_model_dir = context.getModelBaseDir();

  if (!json_file_path.empty()) {
    try {
      std::ifstream filestr(json_file_path);
      if (!filestr.good()) {
        LOGIP("json 文件不存在, 创建新的空json文件: %s",
              json_file_path.c_str());
        if (generateEmptyJsonFile(json_file_path)) {
          std::ifstream new_filestr(json_file_path);
          m_json_object = nlohmann::ordered_json::parse(new_filestr);
          new_filestr.close();
        }
      } else {
        m_json_object = nlohmann::ordered_json::parse(filestr);
        filestr.close();
      }
    } catch (const std::exception &e) {
      LOGIP("parse json file %s failed, %s", json_file_path.c_str(), e.what());
    }
  }

  std::cout << "json_file_path: " << json_file_path << "\n"
            << "image_dir_name: " << m_image_dir << "\n"
            << "model_dir: " << m_model_dir << std::endl;
}

bool CVI_TDLModelTestSuite::generateEmptyJsonFile(
    const fs::path &json_file_path) {
  try {
    // 确保目录存在
    std::string model_name = json_file_path.stem().string();
    fs::path image_dir = m_image_dir / model_name;
    if (!fs::exists(image_dir)) {
      LOGE("image dir: %s not exists", image_dir.c_str());
      return false;
    }

    std::string extension;
    for (const auto &entry : fs::directory_iterator(image_dir)) {
      if (entry.path().filename().string().find("_mask_") ==
          std::string::npos) {
        extension = entry.path().extension().string();
        break;
      }
    }

    // 创建空的JSON对象，包含必要的key值
    nlohmann::ordered_json empty_json;
    empty_json["model_id"] = extractModelIdFromName(model_name);
    empty_json["model_name"] = model_name;
    empty_json["image_dir"] = model_name;
    empty_json["data_extension"] = extension;
    std::map<std::string, float> custom_config =
        getCustomRegressionConfig(model_name);
    for (auto &entry : custom_config) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1) << entry.second;
      empty_json[entry.first] = nlohmann::ordered_json::parse(oss.str());
    }

    // 写入文件
    std::ofstream ofs(json_file_path);
    if (!ofs.is_open()) {
      LOGE("Failed to create json file: %s", json_file_path.c_str());
      return false;
    }

    ofs << empty_json.dump(4);  // 4个空格缩进，格式化输出
    ofs.close();

    LOGIP("Successfully created empty json file: %s", json_file_path.c_str());
    return true;

  } catch (const std::exception &e) {
    LOGE("Exception while creating json file %s: %s", json_file_path.c_str(),
         e.what());
    return false;
  }
}

bool CVI_TDLModelTestSuite::checkToGetProcessResult(
    TestFlag test_flag, const std::string &platform,
    nlohmann::ordered_json &result) {
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();
  if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
    if (m_json_object.find(platform) != m_json_object.end()) {
      LOGIP("platform: %s is in json file %s, no need to generate",
            platform.c_str(), context.getJsonFilePath().c_str());
      return false;
    } else {
      result = getProcessResult();
      if (result.empty()) {
        LOGIP("results is empty,%s", context.getJsonFilePath().c_str());
        return false;
      }
    }
  } else {
    result = m_json_object[platform];
  }
  return true;
}
nlohmann::ordered_json CVI_TDLModelTestSuite::getValidPlatformResult() {
  std::vector<std::string> platform_list = get_platform_list();
  nlohmann::ordered_json function_result;
  for (auto &platform : platform_list) {
    if (m_json_object.find(platform) != m_json_object.end()) {
      function_result = m_json_object[platform];
      break;
    }
  }
  return function_result;
}
nlohmann::ordered_json CVI_TDLModelTestSuite::getProcessResult() {
  nlohmann::ordered_json empty_result;
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  if (m_json_object.find("data_extension") != m_json_object.end()) {
    std::string data_extension =
        m_json_object["data_extension"].get<std::string>();
    std::vector<std::string> file_list = getFileList(image_dir, data_extension);
    for (auto &file : file_list) {
      empty_result[file] = nlohmann::ordered_json();
    }
  } else {
    empty_result = getValidPlatformResult();
  }

  return empty_result;
}

void CVI_TDLModelTestSuite::roundFloatPrecision(
    nlohmann::ordered_json &json_object) {
  if (json_object.is_number_float()) {
    // 处理浮点数，保留两位小数
    double value = json_object.get<double>();
    // 保留两位小数：乘以100，四舍五入，再除以100
    double rounded = std::round(value * std::pow(10, float_precesion_num_)) /
                     std::pow(10, float_precesion_num_);
    json_object = rounded;
  } else if (json_object.is_array()) {
    // 递归处理数组中的每个元素
    for (auto &element : json_object) {
      roundFloatPrecision(element);
    }
  } else if (json_object.is_object()) {
    // 递归处理对象中的每个值
    for (auto it = json_object.begin(); it != json_object.end(); ++it) {
      roundFloatPrecision(it.value());
    }
  }
  // 对于其他类型（字符串、布尔值、null等），不做处理
}

bool CVI_TDLModelTestSuite::writeJsonFile(const std::string &json_file_path,
                                          nlohmann::ordered_json &json_object) {
  std::ofstream ofs(json_file_path);
  if (ofs.is_open()) {
    roundFloatPrecision(json_object);
  } else {
    LOGE("write json file :%s failed", json_file_path.c_str());
    return false;
  }
  ofs << json_object.dump(4);
  ofs.close();
  LOGIP("Successfully updated results at: %s", json_file_path.c_str());
  return true;
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
bool CVI_TDLModelTestSuite::matchObjects(
    const std::vector<std::vector<float>> &gt_objects,
    const std::vector<std::vector<float>> &pred_objects,
    const float iout_thresh, const float score_diff_thresh) {
  std::vector<int> gt_matched(gt_objects.size(), 0);
  std::vector<int> pred_matched(pred_objects.size(), 0);

  bool is_matched = true;
  for (size_t i = 0; i < gt_objects.size(); i++) {
    auto &gt_object = gt_objects[i];
    std::vector<float> matched_dets;
    float max_iout = 0.0f;
    int pred_matched_idx;
    for (size_t j = 0; j < pred_objects.size(); j++) {
      const auto &pred_object = pred_objects[j];
      float iout = iou(gt_object, pred_object);
      if (iout > max_iout) {
        max_iout = iout;
        pred_matched_idx = j;
      }
    }
    if (gt_object[5] == pred_objects[pred_matched_idx][5] &&
        max_iout > iout_thresh) {
      matched_dets = pred_objects[pred_matched_idx];
      float score_diff = std::abs(matched_dets[4] - gt_object[4]);
      if (score_diff > score_diff_thresh) {
        std::cout << "score diff: " << score_diff << ",gtbox:[" << gt_object[0]
                  << "," << gt_object[1] << "," << gt_object[2] << ","
                  << gt_object[3] << "]"
                  << ",score:" << gt_object[4] << ",class_id:" << gt_object[5]
                  << ",predbox:[" << matched_dets[0] << "," << matched_dets[1]
                  << "," << matched_dets[2] << "," << matched_dets[3] << "]"
                  << ",pred_score:" << matched_dets[4]
                  << ",pred_class_id:" << matched_dets[5] << std::endl;
        is_matched = false;
      } else {
        gt_matched[i] = 1;
        pred_matched[pred_matched_idx] = 1;
      }
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
      } else if (gt_objects[i][5] == 0.5) {
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

bool CVI_TDLModelTestSuite::matchScore(const std::vector<float> &gt_info,
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

std::shared_ptr<BaseImage> CVI_TDLModelTestSuite::loadInputData(
    std::string &image_path) {
  std::shared_ptr<BaseImage> frame;

  if (image_path.size() >= 4 &&
      image_path.substr(image_path.size() - 4) == ".txt") {
    std::vector<float> keypoints;
    std::ifstream infile(image_path);
    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      float x, y;
      if (iss >> x >> y) {
        keypoints.push_back(x);
        keypoints.push_back(y);
      }
    }

    if (keypoints.size() != 42) {
      throw std::invalid_argument("txt file err");
    }

    frame = ImageFactory::createImage(42, 1, ImageFormat::GRAY,
                                      TDLDataType::FP32, true);
    float *data_buffer =
        reinterpret_cast<float *>(frame->getVirtualAddress()[0]);

    memcpy(data_buffer, &keypoints[0], 42 * sizeof(float));

  } else if (image_path.size() >= 4 &&
             image_path.substr(image_path.size() - 4) == ".bin") {
    int frame_size = 0;
    FILE *fp = fopen(image_path.c_str(), "rb");
    if (fp) {
      fseek(fp, 0, SEEK_END);
      frame_size = ftell(fp);
      fseek(fp, 0, SEEK_SET);
      frame = ImageFactory::createImage(frame_size, 1, ImageFormat::GRAY,
                                        TDLDataType::UINT8, true);
      uint8_t *data_buffer = frame->getVirtualAddress()[0];
      fread(data_buffer, 1, frame_size, fp);
      fclose(fp);
    }
  } else {
    frame = ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);
  }

  return frame;
};

bool CVI_TDLModelTestSuite::matchKeypoints(
    const std::vector<float> &gt_keypoints_x,
    const std::vector<float> &gt_keypoints_y,
    const std::vector<float> &gt_keypoints_score,
    const std::vector<float> &pred_keypoints_x,
    const std::vector<float> &pred_keypoints_y,
    const std::vector<float> &pred_keypoints_score, const float position_thresh,
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

bool CVI_TDLModelTestSuite::matchSegmentation(const cv::Mat &mat1,
                                              const cv::Mat &mat2,
                                              float mask_thresh) {
  cv::Mat diff = (mat1 != mat2);  // 元素不相等的位置为255，相等为0
  int diffCount = cv::countNonZero(diff);

  int height = mat1.rows;
  int width = mat1.cols;

  return float(diffCount) / (height * width) < mask_thresh;
}

}  // namespace unitest
}  // namespace cvitdl