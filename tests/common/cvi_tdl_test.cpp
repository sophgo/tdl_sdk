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
  std::string platform_str = gen_model_dir();
  std::string model_dir_tmp = model_dir;
  if (model_dir.size() >= platform_str.size() &&
      model_dir.substr(model_dir.size() - platform_str.size(),
                       platform_str.size()) == platform_str) {
    model_dir_tmp = model_dir.substr(0, model_dir.size() - platform_str.size());
    LOGIP("update model_dir: %s", model_dir.c_str());
  }
  if (model_dir_tmp.back() == '/') {
    model_dir_tmp.pop_back();
  }

  model_dir_ = fs::path(model_dir_tmp);
  image_root_ = fs::path(regress_asset_dir);  /// fs::path("input");
  json_root_ = fs::path(regress_asset_dir) / fs::path("json");
  if (!json_file_name.empty()) {
    json_file_path_ = json_root_ / fs::path(json_file_name);
  }
}
bool CVI_TDLTestContext::setTestFlag(const std::string &test_flag) {
  if (test_flag == "func") {
    test_flag_ = TestFlag::FUNCTION;
  } else if (test_flag == "perf") {
    test_flag_ = TestFlag::PERFORMANCE;
  } else if (test_flag == "generate_func") {
    test_flag_ = TestFlag::GENERATE_FUNCTION_RES;
  } else if (test_flag == "generate_performance") {
    test_flag_ = TestFlag::GENERATE_PERFORMANCE_RES;
  } else {
    return false;
  }
  return true;
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
        if (context.getTestFlag() == TestFlag::FUNCTION) {
          LOGIP("需要先生成文件，退出");
          exit(0);
        }
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

}  // namespace unitest
}  // namespace cvitdl