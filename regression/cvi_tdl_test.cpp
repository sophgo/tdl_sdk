#include "cvi_tdl_test.hpp"
#include <inttypes.h>
#include <fstream>
#include "cvi_tdl_model_id.hpp"
#include "tdl_model_defs.hpp"
#include "utils/common_utils.hpp"
// #include "core/utils/vpss_helper.h"

namespace fs = std::experimental::filesystem;

namespace cvitdl {
namespace unitest {

CVI_TDLTestEnvironment::CVI_TDLTestEnvironment(const std::string &model_dir,
                                               const std::string &image_dir,
                                               const std::string &json_dir) {
  CVI_TDLTestContext::getInstance().init(model_dir, image_dir, json_dir);
}

CVI_TDLTestContext &CVI_TDLTestContext::getInstance() {
  static CVI_TDLTestContext instance;
  return instance;
}

fs::path CVI_TDLTestContext::getImageBaseDir() { return m_image_dir; }

fs::path CVI_TDLTestContext::getModelBaseDir() { return m_model_dir; }

fs::path CVI_TDLTestContext::getJsonBaseDir() { return m_json_dir; }

CVI_TDLTestContext::CVI_TDLTestContext() : m_inited(false) {}

void CVI_TDLTestContext::init(std::string model_dir, std::string image_dir,
                              std::string json_dir) {
  if (!m_inited) {
    m_model_dir = model_dir;
    m_image_dir = image_dir;
    m_json_dir = json_dir;
    m_inited = true;
  }
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
  int64_t ion_size = get_ion_memory_size();

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

CVI_TDLModelTestSuite::CVI_TDLModelTestSuite(
    const std::string &json_file_name, const std::string &image_dir_name) {
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  fs::path json_file_path = context.getJsonBaseDir();
  json_file_path /= json_file_name;
  std::cout << "to parse json file: " << json_file_path << std::endl;
  if (!json_file_name.empty()) {
    std::ifstream filestr(json_file_path);
    filestr >> m_json_object;
    filestr.close();
    std::cout << "m_json_object: " << m_json_object << std::endl;
  }
  if (m_json_object.contains("image_dir")) {
    m_image_dir = context.getImageBaseDir() /
                  fs::path(m_json_object["image_dir"].get<std::string>());
  } else {
    m_image_dir = context.getImageBaseDir() / fs::path(image_dir_name);
  }
  m_model_dir = context.getModelBaseDir();

  std::cout << "json_file_path: " << json_file_path
            << ",image_dir_name: " << m_image_dir
            << ",model_dir: " << m_model_dir << std::endl;
}

CVI_TDLModelTestSuite::CVI_TDLModelTestSuite() {
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();
  fs::path json_file_path = context.getJsonBaseDir();
  std::ifstream filestr(json_file_path);
  filestr >> m_json_object;
  filestr.close();
  // std::cout << "m_json_object: " << m_json_object << "\n" << std::endl;

  m_image_dir = context.getImageBaseDir();
  m_model_dir = context.getModelBaseDir();
  std::cout << "json_file_path: " << json_file_path << "\n"
            << "image_dir_name: " << m_image_dir << "\n"
            << "model_dir: " << m_model_dir << std::endl;
}

float iou(const std::vector<float> &gt_object,
          const std::vector<float> &pred_object) {
  float iout = 0.0f;
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
    const float iout_thresh, const float score_thresh) {
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
      if (score_diff > score_thresh) {
        std::cout << "score diff: " << score_diff << ",gtbox:[" << gt_object[0]
                  << "," << gt_object[1] << "," << gt_object[2] << ","
                  << gt_object[3] << "]" << ",score:" << gt_object[4]
                  << ",class_id:" << gt_object[5] << ",predbox:["
                  << matched_dets[0] << "," << matched_dets[1] << ","
                  << matched_dets[2] << "," << matched_dets[3] << "]"
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
      if (gt_objects[i][5] == 0.5) {
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
    if (pred_matched[i] == 0) {
      std::cout << "pred box not matched ,predbox:[" << pred_objects[i][0]
                << "," << pred_objects[i][1] << "," << pred_objects[i][2] << ","
                << pred_objects[i][3] << "],score:" << pred_objects[i][4]
                << ",class_id:" << pred_objects[i][5] << std::endl;
      is_matched = false;
    }
  }
  return is_matched;
}

bool CVI_TDLModelTestSuite::matchScore(
    const std::vector<std::vector<float>> &gt_info,
    const std::vector<std::vector<float>> &pred_info,
    const float score_thresh) {
  if (gt_info.size() != pred_info.size()) {
    return false;
  }

  for (size_t i = 0; i < gt_info.size(); ++i) {
    const auto &gt = gt_info[i];
    const auto &pred = pred_info[i];

    if (std::abs(gt[0] - pred[0]) > score_thresh ||
        std::abs(gt[1] - pred[1]) > score_thresh ||
        std::abs(gt[2] - pred[2]) > score_thresh ||
        std::abs(gt[3] - pred[3]) > score_thresh) {
      return false;
    }
  }
  return true;
};

ModelType CVI_TDLModelTestSuite::stringToModelType(
    const std::string &model_type_str) {
  auto it = model_type_map.find(model_type_str);
  if (it != model_type_map.end()) {
    return it->second;
  } else {
    throw std::invalid_argument("Invalid model type: " + model_type_str);
  }
}

std::shared_ptr<BaseImage> CVI_TDLModelTestSuite::getInputData(
    std::string &image_path, ModelType model_id) {
  std::shared_ptr<BaseImage> frame;
  if (image_path.size() >= 4 &&
      image_path.substr(image_path.size() - 4) != ".bin") {
    frame = ImageFactory::readImage(image_path, true);
  } else {
    int frame_size = 0;
    if (model_id == ModelType::CLS_SOUND_BABAY_CRY) {
      frame_size = 96000;
    } else if (model_id == ModelType::CLS_SOUND_COMMAND) {
      frame_size = 32000;
    } else {
      std::cout << "model_id not supported" << std::endl;
    }
    unsigned char buffer[frame_size];
    read_binary_file(image_path, buffer, frame_size);
    frame = ImageFactory::createImage(frame_size, 1, ImageFormat::GRAY,
                                      TDLDataType::UINT8, true);

    uint8_t *data_buffer = frame->getVirtualAddress()[0];
    memcpy(data_buffer, buffer, frame_size * sizeof(uint8_t));
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
    const float score_thresh) {
  if (gt_keypoints_x.size() != pred_keypoints_x.size() ||
      gt_keypoints_y.size() != pred_keypoints_y.size() ||
      gt_keypoints_x.size() != gt_keypoints_y.size() ||
      gt_keypoints_score.size() != pred_keypoints_score.size()) {
    return false;
  }

  for (size_t i = 0; i < gt_keypoints_score.size(); i++) {
    float score_diff =
        std::abs(gt_keypoints_score[i] - pred_keypoints_score[i]);

    if (score_diff > score_thresh) {
      return false;
    }
  }

  float total_distance = 0.0f;
  int num_keypoints = gt_keypoints_x.size();
  float score_thresh_for_distance = 0.5f;
  std::vector<float> keypoints_index_for_distance;
  if (gt_keypoints_score.size() == 0) {
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
}  // namespace unitest
}  // namespace cvitdl