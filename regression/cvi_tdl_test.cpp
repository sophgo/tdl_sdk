#include "cvi_tdl_test.hpp"

#include <inttypes.h>

#include <fstream>

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
  std::cout << "model_dir: " << m_model_dir << ", image_dir: " << m_image_dir
            << ", json_dir: " << m_json_dir << std::endl;

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
    m_image_dir =
        context.getImageBaseDir() / fs::path(m_json_object["image_dir"]);
  } else {
    m_image_dir = context.getImageBaseDir() / fs::path(image_dir_name);
  }
  m_model_dir = context.getModelBaseDir();

  std::cout << "json_file_path: " << json_file_path
            << ",image_dir_name: " << m_image_dir
            << ",model_dir: " << m_model_dir << std::endl;
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
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  return area_inter / (area1 + area2 - area_inter);
}

// object info:[x1,y1,x2,y2,score,class_id]
bool CVI_TDLModelTestSuite::matchObjects(
    const std::vector<std::vector<float>> &gt_objects,
    const std::vector<std::vector<float>> &pred_objects,
    const float iout_thresh, const float score_thresh) {
  for (const auto &gt_object : gt_objects) {
    std::vector<std::vector<float>> matched_dets;
    for (const auto &pred_object : pred_objects) {
      float iout = iou(gt_object, pred_object);
      if (gt_object[5] == pred_object[5]) {
        if (iout > iout_thresh) {
          matched_dets.push_back(pred_object);
        }
      }
    }
    if (matched_dets.size() == 0) {
      std::cout << "no matched det,gtbox:[" << gt_object[0] << ","
                << gt_object[1] << "," << gt_object[2] << "," << gt_object[3]
                << "]"
                << ",score:" << gt_object[4] << ",class_id:" << gt_object[5]
                << std::endl;
      for (const auto &pred_object : pred_objects) {
        std::cout << "predbox:[" << pred_object[0] << "," << pred_object[1]
                  << "," << pred_object[2] << "," << pred_object[3] << "]"
                  << ",score:" << pred_object[4]
                  << ",class_id:" << pred_object[5] << std::endl;
      }
      return false;
    } else if (matched_dets.size() > 1) {
      std::cout << "has multiple matched det,gtbox:[" << gt_object[0] << ","
                << gt_object[1] << "," << gt_object[2] << "," << gt_object[3]
                << "]"
                << ",score:" << gt_object[4] << ",class_id:" << gt_object[5]
                << std::endl;
      for (const auto &matched_det : matched_dets) {
        std::cout << "matched det:[" << matched_det[0] << "," << matched_det[1]
                  << "," << matched_det[2] << "," << matched_det[3] << "]"
                  << ",score:" << matched_det[4]
                  << ",class_id:" << matched_det[5] << std::endl;
      }
      return false;
    } else {
      float score_diff = std::abs(matched_dets[0][4] - gt_object[4]);
      if (score_diff > score_thresh) {
        std::cout << "score diff: " << score_diff << ",gtbox:[" << gt_object[0]
                  << "," << gt_object[1] << "," << gt_object[2] << ","
                  << gt_object[3] << "]" << ",score:" << gt_object[4]
                  << ",class_id:" << gt_object[5]
                  << ",pred_score:" << matched_dets[0][4] << ",predbox:["
                  << matched_dets[0][0] << "," << matched_dets[0][1] << ","
                  << matched_dets[0][2] << "," << matched_dets[0][3] << "]"
                  << ",pred_class_id:" << matched_dets[0][5] << std::endl;
        return false;
      }
    }
  }
  return true;
}
}  // namespace unitest
}  // namespace cvitdl