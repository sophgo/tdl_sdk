#include "cviai_test.hpp"
#include <inttypes.h>
#include <fstream>
#include "core/utils/vpss_helper.h"

namespace fs = std::experimental::filesystem;

namespace cviai {
namespace unitest {

CVIAITestEnvironment::CVIAITestEnvironment(const std::string &model_dir,
                                           const std::string &image_dir,
                                           const std::string &json_dir) {
  CVIAITestContext::getInstance().init(model_dir, image_dir, json_dir);
}

CVIAITestContext &CVIAITestContext::getInstance() {
  static CVIAITestContext instance;
  return instance;
}

fs::path CVIAITestContext::getImageBaseDir() { return m_image_dir; }

fs::path CVIAITestContext::getModelBaseDir() { return m_model_dir; }

fs::path CVIAITestContext::getJsonBaseDir() { return m_json_dir; }

CVIAITestContext::CVIAITestContext() : m_inited(false) {}

void CVIAITestContext::init(std::string model_dir, std::string image_dir, std::string json_dir) {
  if (!m_inited) {
    m_model_dir = model_dir;
    m_image_dir = image_dir;
    m_json_dir = json_dir;
    m_inited = true;
  }
}

int64_t CVIAITestSuite::get_ion_memory_size() {
  const char ION_SUMMARY_PATH[255] = "/sys/kernel/debug/ion/cvi_carveout_heap_dump/total_mem";
  std::ifstream ifs(ION_SUMMARY_PATH);
  std::string line;
  if (std::getline(ifs, line)) {
    return std::stoll(line);
  }
  return -1;
}

void CVIAITestSuite::SetUpTestCase() {
  int64_t ion_size = get_ion_memory_size();

  const CVI_S32 vpssgrp_width = DEFAULT_IMG_WIDTH;
  const CVI_S32 vpssgrp_height = DEFAULT_IMG_HEIGHT;
  const uint32_t num_buffer = 1;

  // check if ION is enough to use.
  int64_t used_size = vpssgrp_width * vpssgrp_height * num_buffer * 2;
  ASSERT_LT(used_size, ion_size) << "insufficient ion size";

  COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;

  // Init SYS and Common VB,
  VB_CONFIG_S stVbConf;
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  stVbConf.u32MaxPoolCnt = 1;
  CVI_U32 u32BlkSize;
  u32BlkSize = COMMON_GetPicBufferSize(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR,
                                       DATA_BITWIDTH_8, enCompressMode, DEFAULT_ALIGN);
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[0].u32BlkCnt = num_buffer;

  CVI_SYS_Exit();
  CVI_VB_Exit();

  ASSERT_EQ(CVI_VB_SetConfig(&stVbConf), CVI_SUCCESS);

  ASSERT_EQ(CVI_VB_Init(), CVI_SUCCESS);

  ASSERT_EQ(CVI_SYS_Init(), CVI_SUCCESS);
}

void CVIAITestSuite::TearDownTestCase() {
  CVI_SYS_Exit();
  CVI_VB_Exit();
}

CVIAIModelTestSuite::CVIAIModelTestSuite(const std::string &json_file_name,
                                         const std::string &image_dir_name) {
  CVIAITestContext &context = CVIAITestContext::getInstance();

  fs::path json_file_path = context.getJsonBaseDir();
  json_file_path /= json_file_name;
  std::ifstream filestr(json_file_path);
  filestr >> m_json_object;
  filestr.close();

  m_image_dir = context.getImageBaseDir() / fs::path(image_dir_name);
  m_model_dir = context.getModelBaseDir();
}
}  // namespace unitest
}  // namespace cviai