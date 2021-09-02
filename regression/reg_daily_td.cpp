#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"
#include "json.hpp"

#define SCORE_BIAS 0.05

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;
  std::string image_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  int img_num = int(m_json_read["reg_config"][0]["image_num"]);

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }

  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx]);
    // printf("[%d] %s\n", img_idx, image_path.c_str());
    float expected_res = float(m_json_read["reg_config"][0]["expected_results"][img_idx]);

    // Read image using IVE.
    IVE_IMAGE_S ive_frame =
        CVI_IVE_ReadImage(ive_handle, image_path.c_str(), IVE_IMAGE_TYPE_U8C3_PLANAR);
    if (ive_frame.u16Width == 0) {
      printf("Read image failed with %x!\n", ret);
      pass = false;
      return ret;
    }

    // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
    VIDEO_FRAME_INFO_S frame;
    ret = CVI_IVE_Image2VideoFrameInfo(&ive_frame, &frame, false);
    if (ret != CVIAI_SUCCESS) {
      printf("Convert to video frame failed with %#x!\n", ret);
      return ret;
    }

    float moving_score;
    CVI_AI_TamperDetection(ai_handle, &frame, &moving_score);
    // printf("[%d] %f (expected: %f)\n", img_idx, moving_score, expected_res);

    pass &= ABS(moving_score - expected_res) < SCORE_BIAS;
    // printf(" > %s\n", (pass ? "PASS" : "FAILURE"));

    CVI_SYS_FreeI(ive_handle, &ive_frame);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_IVE_DestroyHandle(ive_handle);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();

  return pass ? CVIAI_SUCCESS : CVIAI_FAILURE;
}
