#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  // CVI_S32 ret = CVI_SUCCESS;
  std::string model_dir = std::string(argv[1]);
  std::string image_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  std::string od_model_name = std::string(m_json_read["od_model"]);
  std::string od_model_path = model_dir + "/" + od_model_name;
  printf("od_model_path: %s\n", od_model_path.c_str());

  std::string pose_model_name = std::string(m_json_read["pose_model"]);
  std::string pose_model_path = model_dir + "/" + pose_model_name;
  printf("pose_model_path: %s\n", pose_model_path.c_str());

  int img_num = int(m_json_read["test_images"].size());
  printf("img_num: %d\n", img_num);

  float threshold = float(m_json_read["threshold"]);
  printf("threshold: %f\n", threshold);

  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret =
      CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, od_model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, false);
  CVI_AI_SelectDetectClass(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, 1,
                           CVI_AI_DET_TYPE_PERSON);
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, pose_model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set model alphapose failed with %#x!\n", ret);
    return ret;
  }

  VB_BLK blk1;
  VIDEO_FRAME_INFO_S frame;
  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path = image_dir + "/" + std::string(m_json_read["test_images"][img_idx]);
    // printf("[%d] %s\n", img_idx, image_path.c_str());

    int expected_res = int(m_json_read["expected_results"][img_idx]);
    // printf("expected_res %d\n", expected_res);

    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk1, &frame, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    CVI_AI_MobileDetV2_D0(ai_handle, &frame, &obj);

    CVI_AI_AlphaPose(ai_handle, &frame, &obj);

    CVI_AI_Fall(ai_handle, &obj);

    // if (obj.size > 0 && obj.info[0].pedestrian_properity != NULL) {
    //  printf("; fall score %d ", obj.info[0].pedestrian_properity->fall);
    //}

    bool pass = (abs(expected_res - obj.info[0].pedestrian_properity->fall) <= threshold);

    printf("[%d] pass: %d; fall, expected : %d, result : %d\n", img_idx, pass, expected_res,
           obj.info[0].pedestrian_properity->fall);
    if (!pass) {
      return CVI_FAILURE;
    }

    CVI_AI_Free(&obj);
    CVI_VB_ReleaseBlock(blk1);
  }
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
  printf("fall regression result: all pass\n");
  return CVI_SUCCESS;
}