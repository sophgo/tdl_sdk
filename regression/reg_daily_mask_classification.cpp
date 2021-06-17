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
  CVI_S32 ret = CVI_SUCCESS;
  std::string model_dir = std::string(argv[1]);
  std::string image_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  std::string model_name = std::string(m_json_read["reg_config"][0]["model_name"]);
  std::string model_path = model_dir + "/" + model_name;
  int img_num = int(m_json_read["reg_config"][0]["image_num"]);

  float threshold = float(m_json_read["reg_config"][0]["threshold"]);

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);

  ret |=
      CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set face quality model failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, false);

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx]);
    int expected_res = int(m_json_read["reg_config"][0]["expected_results"][img_idx]);

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image \'%s\' failed with %#x!\n", image_path.c_str(), ret);
      return ret;
    }

    cvai_face_t face_meta;
    memset(&face_meta, 0, sizeof(cvai_face_t));

    face_meta.size = 1;
    face_meta.width = frame.stVFrame.u32Width;
    face_meta.height = frame.stVFrame.u32Height;
    face_meta.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * face_meta.size);
    memset(face_meta.info, 0, sizeof(cvai_face_info_t) * face_meta.size);
    face_meta.info[0].bbox.x1 = 0;
    face_meta.info[0].bbox.y1 = 0;
    face_meta.info[0].bbox.x2 = frame.stVFrame.u32Width;
    face_meta.info[0].bbox.y2 = frame.stVFrame.u32Height;

    face_meta.info[0].pts.size = 5;
    face_meta.info[0].pts.x = (float *)malloc(sizeof(float) * face_meta.info[0].pts.size);
    face_meta.info[0].pts.y = (float *)malloc(sizeof(float) * face_meta.info[0].pts.size);

    CVI_AI_MaskClassification(ai_handle, &frame, &face_meta);

    int predict_mask = face_meta.info[0].mask_score >= threshold;
    if (predict_mask != expected_res) {
      printf("Result not equal!, expected: %d, predict: %d on %s\n", expected_res, predict_mask,
             image_path.c_str());
      CVI_AI_Free(&face_meta);
      CVI_VB_ReleaseBlock(blk_fr);
      CVI_AI_DestroyHandle(ai_handle);
      CVI_SYS_Exit();
      return CVI_FAILURE;
    }

    CVI_AI_Free(&face_meta);
    CVI_VB_ReleaseBlock(blk_fr);
  }

  printf("PASSED\n");
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
  return CVI_SUCCESS;
}
