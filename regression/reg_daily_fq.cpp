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

  float pos_threshold = float(m_json_read["reg_config"][0]["pos_threshold"]);
  float neg_threshold = float(m_json_read["reg_config"][0]["neg_threshold"]);

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 112;
  const CVI_S32 vpssgrp_height = 112;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;

  ret = CVI_AI_CreateHandle2(&ai_handle, 1);
  // ret |= CVI_AI_SetVpssTimeout(ai_handle, 10);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set face quality model failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, false);

  float std_face_landmark_x[5] = {38.29459953, 73.53179932, 56.02519989, 41.54930115, 70.72990036};
  float std_face_landmark_y[5] = {51.69630051, 51.50139999, 71.73660278, 92.3655014, 92.20410156};

  cvai_face_t face_meta;
  memset(&face_meta, 0, sizeof(cvai_face_t));

  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx]);
    // printf("[%d] %s\n", img_idx, image_path.c_str());
    int expected_res = int(m_json_read["reg_config"][0]["expected_results"][img_idx]);

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    face_meta.size = 1;
    face_meta.height = 112;
    face_meta.width = 112;
    face_meta.info = (cvai_face_info_t *)malloc(1 * sizeof(cvai_face_info_t));
    face_meta.info[0].bbox.x1 = 0;
    face_meta.info[0].bbox.y1 = 0;
    face_meta.info[0].bbox.x2 = 111;
    face_meta.info[0].bbox.y2 = 111;
    face_meta.info[0].pts.size = 5;
    face_meta.info[0].pts.x = (float *)malloc(5 * sizeof(float));
    face_meta.info[0].pts.y = (float *)malloc(5 * sizeof(float));
    for (int i = 0; i < 5; i++) {
      face_meta.info[0].pts.x[i] = std_face_landmark_x[i];
      face_meta.info[0].pts.y[i] = std_face_landmark_y[i];
    }
    face_meta.info[0].feature.size = 0;
    face_meta.info[0].feature.ptr = NULL;

    CVI_AI_FaceQuality(ai_handle, &frame, &face_meta);
    // printf("face quality: %f\n", face_meta.info[0].face_quality);
    if (expected_res == 0) {
      pass &= face_meta.info[0].face_quality < neg_threshold;
    } else {
      pass &= face_meta.info[0].face_quality > pos_threshold;
    }

    CVI_AI_Free(&face_meta);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();

  return pass ? CVI_SUCCESS : CVI_FAILURE;
}
