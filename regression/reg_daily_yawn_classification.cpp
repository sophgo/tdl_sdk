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
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;
  std::string model_dir = std::string(argv[1]);
  std::string image_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  std::string model_name_fd = std::string(m_json_read["reg_config"][0]["model_name"][0]);
  std::string model_path_fd = model_dir + "/" + model_name_fd;
  std::string model_name_fl = std::string(m_json_read["reg_config"][0]["model_name"][1]);
  std::string model_path_fl = model_dir + "/" + model_name_fl;
  std::string model_name = std::string(m_json_read["reg_config"][0]["model_name"][2]);
  std::string model_path = model_dir + "/" + model_name;

  int img_num = int(m_json_read["reg_config"][0]["image_num"]);
  float threshold = float(m_json_read["reg_config"][0]["threshold"]);

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  cviai_handle_t facelib_handle = NULL;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle2(&facelib_handle, 1, 0);
  ret |=
      CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, model_path_fd.c_str());
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER,
                             model_path_fl.c_str());
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION,
                             model_path.c_str());

  if (ret != CVIAI_SUCCESS) {
    printf("Set face quality model failed with %#x!\n", ret);
    return ret;
  }

  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx]);
    int expected_res = int(m_json_read["reg_config"][0]["expected_results"][img_idx]);

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVIAI_SUCCESS) {
      printf("Read image \'%s\' failed with %#x!\n", image_path.c_str(), ret);
      return ret;
    }
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_RetinaFace(facelib_handle, &frame, &face);

    face.dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));
    face.dms->dms_od.info = NULL;
    CVI_AI_FaceLandmarker(facelib_handle, &frame, &face);
    CVI_AI_YawnClassification(facelib_handle, &frame, &face);

    if (expected_res)
      pass &= face.dms->yawn_score > threshold;
    else
      pass &= face.dms->yawn_score < threshold;

    CVI_AI_FreeDMS(face.dms);
    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
  return pass ? CVIAI_SUCCESS : CVIAI_FAILURE;
}
