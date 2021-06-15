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

  std::string fd_model_name = std::string(m_json_read["fd_model"]);
  std::string fd_model_path = model_dir + "/" + fd_model_name;
  printf("fd_model_path: %s\n", fd_model_path.c_str());

  std::string liveness_model_name = std::string(m_json_read["liveness_model"]);
  std::string liveness_model_path = model_dir + "/" + liveness_model_name;
  printf("liveness_model_path: %s\n", liveness_model_path.c_str());

  int img_num = int(m_json_read["image_num"]);
  printf("img_num: %d\n", img_num);

  float threshold = float(m_json_read["threshold"]);
  printf("threshold: %f\n", threshold);

  static CVI_S32 vpssgrp_width = 1920;
  static CVI_S32 vpssgrp_height = 1080;
  cviai_handle_t handle = NULL;

  CVI_S32 ret = CVI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_CreateHandle(&handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, fd_model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_SetModelPath(handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, liveness_model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set model liveness failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  // bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string rgb_image_path =
        image_dir + "/" + std::string(m_json_read["test_images"][img_idx][0]);
    std::string ir_image_path =
        image_dir + "/" + std::string(m_json_read["test_images"][img_idx][1]);
    float expected_res = float(m_json_read["expected_results"][img_idx]);

    VB_BLK blk1;
    VIDEO_FRAME_INFO_S frame1;
    CVI_S32 ret = CVI_AI_ReadImage(rgb_image_path.c_str(), &blk1, &frame1, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image1 failed with %#x!\n", ret);
      return ret;
    }

    VB_BLK blk2;
    VIDEO_FRAME_INFO_S frame2;
    ret = CVI_AI_ReadImage(ir_image_path.c_str(), &blk2, &frame2, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image2 failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t rgb_face;
    memset(&rgb_face, 0, sizeof(cvai_face_t));

    cvai_face_t ir_face;
    memset(&ir_face, 0, sizeof(cvai_face_t));

    CVI_AI_RetinaFace(handle, &frame1, &rgb_face);
    CVI_AI_RetinaFace(handle, &frame2, &ir_face);

    if (rgb_face.size > 0) {
      if (ir_face.size > 0) {
        CVI_AI_Liveness(handle, &frame1, &frame2, &rgb_face, &ir_face);
      } else {
        rgb_face.info[0].liveness_score = -2.0;
      }
    }

    bool pass = abs(rgb_face.info[0].liveness_score - expected_res) < threshold;
    printf("[%d] expected: %f, score: %f, pass: %d\n", img_idx, expected_res,
           rgb_face.info[0].liveness_score, pass);
    if (!pass) {
      exit(0);
    }

    CVI_AI_Free(&rgb_face);
    CVI_AI_Free(&ir_face);
    CVI_VB_ReleaseBlock(blk1);
    CVI_VB_ReleaseBlock(blk2);
  }
  CVI_AI_DestroyHandle(handle);
  CVI_SYS_Exit();
  printf("liveness regression result: all pass\n");
}
