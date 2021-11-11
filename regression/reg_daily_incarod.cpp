#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

void dms_init(cvai_face_t* face) {
  cvai_dms_t* dms = (cvai_dms_t*)malloc(sizeof(cvai_dms_t));
  dms->reye_score = 0;
  dms->leye_score = 0;
  dms->yawn_score = 0;
  dms->phone_score = 0;
  dms->smoke_score = 0;
  dms->landmarks_106.size = 0;
  dms->landmarks_5.size = 0;
  dms->head_pose.yaw = 0;
  dms->head_pose.pitch = 0;
  dms->head_pose.roll = 0;
  dms->dms_od.info = NULL;
  dms->dms_od.size = 0;
  face->dms = dms;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  std::string model_dir = std::string(argv[1]);
  std::string image_dir = std::string(argv[2]);

  nlohmann::json m_json_read;
  std::ofstream m_ofs_results;

  std::ifstream filestr(argv[3]);
  filestr >> m_json_read;
  filestr.close();

  std::string od_model_name = std::string(m_json_read["model"]);
  std::string od_model_path = model_dir + "/" + od_model_name;

  int img_num = int(m_json_read["test_images"].size());

  float threshold = float(m_json_read["threshold"]);

  bool pass = true;
  static CVI_S32 vpssgrp_width = 1920;
  static CVI_S32 vpssgrp_height = 1080;
  cviai_handle_t handle = NULL;

  CVI_S32 ret = CVIAI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_CreateHandle2(&handle, 1, 0);
  if (ret != CVIAI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret =
      CVI_AI_OpenModel(handle, CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, od_model_path.c_str());
  if (ret != CVIAI_SUCCESS) {
    printf("Set model incarod failed with %#x!\n", ret);
    return ret;
  }

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path = image_dir + "/" + std::string(m_json_read["test_images"][img_idx]);
    int expected_res_num = int(m_json_read["expected_results"][img_idx][0]);

    VB_BLK blk1;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk1, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVIAI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face_meta;
    memset(&face_meta, 0, sizeof(cvai_face_t));
    dms_init(&face_meta);
    CVI_AI_IncarObjectDetection(handle, &frame, &face_meta);

    pass = (expected_res_num == int(face_meta.dms->dms_od.size));
    if (!pass) {
      printf("[%d] pass: %d; expected det nums: %d, result: %d\n", img_idx, pass, expected_res_num,
             face_meta.dms->dms_od.size);
    }

    for (uint32_t i = 0; i < face_meta.dms->dms_od.size; i++) {
      float expected_res_x1 = float(m_json_read["expected_results"][img_idx][1][i][0]);
      float expected_res_y1 = float(m_json_read["expected_results"][img_idx][1][i][1]);
      float expected_res_x2 = float(m_json_read["expected_results"][img_idx][1][i][2]);
      float expected_res_y2 = float(m_json_read["expected_results"][img_idx][1][i][3]);
      int expected_res_class = int(m_json_read["expected_results"][img_idx][1][i][4]);

      bool pass = (face_meta.dms->dms_od.info[i].classes == expected_res_class);
      pass &= (abs(face_meta.dms->dms_od.info[i].bbox.x1 - expected_res_x1) < threshold) &
              (abs(face_meta.dms->dms_od.info[i].bbox.y1 - expected_res_y1) < threshold) &
              (abs(face_meta.dms->dms_od.info[i].bbox.x2 - expected_res_x2) < threshold) &
              (abs(face_meta.dms->dms_od.info[i].bbox.y2 - expected_res_y2) < threshold);

      if (!pass) {
        printf(
            "[%d][%d] pass: %d, x1, y1, x2, y2, class, expected : [%f, %f, %f, %f %d], result : "
            "[%f, %f, %f, "
            "%f %d]\n",
            img_idx, i, pass, expected_res_x1, expected_res_y1, expected_res_x2, expected_res_y2,
            expected_res_class, face_meta.dms->dms_od.info[i].bbox.x1,
            face_meta.dms->dms_od.info[i].bbox.y1, face_meta.dms->dms_od.info[i].bbox.x2,
            face_meta.dms->dms_od.info[i].bbox.y2, face_meta.dms->dms_od.info[i].classes);
      }
    }
    CVI_AI_Free(&face_meta);
    CVI_VB_ReleaseBlock(blk1);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(handle);
  CVI_SYS_Exit();
  return pass ? CVIAI_SUCCESS : CVIAI_FAILURE;
}
