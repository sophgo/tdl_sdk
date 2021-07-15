#include <dirent.h>
#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

float sqrt3(const float x) {
  union {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
  return u.x;
}

float compute_nme(cvai_pts_t *labels, cvai_pts_t *pts) {
  float sum = 0.0;

  for (int i = 0; i < 5; ++i) {
    float _dist = 0.0;
    _dist += (float)((labels->x[i] - pts->x[i]) * (labels->x[i] - pts->x[i]));
    _dist += (float)((labels->y[i] - pts->y[i]) * (labels->y[i] - pts->y[i]));
    sum += sqrt3(_dist);
  }
  float _nme = sum / 5;
  float dist = sqrt3((float)((labels->x[0] - labels->x[1]) * (labels->x[0] - labels->x[1]) +
                             (labels->y[0] - labels->y[1]) * (labels->y[0] - labels->y[1])));
  _nme /= dist;
  return _nme;
}

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

  std::string model_name = std::string(m_json_read["model"]);
  std::string model_path = model_dir + "/" + model_name;
  int img_num = int(m_json_read["image_num"]);
  float nme_threshold = float(m_json_read["nme_threshold"]);

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER,
                            model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  float nme = 0.0;
  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    float landmark[10];
    float bbox[4];
    std::string image_path = image_dir + "/" + std::string(m_json_read["test_images"][img_idx]);
    for (int i = 0; i < 4; ++i) {
      bbox[i] = float(m_json_read["expected_results"][img_idx][i]);
    }
    for (int i = 0; i < 10; ++i) {
      landmark[i] = float(m_json_read["expected_results"][img_idx][i + 4]);
    }
    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      continue;
    }

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    face.size = 1;
    face.width = frame.stVFrame.u32Width;
    face.height = frame.stVFrame.u32Height;
    face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * face.size);
    memset(face.info, 0, sizeof(cvai_face_info_t) * face.size);
    face.info[0].bbox.x1 = bbox[0];
    face.info[0].bbox.y1 = bbox[2];
    face.info[0].bbox.x2 = bbox[1];
    face.info[0].bbox.y2 = bbox[3];
    face.info[0].pts.size = 5;
    face.info[0].pts.x = (float *)malloc(sizeof(float) * face.info[0].pts.size);
    face.info[0].pts.y = (float *)malloc(sizeof(float) * face.info[0].pts.size);
    for (int i = 0; i < 5; ++i) {
      face.info[0].pts.x[i] = landmark[i];
      face.info[0].pts.y[i] = landmark[i + 1];
    }
    face.dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));
    face.dms->dms_od.info = NULL;
    CVI_AI_FaceLandmarker(facelib_handle, &frame, &face);
    nme += compute_nme(&(face.info[0].pts), &(face.dms->landmarks_5));
    CVI_AI_FreeDMS(face.dms);
    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  pass &= nme < nme_threshold;
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();

  return pass ? CVI_SUCCESS : CVI_FAILURE;
}
