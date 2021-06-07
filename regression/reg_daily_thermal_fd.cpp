#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "json.hpp"

#define MATCH_IOU_THRESHOLD 0.95
#define MATCH_SCORE_DIFF 0.02

float iou(cvai_bbox_t &bbox1, cvai_bbox_t &bbox2) {
  float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
  float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
  float inter_x1 = MAX2(bbox1.x1, bbox2.x1);
  float inter_y1 = MAX2(bbox1.y1, bbox2.y1);
  float inter_x2 = MIN2(bbox1.x2, bbox2.x2);
  float inter_y2 = MIN2(bbox1.y2, bbox2.y2);
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
#if 0
  printf(" area1: %f,  area2: %f,  area_inter: %f,  iou: %f\n", 
    area1, area2, area_inter,
    area_inter / (area1 + area2 - area_inter));
#endif
  return area_inter / (area1 + area2 - area_inter);
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

  CVI_S32 ret = CVI_SUCCESS;
  CVI_S32 vpssgrp_width = 1280;
  CVI_S32 vpssgrp_height = 720;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t facelib_handle = NULL;
  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set model thermalface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, false);
  CVI_AI_SetModelThreshold(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, 0.5);

  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx]);
    // printf("[%d] %s\n", img_idx, image_path.c_str());

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_ThermalFace(facelib_handle, &frame, &face);
#if 0
    printf("find %u faces.\n", face.size);
    for (uint32_t j = 0; j < face.size; j++) {
      printf("face[%u]: %f (%f,%f,%f,%f)\n", j, face.info[j].bbox.score, 
      face.info[j].bbox.x1, face.info[j].bbox.y1, face.info[j].bbox.x2, face.info[j].bbox.y2);
    }
#endif

    uint32_t expected_bbox_num =
        uint32_t(m_json_read["reg_config"][0]["expected_results"][img_idx]["bbox_num"]);

    if (expected_bbox_num != face.size) {
      pass = false;
      CVI_AI_Free(&face);
      CVI_VB_ReleaseBlock(blk_fr);
      continue;
    }

    cvai_bbox_t *expected_result = new cvai_bbox_t[expected_bbox_num];
    for (uint32_t i = 0; i < expected_bbox_num; i++) {
      expected_result[i].score = float(
          m_json_read["reg_config"][0]["expected_results"][img_idx]["bbox_info"][(int)i]["score"]);
      expected_result[i].x1 =
          float(m_json_read["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["x1"]);
      expected_result[i].y1 =
          float(m_json_read["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["y1"]);
      expected_result[i].x2 =
          float(m_json_read["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["x2"]);
      expected_result[i].y2 =
          float(m_json_read["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["y2"]);
#if 0
      printf("(%u) %f (%f,%f,%f,%f)\n", i, expected_result[i].score, 
        expected_result[i].x1, expected_result[i].y1, 
        expected_result[i].x2, expected_result[i].y2);
#endif
    }
    bool *match_result = new bool[expected_bbox_num];
    for (uint32_t j = 0; j < face.size; j++) {
      bool is_match = false;
      for (uint32_t i = 0; i < expected_bbox_num; i++) {
        if (match_result[i]) continue;
        if (iou(face.info[j].bbox, expected_result[i]) < MATCH_IOU_THRESHOLD) continue;
        if (ABS(face.info[j].bbox.score - expected_result[i].score) < MATCH_SCORE_DIFF) {
          match_result[i] = true;
          is_match = true;
        }
        break;
      }
      if (!is_match) break;
    }
    for (uint32_t i = 0; i < expected_bbox_num; i++) {
      pass &= match_result[i];
    }

    delete[] expected_result;
    delete[] match_result;
    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
