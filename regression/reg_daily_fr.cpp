#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

typedef enum {
  FaceRecognition,
  FaceAttribute,
  MaskFR,
} ModelType;

typedef CVI_S32 (*InferenceFunc)(const cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_face_t *);

const float std_face_landmark_x[5] = {38.29459953, 73.53179932, 56.02519989, 41.54930115,
                                      70.72990036};
const float std_face_landmark_y[5] = {51.69630051, 51.50139999, 71.73660278, 92.3655014,
                                      92.20410156};

void init_face_meta(cvai_face_t &face_meta) {
  memset(&face_meta, 0, sizeof(cvai_face_t));
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
}

CVI_S32 compare_face(cviai_handle_t ai_handle, cviai_service_handle_t service_handle,
                     InferenceFunc inference, std::string image_path1, std::string image_path2,
                     float *score) {
  VB_BLK blk_fr;
  VIDEO_FRAME_INFO_S frame;
  CVI_S32 ret = CVI_AI_ReadImage(image_path1.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("failed to read image: %s\n", image_path1.c_str());
  }
  cvai_face_t face_meta1;
  init_face_meta(face_meta1);
  inference(ai_handle, &frame, &face_meta1);
  CVI_VB_ReleaseBlock(blk_fr);

  ret = CVI_AI_ReadImage(image_path2.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("failed to read image: %s\n", image_path2.c_str());
  }
  cvai_face_t face_meta2;
  init_face_meta(face_meta2);
  inference(ai_handle, &frame, &face_meta2);
  CVI_VB_ReleaseBlock(blk_fr);

  CVI_AI_Service_CalculateSimilarity(service_handle, &face_meta1.info[0].feature,
                                     &face_meta2.info[0].feature, score);

  CVI_AI_Free(&face_meta1);
  CVI_AI_Free(&face_meta2);
  return CVI_SUCCESS;
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
  cviai_service_handle_t service_handle = NULL;
  CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  CVI_AI_Service_CreateHandle(&service_handle, ai_handle);

  for (size_t test_idx = 0; test_idx < m_json_read.size(); test_idx++) {
    auto test_config = m_json_read[test_idx];
    std::string model_name = std::string(std::string(test_config["model_name"]).c_str());
    std::string model_path = model_dir + "/" + model_name;
    int model_type = test_config["model_type"];

    CVI_AI_SUPPORTED_MODEL_E model_id;
    InferenceFunc inference;
    switch (model_type) {
      case FaceRecognition: {
        model_id = CVI_AI_SUPPORTED_MODEL_FACERECOGNITION;
        inference = CVI_AI_FaceRecognition;
      } break;
      case FaceAttribute: {
        model_id = CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE;
        inference = CVI_AI_FaceAttribute;
      } break;
      case MaskFR: {
        model_id = CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION;
        inference = CVI_AI_MaskFaceRecognition;
      } break;
      default:
        printf("unsupported model type: %d\n", model_type);
        return CVI_FAILURE;
    }
    ret |= CVI_AI_SetModelPath(ai_handle, model_id, model_path.c_str());
    ret |= CVI_AI_OpenModel(ai_handle, model_id);
    if (ret != CVI_SUCCESS) {
      printf("Open model failed with %#x!\n", ret);
      return ret;
    }

    for (size_t pair_idx = 0; pair_idx < test_config["same_pairs"].size(); pair_idx++) {
      auto same_pair = test_config["same_pairs"][pair_idx];
      float expected_score = test_config["same_scores"][pair_idx];

      std::string image_path1 = image_dir + "/" + std::string(same_pair[0]);
      std::string image_path2 = image_dir + "/" + std::string(same_pair[1]);
      float score = 0.0;
      ret = compare_face(ai_handle, service_handle, inference, image_path1, image_path2, &score);
      if (ret != CVI_SUCCESS) {
        return CVI_FAILURE;
      }

      if (std::abs(score - expected_score) >= 0.1) {
        printf("FAILED! expect: %f, predict: %f, img1: %s, img2: %s\n", expected_score, score,
               image_path1.c_str(), image_path2.c_str());
        return CVI_FAILURE;
      }
    }

    for (size_t pair_idx = 0; pair_idx < test_config["diff_pairs"].size(); pair_idx++) {
      auto diff_pair = test_config["diff_pairs"][pair_idx];
      float expected_score = test_config["diff_scores"][pair_idx];

      std::string image_path1 = image_dir + "/" + std::string(diff_pair[0]);
      std::string image_path2 = image_dir + "/" + std::string(diff_pair[1]);
      float score = 0.0;
      ret = compare_face(ai_handle, service_handle, inference, image_path1, image_path2, &score);
      if (ret != CVI_SUCCESS) {
        return CVI_FAILURE;
      }

      if (std::abs(score - expected_score) >= 0.1) {
        printf("FAILED! expect: %f, predict: %f, img1: %s, img2: %s\n", expected_score, score,
               image_path1.c_str(), image_path2.c_str());
        return CVI_FAILURE;
      }
    }
  }
  printf("PASSED\n");

  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
  return CVI_SUCCESS;
}
