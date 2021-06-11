#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

#define WRITE_RESULT_TO_FILE 0
#define DISTANCE_BIAS 0.05

float cosine_distance(const cvai_feature_t &features1, const cvai_feature_t &features2) {
  int32_t value1 = 0, value2 = 0, value3 = 0;
  for (uint32_t i = 0; i < features1.size; i++) {
    value1 += (short)features1.ptr[i] * features1.ptr[i];
    value2 += (short)features2.ptr[i] * features2.ptr[i];
    value3 += (short)features1.ptr[i] * features2.ptr[i];
  }
  return 1 - ((float)value3 / (sqrt((double)value1) * sqrt((double)value2)));
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

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("Set ReID model failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);

#if WRITE_RESULT_TO_FILE
  FILE *outFile;
  outFile = fopen("result_DailyRegression_ReID.txt", "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file.\n");
    exit(EXIT_FAILURE);
  }
  fprintf(outFile, "%d\n", img_num);
#endif

  cvai_object_t obj[2];
  bool pass = true;
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    for (int i = 0; i < 2; i++) {
      VB_BLK blk_fr;
      VIDEO_FRAME_INFO_S frame;

      std::string image_path =
          image_dir + "/" + std::string(m_json_read["reg_config"][0]["test_images"][img_idx][i]);
      // printf("[%d:%d] %s\n", img_idx, i, image_path.c_str());
      CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
      if (ret != CVI_SUCCESS) {
        printf("Read image failed with %#x!\n", ret);
        return ret;
      }
      memset(&obj[i], 0, sizeof(cvai_object_t));
      obj[i].size = 1;
      obj[i].info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * obj[i].size);
      memset(obj[i].info, 0, sizeof(cvai_object_info_t) * obj[i].size);
      obj[i].width = frame.stVFrame.u32Width;
      obj[i].height = frame.stVFrame.u32Height;
      obj[i].info[0].bbox.x1 = 0;
      obj[i].info[0].bbox.y1 = 0;
      obj[i].info[0].bbox.x2 = frame.stVFrame.u32Width - 1;
      obj[i].info[0].bbox.y2 = frame.stVFrame.u32Height - 1;
      obj[i].info[0].bbox.score = 1.0;
      obj[i].info[0].classes = 0;
      memset(&obj[i].info[0].feature, 0, sizeof(cvai_feature_t));
      CVI_AI_OSNet(ai_handle, &frame, &obj[i]);

      CVI_VB_ReleaseBlock(blk_fr);
    }
    float distance = cosine_distance(obj[0].info[0].feature, obj[1].info[0].feature);
    float expected_distance = float(m_json_read["reg_config"][0]["expected_results"][img_idx]);
    // printf("cos distance: %f (expected: %f)\n", distance, expected_distance);

    pass &= ABS(distance - expected_distance) < DISTANCE_BIAS;

#if WRITE_RESULT_TO_FILE
    fprintf(outFile, "%f\n", distance);
#endif

    for (int i = 0; i < 2; i++) {
      CVI_AI_Free(&obj[i]);
    }
  }
  printf("Regression Result: %s\n", (pass ? "PASS" : "FAILURE"));

#if WRITE_RESULT_TO_FILE
  fclose(outFile);
#endif

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();

  return pass ? CVI_SUCCESS : CVI_FAILURE;
}
