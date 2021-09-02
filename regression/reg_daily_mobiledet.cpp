#include <fstream>
#include <string>
#include <unordered_map>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *);

static const std::unordered_map<std::string, std::pair<CVI_AI_SUPPORTED_MODEL_E, InferenceFunc>>
    MODEL_MAP = {
        {"mobiledetv2-d0-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, CVI_AI_MobileDetV2_D0}},
        {"mobiledetv2-d0.cvimodel", {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, CVI_AI_MobileDetV2_D0}},
        {"mobiledetv2-d1-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1, CVI_AI_MobileDetV2_D1}},
        {"mobiledetv2-d1.cvimodel", {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1, CVI_AI_MobileDetV2_D1}},
        {"mobiledetv2-d2-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2, CVI_AI_MobileDetV2_D2}},
        {"mobiledetv2-d2.cvimodel", {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2, CVI_AI_MobileDetV2_D2}},
        {"mobiledetv2-lite-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE, CVI_AI_MobileDetV2_Lite}},
        {"mobiledetv2-lite-person-pets-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS,
          CVI_AI_MobileDetV2_Lite_Person_Pets}},
        {"mobiledetv2-lite-person-pets.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS,
          CVI_AI_MobileDetV2_Lite_Person_Pets}},
        {"mobiledetv2-lite.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE, CVI_AI_MobileDetV2_Lite}},
        {"mobiledetv2-pedestrian-d0-ls-640.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, CVI_AI_MobileDetV2_Pedestrian_D0}},
        {"mobiledetv2-pedestrian-d0-ls-768.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, CVI_AI_MobileDetV2_Pedestrian_D0}},
        {"mobiledetv2-pedestrian-d0-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, CVI_AI_MobileDetV2_Pedestrian_D0}},
        {"mobiledetv2-pedestrian-d0.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, CVI_AI_MobileDetV2_Pedestrian_D0}},
        {"mobiledetv2-pedestrian-d1-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, CVI_AI_MobileDetV2_Pedestrian_D0}},
        {"mobiledetv2-pedestrian-d1.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0, CVI_AI_MobileDetV2_Pedestrian_D0}},
        {"mobiledetv2-vehicle-d0-ls.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, CVI_AI_MobileDetV2_Vehicle_D0}},
        {"mobiledetv2-vehicle-d0.cvimodel",
         {CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, CVI_AI_MobileDetV2_Vehicle_D0}},
};

const float bbox_threhold = 0.90;
const float score_threshold = 0.1;

float iou(cvai_bbox_t &bbox1, cvai_bbox_t &bbox2) {
  float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
  float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
  float inter_x1 = MAX2(bbox1.x1, bbox2.x1);
  float inter_y1 = MAX2(bbox1.y1, bbox2.y1);
  float inter_x2 = MIN2(bbox1.x2, bbox2.x2);
  float inter_y2 = MIN2(bbox1.y2, bbox2.y2);
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
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

  for (size_t test_index = 0; test_index < m_json_read.size(); test_index++) {
    std::string model_name = std::string(m_json_read[test_index]["model_name"]);
    printf("model name=%s\n", model_name.c_str());
    std::string model_path = model_dir + "/" + model_name;
    printf("model_path: %s\n", model_path.c_str());

    auto iter = MODEL_MAP.find(model_name);
    if (iter == MODEL_MAP.end()) {
      printf("Unknown model from json file: %s\n", model_name.c_str());
      return CVI_FAILURE;
    }

    CVI_AI_SUPPORTED_MODEL_E model_index = iter->second.first;
    InferenceFunc inference = iter->second.second;

    ret = CVI_AI_SetModelPath(handle, model_index, model_path.c_str());
    if (ret != CVI_SUCCESS) {
      printf("Set model path failed with %#x!\n", ret);
      return ret;
    }

    CVI_AI_SetSkipVpssPreprocess(handle, model_index, false);
    CVI_AI_SetModelThreshold(handle, model_index, 0.6);
    auto results = m_json_read[test_index]["results"];

    for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
      std::string image_path = image_dir + "/" + iter.key();

      VB_BLK blk1;
      VIDEO_FRAME_INFO_S frame;
      CVI_S32 ret = CVI_AI_ReadImage(image_path.c_str(), &blk1, &frame, PIXEL_FORMAT_BGR_888);
      if (ret != CVI_SUCCESS) {
        printf("Read image failed with %#x!\n", ret);
        return ret;
      }

      cvai_object_t obj_meta;
      memset(&obj_meta, 0, sizeof(cvai_object_t));

      if (inference(handle, &frame, &obj_meta) != CVI_SUCCESS) {
        printf("failed to inference\n");
        return CVI_FAILURE;
      }

      bool pass = true;

      auto expected_dets = iter.value();

      if (obj_meta.size != expected_dets.size()) {
        printf("[%s] num dets not equal!, expected: %u, actual: %zu\n", image_path.c_str(),
               obj_meta.size, expected_dets.size());
        pass = false;
      }

      for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
        auto bbox = expected_dets[det_index]["bbox"];
        int catId = int(expected_dets[det_index]["category_id"]) - 1;

        cvai_bbox_t expected_bbox = {
            .x1 = float(bbox[0]),
            .y1 = float(bbox[1]),
            .x2 = float(bbox[2]) + float(bbox[0]),
            .y2 = float(bbox[3]) + float(bbox[1]),
            .score = float(expected_dets[det_index]["score"]),
        };

        pass = false;
        for (uint32_t actual_det_index = 0; actual_det_index < obj_meta.size; actual_det_index++) {
          if (obj_meta.info[actual_det_index].classes == catId) {
            pass = iou(obj_meta.info[actual_det_index].bbox, expected_bbox) >= bbox_threhold &&
                   abs(obj_meta.info[actual_det_index].bbox.score - expected_bbox.score) <
                       score_threshold;
            if (pass) {
              break;
            }
          }
        }

        printf("img=%s, cat=%d, score=%f, bbox={%f, %f, %f, %f}\n", iter.key().c_str(), catId,
               expected_bbox.score, expected_bbox.x1, expected_bbox.x2, expected_bbox.y1,
               expected_bbox.y2);

        if (!pass) {
          printf("detections aren't matched: %s\n", image_path.c_str());
          for (uint32_t actual_det_index = 0; actual_det_index < obj_meta.size;
               actual_det_index++) {
            printf(
                "predict img=%s, cat=%d, score=%f, bbox={%f, %f, %f, %f}\n", iter.key().c_str(),
                obj_meta.info[actual_det_index].classes, obj_meta.info[actual_det_index].bbox.score,
                obj_meta.info[actual_det_index].bbox.x1, obj_meta.info[actual_det_index].bbox.x2,
                obj_meta.info[actual_det_index].bbox.y1, obj_meta.info[actual_det_index].bbox.y2);
          }
          return CVI_FAILURE;
        }
      }

      CVI_AI_Free(&obj_meta);
      CVI_VB_ReleaseBlock(blk1);
    }
    CVI_AI_CloseAllModel(handle);
  }

  CVI_AI_DestroyHandle(handle);
  CVI_SYS_Exit();
  printf("retinaface regression result: all pass\n");
  return CVI_SUCCESS;
}
