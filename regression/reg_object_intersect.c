#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

void bbox_setup(const float x1, const float y1, const float x2, const float y2, cvai_bbox_t *bbox) {
  bbox->x1 = x1;
  bbox->y1 = y1;
  bbox->x2 = x2;
  bbox->y2 = y2;
  bbox->score = 1.f;
}

int main(int argc, char *argv[]) {
  if (argc != 1) {
    printf("Usage: %s.\n", argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 20;
  const CVI_S32 vpssgrp_height = 20;
  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  // Init cviai fr service handle.
  cviai_objservice_handle_t obj_handle = NULL;
  ret = CVI_AI_OBJService_CreateHandle(&obj_handle, ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create obj service handle failed with %#x!\n", ret);
    return ret;
  }

  // Setup fake frame info
  VIDEO_FRAME_INFO_S frame;
  // Fake timestamp
  frame.stVFrame.u64PTS = 0;
  // Fake size
  frame.stVFrame.u32Width = 40;
  frame.stVFrame.u32Height = 50;
  // Setup polygon region
  printf("Testing polygon\n");
  cvai_pts_t pts;
  pts.size = 3;
  pts.x = (float *)malloc(pts.size * sizeof(float));
  pts.y = (float *)malloc(pts.size * sizeof(float));
  pts.x[0] = 0;
  pts.y[0] = 0;
  pts.x[1] = 6;
  pts.y[1] = 0;
  pts.x[2] = 0;
  pts.y[2] = 6;
  CVI_AI_OBJService_SetIntersect(obj_handle, &frame, &pts);
  CVI_AI_Free(&pts);
  // Setup fake paths
  const uint32_t input_length = 3;
  area_detect_pts_t *input = (area_detect_pts_t *)malloc(input_length * sizeof(area_detect_pts_t));
  input[0].unique_id = 12971892392312;
  input[1].unique_id = 14890129723233;
  input[2].unique_id = 23903582342343;
  const uint32_t frame_test = 2;
  float x[input_length][frame_test];
  x[0][0] = 1.f;
  x[1][0] = -6.f;
  x[2][0] = 1.f;
  x[0][1] = 6.f;
  x[1][1] = 2.f;
  x[2][1] = 1.f;
  float y[input_length][frame_test];
  y[0][0] = 1.f;
  y[1][0] = 12.f;
  y[2][0] = 1.f;
  y[0][1] = 6.f;
  y[1][1] = 2.f;
  y[2][1] = 1.f;

  // Test the result
  for (uint32_t i = 0; i < frame_test; i++) {
    for (uint32_t j = 0; j < input_length; j++) {
      input[j].x = x[j][i];
      input[j].y = y[j][i];
    }
    cvai_area_detect_e *status = NULL;
    CVI_AI_OBJService_DetectIntersect(obj_handle, &frame, input, input_length, &status);
    for (uint32_t j = 0; j < input_length; j++) {
      printf("[frame %u][id %lu](%f, %f) status %u \n", i, (long unsigned int)input[j].unique_id,
             input[j].x, input[j].y, status[j]);
    }
    frame.stVFrame.u64PTS += 33;
    free(status);
  }

  // Again setup line
  printf("Testing line\n");
  pts.size = 2;
  pts.x = (float *)malloc(pts.size * sizeof(float));
  pts.y = (float *)malloc(pts.size * sizeof(float));
  pts.x[0] = 3;
  pts.y[0] = 3;
  pts.x[1] = 3;
  pts.y[1] = 8;
  CVI_AI_OBJService_SetIntersect(obj_handle, &frame, &pts);
  CVI_AI_Free(&pts);
  // Setup fake paths
  input[0].unique_id = 12984014844833;
  input[1].unique_id = 34139846148383;
  input[2].unique_id = 28282090010294;
  x[0][0] = 3.f;
  x[1][0] = 0.f;
  x[2][0] = 3.f;
  x[0][1] = 4.f;
  x[1][1] = 6.f;
  x[2][1] = 3.f;
  y[0][0] = 4.f;
  y[1][0] = 5.f;
  y[2][0] = 10.f;
  y[0][1] = 4.f;
  y[1][1] = 5.f;
  y[2][1] = 10.f;
  // Test the result
  for (uint32_t i = 0; i < frame_test; i++) {
    for (uint32_t j = 0; j < input_length; j++) {
      input[j].x = x[j][i];
      input[j].y = y[j][i];
    }
    cvai_area_detect_e *status = NULL;
    CVI_AI_OBJService_DetectIntersect(obj_handle, &frame, input, input_length, &status);
    for (uint32_t j = 0; j < input_length; j++) {
      printf("[frame %u][id %lu](%f, %f) status %u \n", i, (long unsigned int)input[j].unique_id,
             input[j].x, input[j].y, status[j]);
    }
    frame.stVFrame.u64PTS += 33;
    free(status);
  }

  CVI_AI_OBJService_DestroyHandle(obj_handle);
  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
