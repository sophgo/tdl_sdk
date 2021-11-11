#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile bool bExit = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4 && argc != 5) {
    printf(
        "Usage: %s <model_name> <model_path> <video output> <threshold>.\n"
        "\t model_name: detection model name should be one of {mobiledetv2-person-vehicle, "
        "mobiledetv2-person-pets, "
        "mobiledetv2-coco80, "
        "mobiledetv2-vehicle, "
        "mobiledetv2-pedestrian, "
        "yolov3}\n"
        "\t video output, 0: disable, 1: output to panel, 2: output through rtsp\n"
        "\t threshold (optional): threshold for detection model\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[3]);

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_pd_model_info(argv[1], &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1920, .u32Height = 1080};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, VI_PIXEL_FORMAT, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), s32Ret,
                 create_service_fail);

  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, od_model_id, argv[2]), s32Ret, setup_ai_fail);

  if (argc == 5) {
    float threshold = atof(argv[4]);
    if (threshold < 0.0 || threshold > 1.0) {
      printf("wrong threshold value: %f\n", threshold);
      return CVI_FAILURE;
    } else {
      printf("set threshold to %f\n", threshold);
    }
    GOTO_IF_FAILED(CVI_AI_SetModelThreshold(ai_handle, od_model_id, threshold), s32Ret,
                   setup_ai_fail);
  }

  GOTO_IF_FAILED(CVI_AI_SelectDetectClass(ai_handle, od_model_id, 1, CVI_AI_DET_TYPE_PERSON),
                 s32Ret, setup_ai_fail);

  //========================================
  // setting intrusion detection

  float r0[2][8] = {{0, 50, 0, 100, 200, 150, 200, 100}, {0, 100, 200, 150, 200, 100, 0, 50}};
  float r1[2][5] = {{380, 560, 500, 320, 260}, {160, 250, 500, 580, 220}};
  float r2[2][4] = {{780, 880, 840, 675}, {400, 420, 620, 580}};

  cvai_pts_t test_region_0;
  cvai_pts_t test_region_1;
  cvai_pts_t test_region_2;

  test_region_0.size = (uint32_t)sizeof(r0) / (sizeof(float) * 2);
  test_region_0.x = malloc(sizeof(float) * test_region_0.size);
  test_region_0.y = malloc(sizeof(float) * test_region_0.size);
  memcpy(test_region_0.x, r0[0], sizeof(float) * test_region_0.size);
  memcpy(test_region_0.y, r0[1], sizeof(float) * test_region_0.size);

  test_region_1.size = (uint32_t)sizeof(r1) / (sizeof(float) * 2);
  test_region_1.x = malloc(sizeof(float) * test_region_1.size);
  test_region_1.y = malloc(sizeof(float) * test_region_1.size);
  memcpy(test_region_1.x, r1[0], sizeof(float) * test_region_1.size);
  memcpy(test_region_1.y, r1[1], sizeof(float) * test_region_1.size);

  test_region_2.size = (uint32_t)sizeof(r2) / (sizeof(float) * 2);
  test_region_2.x = malloc(sizeof(float) * test_region_2.size);
  test_region_2.y = malloc(sizeof(float) * test_region_2.size);
  memcpy(test_region_2.x, r2[0], sizeof(float) * test_region_2.size);
  memcpy(test_region_2.y, r2[1], sizeof(float) * test_region_2.size);

  CVI_AI_Service_Polygon_SetTarget(service_handle, &test_region_0);
  // CVI_AI_Service_Polygon_CleanAll(service_handle);
  CVI_AI_Service_Polygon_SetTarget(service_handle, &test_region_1);
  CVI_AI_Service_Polygon_SetTarget(service_handle, &test_region_2);

  /* get the vertices of convex */
  cvai_pts_t **convex_pts = NULL;
  uint32_t convex_num;
  CVI_AI_Service_Polygon_GetTarget(service_handle, &convex_pts, &convex_num);

  //========================================
  cvai_service_brush_t region_brush = CVI_AI_Service_GetDefaultBrush();
  region_brush.color.r = 0;
  region_brush.color.g = 255;
  region_brush.color.b = 255;
  //========================================

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  cvai_object_t obj_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  uint32_t frame_counter = 0;
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }
    frame_counter += 1;

    inference(ai_handle, &stfdFrame, &obj_meta);
    printf("nums of object %u\n", obj_meta.size);

    int s32Ret = CVI_SUCCESS;
    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      printf("Frame [%u]\n", frame_counter);
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                    vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
      float x_scale = (float)stVOFrame.stVFrame.u32Width / obj_meta.width;
      float y_scale = (float)stVOFrame.stVFrame.u32Height / obj_meta.height;

      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      // CVI_AI_Service_DrawPolygon(service_handle, &stVOFrame, &test_region_1, region_brush);
      for (uint32_t i = 0; i < convex_num; i++) {
        CVI_AI_Service_DrawPolygon(service_handle, &stVOFrame, convex_pts[i], region_brush);
      }
      CVI_AI_Service_ObjectDrawRect(service_handle, &obj_meta, &stVOFrame, true,
                                    CVI_AI_Service_GetDefaultBrush());
      for (uint32_t i = 0; i < obj_meta.size; i++) {
        bool is_intrusion;
        cvai_bbox_t t_bbox = obj_meta.info[i].bbox;
        t_bbox.x1 *= x_scale;
        t_bbox.y1 *= y_scale;
        t_bbox.x2 *= x_scale;
        t_bbox.y2 *= y_scale;
        CVI_AI_Service_Polygon_Intersect(service_handle, &t_bbox, &is_intrusion);
        if (is_intrusion) {
          printf("[%u] intrusion! (%.1f,%.1f,%.1f,%.1f)\n", i, t_bbox.x1, t_bbox.y1, t_bbox.x2,
                 t_bbox.y2);
          CVI_AI_Service_ObjectWriteText("Intrusion", t_bbox.x1, t_bbox.y1, &stVOFrame, 1, 0, 0);
        }
      }
      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
        break;
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&obj_meta);
  }

  CVI_AI_Free(&test_region_0);
  CVI_AI_Free(&test_region_1);
  CVI_AI_Free(&test_region_2);

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(service_handle);
create_service_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}