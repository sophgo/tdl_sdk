#include "app/cviai_app.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ive/ive.h"

static volatile bool bExit = false;

int getNumDigits(uint64_t num) {
  int digits = 0;
  do {
    num /= 10;
    digits++;
  } while (num != 0);
  return digits;
}

char *uint64ToString(uint64_t number) {
  int n = getNumDigits(number);
  int i;
  char *numArray = calloc(n, sizeof(char));
  for (i = n - 1; i >= 0; --i, number /= 10) {
    numArray[i] = (number % 10) + '0';
  }
  return numArray;
}

char *floatToString(float number) {
  char *numArray = calloc(64, sizeof(char));
  sprintf(numArray, "%g", number);
  return numArray;
}

int get_alive_num(face_capture_t *face_cpt_info) {
  int counter = 0;
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE) {
      counter += 1;
    }
  }
  return counter;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf(
        "Usage: %s <face_detection_model_path>\n"
        "          <face_attribute_model_path>\n"
        "          <face_quality_model_path>\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;

  CVI_S32 voType = atoi(argv[4]);

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, PIXEL_FORMAT_RGB_888, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  cviai_app_handle_t app_handle = NULL;

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);

  ret |= CVI_AI_Service_EnableTPUDraw(service_handle, true);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, argv[2]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, argv[3]);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, false);
  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle, false);
#if 1
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  ds_conf.ktracker_conf.max_unmatched_num = 10;
  ds_conf.ktracker_conf.accreditation_threshold = 10;
  ds_conf.ktracker_conf.P_std_beta[2] = 0.1;
  ds_conf.ktracker_conf.P_std_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.Q_std_beta[2] = 0.1;
  ds_conf.kfilter_conf.Q_std_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.R_std_beta[2] = 0.1;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);
#endif

  ret = CVI_AI_APP_CreateHandle(&app_handle, ai_handle);
  CVI_AI_APP_FaceCapture_Init(app_handle);

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  size_t counter = 0;
  while (bExit == false) {
    counter += 1;
    printf("\nGet Frame %zu\n", counter);

    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    int trk_num = get_alive_num(app_handle->face_cpt_info);
    printf("ALIVE face num = %d\n", trk_num);

    CVI_AI_APP_FaceCapture_Run(app_handle, &stfdFrame);

    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                    vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      cvai_face_t *face_meta = &app_handle->face_cpt_info->last_faces;
      CVI_AI_Service_FaceDrawRect(service_handle, face_meta, &stVOFrame, false,
                                  CVI_AI_Service_GetDefaultBrush());
      for (uint32_t j = 0; j < face_meta->size; j++) {
        char *id_num = uint64ToString(face_meta->info[j].unique_id);
        CVI_AI_Service_ObjectWriteText(id_num, face_meta->info[j].bbox.x1,
                                       face_meta->info[j].bbox.y1, &stVOFrame, -1, -1, -1);
        free(id_num);
        char *fq_num = floatToString(face_meta->info[j].face_quality);
        CVI_AI_Service_ObjectWriteText(fq_num, face_meta->info[j].bbox.x1,
                                       face_meta->info[j].bbox.y1 + 30, &stVOFrame, -1, -1, -1);
        free(fq_num);
      }

      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }
  }
  CVI_AI_APP_DestroyHandle(app_handle);
  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}
