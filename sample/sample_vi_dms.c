#define LOG_TAG "SampleDMS"
#define LOG_LEVEL LOG_LEVEL_INFO

#include "middleware_utils.h"
#include "sample_log.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <core/utils/vpss_helper.h>
#include <cvi_comm.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include <cviai.h>
#include <rtsp.h>
#include <sample_comm.h>

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile bool bExit = false;

// smooth value structure
typedef struct {
  float mask_score;
  cvai_dms_t dms;
} SAMPLE_AI_SMOOTH_FACE_INFO_S;

// threshold structure
typedef struct {
  int fEyeWinTh;
  int fYawnWinTh;
  float fYawTh;
  float fPitchTh;
  float fEyeTh;
  float fSmoothEyeTh;
  float fYawnTh;
  float fMaskTh;
} SAMPLE_AI_THRESHOLD_S;

typedef enum {
  DROWSINESS,
  DISTRACTION,
  NORMAL,
} SAMPLE_AI_DMS_STATUS_E;

typedef struct {
  SAMPLE_AI_DMS_STATUS_E enOverallStaus;
  bool bHasMask;
  bool bYawning;
  bool bREyeClosing;
  char bLEyeClosing;
  cvai_pts_t stLandmarks;
  struct {
    float fYaw;
    float fPitch;
    float fRoll;
  } face_angle;
} SAMPLE_AI_DMS_RESULT_S;

SAMPLE_AI_DMS_RESULT_S g_stDMSResult;

MUTEXAUTOLOCK_INIT(ResultMutex);

void update_moving_avg(SAMPLE_AI_SMOOTH_FACE_INFO_S *pstSmoothInfo, cvai_face_t *pstFaceMeta) {
  // Yawn score
  pstSmoothInfo->dms.yawn_score =
      pstSmoothInfo->dms.yawn_score * 0.5 + pstFaceMeta->dms->yawn_score * 0.5;

  // Eye closing
  pstSmoothInfo->dms.reye_score =
      pstSmoothInfo->dms.reye_score * 0.5 + pstFaceMeta->dms->reye_score * 0.5;
  pstSmoothInfo->dms.leye_score =
      pstSmoothInfo->dms.leye_score * 0.5 + pstFaceMeta->dms->leye_score * 0.5;

  // Mask score
  pstSmoothInfo->mask_score =
      pstSmoothInfo->mask_score * 0.5 + pstFaceMeta->info[0].mask_score * 0.5;

  // Face angle
  pstSmoothInfo->dms.head_pose.yaw =
      pstSmoothInfo->dms.head_pose.yaw * 0.5 + pstFaceMeta->dms->head_pose.yaw * 0.5;
  pstSmoothInfo->dms.head_pose.pitch =
      pstSmoothInfo->dms.head_pose.pitch * 0.5 + pstFaceMeta->dms->head_pose.pitch * 0.5;
  pstSmoothInfo->dms.head_pose.roll =
      pstSmoothInfo->dms.head_pose.roll * 0.5 + pstFaceMeta->dms->head_pose.roll * 0.5;
}

void update_score_window(SAMPLE_AI_SMOOTH_FACE_INFO_S *pstSmoothInfo,
                         SAMPLE_AI_THRESHOLD_S *pstThresh, int *pEyeScoreWindow,
                         int *pYawnScoreWindow) {
  // Update Yawn window score
  if (pstSmoothInfo->mask_score < pstThresh->fMaskTh) {
    if (pstSmoothInfo->dms.yawn_score < pstThresh->fYawnTh) {
      if (*pYawnScoreWindow < 30) {
        *pYawnScoreWindow += 1;
      }
    } else {
      if (*pYawnScoreWindow > 0) {
        *pYawnScoreWindow -= 1;
      }
    }
  } else {
    if (*pYawnScoreWindow < 0) {
      *pYawnScoreWindow += 1;
    }
  }

  // Update Eye closing window score
  if (pstSmoothInfo->dms.reye_score + pstSmoothInfo->dms.leye_score > pstThresh->fEyeTh) {
    if (*pEyeScoreWindow < 30) {
      *pEyeScoreWindow += 1;
    }
  } else {
    if (*pEyeScoreWindow > 0) {
      *pEyeScoreWindow -= 1;
    }
  }
}

void update_status(SAMPLE_AI_SMOOTH_FACE_INFO_S *pstSmoothFaceInfo, cvai_face_t *pstFaceMeta,
                   SAMPLE_AI_THRESHOLD_S *pstThreshold, int eyeScoreWindow, int yawnScoreWindow,
                   SAMPLE_AI_DMS_RESULT_S *pstDMSResult) {
  pstDMSResult->bYawning = pstFaceMeta->dms->yawn_score >= pstThreshold->fYawnTh;
  pstDMSResult->bREyeClosing = pstFaceMeta->dms->reye_score < pstThreshold->fEyeTh;
  pstDMSResult->bLEyeClosing = pstFaceMeta->dms->leye_score < pstThreshold->fEyeTh;
  pstDMSResult->bHasMask = pstSmoothFaceInfo->mask_score >= pstThreshold->fMaskTh;
  pstDMSResult->face_angle.fYaw = pstFaceMeta->dms->head_pose.yaw;
  pstDMSResult->face_angle.fPitch = pstFaceMeta->dms->head_pose.pitch;
  pstDMSResult->face_angle.fRoll = pstFaceMeta->dms->head_pose.roll;
  free(pstDMSResult->stLandmarks.x);
  free(pstDMSResult->stLandmarks.y);

  pstDMSResult->stLandmarks.size = pstFaceMeta->dms->landmarks_106.size;
  pstDMSResult->stLandmarks.x = (float *)malloc(sizeof(float) * pstDMSResult->stLandmarks.size);
  memcpy(pstDMSResult->stLandmarks.x, pstFaceMeta->dms->landmarks_106.x,
         sizeof(float) * pstDMSResult->stLandmarks.size);

  pstDMSResult->stLandmarks.y = (float *)malloc(sizeof(float) * pstDMSResult->stLandmarks.size);
  memcpy(pstDMSResult->stLandmarks.y, pstFaceMeta->dms->landmarks_106.y,
         sizeof(float) * pstDMSResult->stLandmarks.size);

  if (fabs(pstSmoothFaceInfo->dms.head_pose.yaw) > pstThreshold->fYawTh ||
      fabs(pstSmoothFaceInfo->dms.head_pose.pitch) > pstThreshold->fPitchTh) {
    pstDMSResult->enOverallStaus = DISTRACTION;
  } else if (eyeScoreWindow < pstThreshold->fEyeWinTh ||
             yawnScoreWindow < pstThreshold->fYawnWinTh) {
    pstDMSResult->enOverallStaus = DROWSINESS;
  } else {
    pstDMSResult->enOverallStaus = NORMAL;
  }
}

/**
 * @brief Arguments for video encoder thread
 *
 */
typedef struct {
  SAMPLE_AI_MW_CONTEXT *pstMWContext;
  cviai_service_handle_t stServiceHandle;
} SAMPLE_AI_VENC_THREAD_ARG_S;

/**
 * @brief Arguments for ai thread
 *
 */
typedef struct {
  ODInferenceFunc inference_func;
  CVI_AI_SUPPORTED_MODEL_E enOdModelId;
  cviai_handle_t stAIHandle;
} SAMPLE_AI_AI_THREAD_ARG_S;

void *run_venc(void *args) {
  AI_LOGI("Enter encoder thread\n");
  SAMPLE_AI_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_AI_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  SAMPLE_AI_DMS_RESULT_S stDMSResult = {0};

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      AI_LOGE("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    {
      // Get detection result from global
      MutexAutoLock(ResultMutex, lock);
      memcpy(&stDMSResult, &g_stDMSResult, sizeof(SAMPLE_AI_DMS_RESULT_S));

      stDMSResult.stLandmarks.x = (float *)malloc(sizeof(float) * g_stDMSResult.stLandmarks.size);
      memcpy(stDMSResult.stLandmarks.x, g_stDMSResult.stLandmarks.x,
             sizeof(float) * stDMSResult.stLandmarks.size);

      stDMSResult.stLandmarks.y = (float *)malloc(sizeof(float) * g_stDMSResult.stLandmarks.size);
      memcpy(stDMSResult.stLandmarks.y, g_stDMSResult.stLandmarks.y,
             sizeof(float) * stDMSResult.stLandmarks.size);
    }

    if (stDMSResult.bYawning) {
      CVI_AI_Service_ObjectWriteText("Yawn: open", 1000, 190, &stFrame, -1, -1, -1);
    } else {
      CVI_AI_Service_ObjectWriteText("Yawn: close", 1000, 190, &stFrame, 255, 0, 255);
    }

    if (stDMSResult.bREyeClosing) {
      CVI_AI_Service_ObjectWriteText("Right Eye: close", 700, 100, &stFrame, -1, -1, -1);
    } else {
      CVI_AI_Service_ObjectWriteText("Right Eye: close", 700, 100, &stFrame, 255, 0, 255);
    }

    if (stDMSResult.bLEyeClosing) {
      CVI_AI_Service_ObjectWriteText("Right Eye: close", 1000, 100, &stFrame, -1, -1, -1);
    } else {
      CVI_AI_Service_ObjectWriteText("Right Eye: open", 1000, 100, &stFrame, 255, 0, 255);
    }

    if (stDMSResult.bHasMask) {
      CVI_AI_Service_ObjectWriteText("Mask: Yes", 1000, 150, &stFrame, -1, 0, -1);
    } else {
      CVI_AI_Service_ObjectWriteText("Mask: No", 1000, 150, &stFrame, 255, 0, 255);
    }

    CVI_AI_Service_FaceDrawPts(&(stDMSResult.stLandmarks), &stFrame);

    {
      char acAngleString[255];

      snprintf(acAngleString, sizeof(acAngleString), "Yaw: %.2f Pitch: %.2f Roll: %.2f",
               stDMSResult.face_angle.fYaw, stDMSResult.face_angle.fPitch,
               stDMSResult.face_angle.fRoll);

      CVI_AI_Service_ObjectWriteText(acAngleString, 700, 50, &stFrame, -1, -1,
                                     -1);  // per frame info
    }

    {
      char acOverallStatus[255];
      if (stDMSResult.enOverallStaus == DROWSINESS) {
        snprintf(acOverallStatus, sizeof(acOverallStatus), "Status: Drowsiness");
      } else if (stDMSResult.enOverallStaus == DISTRACTION) {
        snprintf(acOverallStatus, sizeof(acOverallStatus), "Status: Distraction");
      } else {
        snprintf(acOverallStatus, sizeof(acOverallStatus), "Status: Normal");
      }
      CVI_AI_Service_ObjectWriteText(acOverallStatus, 30, 70, &stFrame, -1, -1, -1);
    }

    if (s32Ret != CVIAI_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      AI_LOGE("Draw fame fail!, ret=%x\n", s32Ret);
      goto error;
    }

    s32Ret = SAMPLE_AI_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
  error:
    free(stDMSResult.stLandmarks.x);
    free(stDMSResult.stLandmarks.y);
    CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }
  AI_LOGI("Exit encoder thread\n");
  pthread_exit(NULL);
}

void *run_ai_thread(void *args) {
  AI_LOGI("Enter AI thread\n");
  SAMPLE_AI_AI_THREAD_ARG_S *pstAIArgs = (SAMPLE_AI_AI_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  cvai_face_t stFaceMeta = {0};
  SAMPLE_AI_DMS_RESULT_S stDMSResult = {0};
  SAMPLE_AI_SMOOTH_FACE_INFO_S stSmoothFaceInfo = {0};
  int eye_score_window = 0;   // 30 frames
  int yawn_score_window = 0;  // 30 frames

  SAMPLE_AI_THRESHOLD_S stThreshold = {.fEyeWinTh = 11,
                                       .fYawnWinTh = 11,
                                       .fYawTh = 0.25,
                                       .fPitchTh = 0.25,
                                       .fEyeTh = 0.45,
                                       .fSmoothEyeTh = 0.65 * 2,  // 0.65 is one eye
                                       .fYawnTh = 0.75,
                                       .fMaskTh = 0.5};

  CVI_S32 s32Ret;
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000);

    if (s32Ret != CVI_SUCCESS) {
      AI_LOGE("CVI_VPSS_GetChnFrame failed with %#x\n", s32Ret);
      goto get_frame_failed;
    }

    CVI_AI_RetinaFace(pstAIArgs->stAIHandle, &stFrame, &stFaceMeta);
    // Just calculate the first one
    if (stFaceMeta.size > 0) {
      // Detect phones, foods
      GOTO_IF_FAILED(CVI_AI_IncarObjectDetection(pstAIArgs->stAIHandle, &stFrame, &stFaceMeta),
                     s32Ret, inf_error);

      // Predict facial landmark
      GOTO_IF_FAILED(CVI_AI_FaceLandmarker(pstAIArgs->stAIHandle, &stFrame, &stFaceMeta), s32Ret,
                     inf_error);

      // Calculate face angle according to facial landmarks.
      GOTO_IF_FAILED(
          CVI_AI_Service_FaceAngle(&(stFaceMeta.dms->landmarks_5), &stFaceMeta.dms->head_pose),
          s32Ret, inf_error);

      // Predict mask detection
      GOTO_IF_FAILED(CVI_AI_MaskClassification(pstAIArgs->stAIHandle, &stFrame, &stFaceMeta),
                     s32Ret, inf_error);

      // Predict eye closing
      GOTO_IF_FAILED(CVI_AI_EyeClassification(pstAIArgs->stAIHandle, &stFrame, &stFaceMeta), s32Ret,
                     inf_error);

      if (stFaceMeta.info[0].mask_score < stThreshold.fMaskTh) {
        // Yawn classification
        GOTO_IF_FAILED(CVI_AI_YawnClassification(pstAIArgs->stAIHandle, &stFrame, &stFaceMeta),
                       s32Ret, inf_error);
      }

      update_moving_avg(&stSmoothFaceInfo, &stFaceMeta);
      update_score_window(&stSmoothFaceInfo, &stThreshold, &eye_score_window, &yawn_score_window);
      update_status(&stSmoothFaceInfo, &stFaceMeta, &stThreshold, eye_score_window,
                    yawn_score_window, &stDMSResult);
    }

    {
      // Copy dms results to global.
      MutexAutoLock(ResultMutex, lock);
      free(g_stDMSResult.stLandmarks.x);
      free(g_stDMSResult.stLandmarks.y);

      memcpy(&g_stDMSResult, &stDMSResult, sizeof(SAMPLE_AI_DMS_RESULT_S));

      g_stDMSResult.stLandmarks.x = (float *)malloc(sizeof(float) * g_stDMSResult.stLandmarks.size);
      memcpy(g_stDMSResult.stLandmarks.x, stDMSResult.stLandmarks.x,
             sizeof(float) * stDMSResult.stLandmarks.size);

      g_stDMSResult.stLandmarks.y = (float *)malloc(sizeof(float) * g_stDMSResult.stLandmarks.size);
      memcpy(g_stDMSResult.stLandmarks.y, stDMSResult.stLandmarks.y,
             sizeof(float) * stDMSResult.stLandmarks.size);
    }

  inf_error:
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
    CVI_AI_Free(&stFaceMeta);
    free(stDMSResult.stLandmarks.x);
    free(stDMSResult.stLandmarks.y);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }

  AI_LOGI("Exit AI thread\n");
  pthread_exit(NULL);
}

CVI_S32 get_middleware_config(SAMPLE_AI_MW_CONFIG_S *pstMWConfig) {
  // Video Pipeline of this sample:
  //                                                       +------+
  //                                    CHN0 (VBPool 0)    | VENC |--------> RTSP
  //  +----+      +----------------+---------------------> +------+
  //  | VI |----->| VPSS 0 (DEV 1) |            +-----------------------+
  //  +----+      +----------------+----------> | VPSS 1 (DEV 0) AI SDK |------------> AI model
  //                            CHN1 (VBPool 1) +-----------------------+  CHN0 (VBPool 2)

  // VI configuration
  //////////////////////////////////////////////////
  // Get VI configurations from ini file.
  CVI_S32 s32Ret = SAMPLE_AI_Get_VI_Config(&pstMWConfig->stViConfig);
  if (s32Ret != CVI_SUCCESS || pstMWConfig->stViConfig.s32WorkingViNum <= 0) {
    AI_LOGE("Failed to get senor infomation from ini file (/mnt/data/sensor_cfg.ini).\n");
    return -1;
  }

  // Get VI size
  PIC_SIZE_E enPicSize;
  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(pstMWConfig->stViConfig.astViInfo[0].stSnsInfo.enSnsType,
                                          &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("Cannot get senor size\n");
    return s32Ret;
  }

  SIZE_S stSensorSize;
  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSensorSize);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("Cannot get senor size\n");
    return s32Ret;
  }

  // Setup frame size of video encoder to 1080p
  SIZE_S stVencSize = {
      .u32Width = 1920,
      .u32Height = 1080,
  };

  // VBPool configurations
  //////////////////////////////////////////////////
  pstMWConfig->stVBPoolConfig.u32VBPoolCount = 3;

  // VBPool 0 for VI
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].enFormat = VI_PIXEL_FORMAT;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32BlkCount = 3;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32Height = stSensorSize.u32Height;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32Width = stSensorSize.u32Width;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].bBind = true;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32VpssChnBinding = VPSS_CHN0;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 1 for AI frame
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].enFormat = PIXEL_FORMAT_RGB_888;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 3;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32Height = stVencSize.u32Height;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32Width = stVencSize.u32Width;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].bBind = true;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 2 for AI preprocessing.
  // The input pixel format of AI SDK models is eighter RGB 888 or RGB 888 Planar.
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_RGB_888_PLANAR;
  // AI SDK use only 1 buffer at the same time.
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
  // Considering the maximum input size of object detection model is 1024x768, we set same size
  // here.
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[2].u32Height = 768;
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[2].u32Width = 1024;
  // Don't bind with VPSS here, AI SDK would bind this pool automatically when user assign this pool
  // through CVI_AI_SetVBPool.
  pstMWConfig->stVBPoolConfig.astVBPoolSetup[2].bBind = false;

  // VPSS configurations
  //////////////////////////////////////////////////

  // Create a VPSS Grp0 for main stream, video encoder, and AI frame.
  pstMWConfig->stVPSSPoolConfig.u32VpssGrpCount = 1;
  pstMWConfig->stVPSSPoolConfig.stVpssMode.aenInput[0] = VPSS_INPUT_MEM;
  pstMWConfig->stVPSSPoolConfig.stVpssMode.enMode = VPSS_MODE_DUAL;
  pstMWConfig->stVPSSPoolConfig.stVpssMode.ViPipe[0] = 0;
  pstMWConfig->stVPSSPoolConfig.stVpssMode.aenInput[1] = VPSS_INPUT_ISP;
  pstMWConfig->stVPSSPoolConfig.stVpssMode.ViPipe[1] = 0;

  SAMPLE_AI_VPSS_CONFIG_S *pstVpssConfig = &pstMWConfig->stVPSSPoolConfig.astVpssConfig[0];
  pstVpssConfig->bBindVI = true;

  // Assign device 1 to VPSS Grp0, because device1 has 3 outputs in dual mode.
  VPSS_GRP_DEFAULT_HELPER2(&pstVpssConfig->stVpssGrpAttr, stSensorSize.u32Width,
                           stSensorSize.u32Height, VI_PIXEL_FORMAT, 1);

  // Enable two channels for VENC and AI frame
  pstVpssConfig->u32ChnCount = 2;

  // Bind VPSS Grp0 Ch0 with VI
  pstVpssConfig->u32ChnBindVI = VPSS_CHN0;
  VPSS_CHN_DEFAULT_HELPER(&pstVpssConfig->astVpssChnAttr[0], stVencSize.u32Width,
                          stVencSize.u32Height, VI_PIXEL_FORMAT, true);
  VPSS_CHN_DEFAULT_HELPER(&pstVpssConfig->astVpssChnAttr[1], stVencSize.u32Width,
                          stVencSize.u32Height, PIXEL_FORMAT_RGB_888, true);

  // VENC
  //////////////////////////////////////////////////
  // Get default VENC configurations
  SAMPLE_AI_Get_Input_Config(&pstMWConfig->stVencConfig.stChnInputCfg);
  pstMWConfig->stVencConfig.u32FrameWidth = stVencSize.u32Width;
  pstMWConfig->stVencConfig.u32FrameHeight = stVencSize.u32Height;

  // RTSP
  //////////////////////////////////////////////////
  // Get default RTSP configurations
  SAMPLE_AI_Get_RTSP_Config(&pstMWConfig->stRTSPConfig.stRTSPConfig);

  return s32Ret;
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);
  AI_LOGI("handle signal, signo: %d\n", signo);
  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "\nUsage: %s DET_MODEL_NAME DET_MODEL_PATH POSE_MODEL_PATH\n\n"
        "\tDET_MODEL_NAME, person detection model name should be one of "
        "{mobiledetv2-person-vehicle, "
        "mobiledetv2-person-pets, "
        "mobiledetv2-coco80, "
        "mobiledetv2-pedestrian, "
        "yolov3}.\n"
        "\tDET_MODEL_PATH, detection cvimodel path.\n"
        "\tPOSE_MODEL_PATH, alpha pose cvimodel path.\n",
        argv[0]);
    return -1;
  }

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  //  Step 1: Initialize middleware stuff.
  ////////////////////////////////////////////////////

  // Get middleware configurations including VI, VB, VPSS
  SAMPLE_AI_MW_CONFIG_S stMWConfig = {0};
  CVI_S32 s32Ret = get_middleware_config(&stMWConfig);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("get middleware configuration failed! ret=%x\n", s32Ret);
    return -1;
  }

  // Initialize middleware.
  SAMPLE_AI_MW_CONTEXT stMWContext = {0};
  s32Ret = SAMPLE_AI_Init_WM(&stMWConfig, &stMWContext);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("init middleware failed! ret=%x\n", s32Ret);
    return -1;
  }

  // Step 2: Create and setup AI SDK
  ///////////////////////////////////////////////////

  // Create AI handle and assign VPSS Grp1 Device 0 to AI SDK. VPSS Grp1 is created
  // during initialization of AI SDK.
  cviai_handle_t stAIHandle = NULL;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&stAIHandle, 1, 0), s32Ret, create_ai_fail);

  // Assign VBPool ID 2 to the first VPSS in AI SDK.
  GOTO_IF_FAILED(CVI_AI_SetVBPool(stAIHandle, 0, 2), s32Ret, create_service_fail);

  CVI_AI_SetVpssTimeout(stAIHandle, 1000);

  cviai_service_handle_t stServiceHandle = NULL;
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&stServiceHandle, stAIHandle), s32Ret,
                 create_service_fail);

  // Step 3: Open and setup AI models
  ///////////////////////////////////////////////////
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]), s32Ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, argv[2]),
                 s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, argv[3]),
                 s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, argv[4]),
                 s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, argv[5]),
                 s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, argv[6]),
                 s32Ret, setup_ai_fail);

  // Step 4: Run models in thread.
  ///////////////////////////////////////////////////

  pthread_t stVencThread, stAIThread;
  SAMPLE_AI_VENC_THREAD_ARG_S venc_args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
  };

  SAMPLE_AI_AI_THREAD_ARG_S ai_args = {
      .stAIHandle = stAIHandle,
  };

  pthread_create(&stVencThread, NULL, run_venc, &venc_args);
  pthread_create(&stAIThread, NULL, run_ai_thread, &ai_args);

  // Thread for video encoder
  pthread_join(stVencThread, NULL);

  // Thread for AI inference
  pthread_join(stAIThread, NULL);

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(stServiceHandle);
create_service_fail:
  CVI_AI_DestroyHandle(stAIHandle);
create_ai_fail:
  SAMPLE_AI_Destroy_MW(&stMWContext);

  return 0;
}