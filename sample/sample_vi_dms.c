#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cviai.h"
#include "ive/ive.h"

#include <cvi_ae.h>
#include <cvi_ae_comm.h>
#include <cvi_awb_comm.h>
#include <cvi_buffer.h>
#include <cvi_comm_isp.h>
#include <cvi_comm_vpss.h>
#include <cvi_isp.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_venc.h>
#include <cvi_vi.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include "acodec.h"
#include "cvi_audio.h"
#include "sample_comm.h"

#include "vi_vo_utils.h"
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef struct {
  float mask_score;
  cvai_dms_t dms;
} SMOOTH_FACE_INFO;

static volatile bool bExit = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}
//======== VIDEO =======

bool gRun = true;
SAMPLE_VI_CONFIG_S stViConfig;
char codec[] = "h264";

int main(int argc, char **argv) {
  if (argc != 8) {
    printf(
        "Usage: %s <retina_model_path> <mask_classification_model> <eye_classification_model> "
        "<yawn_classification_model> <face_landmark_model> <in_car_od_model> <video output>.\n"
        "\tretina_model_path, path to retinaface model\n"
        "\tmask_classification_model, path to mask classification model\n"
        "\teye_classification_model, path to eye classification model\n"
        "\tvideo output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVI_FAILURE;
  }

  // VIDEO
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 voType = atoi(argv[7]);

  CVI_S32 s32Ret = CVI_SUCCESS;
  //****************************************************************
  // Init VI, VO, Vpss
  CVI_U32 DevNum = 0;
  VI_PIPE ViPipe = 0;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_CHN VpssChnVO = VPSS_CHN2;
  CVI_S32 GrpWidth = 1920;
  CVI_S32 GrpHeight = 1080;

  SAMPLE_VI_CONFIG_S stViConfig;

  s32Ret = InitVI(&stViConfig, &DevNum);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video input failed with %d\n", s32Ret);
    return s32Ret;
  }
  if (ViPipe >= DevNum) {
    printf("Not enough devices. Found %u, required index %u.\n", DevNum, ViPipe);
    return CVI_FAILURE;
  }

  const CVI_U32 voWidth = 1280;
  const CVI_U32 voHeight = 720;

  OutputContext outputContext = {0};
  if (voType) {
    OutputType outputType = voType == 1 ? OUTPUT_TYPE_PANEL : OUTPUT_TYPE_RTSP;
    s32Ret = InitOutput(outputType, voWidth, voHeight, &outputContext);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
      return s32Ret;
    }
  }

  s32Ret = InitVPSS(VpssGrp, VpssChn, VpssChnVO, GrpWidth, GrpHeight, voWidth, voHeight, ViPipe,
                    voType != 0);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }

  cviai_handle_t facelib_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&facelib_handle, 1);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, argv[2]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, argv[3]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, argv[4]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, argv[5]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, argv[6]);
  if (ret != CVI_SUCCESS) {
    printf("Set model failed with %#x!\n", ret);
    return ret;
  }
  // Do vpss frame transform in retina face
  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  {
    CVI_S32 s32Ret = CVI_SUCCESS;
    char status_n[256];
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    SMOOTH_FACE_INFO smooth_face_info;
    memset(&smooth_face_info, 0, sizeof(smooth_face_info));
    bool mask_flag = false;
    int start_count = 0;
    int eye_score_window = 0;   // 30 frames
    int yawn_score_window = 0;  // 30 frames

    float yaw_th = 0.5;
    float pitch_th = 0.5;
    float yawn_th = 0.75;
    float eye_th = 0.65;
    int eye_win_th = 11;
    int yawn_win_th = 11;

    while (gRun) {
      if (bExit) break;
      VIDEO_FRAME_INFO_S stVencFrame, stVOFrame;
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVencFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        SAMPLE_PRT("CVI_VPSS_GetChnFrame grp0 chn0 failed with %#x\n", s32Ret);
        continue;
      }
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_RetinaFace(facelib_handle, &stVencFrame, &face);
      // Just calculate the first one
      if (face.size > 0) {
        face.dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));

        // calculate od
        CVI_AI_IncarObjectDetection(facelib_handle, &stVencFrame, &face);
        CVI_AI_Service_Incar_ObjectDrawRect((&face.dms->dms_od), &stVOFrame, true);

        // calculate landmark
        CVI_AI_FaceLandmarker(facelib_handle, &stVencFrame, &face);
        CVI_AI_Service_FaceDrawLandmarks(&(face.dms->landmarks_106), &stVOFrame);

        // face angle
        CVI_AI_Service_FaceAngle(&(face.dms->landmarks_5), &face.dms->head_pose);
        memset(status_n, 0, sizeof(status_n));
        sprintf(status_n, "Yaw: %f Pitch: %f Roll: %f", face.dms->head_pose.yaw,
                face.dms->head_pose.pitch, face.dms->head_pose.roll);
        CVI_AI_Service_ObjectWriteText(status_n, 800, 600, &stVOFrame);  // per frame info
        // calculate smooth
        smooth_face_info.dms.head_pose.yaw =
            smooth_face_info.dms.head_pose.yaw * 0.5 + face.dms->head_pose.yaw * 0.5;
        smooth_face_info.dms.head_pose.pitch =
            smooth_face_info.dms.head_pose.pitch * 0.5 + face.dms->head_pose.pitch * 0.5;
        smooth_face_info.dms.head_pose.roll =
            smooth_face_info.dms.head_pose.roll * 0.5 + face.dms->head_pose.roll * 0.5;

        // mask detection
        CVI_AI_MaskClassification(facelib_handle, &stVencFrame, &face);

        smooth_face_info.mask_score =
            smooth_face_info.mask_score * 0.5 + face.info[0].mask_score * 0.5;

        // show the smooth mask score
        memset(status_n, 0, sizeof(status_n));
        if (smooth_face_info.mask_score < 0.5) {
          strcpy(status_n, "Status: no mask");
          mask_flag = true;
        } else {
          strcpy(status_n, "Status: mask");
          mask_flag = false;
        }
        CVI_AI_Service_ObjectWriteText(status_n, 800, 550, &stVOFrame);

        // eye classification
        CVI_AI_EyeClassification(facelib_handle, &stVencFrame, &face);
        // calculate smooth eye score

        smooth_face_info.dms.reye_score =
            smooth_face_info.dms.reye_score * 0.5 + face.dms->reye_score * 0.5;
        smooth_face_info.dms.leye_score =
            smooth_face_info.dms.leye_score * 0.5 + face.dms->leye_score * 0.5;

        if (smooth_face_info.dms.reye_score + smooth_face_info.dms.leye_score > 1.3) {
          if (eye_score_window < 30) eye_score_window += 1;
        } else {
          if (eye_score_window > 0) eye_score_window -= 1;
        }

        // show the eye score per frame
        memset(status_n, 0, sizeof(status_n));
        if (face.dms->reye_score < eye_th) {
          strcpy(status_n, "Right Eye: close");
        } else {
          strcpy(status_n, "Right Eye: open");
        }
        CVI_AI_Service_ObjectWriteText(status_n, 800, 650, &stVOFrame);

        memset(status_n, 0, sizeof(status_n));
        if (face.dms->leye_score < eye_th) {
          strcpy(status_n, "Left Eye: close");
        } else {
          strcpy(status_n, "Left Eye: open");
        }
        CVI_AI_Service_ObjectWriteText(status_n, 1100, 650, &stVOFrame);
#if 0
        // show smooth eye score
        memset(status_n, 0, sizeof(status_n));
        if (smooth_face_info.dms.reye_score < eye_th) {
          strcpy(status_n, "Right Eye: close");
        } else {
          strcpy(status_n, "Right Eye: open");
        }
        CVI_AI_Service_ObjectWriteText(status_n, 30, 170, &stVOFrame, true);

        memset(status_n, 0, sizeof(status_n));
        if (smooth_face_info.dms.leye_score < eye_th) {
          strcpy(status_n, "Left Eye: close");
        } else {
          strcpy(status_n, "Left Eye: open");
        }
        CVI_AI_Service_ObjectWriteText(status_n, 300, 170, &stVOFrame, true);
#endif
        if (mask_flag) {
          // Yawn classification
          CVI_AI_YawnClassification(facelib_handle, &stVencFrame, &face);
          // calculate smooth yawn score
          smooth_face_info.dms.yawn_score =
              smooth_face_info.dms.yawn_score * 0.5 + face.dms->yawn_score * 0.5;

          if (smooth_face_info.dms.yawn_score < yawn_th) {
            if (yawn_score_window < 30) yawn_score_window += 1;
          } else {
            if (yawn_score_window > 0) yawn_score_window -= 1;
          }
          memset(status_n, 0, sizeof(status_n));
          if (face.dms->yawn_score < yawn_th) {
            strcpy(status_n, "Yawn: close");
          } else {
            strcpy(status_n, "Yawn: open");
          }
          CVI_AI_Service_ObjectWriteText(status_n, 800, 690, &stVOFrame);
        } else {
          if (yawn_score_window < 0) yawn_score_window += 1;
        }
#if 0
          memset(status_n, 0, sizeof(status_n));
          if (smooth_face_info.dms.yawn_score < 0.6) {
            strcpy(status_n, "Yawn: close");
          } else {
            strcpy(status_n, "Yawn: open");
          }
          CVI_AI_Service_ObjectWriteText(status_n, 30, 220, &stVOFrame);
#endif
        memset(status_n, 0, sizeof(status_n));
        if (start_count > 90) {
          if (eye_score_window < eye_win_th || yawn_score_window < yawn_win_th) {
            strcpy(status_n, "Status: Drowsiness");
            CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame);
          } else if ((smooth_face_info.dms.head_pose.yaw < yaw_th &&
                      smooth_face_info.dms.head_pose.yaw > -yaw_th &&
                      smooth_face_info.dms.head_pose.pitch < pitch_th &&
                      smooth_face_info.dms.head_pose.pitch > -pitch_th) ||
                     smooth_face_info.dms.yawn_score > yawn_th) {
            strcpy(status_n, "Status: Normal");
            CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame);
          } else {
            strcpy(status_n, "Status: Distraction");
            CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame);
          }
        } else {
          start_count++;
        }
        CVI_AI_FreeDMS(face.dms);

      } else {
        memset(status_n, 0, sizeof(status_n));
        sprintf(status_n, "No face");
        CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame);
      }

      s32Ret = SendOutputFrame(&stVOFrame, &outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
        break;
      }

      if (CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVencFrame) != 0) {
        SAMPLE_PRT("CVI_VPSS_ReleaseChnFrame chn1 NG\n");
      }
      s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
      CVI_AI_Free(&face);
    }
  }

  DestoryOutput(&outputContext);
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn, VpssGrp);
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  abChnEnable[VpssChn] = CVI_TRUE;
  abChnEnable[VpssChnVO] = CVI_TRUE;
  SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
  CVI_AI_DestroyHandle(facelib_handle);

  return 0;
}
