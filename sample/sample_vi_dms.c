#define _GNU_SOURCE
#include <signal.h>
#include "cviai.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

// smooth value structure
typedef struct {
  float mask_score;
  cvai_dms_t dms;
} SMOOTH_FACE_INFO;
SMOOTH_FACE_INFO smooth_face_info;

// threshold structure
typedef struct {
  int eye_win_th;
  int yawn_win_th;
  float yaw_th;
  float pitch_th;
  float eye_th;
  float smooth_eye_th;
  float yawn_th;
  float mask_th;
} THRESHOLD_S;
THRESHOLD_S _threshold;

// ai handle
cviai_handle_t ai_handle = NULL;
cviai_service_handle_t service_handle = NULL;

// siganl exit
static volatile bool bExit = true;
// status_name
char status_n[256];

static void HandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = false;
  }
}

// dms initial
void dms_init(cvai_face_t* face) {
  cvai_dms_t* dms = (cvai_dms_t*)malloc(sizeof(cvai_dms_t));
  dms->reye_score = 0;
  dms->leye_score = 0;
  dms->yawn_score = 0;
  dms->phone_score = 0;
  dms->smoke_score = 0;
  dms->landmarks_106.size = 0;
  dms->landmarks_5.size = 0;
  dms->head_pose.yaw = 0;
  dms->head_pose.pitch = 0;
  dms->head_pose.roll = 0;
  dms->dms_od.info = NULL;
  dms->dms_od.size = 0;
  face->dms = dms;
}

void YawnClassifition(cvai_face_t* face, VIDEO_FRAME_INFO_S stVencFrame,
                      VIDEO_FRAME_INFO_S stVOFrame, int* yawn_score_window,
                      const THRESHOLD_S threshold, const bool mask_flag, const int display) {
  CVI_AI_YawnClassification(ai_handle, &stVencFrame, face);

  // calculate smooth yawn score
  smooth_face_info.dms.yawn_score =
      smooth_face_info.dms.yawn_score * 0.5 + face->dms->yawn_score * 0.5;
  if (smooth_face_info.dms.yawn_score < threshold.yawn_th) {
    if (*yawn_score_window < 30) *yawn_score_window += 1;
  } else {
    if (*yawn_score_window > 0) *yawn_score_window -= 1;
  }

  if (display) {
    memset(status_n, 0, sizeof(status_n));
    if (face->dms->yawn_score < threshold.yawn_th) {
      strcpy(status_n, "Yawn: close");
      CVI_AI_Service_ObjectWriteText(status_n, 1000, 190, &stVOFrame, 255, 0, 255);
    } else {
      strcpy(status_n, "Yawn: open");
      CVI_AI_Service_ObjectWriteText(status_n, 1000, 190, &stVOFrame, -1, -1, -1);
    }
  }
}

void EyeClassifition(cvai_face_t* face, VIDEO_FRAME_INFO_S stVencFrame,
                     VIDEO_FRAME_INFO_S stVOFrame, int* eye_score_window,
                     const THRESHOLD_S threshold, const int display) {
  CVI_AI_EyeClassification(ai_handle, &stVencFrame, face);

  // calculate smooth eye score
  smooth_face_info.dms.reye_score =
      smooth_face_info.dms.reye_score * 0.5 + face->dms->reye_score * 0.5;
  smooth_face_info.dms.leye_score =
      smooth_face_info.dms.leye_score * 0.5 + face->dms->leye_score * 0.5;

  if (smooth_face_info.dms.reye_score + smooth_face_info.dms.leye_score > threshold.smooth_eye_th) {
    if (*eye_score_window < 30) *eye_score_window += 1;
  } else {
    if (*eye_score_window > 0) *eye_score_window -= 1;
  }

  if (display) {
    // show the eye score per frame
    memset(status_n, 0, sizeof(status_n));
    if (face->dms->reye_score < threshold.eye_th) {
      strcpy(status_n, "Right Eye: close");
      CVI_AI_Service_ObjectWriteText(status_n, 700, 100, &stVOFrame, -1, -1, -1);
    } else {
      strcpy(status_n, "Right Eye: open");
      CVI_AI_Service_ObjectWriteText(status_n, 700, 100, &stVOFrame, 255, 0, 255);
    }
    memset(status_n, 0, sizeof(status_n));
    if (face->dms->leye_score < threshold.eye_th) {
      strcpy(status_n, "Left Eye: close");
      CVI_AI_Service_ObjectWriteText(status_n, 1000, 100, &stVOFrame, -1, -1, -1);
    } else {
      strcpy(status_n, "Left Eye: open");
      CVI_AI_Service_ObjectWriteText(status_n, 1000, 100, &stVOFrame, 255, 0, 255);
    }
  }
}

void MaskClassifition(cvai_face_t* face, VIDEO_FRAME_INFO_S stVencFrame,
                      VIDEO_FRAME_INFO_S stVOFrame, bool* mask_flag, const THRESHOLD_S threshold,
                      const int display) {
  CVI_AI_MaskClassification(ai_handle, &stVencFrame, face);

  // calculate smooth mask score
  smooth_face_info.mask_score = smooth_face_info.mask_score * 0.5 + face->info[0].mask_score * 0.5;

  if (display) {
    memset(status_n, 0, sizeof(status_n));
    if (smooth_face_info.mask_score < threshold.mask_th) {
      strcpy(status_n, "Status: no mask");
      *mask_flag = true;
      if (display) CVI_AI_Service_ObjectWriteText(status_n, 1000, 150, &stVOFrame, 255, 0, 255);
    } else {
      strcpy(status_n, "Status: mask");
      *mask_flag = false;
      if (display) CVI_AI_Service_ObjectWriteText(status_n, 1000, 150, &stVOFrame, -1, -1, -1);
    }
  }
}

void FaceLandmarks(cvai_face_t* face, VIDEO_FRAME_INFO_S stVencFrame, VIDEO_FRAME_INFO_S stVOFrame,
                   const int display) {
  CVI_AI_FaceLandmarker(ai_handle, &stVencFrame, face);

  if (display) CVI_AI_Service_FaceDrawPts(&(face->dms->landmarks_106), &stVOFrame);
}

void FaceAngle(cvai_face_t* face, VIDEO_FRAME_INFO_S stVOFrame, const int display) {
  CVI_AI_Service_FaceAngle(&(face->dms->landmarks_5), &face->dms->head_pose);

  // calculate smooth face angle
  smooth_face_info.dms.head_pose.yaw =
      smooth_face_info.dms.head_pose.yaw * 0.5 + face->dms->head_pose.yaw * 0.5;
  smooth_face_info.dms.head_pose.pitch =
      smooth_face_info.dms.head_pose.pitch * 0.5 + face->dms->head_pose.pitch * 0.5;
  smooth_face_info.dms.head_pose.roll =
      smooth_face_info.dms.head_pose.roll * 0.5 + face->dms->head_pose.roll * 0.5;

  if (display) {
    memset(status_n, 0, sizeof(status_n));
    sprintf(status_n, "Yaw: %f Pitch: %f Roll: %f", face->dms->head_pose.yaw,
            face->dms->head_pose.pitch, face->dms->head_pose.roll);
    CVI_AI_Service_ObjectWriteText(status_n, 700, 50, &stVOFrame, -1, -1,
                                   -1);  // per frame info
  }
}

void StatusUpdate(VIDEO_FRAME_INFO_S stVOFrame, const int eye_score_window,
                  const int yawn_score_window, const THRESHOLD_S threshold) {
  memset(status_n, 0, sizeof(status_n));
  if ((smooth_face_info.dms.head_pose.yaw > threshold.yaw_th ||
       smooth_face_info.dms.head_pose.yaw < -threshold.yaw_th ||
       smooth_face_info.dms.head_pose.pitch > threshold.pitch_th ||
       smooth_face_info.dms.head_pose.pitch < -threshold.pitch_th)) {
    strcpy(status_n, "Status: Distraction");
    CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame, -1, -1, -1);
  } else if (eye_score_window < threshold.eye_win_th || yawn_score_window < threshold.yawn_win_th) {
    strcpy(status_n, "Status: Drowsiness");
    CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame, -1, -1, -1);
  } else {
    strcpy(status_n, "Status: Normal");
    CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame, -1, -1, -1);
  }
}

int main(int argc, char** argv) {
  if (argc != 9) {
    printf(
        "Usage: %s <retina_model_path> <mask_classification_model> <eye_classification_model> "
        "<yawn_classification_model> <face_landmark_model> <in_car_od_model> <video output> "
        "<display detial>.\n"
        "\tretina_model_path, path to retinaface model\n"
        "\tmask_classification_model, path to mask classification model\n"
        "\teye_classification_model, path to eye classification model\n"
        "\tvideo output, 0: disable, 1: output to panel, 2: output through rtsp\n"
        "\tdisplay detial, 0: disable, 1: enable\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  // VIDEO
  // Set signal catch
  signal(SIGINT, HandleSig);
  signal(SIGTERM, HandleSig);

  CVI_S32 display = atoi(argv[8]);
  CVI_S32 voType = atoi(argv[7]);
  CVI_S32 s32Ret;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, PIXEL_FORMAT_RGB_888, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  // Load model
  CVI_S32 ret;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), ret, create_service_fail);

  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]), ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, argv[2]),
                 ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION, argv[3]),
                 ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, argv[4]),
                 ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, argv[5]), ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, argv[6]),
                 ret, setup_ai_fail);

  {
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    memset(&smooth_face_info, 0, sizeof(smooth_face_info));
    memset(&_threshold, 0, sizeof(_threshold));
    bool mask_flag = false;
    int start_count = 0;
    int eye_score_window = 0;   // 30 frames
    int yawn_score_window = 0;  // 30 frames

    // set the threshold
    _threshold.eye_win_th = 11;
    _threshold.yawn_win_th = 11;
    _threshold.yaw_th = 0.25;
    _threshold.pitch_th = 0.25;
    _threshold.eye_th = 0.45;
    _threshold.smooth_eye_th = 0.65 * 2;  // 0.65 is one eye
    _threshold.yawn_th = 0.75;
    _threshold.mask_th = 0.5;

    while (bExit) {
      VIDEO_FRAME_INFO_S stVencFrame, stVOFrame;
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                    &stVencFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        SAMPLE_PRT("CVI_VPSS_GetChnFrame grp0 chn0 failed with %#x\n", s32Ret);
        continue;
      }
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                    vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_RetinaFace(ai_handle, &stVencFrame, &face);

      // Just calculate the first one
      if (face.size > 0) {
        dms_init(&face);

        // calculate od
        CVI_AI_IncarObjectDetection(ai_handle, &stVencFrame, &face);

        // calculate landmark
        FaceLandmarks(&face, stVencFrame, stVOFrame, display);

        // face angle
        FaceAngle(&face, stVOFrame, display);

        // mask detection
        MaskClassifition(&face, stVencFrame, stVOFrame, &mask_flag, _threshold, display);

        // eye classification
        EyeClassifition(&face, stVencFrame, stVOFrame, &eye_score_window, _threshold, display);
        if (mask_flag) {
          // Yawn classification
          YawnClassifition(&face, stVencFrame, stVOFrame, &yawn_score_window, _threshold, mask_flag,
                           display);
        } else {
          if (yawn_score_window < 0) yawn_score_window += 1;
        }
        if (start_count > 90)
          StatusUpdate(stVOFrame, eye_score_window, yawn_score_window, _threshold);
        else
          start_count++;
        CVI_AI_FreeDMS(face.dms);
      } else {
        memset(status_n, 0, sizeof(status_n));
        sprintf(status_n, "No face");
        CVI_AI_Service_ObjectWriteText(status_n, 30, 70, &stVOFrame, -1, -1, -1);
      }

      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
        break;
      }

      if (CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                   &stVencFrame) != 0) {
        SAMPLE_PRT("CVI_VPSS_ReleaseChnFrame chn1 NG\n");
      }
      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
      CVI_AI_Free(&face);
    }
  }

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(service_handle);
create_service_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();

  return 0;
}
