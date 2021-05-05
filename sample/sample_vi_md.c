#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"
static volatile bool bExit = false;
typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *,
                             cvai_obj_det_type_e);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;
#define CREATE_WRAPPER(realfunc)                                                     \
  int inference_wrapper_##realfunc(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, \
                                   cvai_object_t *objects, cvai_obj_det_type_e e) {  \
    return realfunc(handle, frame, objects);                                         \
  }
#define WRAPPER(realfunc) inference_wrapper_##realfunc
CREATE_WRAPPER(CVI_AI_MobileDetV2_Vehicle_D0)
CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVI_SUCCESS;
  if (strcmp(model_name, "mobiledetv2-lite") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE;
    config->inference = CVI_AI_MobileDetV2_Lite;
  } else if (strcmp(model_name, "mobiledetv2-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
    config->inference = CVI_AI_MobileDetV2_D0;
  } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
    config->inference = CVI_AI_MobileDetV2_D1;
  } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
    config->inference = CVI_AI_MobileDetV2_D2;
  } else if (strcmp(model_name, "mobiledetv2-vehicle-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0;
    config->inference = WRAPPER(CVI_AI_MobileDetV2_Vehicle_D0);
  } else if (strcmp(model_name, "yolov3") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;
  } else {
    ret = CVI_FAILURE;
  }
  return ret;
}
typedef struct CVI_IVE_BACKGROUND_HANDLE {
  VPSS_GRP grp;
  VPSS_CHN chn;
  IVE_HANDLE handle;
  IVE_SRC_IMAGE_S src[2], tmp, andframe[2];
  VB_BLK blk[2];
  VIDEO_FRAME_INFO_S vbsrc[2];
  int count;
  int i_count;
} CVI_IVE_BACKGROUND_HANDLE_S;

static void FilterOutStaticObject(CVI_IVE_BACKGROUND_HANDLE_S *bk_handle,
                                  VIDEO_FRAME_INFO_S *srcFrame, IVE_IMAGE_S *bk_dst,
                                  cvai_object_t *src, cvai_object_t *result);
static void CVI_MD(CVI_IVE_BACKGROUND_HANDLE_S *bk_handle, VIDEO_FRAME_INFO_S *src,
                   IVE_IMAGE_S *dst);

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
        "\t model_name: detection model name should be one of {mobiledetv2-lite, mobiledetv2-d0, "
        "mobiledetv2-d1, "
        "mobiledetv2-d2, "
        "mobiledetv2-vehicle-d0, "
        "yolov3}\n"
        "\t video output, 0: disable, 1: output to panel, 2: output through rtsp\n"
        "\t threshold (optional): threshold for detection model\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[3]);
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  ModelConfig model_config;
  if (createModelConfig(argv[1], &model_config) == CVI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVI_FAILURE;
  }

  CVI_S32 s32Ret = CVI_SUCCESS;
  /*
   * VPSS pipeline:                                        _______________________
   *                                                      |         CHN0 (YUV 400)|----->MD
   *                                                +---->| GRP 1                 |
   *                                                |     |_______________________|
   *        __________________________________      |      ______________________________
   *       |           CHN0 (RGB 888)         |-----+---->|         CHN0 (RGB 888 PLANER)|-->Model
   * VI -->| GRP 0     CHN1 (YUV 420 PLANER)  |--> VO     | GRP 2 (AISDK)                |
   *       |__________________________________|           |______________________________|
   *
   *
   */

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
  VPSS_GRP BkVpssGrp = 1;
  VPSS_CHN BkVpssChn = 0;
  {
    CVI_S32 s32Ret = CVI_SUCCESS;
    VPSS_GRP_ATTR_S stVpssGrpAttr;
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];
    abChnEnable[BkVpssChn] = CVI_TRUE;
    VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[BkVpssChn], voWidth, voHeight, PIXEL_FORMAT_YUV_400,
                            true);
    VPSS_GRP_DEFAULT_HELPER(&stVpssGrpAttr, voWidth, voHeight, PIXEL_FORMAT_RGB_888);
    /*start vpss*/
    s32Ret = SAMPLE_COMM_VPSS_Init(BkVpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
      printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
      return s32Ret;
    }
    s32Ret = SAMPLE_COMM_VPSS_Start(BkVpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
      printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
      return s32Ret;
    }
  }

  CVI_IVE_BACKGROUND_HANDLE_S bk_handle;
  bk_handle.grp = BkVpssGrp;
  bk_handle.chn = BkVpssChn;
  bk_handle.handle = CVI_IVE_CreateHandle();
  bk_handle.count = 0;
  bk_handle.i_count = 0;
  CREATE_VBFRAME_HELPER(&bk_handle.blk[0], &bk_handle.vbsrc[0], voWidth, voHeight,
                        PIXEL_FORMAT_YUV_400);
  CREATE_VBFRAME_HELPER(&bk_handle.blk[1], &bk_handle.vbsrc[1], voWidth, voHeight,
                        PIXEL_FORMAT_YUV_400);
  CVI_IVE_VideoFrameInfo2Image(&bk_handle.vbsrc[0], &bk_handle.src[0]);
  CVI_IVE_VideoFrameInfo2Image(&bk_handle.vbsrc[1], &bk_handle.src[1]);
  CVI_IVE_CreateImage(bk_handle.handle, &bk_handle.tmp, IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  CVI_IVE_CreateImage(bk_handle.handle, &bk_handle.andframe[0], IVE_IMAGE_TYPE_U8C1, voWidth,
                      voHeight);
  CVI_IVE_CreateImage(bk_handle.handle, &bk_handle.andframe[1], IVE_IMAGE_TYPE_U8C1, voWidth,
                      voHeight);
  IVE_IMAGE_S bk_dst;
  CVI_IVE_CreateImage(bk_handle.handle, &bk_dst, IVE_IMAGE_TYPE_U8C1, voWidth, voHeight);
  // Init end
  //****************************************************************
  cviai_handle_t facelib_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&facelib_handle, 2);
  ret = CVI_AI_SetModelPath(facelib_handle, model_config.model_id, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }
  if (argc == 5) {
    float threshold = atof(argv[4]);
    if (threshold < 0.0 || threshold > 1.0) {
      printf("wrong threshold value: %f\n", threshold);
      return ret;
    } else {
      printf("set threshold to %f\n", threshold);
    }
    CVI_AI_SetModelThreshold(facelib_handle, model_config.model_id, threshold);
  }
  CVI_AI_SetSkipVpssPreprocess(facelib_handle, model_config.model_id, false);

  // get person class only
  ret = CVI_AI_SelectDetectClass(facelib_handle, model_config.model_id, 1, CVI_AI_DET_TYPE_PERSON);

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  cvai_object_t obj_meta, obj_meta_moving;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&obj_meta_moving, 0, sizeof(cvai_object_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }
    model_config.inference(facelib_handle, &stfdFrame, &obj_meta, 0);

    // Filter out staic object
    FilterOutStaticObject(&bk_handle, &stfdFrame, &bk_dst, &obj_meta, &obj_meta_moving);
    printf("detect %u moving objects\n", obj_meta_moving.size);

    int s32Ret = CVI_SUCCESS;
    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_Service_ObjectDrawRect(NULL, &obj_meta_moving, &stVOFrame, true);
      s32Ret = SendOutputFrame(&stVOFrame, &outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
        break;
      }
      s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }
    CVI_AI_Free(&obj_meta);
    CVI_AI_Free(&obj_meta_moving);
  }

  CVI_VB_ReleaseBlock(bk_handle.blk[0]);
  CVI_VB_ReleaseBlock(bk_handle.blk[1]);
  CVI_SYS_FreeI(bk_handle.handle, &bk_handle.tmp);
  CVI_SYS_FreeI(bk_handle.handle, &bk_handle.andframe[0]);
  CVI_SYS_FreeI(bk_handle.handle, &bk_handle.andframe[1]);
  CVI_IVE_DestroyHandle(bk_handle.handle);
  CVI_AI_DestroyHandle(facelib_handle);
  DestoryOutput(&outputContext);
  // Exit vpss stuffs
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn, VpssGrp);
  {
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    abChnEnable[VpssChn] = CVI_TRUE;
    abChnEnable[VpssChnVO] = CVI_TRUE;
    SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);
  }
  {
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    abChnEnable[bk_handle.chn] = CVI_TRUE;
    SAMPLE_COMM_VPSS_Stop(bk_handle.grp, abChnEnable);
  }
  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
}

static void FilterOutStaticObject(CVI_IVE_BACKGROUND_HANDLE_S *bk_handle,
                                  VIDEO_FRAME_INFO_S *srcFrame, IVE_IMAGE_S *bk_dst,
                                  cvai_object_t *src, cvai_object_t *result) {
  CVI_MD(bk_handle, srcFrame, bk_dst);
  CVI_IVE_BufRequest(bk_handle->handle, bk_dst);
  int valid_num = 0;
  int *valid_index = (int *)malloc(src->size * sizeof(int));
  for (int a = 0; a < src->size; a++) {
    int x1 = src->info[a].bbox.x1;
    int y1 = src->info[a].bbox.y1;
    int x2 = src->info[a].bbox.x2;
    int y2 = src->info[a].bbox.y2;
    bool is_valid = false;
    for (int i = y1; i < y2; i++) {
      for (int j = x1; j < x2; j++) {
        if (bk_dst->pu8VirAddr[0][i * bk_dst->u16Stride[0] + j] > 35) {
          is_valid = true;
          valid_index[valid_num] = a;
          valid_num++;
          break;
        }
      }
      if (is_valid) {
        break;
      }
    }
  }

  result->size = valid_num;
  result->width = src->width;
  result->height = src->height;
  result->rescale_type = src->rescale_type;
  result->info = (cvai_object_info_t *)malloc(result->size * sizeof(cvai_object_info_t));
  memset(result->info, 0, sizeof(cvai_object_info_t) * result->size);
  for (int i = 0; i < valid_num; i++) {
    CVI_AI_CopyObjectInfo(&src->info[valid_index[i]], &result->info[i]);
  }
  free(valid_index);
}

static void CVI_MD(CVI_IVE_BACKGROUND_HANDLE_S *bk_handle, VIDEO_FRAME_INFO_S *src,
                   IVE_IMAGE_S *dst) {
  // VIDEO_FRAME_INFO_S ive_src;
  // Color 2 grayscale
  CVI_S32 ret = CVI_VPSS_SendChnFrame(bk_handle->grp, bk_handle->chn,
                                      &bk_handle->vbsrc[bk_handle->count], 1000);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_SendChnFrame %x\n", ret);
  }
  ret = CVI_VPSS_SendFrame(bk_handle->grp, src, 1000);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_SendFrame %x\n", ret);
  }

  ret = CVI_VPSS_GetChnFrame(bk_handle->grp, bk_handle->chn, &bk_handle->vbsrc[bk_handle->count],
                             1000);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame %x\n", ret);
  }

  if (bk_handle->i_count > 0) {
    // Sub - threshold - dilate
    IVE_SUB_CTRL_S iveSubCtrl;
    iveSubCtrl.enMode = IVE_SUB_MODE_ABS;
    ret = CVI_IVE_Sub(bk_handle->handle, &bk_handle->src[bk_handle->count],
                      &bk_handle->src[1 - bk_handle->count], &bk_handle->tmp, &iveSubCtrl, 0);
    if (ret != CVI_SUCCESS) {
      printf("CVI_IVE_Sub fail %x\n", ret);
    }
    IVE_THRESH_CTRL_S iveTshCtrl;
    iveTshCtrl.enMode = IVE_THRESH_MODE_BINARY;
    iveTshCtrl.u8MinVal = 0;
    iveTshCtrl.u8MaxVal = 255;
    iveTshCtrl.u8LowThr = 35;
    ret = CVI_IVE_Thresh(bk_handle->handle, &bk_handle->tmp, &bk_handle->tmp, &iveTshCtrl, 0);
    if (ret != CVI_SUCCESS) {
      printf("CVI_IVE_Sub fail %x\n", ret);
    }
    IVE_DILATE_CTRL_S stDilateCtrl;
    memset(stDilateCtrl.au8Mask, 1, 25);
    CVI_IVE_Dilate(bk_handle->handle, &bk_handle->tmp, &bk_handle->andframe[bk_handle->count],
                   &stDilateCtrl, 0);
    if (bk_handle->i_count > 1) {
      // And two dilated images
      CVI_IVE_And(bk_handle->handle, &bk_handle->andframe[bk_handle->count],
                  &bk_handle->andframe[1 - bk_handle->count], dst, 0);
      CVI_IVE_And(bk_handle->handle, &bk_handle->src[bk_handle->count], dst, dst, 0);
    }
  }
  bk_handle->count = 1 - bk_handle->count;
  if (bk_handle->i_count < 2) {
    bk_handle->i_count++;
  }
}