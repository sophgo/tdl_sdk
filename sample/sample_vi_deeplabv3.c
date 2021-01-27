#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "ive/ive.h"

static volatile bool bExit = false;

CVI_S32 InitVPSSOutput(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn1, const CVI_U32 grpWidth,
                       const CVI_U32 grpHeight, const CVI_U32 voWidth, const CVI_U32 voHeight) {
  CVI_S32 s32Ret = VPSS_INIT_HELPER2(vpssGrp, voWidth, voHeight, PIXEL_FORMAT_RGB_888_PLANAR,
                                     voWidth, voHeight, PIXEL_FORMAT_YUV_PLANAR_420, 1, true);
  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);
  return s32Ret;
}

CVI_S32 InitVPSS_Multi(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn1, const VPSS_CHN vpssChn2,
                       const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                       const CVI_U32 voHeight, const VI_PIPE viPipe) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

  abChnEnable[vpssChn1] = CVI_TRUE;
  VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChn1], voWidth, voHeight, PIXEL_FORMAT_RGB_888, true);

  abChnEnable[vpssChn2] = CVI_TRUE;
  VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChn2], voWidth, voHeight, PIXEL_FORMAT_RGB_888_PLANAR,
                          true);

  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

  VPSS_GRP_DEFAULT_HELPER(&stVpssGrpAttr, grpWidth, grpHeight, PIXEL_FORMAT_YUV_PLANAR_420);

  /*start vpss*/
  s32Ret = SAMPLE_COMM_VPSS_Init(vpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VPSS_Start(vpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_Bind_VPSS(viPipe, vpssChn1, vpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_Bind_VPSS(viPipe, vpssChn2, vpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }
  return s32Ret;
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

static void createGaussianIVECtrl(IVE_FILTER_CTRL_S *iveFltCtrl) {
  CVI_S8 arr[] = {
      2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2,
  };
  iveFltCtrl->u8MaskSize = 5;
  memcpy(iveFltCtrl->as8Mask, arr,
         iveFltCtrl->u8MaskSize * iveFltCtrl->u8MaskSize * sizeof(CVI_S8));
  iveFltCtrl->u32Norm = 115;
}

static CVI_S32 createBlurImage(IVE_HANDLE ive_handle, IVE_FILTER_CTRL_S *iveFltCtrl,
                               VIDEO_FRAME_INFO_S *labelFrame, VIDEO_FRAME_INFO_S *srcFrame,
                               IVE_IMAGE_S *dst_image) {
  IVE_IMAGE_S src_image;
  CVI_U32 imageLength = srcFrame->stVFrame.u32Length[0] + srcFrame->stVFrame.u32Length[1] +
                        srcFrame->stVFrame.u32Length[2];
  srcFrame->stVFrame.pu8VirAddr[0] =
      CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0], imageLength);

  CVI_S32 s32Ret;
  s32Ret = CVI_IVE_VideoFrameInfo2Image(srcFrame, &src_image);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_IVE_VideoFrameInfo2Image failed with %#x\n", s32Ret);
    return s32Ret;
  }

  IVE_IMAGE_S label_image;
  imageLength = labelFrame->stVFrame.u32Length[0] + labelFrame->stVFrame.u32Length[1] +
                labelFrame->stVFrame.u32Length[2];
  labelFrame->stVFrame.pu8VirAddr[0] =
      CVI_SYS_MmapCache(labelFrame->stVFrame.u64PhyAddr[0], imageLength);

  s32Ret = CVI_IVE_VideoFrameInfo2Image(labelFrame, &label_image);

  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_IVE_VideoFrameInfo2Image failed with %#x\n", s32Ret);
    return s32Ret;
  }

  IVE_IMAGE_S blurImage;
  CVI_IVE_CreateImage(ive_handle, &blurImage, IVE_IMAGE_TYPE_U8C3_PLANAR,
                      srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height);

  s32Ret = CVI_IVE_Filter(ive_handle, &src_image, &blurImage, iveFltCtrl, 0);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_IVE_Filter failed with %#x\n", s32Ret);
    return s32Ret;
  }

  CVI_IVE_CreateImage(ive_handle, dst_image, src_image.enType, src_image.u16Width,
                      src_image.u16Height);

  s32Ret = CVI_IVE_Mask(ive_handle, &src_image, &blurImage, &label_image, dst_image, false);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_IVE_Mask failed with %#x\n", s32Ret);
    return s32Ret;
  }

  CVI_SYS_Munmap((void *)srcFrame->stVFrame.pu8VirAddr[0], imageLength);
  CVI_SYS_Munmap((void *)labelFrame->stVFrame.pu8VirAddr[0], imageLength);
  CVI_SYS_FreeI(ive_handle, &src_image);
  CVI_SYS_FreeI(ive_handle, &blurImage);
  CVI_SYS_FreeI(ive_handle, &label_image);
  return s32Ret;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf(
        "Usage: %s <model_path> <open vo 1 or 0>\n"
        "\t deeplabv3 model path\n"
        "\t video output, 0: disable, 1: enable\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_BOOL isVoOpened = (strcmp(argv[2], "1") == 0) ? true : false;

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 s32Ret = CVI_SUCCESS;
  //****************************************************************
  // Init VI, VO, Vpss
  CVI_U32 DevNum = 0;
  VI_PIPE ViPipe = 0;
  VPSS_GRP VpssGrp = 0;
  VPSS_GRP VpssVoGrp = 2;
  VPSS_CHN VpssChn1 = VPSS_CHN0;
  VPSS_CHN VpssChn2 = VPSS_CHN1;
  VPSS_CHN VpssChnVO = VPSS_CHN0;
  CVI_S32 GrpWidth = 1920;
  CVI_S32 GrpHeight = 1080;
  CVI_U32 VoLayer = 0;
  CVI_U32 VoChn = 0;
  SAMPLE_VI_CONFIG_S stViConfig;
  SAMPLE_VO_CONFIG_S stVoConfig;
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
  if (isVoOpened) {
    s32Ret = InitVO(voWidth, voHeight, &stVoConfig);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
      return s32Ret;
    }
    CVI_VO_HideChn(VoLayer, VoChn);
  }

  s32Ret =
      InitVPSS_Multi(VpssGrp, VpssChn1, VpssChn2, GrpWidth, GrpHeight, voWidth, voHeight, ViPipe);

  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }

  printf("InitVPSSOutput\n");
  s32Ret = InitVPSSOutput(VpssVoGrp, VpssChnVO, GrpWidth, GrpHeight, voWidth, voHeight);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }
  printf("InitVPSSOutput done\n");

  // Init end
  //****************************************************************

  cviai_handle_t facelib_handle = NULL;

  int ret = CVI_AI_CreateHandle2(&facelib_handle, 1);
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_DEEPLABV3, argv[1]);

  printf("set model path\n");
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

  printf("create ive handle\n");
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }

  VIDEO_FRAME_INFO_S frame, rgbPackageFrame, stVOFrame, blurFrame;
  cvai_object_t obj_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));

  IVE_FILTER_CTRL_S ctrl;
  createGaussianIVECtrl(&ctrl);

  cvai_class_filter_t filter;
  uint32_t preserved_classes_id[2] = {11, 12};
  filter.num_preserved_classes = 2;
  filter.preserved_class_ids = preserved_classes_id;

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn1, &frame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame %d(%d) failed with %#x\n", VpssGrp, VpssChn1, s32Ret);
      break;
    }

    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn2, &rgbPackageFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame %d(%d) failed with %#x\n", VpssGrp, VpssChn2, s32Ret);
      break;
    }

    VIDEO_FRAME_INFO_S label_frame;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    CVI_AI_DeeplabV3(facelib_handle, &frame, &label_frame, &filter);
    gettimeofday(&t1, NULL);
    unsigned long elapsed = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec);
    printf("%s, elapsed: %lu us\n", "CVI_AI_DeeplabV3", elapsed);

    gettimeofday(&t0, NULL);
    IVE_IMAGE_S blur_image;
    createBlurImage(ive_handle, &ctrl, &label_frame, &rgbPackageFrame, &blur_image);
    gettimeofday(&t1, NULL);
    elapsed = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec);
    printf("%s, elapsed: %lu us\n", "createBlurImage", elapsed);

    s32Ret = CVI_IVE_Image2VideoFrameInfo(&blur_image, &blurFrame, false);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_IVE_Image2VideoFrameInfo failed with %#x\n", s32Ret);
      return s32Ret;
    }

    // Send frame to VO if opened.
    if (isVoOpened) {
      printf("label_frame: %d, %d, format: %d\n", blurFrame.stVFrame.u32Width,
             blurFrame.stVFrame.u32Height, blurFrame.stVFrame.enPixelFormat);
      CVI_VPSS_SendFrame(VpssVoGrp, &blurFrame, -1);
      // s32Ret = CVI_VPSS_SendChnFrame(VpssGrp, VpssChnVO, &rgbPackageFrame, -1);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_SendChnFrame failed with %#x\n", s32Ret);
        break;
      }

      s32Ret = CVI_VPSS_GetChnFrame(VpssVoGrp, VpssChnVO, &stVOFrame, 2000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame %d(%d) failed with %#x\n", VpssVoGrp, VpssChnVO, s32Ret);
        break;
      }

      s32Ret = CVI_VO_SendFrame(VoLayer, VoChn, &stVOFrame, -1);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VO_SendFrame failed with %#x\n", s32Ret);
        break;
      }
      CVI_VO_ShowChn(VoLayer, VoChn);
      s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn2, &rgbPackageFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn1, &frame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    CVI_SYS_FreeI(ive_handle, &blur_image);
    CVI_VPSS_ReleaseChnFrame(0, 0, &label_frame);
    CVI_AI_Free(&obj_meta);
  }

  CVI_IVE_DestroyHandle(ive_handle);
  CVI_AI_DestroyHandle(facelib_handle);

  // Exit vpss stuffs
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn1, VpssGrp);
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn2, VpssGrp);
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  abChnEnable[VpssChn1] = CVI_TRUE;
  abChnEnable[VpssChn2] = CVI_TRUE;
  abChnEnable[VpssChnVO] = CVI_TRUE;
  SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
}