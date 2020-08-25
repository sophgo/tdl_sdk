#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cvi_ae.h"
#include "cvi_ae_comm.h"
#include "cvi_awb_comm.h"
#include "cvi_buffer.h"
#include "cvi_comm_isp.h"
#include "cvi_isp.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"

#include "cviai.h"
#include "draw_utils.h"
#include "sample_comm.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define YOLOV3_SCALE (float)(1 / 255.0)
#define YOLOV3_QUANTIZE_SCALE YOLOV3_SCALE *(128.0 / 1.00000488758)

static volatile bool bExit = false;

cviai_handle_t facelib_handle = NULL;

static SAMPLE_VI_CONFIG_S vi_config;
static VI_PIPE vi_pipe = 0;
static CVI_S32 work_sns_id = 0;
static VPSS_GRP vpss_group = 0;
static VPSS_CHN vpss_channel = VPSS_CHN0;
static VPSS_CHN vpss_channel_vo = VPSS_CHN1;
static CVI_S32 vpss_group_width = 1920;
static CVI_S32 vpss_group_height = 1080;
static CVI_U32 vo_layer = 0;
static CVI_U32 vo_channel = 0;

static int GetVideoframe(VIDEO_FRAME_INFO_S *obj_det_frame,
                         VIDEO_FRAME_INFO_S *video_output_frame) {
  int ret = CVI_SUCCESS;
  ret = CVI_VPSS_GetChnFrame(vpss_group, vpss_channel, obj_det_frame, 1000);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame vpss_channel failed with %#x\n", ret);
    return ret;
  }

  ret = CVI_VPSS_GetChnFrame(vpss_group, vpss_channel_vo, video_output_frame, 1000);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame vpss_channel_vo failed with %#x\n", ret);
    return ret;
  }

  return ret;
}

static int ReleaseVideoframe(VIDEO_FRAME_INFO_S *obj_det_frame,
                             VIDEO_FRAME_INFO_S *video_output_frame) {
  int ret = CVI_SUCCESS;
  ret = CVI_VPSS_ReleaseChnFrame(vpss_group, vpss_channel, obj_det_frame);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_ReleaseChnFrame vpss_channel NG\n");
    return ret;
  }

  ret = CVI_VPSS_ReleaseChnFrame(vpss_group, vpss_channel_vo, video_output_frame);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_ReleaseChnFrame vpss_channel_vo NG\n");
    return ret;
  }

  return ret;
}

static int DoObjDet(cviai_handle_t facelib_handle, VIDEO_FRAME_INFO_S *obj_det_frame,
                    cvai_object_t *obj_meta) {
  CVI_S32 ret = CVI_SUCCESS;

  CVI_AI_Yolov3(facelib_handle, obj_det_frame, obj_meta, 0);

  return ret;
}

static void Exit() {
  SAMPLE_COMM_VI_UnBind_VPSS(vi_pipe, vpss_channel, vpss_group);

  CVI_BOOL chn_enable[VPSS_MAX_PHY_CHN_NUM] = {0};
  chn_enable[vpss_channel] = CVI_TRUE;
  chn_enable[vpss_channel_vo] = CVI_TRUE;
  SAMPLE_COMM_VPSS_Stop(vpss_group, chn_enable);

  SAMPLE_COMM_VI_DestroyVi(&vi_config);
  SAMPLE_COMM_SYS_Exit();
}

static void Run() {
  CVI_S32 ret = CVI_SUCCESS;
  VIDEO_FRAME_INFO_S obj_det_frame, video_output_frame;
  cvai_object_t obj_meta;

  while (bExit == false) {
    ret = GetVideoframe(&obj_det_frame, &video_output_frame);
    if (ret != CVI_SUCCESS) {
      Exit();
      assert(0 && "get video frame error!\n");
    }

    DoObjDet(facelib_handle, &obj_det_frame, &obj_meta);

    DrawObjMeta(&video_output_frame, &obj_meta);

    // set_vpss_aspect(2,0,0,720,1280);
    // ret = CVI_VO_SendFrame(vo_layer, vo_channel, &video_output_frame, -1);
    // if (ret != CVI_SUCCESS) {
    // 	printf("CVI_VO_SendFrame failed with %#x\n", ret);
    // }
    // CVI_VO_ShowChn(vo_layer,vo_channel);

    ret = ReleaseVideoframe(&obj_det_frame, &video_output_frame);
    if (ret != CVI_SUCCESS) {
      Exit();
      assert(0 && "release video frame error!\n");
    }

    if (obj_meta.info != NULL) free(obj_meta.info);
  }
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

static void SetVIConfig(SAMPLE_VI_CONFIG_S *video_input_config) {
  SAMPLE_COMM_VI_GetSensorInfo(video_input_config);

  video_input_config->astViInfo[work_sns_id].stSnsInfo.enSnsType = SONY_IMX307_MIPI_2M_30FPS_12BIT;
  video_input_config->s32WorkingViNum = 1;
  video_input_config->as32WorkingViId[0] = 0;
  video_input_config->astViInfo[work_sns_id].stSnsInfo.MipiDev = 0xFF;
  video_input_config->astViInfo[work_sns_id].stSnsInfo.s32BusId = 3;
  video_input_config->astViInfo[work_sns_id].stDevInfo.ViDev = 0;
  video_input_config->astViInfo[work_sns_id].stDevInfo.enWDRMode = WDR_MODE_NONE;
  video_input_config->astViInfo[work_sns_id].stPipeInfo.enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;
  video_input_config->astViInfo[work_sns_id].stPipeInfo.aPipe[0] = vi_pipe;
  video_input_config->astViInfo[work_sns_id].stPipeInfo.aPipe[1] = -1;
  video_input_config->astViInfo[work_sns_id].stPipeInfo.aPipe[2] = -1;
  video_input_config->astViInfo[work_sns_id].stPipeInfo.aPipe[3] = -1;
  video_input_config->astViInfo[work_sns_id].stChnInfo.ViChn = 0;
  video_input_config->astViInfo[work_sns_id].stChnInfo.enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  video_input_config->astViInfo[work_sns_id].stChnInfo.enDynamicRange = DYNAMIC_RANGE_SDR8;
  video_input_config->astViInfo[work_sns_id].stChnInfo.enVideoFormat = VIDEO_FORMAT_LINEAR;
  video_input_config->astViInfo[work_sns_id].stChnInfo.enCompressMode = COMPRESS_MODE_NONE;
}

static CVI_S32 SetVoConfig(SAMPLE_VO_CONFIG_S *video_output_config) {
  RECT_S default_disp_rect = {0, 0, 720, 1280};
  SIZE_S default_image_size = {720, 1280};
  CVI_S32 ret = CVI_SUCCESS;
  ret = SAMPLE_COMM_VO_GetDefConfig(video_output_config);
  if (ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", ret);
    return ret;
  }

  video_output_config->VoDev = 0;
  video_output_config->enVoIntfType = VO_INTF_MIPI;
  video_output_config->enIntfSync = VO_OUTPUT_720x1280_60;
  video_output_config->stDispRect = default_disp_rect;
  video_output_config->stImageSize = default_image_size;
  video_output_config->enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  video_output_config->enVoMode = VO_MODE_1MUX;
  return ret;
}

static CVI_S32 InitVPSS() {
  CVI_S32 ret = CVI_SUCCESS;
  VPSS_GRP_ATTR_S vpss_group_attr;
  CVI_BOOL channel_enable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S vpss_channel_attr[VPSS_MAX_PHY_CHN_NUM];

  channel_enable[vpss_channel] = CVI_TRUE;
  vpss_channel_attr[vpss_channel].u32Width = 320;
  vpss_channel_attr[vpss_channel].u32Height = 320;
  vpss_channel_attr[vpss_channel].enVideoFormat = VIDEO_FORMAT_LINEAR;
  vpss_channel_attr[vpss_channel].enPixelFormat = PIXEL_FORMAT_RGB_888_PLANAR;
  vpss_channel_attr[vpss_channel].stFrameRate.s32SrcFrameRate = 30;
  vpss_channel_attr[vpss_channel].stFrameRate.s32DstFrameRate = 30;
  vpss_channel_attr[vpss_channel].u32Depth = 1;
  vpss_channel_attr[vpss_channel].bMirror = CVI_FALSE;
  vpss_channel_attr[vpss_channel].bFlip = CVI_FALSE;
  vpss_channel_attr[vpss_channel].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
  vpss_channel_attr[vpss_channel].stAspectRatio.bEnableBgColor = CVI_TRUE;
  vpss_channel_attr[vpss_channel].stAspectRatio.u32BgColor = COLOR_RGB_BLACK;
  vpss_channel_attr[vpss_channel].stNormalize.bEnable = CVI_TRUE;
  vpss_channel_attr[vpss_channel].stNormalize.factor[0] = YOLOV3_QUANTIZE_SCALE;
  vpss_channel_attr[vpss_channel].stNormalize.factor[1] = YOLOV3_QUANTIZE_SCALE;
  vpss_channel_attr[vpss_channel].stNormalize.factor[2] = YOLOV3_QUANTIZE_SCALE;
  vpss_channel_attr[vpss_channel].stNormalize.mean[0] = 0;
  vpss_channel_attr[vpss_channel].stNormalize.mean[1] = 0;
  vpss_channel_attr[vpss_channel].stNormalize.mean[2] = 0;
  vpss_channel_attr[vpss_channel].stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;

  channel_enable[vpss_channel_vo] = CVI_TRUE;
  vpss_channel_attr[vpss_channel_vo].u32Width = 1280;
  vpss_channel_attr[vpss_channel_vo].u32Height = 720;
  vpss_channel_attr[vpss_channel_vo].enVideoFormat = VIDEO_FORMAT_LINEAR;
  vpss_channel_attr[vpss_channel_vo].enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  vpss_channel_attr[vpss_channel_vo].stFrameRate.s32SrcFrameRate = 30;
  vpss_channel_attr[vpss_channel_vo].stFrameRate.s32DstFrameRate = 30;
  vpss_channel_attr[vpss_channel_vo].u32Depth = 1;
  vpss_channel_attr[vpss_channel_vo].bMirror = CVI_FALSE;
  vpss_channel_attr[vpss_channel_vo].bFlip = CVI_FALSE;
  vpss_channel_attr[vpss_channel_vo].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
  // vpss_channel_attr[vpss_channel_vo].stAspectRatio.enMode        = ASPECT_RATIO_MANUAL;
  // vpss_channel_attr[vpss_channel_vo].stAspectRatio.stVideoRect.s32X = 0;
  // vpss_channel_attr[vpss_channel_vo].stAspectRatio.stVideoRect.s32Y = 0;
  // vpss_channel_attr[vpss_channel_vo].stAspectRatio.stVideoRect.u32Width = 1280;
  // vpss_channel_attr[vpss_channel_vo].stAspectRatio.stVideoRect.u32Height = 720;
  vpss_channel_attr[vpss_channel_vo].stAspectRatio.bEnableBgColor = CVI_TRUE;
  vpss_channel_attr[vpss_channel_vo].stNormalize.bEnable = CVI_FALSE;

  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

  vpss_group_attr.stFrameRate.s32SrcFrameRate = -1;
  vpss_group_attr.stFrameRate.s32DstFrameRate = -1;
  vpss_group_attr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  vpss_group_attr.u32MaxW = vpss_group_width;
  vpss_group_attr.u32MaxH = vpss_group_height;
  // only for test here. u8VpssDev should be decided by VPSS_MODE and usage.
  vpss_group_attr.u8VpssDev = 0;

  /*start vpss*/
  ret = SAMPLE_COMM_VPSS_Init(vpss_group, channel_enable, &vpss_group_attr, vpss_channel_attr);
  if (ret != CVI_SUCCESS) {
    printf("init vpss group failed. ret: 0x%x !\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_VPSS_Start(vpss_group, channel_enable, &vpss_group_attr, vpss_channel_attr);
  if (ret != CVI_SUCCESS) {
    printf("start vpss group failed. ret: 0x%x !\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_VI_Bind_VPSS(vi_pipe, vpss_channel, vpss_group);
  if (ret != CVI_SUCCESS) {
    printf("vi bind vpss failed. ret: 0x%x !\n", ret);
    return ret;
  }

  return ret;
}

static CVI_S32 InitVI(SAMPLE_VI_CONFIG_S *video_input_config) {
  VB_CONFIG_S vb_config;
  PIC_SIZE_E pic_size;
  CVI_U32 block_size, block_rot_size, block_rgb_size;
  SIZE_S stSize;
  CVI_S32 ret = CVI_SUCCESS;

  VI_DEV ViDev = 0;
  VI_PIPE_ATTR_S stPipeAttr;

  ret = SAMPLE_COMM_VI_GetSizeBySensor(
      video_input_config->astViInfo[work_sns_id].stSnsInfo.enSnsType, &pic_size);
  if (ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_SYS_GetPicSize(pic_size, &stSize);
  if (ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", ret);
    return ret;
  }

  memset(&vb_config, 0, sizeof(VB_CONFIG_S));
  vb_config.u32MaxPoolCnt = 2;

  block_size = COMMON_GetPicBufferSize(
      stSize.u32Width, stSize.u32Height, SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8,
      video_input_config->astViInfo[work_sns_id].stChnInfo.enCompressMode, DEFAULT_ALIGN);
  block_rot_size = COMMON_GetPicBufferSize(
      stSize.u32Height, stSize.u32Width, SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8,
      video_input_config->astViInfo[work_sns_id].stChnInfo.enCompressMode, DEFAULT_ALIGN);
  block_size = MAX(block_size, block_rot_size);
  block_rgb_size = COMMON_GetPicBufferSize(
      vpss_group_width, vpss_group_height, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8,
      video_input_config->astViInfo[work_sns_id].stChnInfo.enCompressMode, DEFAULT_ALIGN);
  vb_config.astCommPool[0].u32BlkSize = block_size;
  vb_config.astCommPool[0].u32BlkCnt = 32;
  vb_config.astCommPool[0].enRemapMode = VB_REMAP_MODE_NOCACHE;
  vb_config.astCommPool[1].u32BlkSize = block_rgb_size;
  vb_config.astCommPool[1].u32BlkCnt = 2;
  vb_config.astCommPool[1].enRemapMode = VB_REMAP_MODE_NOCACHE;

  ret = SAMPLE_COMM_SYS_Init(&vb_config);
  if (ret != CVI_SUCCESS) {
    printf("system init failed with %#x\n", ret);
    return -1;
  }

  ret = SAMPLE_COMM_VI_StartSensor(video_input_config);
  if (ret != CVI_SUCCESS) {
    printf("system start sensor failed with %#x\n", ret);
    return ret;
  }
  SAMPLE_COMM_VI_StartDev(&video_input_config->astViInfo[ViDev]);
  SAMPLE_COMM_VI_StartMIPI(video_input_config);

  memset(&stPipeAttr, 0, sizeof(VI_PIPE_ATTR_S));
  stPipeAttr.bYuvSkip = CVI_FALSE;
  stPipeAttr.u32MaxW = stSize.u32Width;
  stPipeAttr.u32MaxH = stSize.u32Height;
  stPipeAttr.enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP;
  stPipeAttr.enBitWidth = DATA_BITWIDTH_12;
  stPipeAttr.stFrameRate.s32SrcFrameRate = -1;
  stPipeAttr.stFrameRate.s32DstFrameRate = -1;
  stPipeAttr.bNrEn = CVI_TRUE;
  ret = CVI_VI_CreatePipe(vi_pipe, &stPipeAttr);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VI_CreatePipe failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_VI_StartPipe(vi_pipe);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VI_StartPipe failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_VI_GetPipeAttr(vi_pipe, &stPipeAttr);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VI_StartPipe failed with %#x!\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_VI_CreateIsp(video_input_config);
  if (ret != CVI_SUCCESS) {
    printf("VI_CreateIsp failed with %#x!\n", ret);
    return ret;
  }

  return SAMPLE_COMM_VI_StartViChn(&video_input_config->astViInfo[ViDev]);
}

int InitVO(SAMPLE_VO_CONFIG_S *video_output_config) {
  CVI_S32 ret = CVI_SUCCESS;

  ret = SAMPLE_COMM_VO_StartVO(video_output_config);
  if (ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_StartVO failed with %#x\n", ret);
  }

  printf("SAMPLE_COMM_VO_StartVO done\n");
  return ret;
}

int main(void) {
  CVI_S32 ret = CVI_SUCCESS;
  SAMPLE_VO_CONFIG_S video_output_config;

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  SetVIConfig(&vi_config);
  ret = InitVI(&vi_config);
  if (ret != CVI_SUCCESS) {
    printf("Init video input failed with %d\n", ret);
    return ret;
  }

  ret = SetVoConfig(&video_output_config);
  if (ret != CVI_SUCCESS) {
    printf("SetVoConfig failed with %d\n", ret);
    return ret;
  }

  ret = InitVO(&video_output_config);
  if (ret != CVI_SUCCESS) {
    printf("CVI_Init_Video_Output failed with %d\n", ret);
    return ret;
  }
  CVI_VO_HideChn(vo_layer, vo_channel);

  ret = InitVPSS();
  if (ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_YOLOV3,
                             "/mnt/data/yolo_v3_320.cvimodel");
  ret |= CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_YOLOV3, true);
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }

  Run();

  CVI_AI_DestroyHandle(facelib_handle);
  Exit();
}
