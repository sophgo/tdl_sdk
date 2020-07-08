#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>

#include "cvi_buffer.h"
#include "cvi_ae_comm.h"
#include "cvi_awb_comm.h"
#include "cvi_comm_isp.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"
#include "cvi_isp.h"
#include "cvi_ae.h"

#include "cv183x_facelib_v0.0.1.h"
#include "sample_comm.h"
#include "draw_utils.h"


#define MAX(a,b) (((a)>(b))?(a):(b))

cv183x_facelib_handle_t facelib_handle = NULL;
static 	SAMPLE_VI_CONFIG_S stViConfig;
static	SAMPLE_VO_CONFIG_S stVoConfig;

static VI_PIPE ViPipe = 0;
static VPSS_GRP VpssGrp = 0;
static VPSS_CHN VpssChn = VPSS_CHN0;
static VPSS_CHN VpssChnLiveness = VPSS_CHN1;
static VPSS_CHN VpssChnVO = VPSS_CHN2;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;
static CVI_U32 VoLayer = 0;
static CVI_U32 VoChn = 0;


static int GetVideoframe(VIDEO_FRAME_INFO_S *stfdFrame, VIDEO_FRAME_INFO_S *stLivenessFrame,
						 VIDEO_FRAME_INFO_S *stVOFrame)
{
	int s32Ret = CVI_SUCCESS;
	s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, stfdFrame, 1000);
	if (s32Ret != CVI_SUCCESS) {
		printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
		return s32Ret;
	}

	s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnLiveness, stLivenessFrame, 1000);
	if (s32Ret != CVI_SUCCESS) {
		printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
		return s32Ret;
	}

	s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, stVOFrame, 1000);
	if (s32Ret != CVI_SUCCESS) {
		printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
		return s32Ret;
	}

	return s32Ret;
}

static int ReleaseVideoframe(VIDEO_FRAME_INFO_S *stfdFrame, VIDEO_FRAME_INFO_S *stLivenessFrame,
							 VIDEO_FRAME_INFO_S *stVOFrame)
{
	int s32Ret = CVI_SUCCESS;
	s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, stfdFrame);
	if (s32Ret != CVI_SUCCESS) {
		printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
		return s32Ret;
	}

	s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnLiveness, stLivenessFrame);
	if (s32Ret != CVI_SUCCESS) {
		printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
		return s32Ret;
	}

	s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, stVOFrame);
	if (s32Ret != CVI_SUCCESS) {
		printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
		return s32Ret;
	}

	return s32Ret;
}

static int DoFd(cv183x_facelib_handle_t facelib_handle, VIDEO_FRAME_INFO_S *stfdFrame,
				VIDEO_FRAME_INFO_S *stLivenessFrame, cvi_face_t *face)
{
	int face_count = 0;
	CVI_S32 s32Ret = CVI_SUCCESS;

	Cv183xFaceDetect(facelib_handle, stfdFrame, face, &face_count);

	printf("face_count %d\n", face_count);
	if (face_count > 0) {
		Cv183xFaceLivenessDetect(facelib_handle, stLivenessFrame, stLivenessFrame, face);
		Cv183xFaceRecognize(facelib_handle, stLivenessFrame, face);
	}

	return 0;
}

static void Exit()
{
	SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn, VpssGrp);

	CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
	abChnEnable[VpssChn] = CVI_TRUE;
	abChnEnable[VpssChnLiveness] = CVI_TRUE;
	abChnEnable[VpssChnVO] = CVI_TRUE;

	SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);
	SAMPLE_COMM_SYS_Exit();
}

static void Run()
{
	CVI_S32 s32Ret = CVI_SUCCESS;
	VIDEO_FRAME_INFO_S stfdFrame, stLivenessFrame, stVOFrame;
	cvi_face_t face;

	while (true) {
		s32Ret = GetVideoframe(&stfdFrame, &stLivenessFrame, &stVOFrame);
		if(s32Ret != CVI_SUCCESS)  {
			Exit();
			assert(0 && "get video frame error!\n");
		}

		DoFd(facelib_handle, &stfdFrame, &stLivenessFrame, &face);

		draw_face_meta(&stVOFrame, &face);

		// set_vpss_aspect(2,0,0,720,1280);
		s32Ret = CVI_VO_SendFrame(VoLayer, VoChn, &stVOFrame, -1);
		if (s32Ret != CVI_SUCCESS) {
			printf("CVI_VO_SendFrame failed with %#x\n", s32Ret);
		}
		CVI_VO_ShowChn(VoLayer,VoChn);

		s32Ret = ReleaseVideoframe(&stfdFrame, &stLivenessFrame, &stVOFrame);
		if(s32Ret != CVI_SUCCESS) {
			Exit();
			assert(0 && "release video frame error!\n");
		}

		if (face.face_info != NULL) free(face.face_info);
	}
}

static void SampleHandleSig(CVI_S32 signo)
{
	signal(SIGINT, SIG_IGN);
	signal(SIGTERM, SIG_IGN);

	if (SIGINT == signo || SIGTERM == signo) {
		Exit();
	}
}

static void SetFacelibAttr(cv183x_facelib_config_t *facelib_config)
{
	memset(facelib_config,0,sizeof(cv183x_facelib_config_t));
	facelib_config->fd_en = 1;
	facelib_config->facereg_en = 0;
	facelib_config->face_matching_en = 0;
	facelib_config->config_liveness = 1;
	facelib_config->model_face_fd = "/mnt/data/retina_mobile.cvimodel";
	facelib_config->model_face_liveness = "/mnt/data/liveness_batch9.cvimodel";
	facelib_config->model_face_extr = "/mnt/data/bmface.cvimodel";
	// facelib_config->model_yolo3 = "/mnt/data/cvimodel/yolo_v3_320_int8_lw_memopt.cvimodel";
	// facelib_config->model_face_thermal = "mnt/data/thermal_face_detection.bf16sigmoid.cvimodel";
}

static void SetVIConfig(SAMPLE_VI_CONFIG_S* stViConfig)
{
	CVI_S32 s32WorkSnsId = 0;

	SAMPLE_SNS_TYPE_E  enSnsType        = SONY_IMX307_MIPI_2M_30FPS_12BIT;
	WDR_MODE_E	   enWDRMode	    	= WDR_MODE_NONE;
	DYNAMIC_RANGE_E    enDynamicRange   = DYNAMIC_RANGE_SDR8;
	PIXEL_FORMAT_E     enPixFormat	    = PIXEL_FORMAT_YUV_PLANAR_420;
	VIDEO_FORMAT_E     enVideoFormat    = VIDEO_FORMAT_LINEAR;
	COMPRESS_MODE_E    enCompressMode   = COMPRESS_MODE_NONE;
	VI_VPSS_MODE_E	   enMastPipeMode   = VI_OFFLINE_VPSS_OFFLINE;

	SAMPLE_COMM_VI_GetSensorInfo(stViConfig);

	stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType	     = enSnsType;
	stViConfig->s32WorkingViNum				     = 1;
	stViConfig->as32WorkingViId[0]				     = 0;
	stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.MipiDev	     = 0xFF;
	stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.s32BusId	     = 3;
	stViConfig->astViInfo[s32WorkSnsId].stDevInfo.ViDev	     = 0;
	stViConfig->astViInfo[s32WorkSnsId].stDevInfo.enWDRMode	     = enWDRMode;
	stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.enMastPipeMode = enMastPipeMode;
	stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[0]	     = ViPipe;
	stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[1]	     = -1;
	stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[2]	     = -1;
	stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[3]	     = -1;
	stViConfig->astViInfo[s32WorkSnsId].stChnInfo.ViChn	     = 0;
	stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enPixFormat     = enPixFormat;
	stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enDynamicRange  = enDynamicRange;
	stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enVideoFormat   = enVideoFormat;
	stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode  = enCompressMode;
}

static CVI_S32 SetVoConfig(SAMPLE_VO_CONFIG_S* stVoConfig)
{
	RECT_S stDefDispRect  = {0, 0, 720, 1280};
	SIZE_S stDefImageSize = {720, 1280};
	CVI_S32 s32Ret = CVI_SUCCESS;
	s32Ret = SAMPLE_COMM_VO_GetDefConfig(stVoConfig);
	if (s32Ret != CVI_SUCCESS) {
		printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", s32Ret);
		return s32Ret;
	}

	stVoConfig->VoDev	 = 0;
	stVoConfig->enVoIntfType  = VO_INTF_MIPI;
	stVoConfig->enIntfSync    = VO_OUTPUT_720x1280_60;
	stVoConfig->stDispRect	 = stDefDispRect;
	stVoConfig->stImageSize	 = stDefImageSize;
	stVoConfig->enPixFormat	 = PIXEL_FORMAT_YUV_PLANAR_420;
	stVoConfig->enVoMode	 = VO_MODE_1MUX;
	return s32Ret;
}

static CVI_S32 InitVPSS()
{
	CVI_S32 s32Ret = CVI_SUCCESS;
	VPSS_GRP_ATTR_S stVpssGrpAttr;
	CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = { 0 };
	VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

	abChnEnable[VpssChn] = CVI_TRUE;
	stVpssChnAttr[VpssChn].u32Width = 608;
	stVpssChnAttr[VpssChn].u32Height = 608;
	stVpssChnAttr[VpssChn].enVideoFormat = VIDEO_FORMAT_LINEAR;
	stVpssChnAttr[VpssChn].enPixelFormat = PIXEL_FORMAT_RGB_888_PLANAR;
	stVpssChnAttr[VpssChn].stFrameRate.s32SrcFrameRate = 30;
	stVpssChnAttr[VpssChn].stFrameRate.s32DstFrameRate = 30;
	stVpssChnAttr[VpssChn].u32Depth = 1;
	stVpssChnAttr[VpssChn].bMirror = CVI_FALSE;
    stVpssChnAttr[VpssChn].bFlip = CVI_FALSE;
	stVpssChnAttr[VpssChn].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
	stVpssChnAttr[VpssChn].stAspectRatio.bEnableBgColor = CVI_TRUE;
	stVpssChnAttr[VpssChn].stAspectRatio.u32BgColor  = COLOR_RGB_BLACK;
	stVpssChnAttr[VpssChn].stNormalize.bEnable = CVI_FALSE;
	stVpssChnAttr[VpssChn].stNormalize.factor[0] = (128/ 255.001236);
	stVpssChnAttr[VpssChn].stNormalize.factor[1] = (128/ 255.001236);
	stVpssChnAttr[VpssChn].stNormalize.factor[2] = (128 / 255.001236);
	stVpssChnAttr[VpssChn].stNormalize.mean[0] = 0;
	stVpssChnAttr[VpssChn].stNormalize.mean[1] = 0;
	stVpssChnAttr[VpssChn].stNormalize.mean[2] = 0;
	stVpssChnAttr[VpssChn].stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;

	abChnEnable[VpssChnLiveness] = CVI_TRUE;
	stVpssChnAttr[VpssChnLiveness].u32Width = 640;
	stVpssChnAttr[VpssChnLiveness].u32Height = 480;
	stVpssChnAttr[VpssChnLiveness].enVideoFormat = VIDEO_FORMAT_LINEAR;
	stVpssChnAttr[VpssChnLiveness].enPixelFormat = PIXEL_FORMAT_RGB_888;
	stVpssChnAttr[VpssChnLiveness].stFrameRate.s32SrcFrameRate = 30;
	stVpssChnAttr[VpssChnLiveness].stFrameRate.s32DstFrameRate = 30;
	stVpssChnAttr[VpssChnLiveness].u32Depth = 1;
	stVpssChnAttr[VpssChnLiveness].bMirror = CVI_FALSE;
    stVpssChnAttr[VpssChnLiveness].bFlip = CVI_FALSE;
	stVpssChnAttr[VpssChnLiveness].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
	stVpssChnAttr[VpssChnLiveness].stAspectRatio.bEnableBgColor = CVI_TRUE;
	stVpssChnAttr[VpssChnLiveness].stAspectRatio.u32BgColor  = COLOR_RGB_BLACK;
	stVpssChnAttr[VpssChnLiveness].stNormalize.bEnable         = CVI_FALSE;

	abChnEnable[VpssChnVO] = CVI_TRUE;
	stVpssChnAttr[VpssChnVO].u32Width                    = 1280;
	stVpssChnAttr[VpssChnVO].u32Height                   = 720;
	stVpssChnAttr[VpssChnVO].enVideoFormat               = VIDEO_FORMAT_LINEAR;
	stVpssChnAttr[VpssChnVO].enPixelFormat               = PIXEL_FORMAT_YUV_PLANAR_420;
	stVpssChnAttr[VpssChnVO].stFrameRate.s32SrcFrameRate = 30;
	stVpssChnAttr[VpssChnVO].stFrameRate.s32DstFrameRate = 30;
	stVpssChnAttr[VpssChnVO].u32Depth                    = 1;
	stVpssChnAttr[VpssChnVO].bMirror                     = CVI_FALSE;
	stVpssChnAttr[VpssChnVO].bFlip                       = CVI_FALSE;
	stVpssChnAttr[VpssChnVO].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
	// stVpssChnAttr[VpssChnVO].stAspectRatio.enMode        = ASPECT_RATIO_MANUAL;
	// stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.s32X = 0;
	// stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.s32Y = 0;
	// stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.u32Width = 1280;
	// stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.u32Height = 720;
	stVpssChnAttr[VpssChnVO].stAspectRatio.bEnableBgColor = CVI_TRUE;
	stVpssChnAttr[VpssChnVO].stNormalize.bEnable         = CVI_FALSE;

	CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

	stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
	stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
	stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
	stVpssGrpAttr.u32MaxW = vpssgrp_width;
	stVpssGrpAttr.u32MaxH = vpssgrp_height;
	// only for test here. u8VpssDev should be decided by VPSS_MODE and usage.
	stVpssGrpAttr.u8VpssDev = 0;

	/*start vpss*/
	s32Ret = SAMPLE_COMM_VPSS_Init(VpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
	if (s32Ret != CVI_SUCCESS) {
		printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
		return s32Ret;
	}

	s32Ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
	if (s32Ret != CVI_SUCCESS) {
		printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
		return s32Ret;
	}

	s32Ret = SAMPLE_COMM_VI_Bind_VPSS(ViPipe, VpssChn, VpssGrp);
	if (s32Ret != CVI_SUCCESS) {
		printf("vi bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
		return s32Ret;
	}

	return s32Ret;
}

static int InitVI(SAMPLE_VI_CONFIG_S* stViConfig)
{
	VB_CONFIG_S stVbConf;
	PIC_SIZE_E enPicSize;
	CVI_U32 u32BlkSize, u32BlkRotSize,u32BlkRGBSize;
	SIZE_S stSize;
	CVI_S32 s32Ret = CVI_SUCCESS;

	VI_DEV ViDev = 0;
	VI_CHN ViChn = 0;
	CVI_S32 s32WorkSnsId = 0;
	ISP_BIND_ATTR_S stBindAttr;

	VI_DEV_ATTR_S      stViDevAttr;
	VI_CHN_ATTR_S      stChnAttr;
	VI_PIPE_ATTR_S     stPipeAttr;

	s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType, &enPicSize);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
		return s32Ret;
	}

	s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
		return s32Ret;
	}

	memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
	stVbConf.u32MaxPoolCnt		= 2;

	u32BlkSize = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height, SAMPLE_PIXEL_FORMAT,
										 DATA_BITWIDTH_8, stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode,
										 DEFAULT_ALIGN);
	u32BlkRotSize = COMMON_GetPicBufferSize(stSize.u32Height, stSize.u32Width, SAMPLE_PIXEL_FORMAT,
											DATA_BITWIDTH_8, stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode,
											DEFAULT_ALIGN);
	u32BlkSize = MAX(u32BlkSize, u32BlkRotSize);
	u32BlkRGBSize = COMMON_GetPicBufferSize(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888,
											DATA_BITWIDTH_8, stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode,
											DEFAULT_ALIGN);
	stVbConf.astCommPool[0].u32BlkSize	= u32BlkSize;
	stVbConf.astCommPool[0].u32BlkCnt	= 32;
	stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
	stVbConf.astCommPool[1].u32BlkSize	= u32BlkRGBSize;
	stVbConf.astCommPool[1].u32BlkCnt	= 2;
	stVbConf.astCommPool[1].enRemapMode = VB_REMAP_MODE_CACHED;
	SAMPLE_PRT("common pool[0] BlkSize %d\n", u32BlkSize);

	s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("system init failed with %#x\n", s32Ret);
		return -1;
	}

	s32Ret = SAMPLE_COMM_VI_StartSensor(stViConfig);
	if (s32Ret != CVI_SUCCESS) {
		CVI_TRACE_LOG(CVI_DBG_ERR, "system start sensor failed with %#x\n", s32Ret);
		return s32Ret;
	}
	SAMPLE_COMM_VI_StartDev(&stViConfig->astViInfo[ViDev]);
	SAMPLE_COMM_VI_StartMIPI(stViConfig);

	stPipeAttr.bYuvSkip = CVI_FALSE;
	stPipeAttr.u32MaxW = stSize.u32Width;
	stPipeAttr.u32MaxH = stSize.u32Height;
	stPipeAttr.enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP;
	stPipeAttr.enBitWidth = DATA_BITWIDTH_12;
	stPipeAttr.stFrameRate.s32SrcFrameRate = -1;
	stPipeAttr.stFrameRate.s32DstFrameRate = -1;
	stPipeAttr.bNrEn = CVI_TRUE;
	s32Ret = CVI_VI_CreatePipe(ViPipe, &stPipeAttr);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("CVI_VI_CreatePipe failed with %#x!\n", s32Ret);
		return s32Ret;
	}

	s32Ret = CVI_VI_StartPipe(ViPipe);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("CVI_VI_StartPipe failed with %#x!\n", s32Ret);
		return s32Ret;
	}

	s32Ret = CVI_VI_GetPipeAttr(ViPipe, &stPipeAttr);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("CVI_VI_StartPipe failed with %#x!\n", s32Ret);
		return s32Ret;
	}

    s32Ret = SAMPLE_COMM_VI_CreateIsp(stViConfig);
    if (s32Ret != CVI_SUCCESS) {
        CVI_TRACE_LOG(CVI_DBG_ERR, "VI_CreateIsp failed with %#x!\n", s32Ret);
        return s32Ret;
    }

	SAMPLE_COMM_VI_StartViChn(&stViConfig->astViInfo[ViDev]);
}

int InitVO(SAMPLE_VO_CONFIG_S* stVoConfig)
{
	CVI_S32 s32Ret = CVI_SUCCESS;

	s32Ret = SAMPLE_COMM_VO_StartVO(stVoConfig);
	if (s32Ret != CVI_SUCCESS) {
		printf("SAMPLE_COMM_VO_StartVO failed with %#x\n", s32Ret);
	}

	printf("SAMPLE_COMM_VO_StartVO done\n");
	return s32Ret;
}

int main(void)
{
	CVI_S32 s32Ret = CVI_SUCCESS;
	VPSS_GRP_ATTR_S stVpssGrpAttr;
	cv183x_facelib_config_t facelib_config;

	signal(SIGINT, SampleHandleSig);
	signal(SIGTERM, SampleHandleSig);

	SetVIConfig(&stViConfig);
	s32Ret = InitVI(&stViConfig);
	if (s32Ret != CVI_SUCCESS) {
		printf("Init video input failed with %d\n", s32Ret);
		return s32Ret;
	}

	s32Ret = SetVoConfig(&stVoConfig);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("SetVoConfig failed with %d\n", s32Ret);
		return s32Ret;
	}

	s32Ret = InitVO(&stVoConfig);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("CVI_Init_Video_Output failed with %d\n", s32Ret);
		return s32Ret;
	}
	CVI_VO_HideChn(VoLayer,VoChn);

	s32Ret = InitVPSS();
	if (s32Ret != CVI_SUCCESS) {
		printf("Init video process group 0 failed with %d\n", s32Ret);
		return s32Ret;
	}

	SetFacelibAttr(&facelib_config);
	s32Ret = Cv183xFaceLibOpen(&facelib_config, &facelib_handle);
    if (s32Ret != CVI_SUCCESS) {
        printf("Facelib open failed with %#x!\n", s32Ret);
        return s32Ret;
    }

	Run();

	Exit();
}
