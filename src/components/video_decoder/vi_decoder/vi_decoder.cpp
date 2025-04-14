#include "framework/image/base_image.hpp"
#include "vi_decoder/vi_cfg.hpp"
#include "vi_decoder/vi_decoder.hpp"

static pthread_t g_IspPid[VI_MAX_DEV_NUM];
static SIZE_S stSize[VI_MAX_DEV_NUM] = {0};
static TDLViCfg stIniCfg;

static void callback_FPS(int fps)
{
  static CVI_FLOAT uMaxFPS[VI_MAX_DEV_NUM] = {0};
  CVI_U32 i;
  for (i = 0; i < VI_MAX_DEV_NUM && g_IspPid[i]; i++) {
    ISP_PUB_ATTR_S pubAttr = {0};

    CVI_ISP_GetPubAttr(i, &pubAttr);
    if (uMaxFPS[i] == 0) {
      uMaxFPS[i] = pubAttr.f32FrameRate;
    }
    if (fps == 0) {
      pubAttr.f32FrameRate = uMaxFPS[i];
    } else {
      pubAttr.f32FrameRate = (CVI_FLOAT) fps;
    }
    CVI_ISP_SetPubAttr(i, &pubAttr);
  }
}

static void *ISP_Thread(void *arg)
{
	CVI_S32 s32Ret = 0;
	CVI_U8 IspDev = *(CVI_U8 *)arg;
	char szThreadName[20];

  free(arg);
	snprintf(szThreadName, sizeof(szThreadName), "ISP%d_RUN", IspDev);
	prctl(PR_SET_NAME, szThreadName, 0, 0, 0);

	if (IspDev > 0) {
		printf("ISP Dev %d return\n", IspDev);
		return NULL;
	}

	CVI_SYS_RegisterThermalCallback(callback_FPS);

	printf("ISP Dev %d running!\n", IspDev);
	s32Ret = CVI_ISP_Run(IspDev);
	if (s32Ret != 0)
    printf("CVI_ISP_Run failed with %#x!\n", s32Ret);

	return NULL;
}

static int TDL_Vi_PQBinLoad(void)
{
  CVI_S32 ret = CVI_SUCCESS;
  FILE *fp = NULL;
  CVI_U8 *buf = NULL;
  CVI_U64 file_size;

  CVI_CHAR binName[BIN_FILE_LENGTH] = {0};
  ret = CVI_BIN_GetBinName(binName);
  if (ret != 0) {
      printf("CVI_BIN_GetBinName failed\n");
      return CVI_FAILURE;
  }

  fp = fopen((const CVI_CHAR *)binName, "rb");
  if (fp == NULL) {
    printf("Can't find bin(%s), use default parameters\n", binName);
    return CVI_FAILURE;
  }

  fseek(fp, 0L, SEEK_END);
  file_size = ftell(fp);
  rewind(fp);

  buf = (CVI_U8 *)malloc(file_size);
  if (buf == NULL) {
    printf("%s\n", "Allocae memory fail");
    fclose(fp);
    return CVI_FAILURE;
  }

  fread(buf, file_size, 1, fp);

  if (fp != NULL) {
    fclose(fp);
  }
  ret = CVI_BIN_ImportBinData(buf, (CVI_U32)file_size);
  if (ret != CVI_SUCCESS) {
    printf("CVI_BIN_ImportBinData error! value:(0x%x)\n", ret);
    free(buf);
    return CVI_FAILURE;
  }

  free(buf);

  return CVI_SUCCESS;
}

static int TDL_Vi_SysInit() {

  int ret = 0;

  VB_CONFIG_S stVbConf;
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  stVbConf.u32MaxPoolCnt = stIniCfg.devNum;
  for (int i = 0; i < stIniCfg.devNum; i++) {
    TDL_Vi_GetSize(stIniCfg.enSnsType[i], &stSize[i]);
    stVbConf.astCommPool[i].u32BlkSize = stSize[i].u32Height * stSize[i].u32Width * 3 / 2;
    stVbConf.astCommPool[i].u32BlkCnt = 2;
    stVbConf.astCommPool[i].enRemapMode	= VB_REMAP_MODE_CACHED;
  }

  ret = CVI_VB_SetConfig(&stVbConf);
  if (ret != 0) {
    printf("CVI_VB_SetConfig failed with %d\n", ret);
    return ret;
  }

  ret = CVI_VB_Init();
  if (ret != 0) {
    printf("CVI_VB_Init failed with %d\n", ret);
    return ret;
  }

  ret = CVI_SYS_Init();
  if (ret != 0) {
    printf("CVI_SYS_Init failed with %d\n", ret);
    CVI_VB_Exit();
    return ret;
  }
}

static int TDL_Vi_StartSensor() {
  int ret = 0;
  ISP_SNS_OBJ_S *pfnSnsObj = CVI_NULL;
  RX_INIT_ATTR_S rx_init_attr;
  ISP_INIT_ATTR_S isp_init_attr;
  ISP_SNS_COMMBUS_U sns_bus_info;
  ALG_LIB_S ae_lib;
  ALG_LIB_S awb_lib;
  ISP_CMOS_SENSOR_IMAGE_MODE_S isp_cmos_sensor_image_mode;
  ISP_SENSOR_EXP_FUNC_S isp_sensor_exp_func;
  ISP_PUB_ATTR_S stPubAttr;

  for (int i = 0; i < stIniCfg.devNum; i++) {
    pfnSnsObj = TDL_Vi_SnsObjGet(stIniCfg.enSnsType[i]);
    if (pfnSnsObj == CVI_NULL) {
      printf("sensor obj(%d) is null\n", i);
      return -1;
    }

    memset(&rx_init_attr, 0, sizeof(RX_INIT_ATTR_S));
    rx_init_attr.MipiDev = stIniCfg.MipiDev[i];
    if (stIniCfg.stMclkAttr[i].bMclkEn) {
      rx_init_attr.stMclkAttr.bMclkEn = CVI_TRUE;
      rx_init_attr.stMclkAttr.u8Mclk  = stIniCfg.stMclkAttr[i].u8Mclk;
    }

    for (int j = 0; j <= MIPI_LANE_NUM; j++) {
      rx_init_attr.as16LaneId[j] = stIniCfg.as16LaneId[i][j];
    }
    for (int j = 0; j <= MIPI_LANE_NUM; j++) {
      rx_init_attr.as8PNSwap[j] = stIniCfg.as8PNSwap[i][j];
    }

    if (pfnSnsObj->pfnPatchRxAttr) {
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || defined(__CV183X__)
      ret = pfnSnsObj->pfnPatchRxAttr(&rx_init_attr);
#else
      rx_init_attr.MipiMode = stIniCfg.enSnsMode;
      for (int j = 0; j < 20; j++) {
        rx_init_attr.as16FuncId[i] = stIniCfg.as16FuncId[i][j];
      }
      ret = pfnSnsObj->pfnPatchRxAttr(i, &rx_init_attr);
#endif
      if (ret != 0) {
        printf("pfnPatchRxAttr(%d) failed!\n", i);
      }
    }

    memset(&isp_init_attr, 0, sizeof(ISP_INIT_ATTR_S));
    isp_init_attr.u16UseHwSync = stIniCfg.u8HwSync[i];
    if (pfnSnsObj->pfnSetInit) {
      ret = pfnSnsObj->pfnSetInit(i, &isp_init_attr);
      if (ret != 0) {
        printf("pfnSetInit(%d) fail with %d\n", i, ret);
      }
    }

    memset(&sns_bus_info, 0, sizeof(ISP_SNS_COMMBUS_U));
    sns_bus_info.s8I2cDev = (stIniCfg.s32BusId[i] >= 0) ? (CVI_S8)stIniCfg.s32BusId[i] : 0x3;
    if (pfnSnsObj->pfnSetBusInfo) {
      ret = pfnSnsObj->pfnSetBusInfo(i, sns_bus_info);
      if (ret != 0) {
        printf("pfnSetBusInfo(%d) fail with %d\n", i, ret);
      }
    }

    if (pfnSnsObj->pfnPatchI2cAddr) {
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || defined(__CV183X__)
      pfnSnsObj->pfnPatchI2cAddr(stIniCfg.s32SnsI2cAddr[i]);
#else
      pfnSnsObj->pfnPatchI2cAddr(i, stIniCfg.s32SnsI2cAddr[i]);
#endif
    }

    awb_lib.s32Id = i;
    ae_lib.s32Id = i;
    strncpy(ae_lib.acLibName, CVI_AE_LIB_NAME, sizeof(ae_lib.acLibName));
    strncpy(awb_lib.acLibName, CVI_AWB_LIB_NAME, sizeof(awb_lib.acLibName));
    if (pfnSnsObj->pfnRegisterCallback) {
      pfnSnsObj->pfnRegisterCallback(i, &ae_lib, &awb_lib);
    }

    CVI_AE_Register(i, &ae_lib);
    CVI_AWB_Register(i, &awb_lib);

    memset(&isp_cmos_sensor_image_mode, 0, sizeof(ISP_CMOS_SENSOR_IMAGE_MODE_S));
    TDL_Vi_IspPubAttr(stIniCfg.enSnsType[i], &stPubAttr);

    isp_cmos_sensor_image_mode.u16Width  = stSize[i].u32Width;
    isp_cmos_sensor_image_mode.u16Height = stSize[i].u32Height;
    isp_cmos_sensor_image_mode.f32Fps    = stPubAttr.f32FrameRate;
    if (pfnSnsObj->pfnExpSensorCb) {
      ret = pfnSnsObj->pfnExpSensorCb(&isp_sensor_exp_func);
      if (ret != 0) {
        printf("pfnExpSensorCb(%d) fail with %d", i, ret);
        return ret;
      }

      ret = isp_sensor_exp_func.pfn_cmos_set_image_mode(i, &isp_cmos_sensor_image_mode);
      if (ret != 0) {
        printf("pfn_cmos_set_image_mode(%d) fail with %d", i, ret);
        return ret;
      }

      ret = isp_sensor_exp_func.pfn_cmos_set_wdr_mode(i, stIniCfg.enWDRMode[i]);
      if (ret != 0) {
        printf("pfn_cmos_set_wdr_mode(%d) fail with %d", i, ret);
        return ret;
      }
    }
  }

  return ret;
}

static int TDL_Vi_StartDev() {
  int ret = 0;
  VI_DEV_ATTR_S  stViDevAttr;

  for (int i = 0; i < stIniCfg.devNum; i++) {
    memset(&stViDevAttr, 0, sizeof(VI_DEV_ATTR_S));
    TDL_Vi_DevAttrGet(stIniCfg.enSnsType[i], &stViDevAttr);
    stViDevAttr.stSize.u32Width = stSize[i].u32Width;
    stViDevAttr.stSize.u32Height = stSize[i].u32Height;
    stViDevAttr.stWDRAttr.enWDRMode = stIniCfg.enWDRMode[i];

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || defined(__CV183X__)
    if (stIniCfg.u8MuxDev[i]) {
      stViDevAttr.isMux = true;
      stViDevAttr.switchGpioPin = stIniCfg.s16SwitchGpio[i];
      stViDevAttr.switchGPioPol = stIniCfg.u8SwitchPol[i];
    }
#endif

#if !(defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || defined(__CV183X__))
    VI_DEV_BIND_PIPE_S  stViDevBindAttr;
    stViDevBindAttr.PipeId[0] = (CVI_S32)stIniCfg.MipiDev[i];
    stViDevBindAttr.u32Num = 1;
    ret = CVI_VI_SetDevBindAttr(i, &stViDevBindAttr);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VI_SetDevBindAttr(%d) fail with %d", i, ret);
      return ret;
    }
#endif

    ret = CVI_VI_SetDevAttr(i, &stViDevAttr);
    if (ret != 0) {
      printf("CVI_VI_SetDevAttr(%d) fail with %d", i, ret);
      return ret;
    }

    ret = CVI_VI_EnableDev(i);
    if (ret != 0) {
      printf("CVI_VI_EnableDev(%d) fail with %d", i, ret);
      return ret;
    }
  }

  return ret;
}

static int TDL_Vi_StartMipi() {
  int ret = 0;
  SNS_COMBO_DEV_ATTR_S combo_dev_attr;
  ISP_SNS_OBJ_S *pfnSnsObj = CVI_NULL;

  for (int i = 0; i < stIniCfg.devNum; i++) {
    pfnSnsObj = TDL_Vi_SnsObjGet(stIniCfg.enSnsType[i]);
    if (pfnSnsObj == CVI_NULL) {
      printf("sensor obj(%d) is null\n", i);
      return -1;
    }

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || defined(__CV183X__)
    CVI_MIPI_SetSensorReset(stIniCfg.MipiDev[i], 1);
#else
    SNS_RST_CONFIG pstSnsrstInfo;
    pstSnsrstInfo.devno = stIniCfg.MipiDev[i];
    pstSnsrstInfo.gpio_pin = stIniCfg.s32RstPin[i];
    pstSnsrstInfo.gpio_active = stIniCfg.s32RstActive[i] == 0 ? RST_ACTIVE_LOW : RST_ACTIVE_HIGH;
    CVI_MIPI_SetSensorReset(&pstSnsrstInfo, 1);
#endif
    CVI_MIPI_SetMipiReset(stIniCfg.MipiDev[i], 1);

    memset(&combo_dev_attr, 0, sizeof(SNS_COMBO_DEV_ATTR_S));
    if (pfnSnsObj->pfnGetRxAttr) {
      ret = pfnSnsObj->pfnGetRxAttr(i, &combo_dev_attr);
      if (ret != 0) {
        printf("pfnGetRxAttr(%d) fail with %d", i, ret);
        return ret;
      }
    }

    ret = CVI_MIPI_SetMipiAttr(i, (CVI_VOID*)&combo_dev_attr);
    if (ret != 0) {
      printf("CVI_MIPI_SetMipiAttr(%d) fail with %d", i, ret);
      return ret;
    }

    CVI_MIPI_SetSensorClock(stIniCfg.MipiDev[i], 1);
    usleep(20);
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || defined(__CV183X__)
    CVI_MIPI_SetSensorReset(stIniCfg.MipiDev[i], 0);
#else
    CVI_MIPI_SetSensorReset(&pstSnsrstInfo, 0);
#endif
    if (pfnSnsObj->pfnSnsProbe) {
      ret = pfnSnsObj->pfnSnsProbe(i);
      if (ret != 0) {
        printf("pfnSnsProbe(%d) fail with %d", i, ret);
        return ret;
      }
    }
  }
  return ret;
}

static int TDL_Vi_StartPipe() {
  int ret = 0;
  VI_PIPE_ATTR_S stViPipeAttr;

  for (int i = 0; i < stIniCfg.devNum; i++) {
    memset(&stViPipeAttr, 0, sizeof(VI_PIPE_ATTR_S));
    TDL_Vi_PipeAttrGet(stIniCfg.enSnsType[i], &stViPipeAttr);

    ret = CVI_VI_CreatePipe(i, &stViPipeAttr);
    if (ret != 0) {
      printf("CVI_VI_CreatePipe(%d) fail with %d", i, ret);
      return ret;
    }

    ret = CVI_VI_StartPipe(i);
    if (ret != 0) {
      printf("CVI_VI_StartPipe(%d) fail with %d", i, ret);
      return ret;
    }
  }
  return ret;
}

static int TDL_Vi_InitIsp() {
  int ret = 0;
  ISP_BIND_ATTR_S stBindAttr;
  ISP_PUB_ATTR_S stPubAttr;

  for (int i = 0; i < stIniCfg.devNum; i++) {
    memset(&stBindAttr, 0, sizeof(ISP_BIND_ATTR_S));
    snprintf(stBindAttr.stAeLib.acLibName, sizeof(CVI_AE_LIB_NAME), "%s", CVI_AE_LIB_NAME);
    stBindAttr.stAeLib.s32Id = i;
    snprintf(stBindAttr.stAwbLib.acLibName, sizeof(CVI_AWB_LIB_NAME), "%s", CVI_AWB_LIB_NAME);
    stBindAttr.stAwbLib.s32Id = i;
    ret = CVI_ISP_SetBindAttr(i, &stBindAttr);
    if (ret != 0) {
      printf("CVI_ISP_SetBindAttr(%d) fail with %d", i, ret);
      return ret;
    }

    ret = CVI_ISP_MemInit(i);
    if (ret != 0) {
      printf("CVI_ISP_MemInit(%d) fail with %d", i, ret);
      return ret;
    }

    memset(&stPubAttr, 0, sizeof(ISP_PUB_ATTR_S));
    TDL_Vi_IspPubAttr(stIniCfg.enSnsType[i], &stPubAttr);
    stPubAttr.stWndRect.u32Width  = stSize[i].u32Width;
    stPubAttr.stWndRect.u32Height = stSize[i].u32Height;
    stPubAttr.stSnsSize.u32Width  = stSize[i].u32Width;
    stPubAttr.stSnsSize.u32Height = stSize[i].u32Height;

    ret = CVI_ISP_SetPubAttr(i, &stPubAttr);
    if (ret != 0) {
      printf("CVI_ISP_SetPubAttr(%d) fail with %d", i, ret);
      return ret;
    }

    ret = CVI_ISP_Init(i);
    if (ret != 0) {
      printf("CVI_ISP_Init(%d) fail with %d", i, ret);
      return ret;
    }
  }
  TDL_Vi_PQBinLoad();
  return ret;
}

static int TDL_Vi_IspRun(CVI_U8 IspDev) {
  CVI_S32 s32Ret = 0;
	CVI_U8 *arg = (CVI_U8 *)malloc(sizeof(*arg));
	struct sched_param param;
	pthread_attr_t attr;

	if (arg == NULL) {
		CVI_TRACE_LOG(CVI_DBG_ERR, "malloc failed\n");
	}

	*arg = IspDev;
	param.sched_priority = 80;

	pthread_attr_init(&attr);
	pthread_attr_setschedpolicy(&attr, SCHED_RR);
	pthread_attr_setschedparam(&attr, &param);
	pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
	s32Ret = pthread_create(&g_IspPid[IspDev], &attr, ISP_Thread, arg);
	if (s32Ret != 0) {
		printf("create isp running thread failed!, error: %d, %s\r\n",
					s32Ret, strerror(s32Ret));
	}
  pthread_attr_destroy(&attr);
  return s32Ret;
}

static int TDL_Vi_StartIsp() {
  int ret = 0;
  for (int i = 0; i < stIniCfg.devNum; i ++) {
    TDL_Vi_IspRun(i);
  }
  return ret;
}

static int TDL_Vi_StartChn() {
  int ret = 0;
  VI_CHN_ATTR_S stViChnAttr;
  ISP_SNS_OBJ_S *pfnSnsObj = NULL;

  for (int i = 0; i < stIniCfg.devNum; i ++) {
    pfnSnsObj = TDL_Vi_SnsObjGet(stIniCfg.enSnsType[i]);
    if (pfnSnsObj == NULL) {
      printf("sensor obj(%d) is null\n", i);
      return -1;
    }

    memset(&stViChnAttr, 0, sizeof(VI_CHN_ATTR_S));
    stViChnAttr.stSize.u32Width  = stSize[i].u32Width;
    stViChnAttr.stSize.u32Height = stSize[i].u32Height;
    stViChnAttr.enPixelFormat = VI_PIXEL_FORMAT;
    stViChnAttr.u32Depth       = 0;
    stViChnAttr.enCompressMode	= COMPRESS_MODE_TILE;
    stViChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    stViChnAttr.bMirror = stIniCfg.u8Orien[i] & 0x1;
    stViChnAttr.bMirror = stIniCfg.u8Orien[i] & 0x2;

    ret = CVI_VI_SetChnAttr(i, i, &stViChnAttr);
    if (ret != 0) {
      printf("CVI_VI_SetChnAttr(%d) fail with %d", i, ret);
      return ret;
    }

    if (pfnSnsObj && pfnSnsObj->pfnMirrorFlip) {
      CVI_VI_RegChnFlipMirrorCallBack(i, i, (void *)pfnSnsObj->pfnMirrorFlip);
    }

    ret = CVI_VI_EnableChn(i, i);
    if (ret != 0) {
      printf("CVI_VI_EnableChn(%d) fail with %d", i, ret);
      return ret;
    }
  }
  return ret;
}

static int TDL_Vi_DestroyIsp() {
  int ret = 0;
  ISP_SNS_OBJ_S *pfnSnsObj = NULL;
  ALG_LIB_S stAeLib;
  ALG_LIB_S stAwbLib;
  for (int i = 0; i < stIniCfg.devNum; i ++) {
    if (g_IspPid[i]) {
      ret = CVI_ISP_Exit(i);
      if (ret != CVI_SUCCESS) {
        printf("CVI_ISP_Exit(%d) fail with %d", i, ret);
        return ret;
      }
      pthread_join(g_IspPid[i], NULL);
      g_IspPid[i] = 0;
    }

    pfnSnsObj = TDL_Vi_SnsObjGet(stIniCfg.enSnsType[i]);
    if (pfnSnsObj == NULL) {
      printf("sensor obj(%d) is null\n", i);
      return -1;
    }

    stAeLib.s32Id = i;
    stAwbLib.s32Id = i;
    strncpy(stAeLib.acLibName, CVI_AE_LIB_NAME, sizeof(stAeLib.acLibName));
    strncpy(stAwbLib.acLibName, CVI_AWB_LIB_NAME, sizeof(stAwbLib.acLibName));
    if (pfnSnsObj->pfnUnRegisterCallback) {
      ret = pfnSnsObj->pfnUnRegisterCallback(i, &stAeLib, &stAwbLib);
      if (ret != CVI_SUCCESS) {
        printf("fnUnRegisterCallback(%d) failed\n", i);
        return ret;
      }
    }

    CVI_AE_UnRegister(i, &stAeLib);
    CVI_AWB_UnRegister(i, &stAwbLib);
  }
  return ret;
}

static int TDL_Vi_DestroyVi() {
  int ret = 0;
  for (int i = 0; i < stIniCfg.devNum; i ++) {
    ret = CVI_VI_DisableChn(i, i);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VI_DisableChn(%d) failed\n", i);
      return ret;
    }

    ret = CVI_VI_StopPipe(i);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VI_StopPipe(%d) failed\n", i);
      return ret;
    }

    ret = CVI_VI_DestroyPipe(i);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VI_DestroyPipe(%d) failed\n", i);
      return ret;
    }

    ret = CVI_VI_DisableDev(i);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VI_DisableDev(%d) failed\n", i);
      return ret;
    }

    CVI_VI_UnRegChnFlipMirrorCallBack(0, i);
  }
  return ret;
}

static int TDL_Vi_SysExit() {
  int ret = 0;
  CVI_SYS_Exit();
  CVI_VB_Exit();
  return ret;
}

int ViDecoder::initialize() {
  int ret = 0;
  VB_CONFIG_S stVbConf;
  memset(&stIniCfg, 0, sizeof(TDLViCfg));
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  //Parse ini
  ret = TDL_Vi_ParseIni(&stIniCfg);
  if (ret != 0) {
    printf("TDL_Vi_ParseIni failed with %d\n", ret);
    return ret;
  }

  //Set sensor number
  CVI_VI_SetDevNum(stIniCfg.devNum);

  ret = TDL_Vi_SysInit();
  if (ret != 0) {
    printf("TDL_Vi_SysInit failed\n");
    return ret;
  }

  ret = TDL_Vi_StartSensor();
  if (ret != 0) {
    printf("TDL_Vi_StartSensor failed\n");
    return ret;
  }

  ret = TDL_Vi_StartDev();
  if (ret != 0) {
    printf("TDL_Vi_StartDev failed\n");
    return ret;
  }

  ret = TDL_Vi_StartMipi();
  if (ret != 0) {
    printf("TDL_Vi_StartMipi failed\n");
    return ret;
  }

  ret = TDL_Vi_StartPipe();
  if (ret != 0) {
    printf("TDL_Vi_StartPipe failed\n");
    return ret;
  }

  ret = TDL_Vi_InitIsp();
  if (ret != 0) {
    printf("TDL_Vi_InitIsp failed\n");
    return ret;
  }

  ret = TDL_Vi_StartIsp();
  if (ret != 0) {
    printf("TDL_Vi_StartIsp failed\n");
    return ret;
  }

  ret = TDL_Vi_StartChn();
  if (ret != 0) {
    printf("TDL_Vi_StarChn failed\n");
    return ret;
  }

  isInitialized = true;

  return ret;
}

int ViDecoder::deinitialize() {
  int ret = 0;
  TDL_Vi_DestroyIsp();
  TDL_Vi_DestroyVi();
  TDL_Vi_SysExit();
  isInitialized = false;
  return ret;
}

ViDecoder::ViDecoder() {
  type_ = VideoDecoderType::VI;
  if (!isInitialized) {
    initialize();
  }
}

ViDecoder::~ViDecoder() {
  if (isInitialized) {
    deinitialize();
  }
}

int32_t ViDecoder::init(const std::string &path, 
                        const std::map<std::string, int> &config) {
  path_ = path;

  if (!isInitialized) {
  return initialize();
  }

  return 0;
}

int32_t ViDecoder::read(std::shared_ptr<BaseImage> &image, int chn) {
  int ret = 0;
  VIDEO_FRAME_INFO_S frame_info;
  ret = CVI_VI_GetChnFrame(chn,chn, &frame_info, 3000);
  if (ret != 0) {
    printf("CVI_VI_GetChnFrame(%d) failed with %d\n", chn, ret);
    return ret;
  }

  image = ImageFactory::wrapVPSSFrame(&frame_info, false);

  ret = CVI_VI_ReleaseChnFrame(chn, chn, &frame_info);
  if (ret != 0) {
    printf("CVI_VI_GetChnFrame(%d) failed with %d\n", chn, ret);
    return ret;
  }

  return 0;
}