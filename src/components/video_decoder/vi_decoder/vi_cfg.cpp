#include "vi_decoder/vi_cfg.hpp"

static const char *snsr_type_name[SAMPLE_SNS_TYPE_BUTT] = {
    "GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT",
    "GCORE_GC2093_MIPI_2M_30FPS_10BIT",
    "GCORE_GC4653_MIPI_4M_30FPS_10BIT",
};

/*=== Source section parser handler begin === */
static void parse_source_type(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                              CVI_U32 param1, CVI_U32 param2) {
  (CVI_VOID) param0;
  (CVI_VOID) param1;
  (CVI_VOID) param2;

  printf("source type =  %s\n", value);
  if (strcmp(value, "SOURCE_USER_FE") == 0) {
    cfg->enSource = VI_PIPE_FRAME_SOURCE_USER_FE;
  }
}

static void parse_source_devnum(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  int devno = atoi(value);

  (CVI_VOID) param0;
  (CVI_VOID) param1;
  (CVI_VOID) param2;

  printf("devNum =  %s\n", value);

  if (devno >= 1 && devno <= VI_MAX_DEV_NUM)
    cfg->devNum = devno;
  else
    cfg->devNum = 1;
}

static void parse_source_enmode(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  int devmode = atoi(value);

  (CVI_VOID) param0;
  (CVI_VOID) param1;
  (CVI_VOID) param2;

  printf("devmode =  %s\n", value);

  if (devmode >= 0 && devmode <= 6)
    cfg->enSnsMode = static_cast<TDLSnsType>(devmode);
  // else
  //	cfg->enSnsMode = 0;
}
/* === Source section parser handler end === */

/* === Sensor section parser handler begin === */
static int parse_lane_id(CVI_S16 *LaneId, const char *value) {
  char buf[8];
  int offset = 0, idx = 0, k;

  for (k = 0; k < MIPI_LANE_NUM * 6; k++) {
    /* find next ',' */
    if (value[k] == ',' || value[k] == '\0') {
      if (k == offset) {
        printf("lane_id parse error, is the format correct?\n");
        return -1;
      }
      memset(buf, 0, sizeof(buf));
      memcpy(buf, &value[offset], k - offset);
      buf[k - offset] = '\0';
      LaneId[idx++] = atoi(buf);
      offset = k + 1;
    }

    if (value[k] == '\0' || idx == MIPI_LANE_NUM + 1) break;
  }

  if (k == 60) {
    printf("lane_id parse error, is the format correct?\n");
    return -1;
  }

  return 0;
}

static int parse_lane_id_mars(CVI_S16 *LaneId, const char *value) {
  char buf[8];
  int offset = 0, idx = 0, k;

  for (k = 0; k < 30; k++) {
    /* find next ',' */
    if (value[k] == ',' || value[k] == '\0') {
      if (k == offset) {
        printf("lane_id parse error, is the format correct?\n");
        return -1;
      }
      memset(buf, 0, sizeof(buf));
      memcpy(buf, &value[offset], k - offset);
      buf[k - offset] = '\0';
      LaneId[idx++] = atoi(buf);
      offset = k + 1;
    }

    if (value[k] == '\0' || idx == 5) break;
  }

  if (k == 30) {
    printf("lane_id parse error, is the format correct?\n");
    return -1;
  }

  return 0;
}

static int parse_pn_swap(CVI_S8 *PNSwap, const char *value) {
  char buf[8];
  int offset = 0, idx = 0, k;

  for (k = 0; k < 30; k++) {
    /* find next ',' */
    if (value[k] == ',' || value[k] == '\0') {
      if (k == offset) {
        printf("lane_id parse error, is the format correct?\n");
        return -1;
      }
      memset(buf, 0, sizeof(buf));
      memcpy(buf, &value[offset], k - offset);
      buf[k - offset] = '\0';
      PNSwap[idx++] = atoi(buf);
      offset = k + 1;
    }

    if (value[k] == '\0' || idx == 5) break;
  }

  if (k == 30) {
    printf("lane_id parse error, is the format correct?\n");
    return -1;
  }

  return 0;
}

static void parse_sensor_name(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                              CVI_U32 param1, CVI_U32 param2) {
#define NAME_SIZE 20
  CVI_U32 index = param0;
  CVI_U32 i;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("sensor =  %s\n", value);
  char sensorNameEnv[NAME_SIZE];

  snprintf(sensorNameEnv, NAME_SIZE, "SENSORNAME%d", index);
  setenv(sensorNameEnv, value, 1);

  for (i = 0; i < SAMPLE_SNS_TYPE_BUTT; i++) {
    if (strcmp(value, snsr_type_name[i]) == 0) {
      cfg->enSnsType[index] = static_cast<TDLSnsType>(i);
      cfg->enWDRMode[index] =
          (static_cast<TDLSnsType>(i) < SAMPLE_SNS_TYPE_LINEAR_BUTT)
              ? WDR_MODE_NONE
              : WDR_MODE_2To1_LINE;
      break;
    }
  }
  if (i == SAMPLE_SNS_TYPE_BUTT) {
    cfg->enSnsType[index] = SAMPLE_SNS_TYPE_BUTT;
    cfg->enWDRMode[index] = WDR_MODE_NONE;
    cfg->u8UseMultiSns = index;
    printf("ERROR: can not find sensor ini in /mnt/data/\n");
  }
}

static void parse_sensor_busid(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                               CVI_U32 param1, CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("bus_id =  %s\n", value);
  cfg->s32BusId[index] = atoi(value);
}

static void parse_sensor_i2caddr(TDLViCfg *cfg, const char *value,
                                 CVI_U32 param0, CVI_U32 param1,
                                 CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("sns_i2c_addr =  %s\n", value);
  cfg->s32SnsI2cAddr[index] = atoi(value);
}

static void parse_sensor_mipidev(TDLViCfg *cfg, const char *value,
                                 CVI_U32 param0, CVI_U32 param1,
                                 CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("mipi_dev =  %s\n", value);
  cfg->MipiDev[index] = atoi(value);
}

static void parse_sensor_laneid(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("Lane_id =  %s\n", value);
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__)
  parse_lane_id_mars(cfg->as16LaneId[index], value);
#else
  parse_lane_id(cfg->as16LaneId[index], value);
#endif
}

static int parse_func_id(CVI_S16 *FuncId, const char *value) {
  char buf[8];
  int offset = 0, idx = 0, k;

  for (k = 0; k < 21 * 6; k++) {
    /* find next ',' */
    if (value[k] == ',' || value[k] == '\0') {
      if (k == offset) {
        printf("func_id parse error, is the format correct?\n");
        return -1;
      }
      memset(buf, 0, sizeof(buf));
      memcpy(buf, &value[offset], k - offset);
      buf[k - offset] = '\0';
      FuncId[idx++] = atoi(buf);
      offset = k + 1;
    }

    if (value[k] == '\0' || idx == 20 + 1) break;
  }

  if (k == 60) {
    printf("func_id parse error, is the format correct?\n");
    return -1;
  }

  return 0;
}

static void parse_sensor_funcid(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("Func_id =  %s\n", value);
  parse_func_id(cfg->as16FuncId[index], value);
}

static void parse_sensor_pnswap(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("pn_swap =  %s\n", value);
  parse_pn_swap(cfg->as8PNSwap[index], value);
}

static void parse_sensor_hwsync(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("hw_sync =  %s\n", value);
  cfg->u8HwSync[index] = atoi(value);
}

static void parse_sensor_mclken(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("mclk_en =  %s\n", value);
  cfg->stMclkAttr[index].bMclkEn = atoi(value);
}

static void parse_sensor_mclk(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                              CVI_U32 param1, CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("mclk =  %s\n", value);
  cfg->stMclkAttr[index].u8Mclk = atoi(value);
}

static void parse_sensor_orien(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                               CVI_U32 param1, CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("orien =  %s\n", value);
  cfg->u8Orien[index] = atoi(value);
}

static void parse_sensor_hsettlen(TDLViCfg *cfg, const char *value,
                                  CVI_U32 param0, CVI_U32 param1,
                                  CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("hs_settle enable =  %s\n", value);
  cfg->bHsettlen[index] = atoi(value);
}

static void parse_sensor_hsettle(TDLViCfg *cfg, const char *value,
                                 CVI_U32 param0, CVI_U32 param1,
                                 CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("hs_settle =  %s\n", value);
  cfg->u8Hsettle[index] = atoi(value);
}

static void parse_sensor_rstpin(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("rstpin =  %s\n", value);
  cfg->s32RstPin[index] = atoi(value);
}

static void parse_sensor_rstactive(TDLViCfg *cfg, const char *value,
                                   CVI_U32 param0, CVI_U32 param1,
                                   CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("rst_active =  %s\n", value);
  cfg->s32RstActive[index] = atoi(value);
}

static void parse_sensor_muxdev(TDLViCfg *cfg, const char *value,
                                CVI_U32 param0, CVI_U32 param1,
                                CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("muxdev =  %s\n", value);
  cfg->u8MuxDev[index] = atoi(value);
}

static void parse_sensor_attachdev(TDLViCfg *cfg, const char *value,
                                   CVI_U32 param0, CVI_U32 param1,
                                   CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("attach_dev =  %s\n", value);
  cfg->u8AttachDev[index] = atoi(value);
}

static void parse_sensor_switchgpio(TDLViCfg *cfg, const char *value,
                                    CVI_U32 param0, CVI_U32 param1,
                                    CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("switch_gpio =  %s\n", value);
  cfg->s16SwitchGpio[index] = atoi(value);
}

static void parse_sensor_switchpol(TDLViCfg *cfg, const char *value,
                                   CVI_U32 param0, CVI_U32 param1,
                                   CVI_U32 param2) {
  CVI_U32 index = param0;

  (CVI_VOID) param1;
  (CVI_VOID) param2;
  printf("switch_pol =  %s\n", value);
  cfg->u8SwitchPol[index] = atoi(value);
}

/* === Sensor section parser handler end === */
typedef CVI_VOID (*parser)(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                           CVI_U32 param1, CVI_U32 param2);

TDLIniHdlr stSectionSource[INI_SOURCE_NUM] = {
    [INI_SOURCE_TYPE] = {"type", 0, 0, 0, parse_source_type},
    [INI_SOURCE_DEVNUM] = {"dev_num", 0, 0, 0, parse_source_devnum},
    [INI_SOURCE_ENMODE] = {"en_mode", 0, 0, 0, parse_source_enmode},
};

TDLIniHdlr stSectionSensor[INI_SENSOR_NUM] = {
    [INI_SENSOR_NAME] = {"name", 0, 0, 0, parse_sensor_name},
    [INI_SENSOR_BUSID] = {"bus_id", 0, 0, 0, parse_sensor_busid},
    [INI_SENSOR_I2CADDR] = {"sns_i2c_addr", 0, 0, 0, parse_sensor_i2caddr},
    [INI_SENSOR_MIPIDEV] = {"mipi_dev", 0, 0, 0, parse_sensor_mipidev},
    [INI_SENSOR_LANEID] = {"lane_id", 0, 0, 0, parse_sensor_laneid},
    [INI_SENSOR_FUNCID] = {"func_id", 0, 0, 0, parse_sensor_funcid},
    [INI_SENSOR_PNSWAP] = {"pn_swap", 0, 0, 0, parse_sensor_pnswap},
    [INI_SENSOR_HWSYNC] = {"hw_sync", 0, 0, 0, parse_sensor_hwsync},
    [INI_SENSOR_MCLKEN] = {"mclk_en", 0, 0, 0, parse_sensor_mclken},
    [INI_SENSOR_MCLK] = {"mclk", 0, 0, 0, parse_sensor_mclk},
    [INI_SENSOR_SETTLEEN] = {"hs_settle_en", 0, 0, 0, parse_sensor_hsettlen},
    [INI_SENSOR_SETTLE] = {"hs_settle", 0, 0, 0, parse_sensor_hsettle},
    [INI_SENSOR_ORIEN] = {"orien", 0, 0, 0, parse_sensor_orien},
    [INI_SENSOR_RSTPIN] = {"rst_pin", 0, 0, 0, parse_sensor_rstpin},
    [INI_SENSOR_ACTIVE] = {"rst_active", 0, 0, 0, parse_sensor_rstactive},
    [INI_SENSOR_MUXDEV] = {"mux_dev", 0, 0, 0, parse_sensor_muxdev},
    [INI_SENSOR_ATTACHDEV] = {"attach_dev", 0, 0, 0, parse_sensor_attachdev},
    [INI_SENSOR_SWITCHGPIO] = {"switch_gpio", 0, 0, 0, parse_sensor_switchgpio},
    [INI_SENSOR_SWITCHPOL] = {"switch_pol", 0, 0, 0, parse_sensor_switchpol},
};

int parse_handler(void *user, const char *section, const char *name,
                  const char *value) {
  TDLViCfg *cfg = (TDLViCfg *)user;
  const TDLIniHdlr *hdler;
  int i, size, index = 0;

  if (strcmp(section, "source") == 0) {
    hdler = stSectionSource;
    size = INI_SOURCE_NUM;
  } else if (strcmp(section, "sensor") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 0;
  } else if (strcmp(section, "sensor2") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 1;
  } else if (strcmp(section, "sensor3") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 2;
  } else if (strcmp(section, "sensor4") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 3;
  } else if (strcmp(section, "sensor5") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 4;
  } else if (strcmp(section, "sensor6") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 5;
  } else if (strcmp(section, "sensor7") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 6;
  } else if (strcmp(section, "sensor8") == 0) {
    hdler = stSectionSensor;
    size = INI_SENSOR_NUM;
    index = 7;
  } else {
    /* unknown section/name */
    return 1;
  }

  if (hdler == stSectionSensor) {
    for (i = 0; i < size; i++) {
      stSectionSensor[i].param0 = index;
    }
  }

  for (i = 0; i < size; i++) {
    if (strcmp(name, hdler[i].name) == 0) {
      hdler[i].pfnJob(cfg, value, hdler[i].param0, hdler[i].param1,
                      hdler[i].param2);
      break;
    }
  }

  return 1;
}

ISP_PUB_ATTR_S isp_pub_attr_base = {
    .stWndRect = {0, 0, 1920, 1080},
    .stSnsSize = {1920, 1080},
    .f32FrameRate = 25.0f,
    .enBayer = BAYER_BGGR,
    .enWDRMode = WDR_MODE_NONE,
    .u8SnsMode = 0,
};

VI_DEV_ATTR_S vi_dev_attr_base = {
    .enIntfMode = VI_MODE_MIPI,
    .enWorkMode = VI_WORK_MODE_1Multiplex,
    .enScanMode = VI_SCAN_PROGRESSIVE,
    .as32AdChnId = {-1, -1, -1, -1},
    .enDataSeq = VI_DATA_SEQ_YUYV,
    .stSynCfg =
        {/*port_vsync    port_vsync_neg    port_hsync port_hsync_neg*/
         VI_VSYNC_PULSE,
         VI_VSYNC_NEG_LOW,
         VI_HSYNC_VALID_SINGNAL,
         VI_HSYNC_NEG_HIGH,
         /*port_vsync_valid     port_vsync_valid_neg*/
         VI_VSYNC_VALID_SIGNAL,
         VI_VSYNC_VALID_NEG_HIGH,

         /*hsync_hfb  hsync_act  hsync_hhb*/
         {0, 1920, 0,
          /*vsync0_vhb vsync0_act vsync0_hhb*/
          0, 1080, 0,
          /*vsync1_vhb vsync1_act vsync1_hhb*/
          0, 0, 0}},
    .enInputDataType = VI_DATA_TYPE_RGB,
    .stSize = {1920, 1080},
    .stWDRAttr = {WDR_MODE_NONE, 1080},
    .enBayerFormat = BAYER_FORMAT_BG,
};

VI_PIPE_ATTR_S vi_pipe_attr_base = {
    .enPipeBypassMode = VI_PIPE_BYPASS_NONE,
    .bYuvSkip = CVI_FALSE,
    .bIspBypass = CVI_FALSE,
    .u32MaxW = 1920,
    .u32MaxH = 1080,
    .enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP,
    .enCompressMode = COMPRESS_MODE_TILE,
    .enBitWidth = DATA_BITWIDTH_12,
    .bNrEn = CVI_TRUE,
    .bSharpenEn = CVI_FALSE,
    .stFrameRate = {-1, -1},
    .bDiscardProPic = CVI_FALSE,
    .bYuvBypassPath = CVI_FALSE,
};

int TDL_Vi_ParseIni(TDLViCfg *vi_cfg) {
  int ret;
  ret = ini_parse(INI_FILE_PATH, parse_handler, vi_cfg);
  if (ret >= 0) {
    return CVI_SUCCESS;
  } else {
    printf("Parse %s failed\n", INI_FILE_PATH);
    return CVI_FAILURE;
  }
  return ret;
}

int TDL_Vi_GetSize(TDLSnsType enMode, SIZE_S *pstSize) {
  if (!pstSize) return -1;

  switch (enMode) {
    case GCORE_GC2093_MIPI_2M_30FPS_10BIT:
    case GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT:
      pstSize->u32Width = 1920;
      pstSize->u32Height = 1080;
      break;
    case GCORE_GC4653_MIPI_4M_30FPS_10BIT:
      pstSize->u32Width = 2560;
      pstSize->u32Height = 1440;
      break;
    default:
      break;
  }
  return 0;
}

ISP_SNS_OBJ_S *TDL_Vi_SnsObjGet(TDLSnsType enSnsType) {
  switch (enSnsType) {
    case GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT:
      return &stSnsGc2053_1l_Obj;

    case GCORE_GC2093_MIPI_2M_30FPS_10BIT:
      return &stSnsGc2093_Obj;

    case GCORE_GC4653_MIPI_4M_30FPS_10BIT:
      return &stSnsGc4653_Obj;

    default:
      return NULL;
      break;
  }
  return NULL;
}

int TDL_Vi_IspPubAttr(TDLSnsType enSnsType, ISP_PUB_ATTR_S *pstIspPubAttr) {
  int s32Ret = CVI_SUCCESS;

  memcpy(pstIspPubAttr, &isp_pub_attr_base, sizeof(ISP_PUB_ATTR_S));

  switch (enSnsType) {
    case GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT:
    case GCORE_GC2093_MIPI_2M_30FPS_10BIT:
      pstIspPubAttr->enBayer = BAYER_RGGB;
      break;

    case GCORE_GC4653_MIPI_4M_30FPS_10BIT:
      pstIspPubAttr->enBayer = BAYER_GRBG;
      break;

    default:
      s32Ret = CVI_FAILURE;
      break;
  }
  return s32Ret;
}

int TDL_Vi_DevAttrGet(TDLSnsType enSnsType, VI_DEV_ATTR_S *pstViDevAttr) {
  int s32Ret = CVI_SUCCESS;

  memcpy(pstViDevAttr, &vi_dev_attr_base, sizeof(VI_DEV_ATTR_S));

  switch (enSnsType) {
    case GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT:
    case GCORE_GC2093_MIPI_2M_30FPS_10BIT:
      pstViDevAttr->enBayerFormat = BAYER_FORMAT_RG;
      break;
    case GCORE_GC4653_MIPI_4M_30FPS_10BIT:
      pstViDevAttr->enBayerFormat = BAYER_FORMAT_GR;
      break;
    default:
      s32Ret = CVI_FAILURE;
      break;
  }
  return s32Ret;
}

int TDL_Vi_PipeAttrGet(TDLSnsType enSnsType, VI_PIPE_ATTR_S *pstViPipeAttr) {
  int s32Ret = CVI_SUCCESS;

  memcpy(pstViPipeAttr, &vi_pipe_attr_base, sizeof(VI_PIPE_ATTR_S));

  switch (enSnsType) {
    case GCORE_GC2093_MIPI_2M_30FPS_10BIT:
    case GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT:
      pstViPipeAttr->u32MaxW = 1920;
      pstViPipeAttr->u32MaxH = 1080;
      break;
    case GCORE_GC4653_MIPI_4M_30FPS_10BIT:
      pstViPipeAttr->u32MaxW = 2560;
      pstViPipeAttr->u32MaxH = 1440;
      break;
    default:
      break;
  }

  return s32Ret;
}
