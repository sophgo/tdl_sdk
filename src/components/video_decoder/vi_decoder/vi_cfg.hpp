#ifndef VI_CFG
#define VI_CFG

#include <stdlib.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <thread>
#include "cvi_ae.h"
#include "cvi_af.h"
#include "cvi_awb.h"
#include "cvi_bin.h"
#include "cvi_comm_sys.h"
#include "cvi_comm_vb.h"
#include "cvi_comm_vi.h"
#include "cvi_comm_video.h"
#include "cvi_isp.h"
#include "cvi_sns_ctrl.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"
#include "ini.h"

#define INI_FILE_PATH "/mnt/data/sensor_cfg.ini"

extern __attribute__((weak)) ISP_SNS_OBJ_S stSnsGc2053_1l_Obj;
extern __attribute__((weak)) ISP_SNS_OBJ_S stSnsGc2093_Obj;
extern __attribute__((weak)) ISP_SNS_OBJ_S stSnsGc4653_Obj;

typedef enum _TDLSnsType {
  GCORE_GC2053_1L_MIPI_2M_30FPS_10BIT,
  GCORE_GC2093_MIPI_2M_30FPS_10BIT,
  GCORE_GC4653_MIPI_4M_30FPS_10BIT,
  SAMPLE_SNS_TYPE_LINEAR_BUTT,
  SAMPLE_SNS_TYPE_BUTT,
} TDLSnsType;

typedef struct _TDLSnsMclkAttr {
  CVI_U8 u8Mclk;
  CVI_BOOL bMclkEn;
} TDLSnsMclkAttr;

typedef struct _TDLViCfg {
  VI_PIPE_FRAME_SOURCE_E enSource;
  CVI_U8 devNum;
  CVI_S32 enSnsMode;
  CVI_U8 u8UseMultiSns;
  TDLSnsType enSnsType[VI_MAX_DEV_NUM];
  WDR_MODE_E enWDRMode[VI_MAX_DEV_NUM];
  CVI_S32 s32BusId[VI_MAX_DEV_NUM];
  CVI_S32 s32SnsI2cAddr[VI_MAX_DEV_NUM];
  unsigned int MipiDev[VI_MAX_DEV_NUM];
  CVI_S16 as16LaneId[VI_MAX_DEV_NUM][MIPI_LANE_NUM + 1];
  CVI_S16 as16FuncId[VI_MAX_DEV_NUM][21];
  CVI_S8 as8PNSwap[VI_MAX_DEV_NUM][MIPI_LANE_NUM + 1];
  CVI_U8 u8HwSync[VI_MAX_DEV_NUM];
  TDLSnsMclkAttr stMclkAttr[VI_MAX_DEV_NUM];
  CVI_U8 u8Orien[VI_MAX_DEV_NUM];
  CVI_U8 u8MuxDev[VI_MAX_DEV_NUM];
  CVI_U8 u8AttachDev[VI_MAX_DEV_NUM];
  CVI_S16 s16SwitchGpio[VI_MAX_DEV_NUM];
  CVI_U8 u8SwitchPol[VI_MAX_DEV_NUM];
  CVI_U8 u8Hsettle[VI_MAX_DEV_NUM];
  CVI_BOOL bHsettlen[VI_MAX_DEV_NUM];
  CVI_S32 s32RstPin[VI_MAX_DEV_NUM];
  CVI_S32 s32RstActive[VI_MAX_DEV_NUM];
} TDLViCfg;

typedef CVI_VOID (*parser)(TDLViCfg *cfg, const char *value, CVI_U32 param0,
                           CVI_U32 param1, CVI_U32 param2);

typedef struct _TDLIniHdlr {
  const char name[16];
  CVI_U32 param0;
  CVI_U32 param1;
  CVI_U32 param2;
  parser pfnJob;
} TDLIniHdlr;

typedef enum _TDLIniSourceName {
  INI_SOURCE_TYPE = 0,
  INI_SOURCE_DEVNUM,
  INI_SOURCE_ENMODE,
  INI_SOURCE_NUM,
} TDLIniSourceName;

typedef enum _TDLIniSensorName {
  INI_SENSOR_NAME = 0,
  INI_SENSOR_BUSID,
  INI_SENSOR_I2CADDR,
  INI_SENSOR_MIPIDEV,
  INI_SENSOR_LANEID,
  INI_SENSOR_FUNCID,
  INI_SENSOR_PNSWAP,
  INI_SENSOR_HWSYNC,
  INI_SENSOR_MCLKEN,
  INI_SENSOR_MCLK,
  INI_SENSOR_SETTLEEN,
  INI_SENSOR_SETTLE,
  INI_SENSOR_ORIEN,
  INI_SENSOR_RSTPIN,
  INI_SENSOR_ACTIVE,
  INI_SENSOR_MUXDEV,
  INI_SENSOR_ATTACHDEV,
  INI_SENSOR_SWITCHGPIO,
  INI_SENSOR_SWITCHPOL,
  INI_SENSOR_NUM,
} TDLIniSensorName;

int parse_handler(void *user, const char *section, const char *name,
                  const char *value);
int TDL_Vi_ParseIni(TDLViCfg *vi_cfg);
int TDL_Vi_GetSize(TDLSnsType enMode, SIZE_S *pstSize);
ISP_SNS_OBJ_S *TDL_Vi_SnsObjGet(TDLSnsType enSnsType);
int TDL_Vi_IspPubAttr(TDLSnsType enSnsType, ISP_PUB_ATTR_S *pstIspPubAttr);
int TDL_Vi_DevAttrGet(TDLSnsType enSnsType, VI_DEV_ATTR_S *pstViDevAttr);
int TDL_Vi_PipeAttrGet(TDLSnsType enSnsType, VI_PIPE_ATTR_S *pstViPipeAttr);

#endif  // VI_CFG