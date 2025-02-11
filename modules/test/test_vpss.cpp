#include <cvi_buffer.h>
#include <cvi_gdc.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vpss.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VPSS_UT_PRT(fmt...) printf(fmt)
// #include <vpss_ut_comm.h>

#define DEFAULT_W 1920
#define DEFAULT_H 1080
#define DEFAULT_FBCTABLE_LENGTH 69632

#define THREAD_CNT 16

#ifndef FPGA_PORTING
#define UT_TIMEOUT_MS 1000
#define TEST_CNT0 10000
#define TEST_CNT1 1000
#else
#define UT_TIMEOUT_MS 60000
#define TEST_CNT0 100
#define TEST_CNT1 100
#endif

#define RANDOM(min, max) ((rand() % ((max) - (min) + 1)) + (min))

// file: http://disk-sophgo-vip.quickconnect.cn/sharing/iglxSGiJB
#define VPSS_DEFAULT_FILE_IN "res/1080p.yuv420"
#define VPSS_LIMIT_FILE_IN "res/4608_2592.rgb"
#define VPSS_TILE_MODE_FILE_IN "res/5000_2000.yuv420"
#define VPSS_MAX_FILE_IN "res/8192_8192.rgb"
#define VPSS_RGB_FILE_IN "res/1080p.rgb"
#define VPSS_ROT_FILE_IN "res/ldc/input/1920x1080.yuv"
#define VPSS_LDC_FILE_IN "res/ldc/input/1920x1080_barrel_0.3.yuv"
#define VPSS_DWA_FILE_IN "res/input/1920x1080_barrel_0.3.yuv"
#define DWA_FILE_IN_FISHEYE "res/input/fisheye_floor_1024x1024.yuv"
#define VPSS_DEFAULT_FBC_FILE_IN0 "res/fbc/1920x1080_table_y.bin"
#define VPSS_DEFAULT_FBC_FILE_IN1 "res/fbc/1920x1080_table_c.bin"
#define VPSS_DEFAULT_FBC_FILE_IN2 "res/fbc/1920x1080_data_y.bin"
#define VPSS_DEFAULT_FBC_FILE_IN3 "res/fbc/1920x1080_data_c.bin"
#define VPSS_STITCH_FILE_IN0 "res/1080p.yuv420"
#define VPSS_STITCH_FILE_IN1 "res/1920_1080_422p.yuv"

#define MD5_BASIC "ba2c34cc33a6ee83732b6ced5ea7c8bf"
#define MD5_FBD_BASIC "4fab5140588725ad4af8ef50c2900bf5"
#define MD5_1_TO_2_CHN0 "d7ce686e6d32700784c6e83abb3916b1"
#define MD5_1_TO_2_CHN1 "ad2ec6fe09dea2b4c7163b62c0fad5ce"
#define MD5_1_TO_3_CHN0 "08dbb8f414ecae651aca06761e073799"
#define MD5_1_TO_3_CHN1 "ac43e14957e0a7d97f0bb10e3b9f3603"
#define MD5_1_TO_3_CHN2 "c6a4ab39a2a61dddcb45b187846adafe"
#define MD5_1_TO_4_CHN0 "08dbb8f414ecae651aca06761e073799"
#define MD5_1_TO_4_CHN1 "0f000356a4060b5709a06e29ad35515d"
#define MD5_1_TO_4_CHN2 "d4489b059164c2590bd470bf50a69716"
#define MD5_1_TO_4_CHN3 "c6a4ab39a2a61dddcb45b187846adafe"
#define MD5_LIMIT_WIDTH "6fa49fc0e5a8dfd6c10a2e2d5386e61b"
#define MD5_MAX_RES "6ab482e5c66eb46fb41369524dc879c1"
#define MD5_MIRROR "aac8aa00f04df8e5a13e985da519885f"
#define MD5_FLIP "890e5209ffa53bfbd594dc7c5cc17edf"
#define MD5_MIRROR_FLIP "79edba95635f1e8f8de793542b571c7e"
#define MD5_ASPECT_RATIO1 "b16632e310a37d2a98c50c05afa7719f"
#define MD5_ASPECT_RATIO2 "4e78f8bdb723fb8726a3531f8654c78c"
#define MD5_ASPECT_RATIO3 "e7715253f0cba6180d7c1bed7ae9f186"
#define MD5_BYTE_ALIGN "85584eb3fee0a24b0f06ab4ef8efbc3e"
#define MD5_DRAW_RECT "ba8a7a4c3de54cf115e67a9b4f3cf8e9"
#define MD5_AMP_BRIGHTNESS "a2f26d7fd035144e34c8f16f510994e7"
#define MD5_AMP_CONTRAST "5bb1c13775c4179850bb9f0682303dc3"
#define MD5_AMP_SATURATION "8a55dfc8e6e4f4da2a93e1a4eb764141"
#define MD5_AMP_HUE "188b1bae2c7fb4fe7789d12b8137be67"
#define MD5_NORMALIZE "47c16ab4426300b684bc16fe0e8f6091"
#define MD5_CONVERT "3cf37a3a754c484bb6a0068fddb99daa"
#define MD5_SCALE_COEF1 "9bf41589235d410777533254016cd579"
#define MD5_SCALE_COEF2 "d4d29d7a0f812d4c6f4474d3de770ac5"
#define MD5_SCALE_COEF3 "58956ab1812e82e79bd6be83da145c04"
#define MD5_SCALE_COEF4 "0225821009938c2f8f33297a8ed35827"
#define MD5_Y_RATIO "74d67e4f04510d5cd9e41cd750663b00"
#define MD5_HIDE "f07d0b060e007107c339f63652210453"
#define MD5_STITCH "d43c764507f86485dab81a36beb47c53"
#define MD5_STITCH_PIP "86c194575a06ce6d2a0c103cbe3f77e4"
#define MD5_STITCH_GRID "f02427fc82a05ceee1efb7a3fca6ee8e"
#define MD5_TILE_CHN0 "1a0a02966deb1066b2e7884aa88384b4"
#define MD5_TILE_CHN1 "f8ee91af9ccb40bfb6c4c5ce7dbe4099"

#define SLT_REF "res/1920_1080.nv21"

#define OUT_FILE_PREFIX "./out"

#define GDC_FILE_IN_LDC_BARREL_0P3_MESH_0 \
  "res/ldc/input/1920x1080_barrel_0.3_r0_ofst_0_0_d-200.mesh"

typedef struct _VPSS_BASIC_TEST_PARAM {
  VPSS_GRP VpssGrp;
  SIZE_S stSizeIn;
  SIZE_S stSizeOut;
  CVI_BOOL bMirror;
  CVI_BOOL bFlip;
  CVI_BOOL bHide;
  PIXEL_FORMAT_E enFormatIn;
  PIXEL_FORMAT_E enFormatOut;
  ASPECT_RATIO_S stAspectRatio;
  VPSS_NORMALIZE_S stNormalize;
  VPSS_CROP_INFO_S stGrpCropInfo;
  VPSS_CROP_INFO_S stChnCropInfo;
  VPSS_DRAW_RECT_S stDrawRect;
  VPSS_CONVERT_S stConvert;
  VPSS_LDC_ATTR_S stLDCAttr;
  FISHEYE_ATTR_S stFishEyeAttr;
  CVI_BOOL bUseLoadMesh;
  ROTATION_E enRotation;
  VPSS_SCALE_COEF_E enCoef;
  CVI_U32 u32ChnAlign;
  CVI_FLOAT YRatio;
  CVI_U32 u32CheckSum;
  CVI_U32 u32FbcTableLength;
  CVI_CHAR aszMD5Sum[33];
  CVI_CHAR aszFileNameIn[64];
  CVI_CHAR aszFileNameFbcIn[4][64];
  CVI_CHAR aszFileNameOut[64];
  CVI_CHAR aszFileNameRef[64];
} VPSS_BASIC_TEST_PARAM;

struct VPSS_CHN_PARAM {
  CVI_BOOL bEnable;
  VPSS_CHN VpssChn;
  SIZE_S stSizeOut;
  CVI_BOOL bMirror;
  CVI_BOOL bFlip;
  PIXEL_FORMAT_E enFormatOut;
  ASPECT_RATIO_S stAspectRatio;
  VPSS_NORMALIZE_S stNormalize;
  CVI_U32 u32CheckSum;
  CVI_CHAR aszMD5Sum[33];
  CVI_CHAR aszFileNameOut[64];
  CVI_CHAR aszFileNameRef[64];
};

typedef struct _VPSS_MULTI_TEST_PARAM {
  VPSS_GRP VpssGrp;
  SIZE_S stSizeIn;
  PIXEL_FORMAT_E enFormatIn;
  CVI_CHAR aszFileNameIn[64];
  struct VPSS_CHN_PARAM astChnParam[VPSS_MAX_CHN_NUM];
} VPSS_MULTI_TEST_PARAM;

typedef enum _VPSS_TEST_OP {
  VPSS_TEST_BASIC = 0,
  VPSS_TEST_1_to_2,
  VPSS_TEST_1_to_3,
  VPSS_TEST_1_to_4,
  VPSS_TEST_LIMIT_WIDTH,
  VPSS_TEST_MAX_RES,
  VPSS_TEST_MIRROR,
  VPSS_TEST_FLIP,
  VPSS_TEST_MIRROR_FLIP,
  VPSS_TEST_ASPECT_RATIO,
  VPSS_TEST_MULTI_GRP,
  VPSS_TEST_MULTI_THREAD,
  VPSS_TEST_BYTE_ALIGN,
  VPSS_TEST_RESIZE,
  VPSS_TEST_MAX_SCALING,
  VPSS_TEST_TILE_MODE,
  VPSS_TEST_DRAW_RECT,
  VPSS_TEST_FORMAT,
  VPSS_TEST_GRP_CROP,
  VPSS_TEST_CHN_CROP,
  VPSS_TEST_AMP_CTRL,
  VPSS_TEST_NORMALIZE,
  VPSS_TEST_CONVERT_TO,
  VPSS_TEST_SCALE_COEF,
  VPSS_TEST_Y_RATIO,
  VPSS_TEST_HIDE,
  VPSS_TEST_ROT,
  VPSS_TEST_LDC,
  VPSS_TEST_FISHEYE,
  VPSS_TEST_FBD,
  VPSS_TEST_PRESSURE,
  VPSS_TEST_PERF,
  VPSS_TEST_MP_GET_CHN_FRM,
  VPSS_TEST_STITCH,
  VPSS_TEST_STITCH_PIP,
  VPSS_TEST_STITCH_FOUR_GRID,
  VPSS_TEST_TILE_1_to_2,
  VPSS_TEST_SLT,
  VPSS_TEST_C_MODEL,
  VPSS_TEST_USER_CONFIG = 100,
  VPSS_TEST_AUTO = 200,
} VPSS_TEST_OP;

static pthread_mutex_t s_SyncMutex = PTHREAD_MUTEX_INITIALIZER;
static CVI_U32 s_u32Flag;

void vpss_ut_HandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    CVI_SYS_Exit();
    CVI_VB_Exit();
    VPSS_UT_PRT("Program termination abnormally\n");
  }
  exit(-1);
}

#define ALIGN_16 16
#define ALIGN_32 32
#define BIT(x) (1 << x)
#define TEST_CHECK_RET(s32Ret)                 \
  do {                                         \
    if (s32Ret == CVI_SUCCESS)                 \
      printf("\n=== %s pass ===\n", __func__); \
    else                                       \
      printf("\n=== %s fail ===\n", __func__); \
  } while (0)

CVI_S32 FileToFrame(SIZE_S *stSize, PIXEL_FORMAT_E enPixelFormat,
                    CVI_CHAR *filename, VIDEO_FRAME_INFO_S *pstVideoFrame) {
  VIDEO_FRAME_INFO_S stVideoFrame;
  VB_BLK blk;
  CVI_U32 u32len;
  VB_CAL_CONFIG_S stVbCalConfig;
  FILE *fp;

  COMMON_GetPicBufferConfig(stSize->u32Width, stSize->u32Height, enPixelFormat,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, ALIGN_16,
                            &stVbCalConfig);

  memset(&stVideoFrame, 0, sizeof(stVideoFrame));
  stVideoFrame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
  stVideoFrame.stVFrame.enPixelFormat = enPixelFormat;
  stVideoFrame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVideoFrame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
  stVideoFrame.stVFrame.u32Width = stSize->u32Width;
  stVideoFrame.stVFrame.u32Height = stSize->u32Height;
  stVideoFrame.stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;

  // stVideoFrame.stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
  // stVideoFrame.stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32TimeRef = 0;
  stVideoFrame.stVFrame.u64PTS = 0;
  stVideoFrame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
  VPSS_UT_PRT("Format(%d) Width(%d) Height(%d) Stride(%d)\n",
              stVideoFrame.stVFrame.enPixelFormat,
              stVideoFrame.stVFrame.u32Width, stVideoFrame.stVFrame.u32Height,
              stVideoFrame.stVFrame.u32Stride[0]);
  blk = CVI_VB_GetBlock(VB_INVALID_POOLID, stVbCalConfig.u32VBSize);
  if (blk == VB_INVALID_HANDLE) {
    printf("CVI_VB_GetBlock fail\n");
    return CVI_FAILURE;
  }

  // open data file & fread into the mmap address
  fp = fopen(filename, "r");
  if (fp == CVI_NULL) {
    printf("open data file error\n");
    CVI_VB_ReleaseBlock(blk);
    return CVI_FAILURE;
  }

  stVideoFrame.u32PoolId = CVI_VB_Handle2PoolId(blk);
  stVideoFrame.stVFrame.u32Length[0] = stVbCalConfig.u32MainYSize;
  // stVideoFrame.stVFrame.u32Length[1] = stVbCalConfig.u32MainCSize;
  stVideoFrame.stVFrame.u64PhyAddr[0] = CVI_VB_Handle2PhysAddr(blk);
  // stVideoFrame.stVFrame.u64PhyAddr[1] = stVideoFrame.stVFrame.u64PhyAddr[0]
  // 	+ ALIGN(stVbCalConfig.u32MainYSize, stVbCalConfig.u16AddrAlign);
  // if (stVbCalConfig.plane_num == 3) {
  // 	stVideoFrame.stVFrame.u32Length[2] = stVbCalConfig.u32MainCSize;
  // 	stVideoFrame.stVFrame.u64PhyAddr[2] =
  // stVideoFrame.stVFrame.u64PhyAddr[1]
  // 		+ ALIGN(stVbCalConfig.u32MainCSize, stVbCalConfig.u16AddrAlign);
  // }

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    stVideoFrame.stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[i],
                               stVideoFrame.stVFrame.u32Length[i]);

    u32len = fread(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i], 1, fp);
    if (u32len <= 0) {
      printf("dpu send frame: fread plane%d error\n", i);
      fclose(fp);
      CVI_VB_ReleaseBlock(blk);
      return CVI_FAILURE;
    }
    CVI_SYS_IonFlushCache(stVideoFrame.stVFrame.u64PhyAddr[i],
                          stVideoFrame.stVFrame.pu8VirAddr[i],
                          stVideoFrame.stVFrame.u32Length[i]);
  }

  printf("length of buffer(%d)\n", stVideoFrame.stVFrame.u32Length[0]);
  printf("phy addr(%#" PRIx64 ")\n", stVideoFrame.stVFrame.u64PhyAddr[0]);
  printf("vir addr(%p)\n", stVideoFrame.stVFrame.pu8VirAddr[0]);

  fclose(fp);

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    CVI_SYS_Munmap(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i]);
  }
  memcpy(pstVideoFrame, &stVideoFrame, sizeof(stVideoFrame));

  return CVI_SUCCESS;
}
CVI_S32 FileSendToVpss(VPSS_GRP VpssGrp, const SIZE_S *stSize,
                       PIXEL_FORMAT_E enPixelFormat, const CVI_CHAR *filename) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VIDEO_FRAME_INFO_S stVideoFrame;
  VB_BLK blk;
  CVI_U32 u32len;
  VB_CAL_CONFIG_S stVbCalConfig;
  FILE *fp;

  COMMON_GetPicBufferConfig(stSize->u32Width, stSize->u32Height, enPixelFormat,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN,
                            &stVbCalConfig);

  memset(&stVideoFrame, 0, sizeof(stVideoFrame));
  stVideoFrame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
  stVideoFrame.stVFrame.enPixelFormat = enPixelFormat;
  stVideoFrame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVideoFrame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
  stVideoFrame.stVFrame.u32Width = stSize->u32Width;
  stVideoFrame.stVFrame.u32Height = stSize->u32Height;
  stVideoFrame.stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;
  stVideoFrame.stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32TimeRef = 0;
  stVideoFrame.stVFrame.u64PTS = 0;
  stVideoFrame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;

  blk = CVI_VB_GetBlock(VB_INVALID_POOLID, stVbCalConfig.u32VBSize);
  if (blk == VB_INVALID_HANDLE) {
    VPSS_UT_PRT("CVI_VB_GetBlock fail\n");
    return CVI_FAILURE;
  }

  // open data file & fread into the mmap address
  fp = fopen(filename, "r");
  if (fp == CVI_NULL) {
    VPSS_UT_PRT("open data file error\n");
    CVI_VB_ReleaseBlock(blk);
    return CVI_FAILURE;
  }

  stVideoFrame.u32PoolId = CVI_VB_Handle2PoolId(blk);
  stVideoFrame.stVFrame.u32Length[0] = stVbCalConfig.u32MainYSize;
  stVideoFrame.stVFrame.u32Length[1] = stVbCalConfig.u32MainCSize;
  stVideoFrame.stVFrame.u64PhyAddr[0] = CVI_VB_Handle2PhysAddr(blk);
  stVideoFrame.stVFrame.u64PhyAddr[1] =
      stVideoFrame.stVFrame.u64PhyAddr[0] +
      ALIGN(stVbCalConfig.u32MainYSize, stVbCalConfig.u16AddrAlign);
  if (stVbCalConfig.plane_num == 3) {
    stVideoFrame.stVFrame.u32Length[2] = stVbCalConfig.u32MainCSize;
    stVideoFrame.stVFrame.u64PhyAddr[2] =
        stVideoFrame.stVFrame.u64PhyAddr[1] +
        ALIGN(stVbCalConfig.u32MainCSize, stVbCalConfig.u16AddrAlign);
  }

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    stVideoFrame.stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[i],
                               stVideoFrame.stVFrame.u32Length[i]);

    u32len = fread(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i], 1, fp);
    if (u32len <= 0) {
      VPSS_UT_PRT("vpss send frame: fread plane%d error\n", i);
      fclose(fp);
      CVI_VB_ReleaseBlock(blk);
      return CVI_FAILURE;
    }
    CVI_SYS_IonFlushCache(stVideoFrame.stVFrame.u64PhyAddr[i],
                          stVideoFrame.stVFrame.pu8VirAddr[i],
                          stVideoFrame.stVFrame.u32Length[i]);
  }

  VPSS_UT_PRT(
      "length of buffer(%d, %d, %d)\n", stVideoFrame.stVFrame.u32Length[0],
      stVideoFrame.stVFrame.u32Length[1], stVideoFrame.stVFrame.u32Length[2]);
  VPSS_UT_PRT("phy addr(%#" PRIx64 ", %#" PRIx64 ", %#" PRIx64 ")\n",
              stVideoFrame.stVFrame.u64PhyAddr[0],
              stVideoFrame.stVFrame.u64PhyAddr[1],
              stVideoFrame.stVFrame.u64PhyAddr[2]);
  VPSS_UT_PRT("vir addr(%p, %p, %p)\n", stVideoFrame.stVFrame.pu8VirAddr[0],
              stVideoFrame.stVFrame.pu8VirAddr[1],
              stVideoFrame.stVFrame.pu8VirAddr[2]);

  fclose(fp);

  VPSS_UT_PRT("read file done and send vpss frame.\n");
  s32Ret = CVI_VPSS_SendFrame(VpssGrp, &stVideoFrame, 1000);
  if (s32Ret != CVI_SUCCESS) VPSS_UT_PRT("CVI_VPSS_SendFrame fail.\n");

  CVI_VB_ReleaseBlock(blk);

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    CVI_SYS_Munmap(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i]);
  }
  return s32Ret;
}

CVI_S32 FrameSaveToFile(const CVI_CHAR *filename,
                        VIDEO_FRAME_INFO_S *pstVideoFrame) {
  FILE *fp;
  CVI_U32 i;
  CVI_S32 c_w_shift, c_h_shift;  // chroma width/height shift
  CVI_S32 s32PixelSize;
  CVI_U32 u32Planar;
  CVI_U8 *w_ptr;
  CVI_U32 image_size = 0;
  CVI_S32 plane_offset = 0;
  CVI_VOID *vir_addr = NULL;
  VIDEO_FRAME_S *pstVFrame = &pstVideoFrame->stVFrame;

  fp = fopen(filename, "w");
  if (fp == CVI_NULL) {
    VPSS_UT_PRT("open data file(%s) error\n", filename);
    return CVI_FAILURE;
  }

  image_size = pstVFrame->u32Length[0] + pstVFrame->u32Length[1] +
               pstVFrame->u32Length[2];
  vir_addr = (CVI_U8 *)CVI_SYS_Mmap(pstVFrame->u64PhyAddr[0], image_size);
  CVI_SYS_IonInvalidateCache(pstVFrame->u64PhyAddr[0], vir_addr, image_size);

  for (i = 0; i < 3; i++) {
    if (pstVFrame->u32Length[i] == 0) continue;
    pstVFrame->pu8VirAddr[i] = (CVI_U8 *)(vir_addr) + plane_offset;
    plane_offset += pstVFrame->u32Length[i];
  }

  VPSS_UT_PRT("u32Width = %d, u32Height = %d\n", pstVFrame->u32Width,
              pstVFrame->u32Height);
  VPSS_UT_PRT("u32Stride[0] = %d, u32Stride[1] = %d, u32Stride[2] = %d\n",
              pstVFrame->u32Stride[0], pstVFrame->u32Stride[1],
              pstVFrame->u32Stride[2]);
  VPSS_UT_PRT("u32Length[0] = %d, u32Length[1] = %d, u32Length[2] = %d\n",
              pstVFrame->u32Length[0], pstVFrame->u32Length[1],
              pstVFrame->u32Length[2]);

  // get_chroma_size_shift_factor(pstVFrame->enPixelFormat, &c_w_shift,
  // &c_h_shift,
  //                              &s32PixelSize, &u32Planar);

  // // save Y
  // w_ptr = pstVFrame->pu8VirAddr[0];
  // for (i = 0; i < pstVFrame->u32Height; i++) {
  //   fwrite(w_ptr + i * pstVFrame->u32Stride[0], s32PixelSize,
  //          pstVFrame->u32Width, fp);
  // }
  // // save U
  // if (u32Planar >= 2) {
  //   w_ptr = pstVFrame->pu8VirAddr[1];
  //   for (i = 0; i < (pstVFrame->u32Height >> c_h_shift); i++) {
  //     fwrite(w_ptr + i * pstVFrame->u32Stride[1], s32PixelSize,
  //            pstVFrame->u32Width >> c_w_shift, fp);
  //   }
  // }
  // // save V
  // if (u32Planar >= 3) {
  //   w_ptr = pstVFrame->pu8VirAddr[2];
  //   for (i = 0; i < (pstVFrame->u32Height >> c_h_shift); i++) {
  //     fwrite(w_ptr + i * pstVFrame->u32Stride[2], s32PixelSize,
  //            pstVFrame->u32Width >> c_w_shift, fp);
  //   }
  // }

  CVI_SYS_Munmap(vir_addr, image_size);
  fclose(fp);

  return 0;
}
static CVI_S32 basic(const VPSS_BASIC_TEST_PARAM *pTestParam) {
  CVI_S32 i, s32Ret = CVI_SUCCESS;
  VPSS_GRP VpssGrp = pTestParam->VpssGrp;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSizeIn, u32BlkSizeOut;
  VIDEO_FRAME_INFO_S stVideoFrame;
  CVI_BOOL bFlag = CVI_FALSE;
  CVI_BOOL bSaveFile = CVI_FALSE;

  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  u32BlkSizeIn = COMMON_GetPicBufferSize(
      pTestParam->stSizeIn.u32Width, pTestParam->stSizeIn.u32Height,
      pTestParam->enFormatIn, DATA_BITWIDTH_8, COMPRESS_MODE_NONE,
      DEFAULT_ALIGN);
  u32BlkSizeOut = COMMON_GetPicBufferSize(
      pTestParam->stSizeOut.u32Width, pTestParam->stSizeOut.u32Height,
      pTestParam->enFormatOut, DATA_BITWIDTH_8, COMPRESS_MODE_NONE,
      DEFAULT_ALIGN);

  stVbConf.u32MaxPoolCnt = 2;
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSizeIn;
  stVbConf.astCommPool[0].u32BlkCnt = 1 + (((pTestParam->stLDCAttr.bEnable) ||
                                            pTestParam->stFishEyeAttr.bEnable)
                                               ? 1
                                               : 0);
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
  stVbConf.astCommPool[1].u32BlkSize = u32BlkSizeOut;
  stVbConf.astCommPool[1].u32BlkCnt =
      1 + (pTestParam->stLDCAttr.bEnable || pTestParam->stFishEyeAttr.bEnable ||
                   pTestParam->enRotation
               ? 1
               : 0);
  stVbConf.astCommPool[1].enRemapMode = VB_REMAP_MODE_CACHED;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSizeIn);
  VPSS_UT_PRT("common pool[1] BlkSize %d\n", u32BlkSizeOut);

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_Init failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    goto exit0;
  }

  /************************************************
   * step2:  Init VPSS
   ************************************************/
  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = pTestParam->enFormatIn;
  stVpssGrpAttr.u32MaxW = pTestParam->stSizeIn.u32Width;
  stVpssGrpAttr.u32MaxH = pTestParam->stSizeIn.u32Height;

  if (pTestParam->stLDCAttr.bEnable &&
      pTestParam->stLDCAttr.stAttr.enRotation != 0) {
    stVpssChnAttr.u32Width =
        ALIGN(pTestParam->stSizeIn.u32Width, DEFAULT_ALIGN);
    stVpssChnAttr.u32Height =
        ALIGN(pTestParam->stSizeIn.u32Height, DEFAULT_ALIGN);
  } else {
    stVpssChnAttr.u32Width = pTestParam->stSizeOut.u32Width;
    stVpssChnAttr.u32Height = pTestParam->stSizeOut.u32Height;
  }

  stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr.enPixelFormat = pTestParam->enFormatOut;
  stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssChnAttr.u32Depth = 1;
  stVpssChnAttr.bMirror = pTestParam->bMirror;
  stVpssChnAttr.bFlip = pTestParam->bFlip;
  stVpssChnAttr.stAspectRatio = pTestParam->stAspectRatio;
  stVpssChnAttr.stNormalize = pTestParam->stNormalize;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    goto exit1;
  }

  s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, 1);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_AttachVbPool failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    goto exit2;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit3;
  }

  // grp crop
  if (pTestParam->stGrpCropInfo.bEnable) {
    s32Ret = CVI_VPSS_SetGrpCrop(VpssGrp, &pTestParam->stGrpCropInfo);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetGrpCrop failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn crop
  if (pTestParam->stChnCropInfo.bEnable) {
    s32Ret = CVI_VPSS_SetChnCrop(VpssGrp, VpssChn, &pTestParam->stChnCropInfo);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnCrop failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn Draw rect
  bFlag = CVI_FALSE;
  for (i = 0; i < VPSS_RECT_NUM; i++)
    if (pTestParam->stDrawRect.astRect[i].bEnable) bFlag = CVI_TRUE;
  if (bFlag) {
    s32Ret = CVI_VPSS_SetChnDrawRect(VpssGrp, VpssChn, &pTestParam->stDrawRect);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnDrawRect failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn Convert to
  if (pTestParam->stConvert.bEnable) {
    s32Ret = CVI_VPSS_SetChnConvert(VpssGrp, VpssChn, &pTestParam->stConvert);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnConvert failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn FishEye
  if (pTestParam->stFishEyeAttr.bEnable) {
    s32Ret =
        CVI_VPSS_SetChnFisheye(VpssGrp, VpssChn, &pTestParam->stFishEyeAttr);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnFisheye failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn LDC
  if (pTestParam->stLDCAttr.bEnable) {
    // if (pTestParam->bUseLoadMesh) {
    // 	MESH_DUMP_ATTR_S MeshDumpAttr;

    // 	strcpy(MeshDumpAttr.binFileName , GDC_FILE_IN_LDC_BARREL_0P3_MESH_0);
    // 	MeshDumpAttr.enModId = CVI_ID_VPSS;
    // 	MeshDumpAttr.vpssMeshAttr.grp = 0;
    // 	MeshDumpAttr.vpssMeshAttr.chn = 0;

    // 	s32Ret = CVI_GDC_LoadMesh(&MeshDumpAttr, &pTestParam->stLDCAttr.stAttr);
    // 	if (s32Ret != CVI_SUCCESS) {
    // 		VPSS_UT_PRT("CVI_GDC_LoadMesh failed with %#x\n", s32Ret);
    // 		goto exit4;
    // 	}
    // } else {
    s32Ret = CVI_VPSS_SetChnLDCAttr(VpssGrp, VpssChn, &pTestParam->stLDCAttr);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnLDCAttr failed with %#x\n", s32Ret);
      goto exit4;
    }
    // }
  }

  // chn rotation
  if (pTestParam->enRotation != ROTATION_0) {
    s32Ret = CVI_VPSS_SetChnRotation(VpssGrp, VpssChn, pTestParam->enRotation);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnRotation failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn coef
  if (pTestParam->enCoef != VPSS_SCALE_COEF_BICUBIC) {
    s32Ret =
        CVI_VPSS_SetChnScaleCoefLevel(VpssGrp, VpssChn, pTestParam->enCoef);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnScaleCoefLevel failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn align
  if (pTestParam->u32ChnAlign > 0) {
    s32Ret = CVI_VPSS_SetChnAlign(VpssGrp, VpssChn, pTestParam->u32ChnAlign);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnAlign failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn YRatio
  if (pTestParam->YRatio > 0) {
    s32Ret = CVI_VPSS_SetChnYRatio(VpssGrp, VpssChn, pTestParam->YRatio);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnYRatio failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // chn hide
  if (pTestParam->bHide) {
    s32Ret = CVI_VPSS_HideChn(VpssGrp, VpssChn);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_HideChn failed with %#x\n", s32Ret);
      goto exit4;
    }
  }

  // send frame
  s32Ret = FileSendToVpss(VpssGrp, &pTestParam->stSizeIn,
                          pTestParam->enFormatIn, pTestParam->aszFileNameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileSendToVpss fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit4;
  }

  s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrame, UT_TIMEOUT_MS);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n", s32Ret);
    goto exit4;
  }
  VPSS_UT_PRT("***CVI_VPSS_GetChnFrame Success***\n");

  // if (pTestParam->aszMD5Sum[0]) {
  //   if (CompareWithMD5(pTestParam->aszMD5Sum, &stVideoFrame)) {
  //     if (pTestParam->aszFileNameRef[0] &&
  //         CompareWithFile(pTestParam->aszFileNameRef, &stVideoFrame) ==
  //             CVI_SUCCESS)
  //       s32Ret = CVI_SUCCESS;
  //     else {
  //       bSaveFile = CVI_TRUE;
  //       s32Ret = CVI_FAILURE;
  //       VPSS_UT_PRT("Compare MD5 fail, MD5:%s\n", pTestParam->aszMD5Sum);
  //     }
  //   } else {
  //     bSaveFile = CVI_FALSE;
  //   }
  // }

  if (bSaveFile && pTestParam->aszFileNameOut[0]) {
    if (FrameSaveToFile(pTestParam->aszFileNameOut, &stVideoFrame)) {
      VPSS_UT_PRT("FrameSaveToFile. s32Ret: 0x%x !\n", s32Ret);
      CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrame);
      goto exit4;
    }
    VPSS_UT_PRT("output file:%s\n", pTestParam->aszFileNameOut);
  }

  CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrame);

exit4:
  CVI_VPSS_StopGrp(VpssGrp);
exit3:
  CVI_VPSS_DisableChn(VpssGrp, VpssChn);
exit2:
  CVI_VPSS_DestroyGrp(VpssGrp);
exit1:
  CVI_SYS_Exit();
exit0:
  CVI_VB_Exit();
  return s32Ret;
}

static CVI_S32 basic_mutli_chn(VPSS_MULTI_TEST_PARAM *pTestParam) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 i, n;
  VPSS_GRP VpssGrp = pTestParam->VpssGrp;
  VPSS_CHN VpssChn;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSizeIn, u32BlkSizeOut;
  VIDEO_FRAME_INFO_S stVideoFrame;
  struct VPSS_CHN_PARAM *pstChnParam;
  CVI_BOOL bSaveFile = CVI_TRUE;

  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  u32BlkSizeIn = COMMON_GetPicBufferSize(
      pTestParam->stSizeIn.u32Width, pTestParam->stSizeIn.u32Height,
      pTestParam->enFormatIn, DATA_BITWIDTH_8, COMPRESS_MODE_NONE,
      DEFAULT_ALIGN);

  stVbConf.astCommPool[0].u32BlkSize = u32BlkSizeIn;
  stVbConf.astCommPool[0].u32BlkCnt = 1;
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSizeIn);
  n = 1;
  for (i = 0; i < VPSS_MAX_CHN_NUM; i++) {
    pstChnParam = &pTestParam->astChnParam[i];
    if (!pstChnParam->bEnable) continue;
    u32BlkSizeOut = COMMON_GetPicBufferSize(
        pstChnParam->stSizeOut.u32Width, pstChnParam->stSizeOut.u32Height,
        pstChnParam->enFormatOut, DATA_BITWIDTH_8, COMPRESS_MODE_NONE,
        DEFAULT_ALIGN);
    stVbConf.astCommPool[n].u32BlkSize = u32BlkSizeOut;
    stVbConf.astCommPool[n].u32BlkCnt = 1;
    stVbConf.astCommPool[n].enRemapMode = VB_REMAP_MODE_CACHED;
    VPSS_UT_PRT("common pool[%d] BlkSize %d\n", n, u32BlkSizeOut);
    n++;
  }
  stVbConf.u32MaxPoolCnt = n;

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_Init failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    goto exit0;
  }

  /************************************************
   * step2:  Init VPSS
   ************************************************/
  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = pTestParam->enFormatIn;
  stVpssGrpAttr.u32MaxW = pTestParam->stSizeIn.u32Width;
  stVpssGrpAttr.u32MaxH = pTestParam->stSizeIn.u32Height;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    goto exit1;
  }

  n = 1;
  for (i = 0; i < VPSS_MAX_CHN_NUM; i++) {
    pstChnParam = &pTestParam->astChnParam[i];
    if (!pstChnParam->bEnable) continue;
    VpssChn = i;
    stVpssChnAttr.u32Width = pstChnParam->stSizeOut.u32Width;
    stVpssChnAttr.u32Height = pstChnParam->stSizeOut.u32Height;
    stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    stVpssChnAttr.enPixelFormat = pstChnParam->enFormatOut;
    stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssChnAttr.u32Depth = 1;
    stVpssChnAttr.bMirror = pstChnParam->bMirror;
    stVpssChnAttr.bFlip = pstChnParam->bFlip;
    stVpssChnAttr.stAspectRatio = pstChnParam->stAspectRatio;
    stVpssChnAttr.stNormalize = pstChnParam->stNormalize;
    s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x, chn(%d)\n", s32Ret, i);
      goto exit2;
    }

    s32Ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, n);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_AttachVbPool failed with %#x\n", s32Ret);
      goto exit2;
    }

    s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x, chn(%d)\n", s32Ret, i);
      goto exit2;
    }
    n++;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit2;
  }

  // send frame
  s32Ret = FileSendToVpss(VpssGrp, &pTestParam->stSizeIn,
                          pTestParam->enFormatIn, pTestParam->aszFileNameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileSendToVpss fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit3;
  }

  for (i = 0; i < VPSS_MAX_CHN_NUM; i++) {
    pstChnParam = &pTestParam->astChnParam[i];
    if (!pstChnParam->bEnable) continue;
    VpssChn = i;
    s32Ret =
        CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrame, UT_TIMEOUT_MS);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT(
          "Grp(%d) Chn(%d), CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n",
          VpssGrp, VpssChn, s32Ret);
      goto exit3;
    }
    VPSS_UT_PRT("***Grp(%d) Chn(%d) CVI_VPSS_GetChnFrame Success***\n", VpssGrp,
                VpssChn);

    // if (pstChnParam->aszMD5Sum[0]) {
    //   if (CompareWithMD5(pstChnParam->aszMD5Sum, &stVideoFrame)) {
    //     bSaveFile = CVI_TRUE;
    //     s32Ret = CVI_FAILURE;
    //     VPSS_UT_PRT("chn%d Compare MD5 fail, MD5:%s\n", i,
    //                 pstChnParam->aszMD5Sum);
    //   } else {
    //     bSaveFile = CVI_FALSE;
    //   }
    // }

    if (bSaveFile && pstChnParam->aszFileNameOut[0]) {
      if (FrameSaveToFile(pstChnParam->aszFileNameOut, &stVideoFrame)) {
        VPSS_UT_PRT("Grp(%d) Chn(%d),FrameSaveToFile. s32Ret: 0x%x !\n",
                    VpssGrp, VpssChn, s32Ret);
      }
      VPSS_UT_PRT("output file:%s\n", pstChnParam->aszFileNameOut);
    }

    // if (pstChnParam->aszFileNameRef[0]) {
    //   if (CompareWithFile(pstChnParam->aszFileNameRef, &stVideoFrame)) {
    //     VPSS_UT_PRT("Grp(%d) Chn(%d),CompareWithFile fail.\n", VpssGrp,
    //                 VpssChn);
    //     CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrame);
    //     s32Ret = CVI_FAILURE;
    //     goto exit3;
    //   }
    // }

    CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrame);

    if (s32Ret) break;
  }

exit3:
  CVI_VPSS_StopGrp(VpssGrp);
exit2:
  for (i = 0; i < VPSS_MAX_CHN_NUM; i++) {
    if (!pTestParam->astChnParam[i].bEnable) continue;
    CVI_VPSS_DisableChn(VpssGrp, i);
  }
  CVI_VPSS_DestroyGrp(VpssGrp);
exit1:
  CVI_SYS_Exit();
exit0:
  CVI_VB_Exit();
  return s32Ret;
}

static CVI_S32 vpss_test_basic(CVI_VOID) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_BASIC_TEST_PARAM stTestParam;

  memset(&stTestParam, 0, sizeof(stTestParam));
  stTestParam.VpssGrp = 0;
  stTestParam.stSizeIn.u32Width = DEFAULT_W;
  stTestParam.stSizeIn.u32Height = DEFAULT_H;
  stTestParam.stSizeOut.u32Width = DEFAULT_W;
  stTestParam.stSizeOut.u32Height = DEFAULT_H;
  stTestParam.bMirror = CVI_FALSE;
  stTestParam.bFlip = CVI_FALSE;
  stTestParam.enFormatIn = PIXEL_FORMAT_YUV_PLANAR_420;
  stTestParam.enFormatOut = PIXEL_FORMAT_NV21;
  stTestParam.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stTestParam.stNormalize.bEnable = CVI_FALSE;
  stTestParam.u32CheckSum = 0x13030706;
  strncpy(stTestParam.aszMD5Sum, MD5_BASIC, sizeof(stTestParam.aszMD5Sum));
  strncpy(stTestParam.aszFileNameIn, VPSS_DEFAULT_FILE_IN,
          sizeof(stTestParam.aszFileNameIn));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret = basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

static CVI_S32 vpss_test_aspect_ratio(CVI_VOID) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_BASIC_TEST_PARAM stTestParam;

  memset(&stTestParam, 0, sizeof(stTestParam));
  stTestParam.VpssGrp = 0;
  stTestParam.stSizeIn.u32Width = DEFAULT_W;
  stTestParam.stSizeIn.u32Height = DEFAULT_H;
  stTestParam.stSizeOut.u32Width = DEFAULT_W;
  stTestParam.stSizeOut.u32Height = DEFAULT_H;
  stTestParam.bMirror = CVI_FALSE;
  stTestParam.bFlip = CVI_FALSE;
  stTestParam.enFormatIn = PIXEL_FORMAT_YUV_PLANAR_420;
  stTestParam.enFormatOut = PIXEL_FORMAT_YUV_PLANAR_420;
  stTestParam.stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
  stTestParam.stAspectRatio.bEnableBgColor = CVI_TRUE;
  stTestParam.stAspectRatio.u32BgColor = 0;
  stTestParam.stAspectRatio.stVideoRect.s32X = 64;
  stTestParam.stAspectRatio.stVideoRect.s32Y = 64;
  stTestParam.stAspectRatio.stVideoRect.u32Width = 1280;
  stTestParam.stAspectRatio.stVideoRect.u32Height = 720;
  stTestParam.stNormalize.bEnable = CVI_FALSE;
  stTestParam.u32CheckSum = 0x1e28594b;
  strncpy(stTestParam.aszMD5Sum, MD5_ASPECT_RATIO1,
          sizeof(stTestParam.aszMD5Sum));
  strncpy(stTestParam.aszFileNameIn, VPSS_DEFAULT_FILE_IN,
          sizeof(stTestParam.aszFileNameIn));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_1_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  // x,y odd
  stTestParam.stAspectRatio.stVideoRect.s32X = 65;
  stTestParam.stAspectRatio.stVideoRect.s32Y = 31;
  stTestParam.stAspectRatio.stVideoRect.u32Width = 1280;
  stTestParam.stAspectRatio.stVideoRect.u32Height = 720;
  stTestParam.u32CheckSum = 0x30186f67;
  strncpy(stTestParam.aszMD5Sum, MD5_ASPECT_RATIO2,
          sizeof(stTestParam.aszMD5Sum));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_2_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  // w,h odd
  stTestParam.stAspectRatio.stVideoRect.s32X = 64;
  stTestParam.stAspectRatio.stVideoRect.s32Y = 30;
  stTestParam.stAspectRatio.stVideoRect.u32Width = 1281;
  stTestParam.stAspectRatio.stVideoRect.u32Height = 721;
  stTestParam.u32CheckSum = 0xd6b83235;
  strncpy(stTestParam.aszMD5Sum, MD5_ASPECT_RATIO3,
          sizeof(stTestParam.aszMD5Sum));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_3_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

/*****************************************************************************
 * function : Create Vpss group & enable channel.
 *****************************************************************************/
CVI_S32 SAMPLE_COMM_VPSS_Init(VPSS_GRP VpssGrp, CVI_BOOL *pabChnEnable,
                              VPSS_GRP_ATTR_S *pstVpssGrpAttr,
                              VPSS_CHN_ATTR_S *pastVpssChnAttr) {
  VPSS_CHN VpssChn;
  CVI_S32 s32Ret;
  CVI_S32 j;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, pstVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    return CVI_FAILURE;
  }

  s32Ret = CVI_VPSS_ResetGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_ResetGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    return CVI_FAILURE;
  }

  for (j = 0; j < VPSS_MAX_PHY_CHN_NUM; j++) {
    if (pabChnEnable[j]) {
      VpssChn = j;
      s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &pastVpssChnAttr[VpssChn]);

      if (s32Ret != CVI_SUCCESS) {
        VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
        return CVI_FAILURE;
      }

      s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);

      if (s32Ret != CVI_SUCCESS) {
        VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
        return CVI_FAILURE;
      }
    }
  }

  return CVI_SUCCESS;
}

/*****************************************************************************
 * function : start vpss grp.
 *****************************************************************************/
CVI_S32 SAMPLE_COMM_VPSS_Start(VPSS_GRP VpssGrp, CVI_BOOL *pabChnEnable,
                               VPSS_GRP_ATTR_S *pstVpssGrpAttr,
                               VPSS_CHN_ATTR_S *pastVpssChnAttr) {
  CVI_S32 s32Ret;
  UNUSED(pabChnEnable);
  UNUSED(pstVpssGrpAttr);
  UNUSED(pastVpssChnAttr);

  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    return CVI_FAILURE;
  }

  return CVI_SUCCESS;
}

CVI_S32 SAMPLE_COMM_ReadFrame(SIZE_S *stSize, PIXEL_FORMAT_E enPixelFormat,
                              const char *filename,
                              VIDEO_FRAME_INFO_S &stVideoFrame) {
  VB_BLK blk;
  FILE *fp;
  CVI_U32 u32len;
  VB_CAL_CONFIG_S stVbCalConfig;

  COMMON_GetPicBufferConfig(stSize->u32Width, stSize->u32Height, enPixelFormat,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN,
                            &stVbCalConfig);

  memset(&stVideoFrame, 0, sizeof(stVideoFrame));
  stVideoFrame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
  stVideoFrame.stVFrame.enPixelFormat = enPixelFormat;
  stVideoFrame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVideoFrame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
  stVideoFrame.stVFrame.u32Width = stSize->u32Width;
  stVideoFrame.stVFrame.u32Height = stSize->u32Height;
  stVideoFrame.stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;
  stVideoFrame.stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32TimeRef = 0;
  stVideoFrame.stVFrame.u64PTS = 0;
  stVideoFrame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;

  // open data file & fread into the mmap address
  fp = fopen(filename, "r");
  if (fp == CVI_NULL) {
    VPSS_UT_PRT("open data file error\n");

    return CVI_FAILURE;
  }

  VIDEO_FRAME_S *vFrame = &(stVideoFrame.stVFrame);
  stVideoFrame.stVFrame.u32Length[0] = stVbCalConfig.u32MainYSize;
  stVideoFrame.stVFrame.u32Length[1] = stVbCalConfig.u32MainCSize;
  if (stVbCalConfig.plane_num == 3) {
    stVideoFrame.stVFrame.u32Length[2] = stVbCalConfig.u32MainCSize;
  }
  CVI_U32 u32MapSize =
      vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2];
  CVI_SYS_IonAlloc(&vFrame->u64PhyAddr[0], (CVI_VOID **)&vFrame->pu8VirAddr[0],
                   "alloc_name", u32MapSize);
  printf(
      "alloc memory size: %d, svFrame->u64PhyAddr[0]: %#lx,plane "
      "size:[%d,%d,%d]\n",
      u32MapSize, vFrame->u64PhyAddr[0], vFrame->u32Length[0],
      vFrame->u32Length[1], vFrame->u32Length[2]);
  stVideoFrame.stVFrame.u64PhyAddr[1] =
      stVideoFrame.stVFrame.u64PhyAddr[0] +
      ALIGN(stVbCalConfig.u32MainYSize, stVbCalConfig.u16AddrAlign);
  if (stVbCalConfig.plane_num == 3) {
    stVideoFrame.stVFrame.u64PhyAddr[2] =
        stVideoFrame.stVFrame.u64PhyAddr[1] +
        ALIGN(stVbCalConfig.u32MainCSize, stVbCalConfig.u16AddrAlign);
  }

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    // stVideoFrame.stVFrame.pu8VirAddr[i] =
    //     (CVI_U8 *)CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[i],
    //                            stVideoFrame.stVFrame.u32Length[i]);
    if (i > 0) {
      stVideoFrame.stVFrame.pu8VirAddr[i] =
          stVideoFrame.stVFrame.pu8VirAddr[i - 1] +
          stVideoFrame.stVFrame.u32Length[i - 1];
    }

    u32len = fread(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i], 1, fp);
    if (u32len <= 0) {
      VPSS_UT_PRT("vpss send frame: fread plane%d error\n", i);
      fclose(fp);
      CVI_SYS_IonFree(stVideoFrame.stVFrame.u64PhyAddr[i],
                      stVideoFrame.stVFrame.pu8VirAddr[i]);
      return CVI_FAILURE;
    }
    CVI_SYS_IonFlushCache(stVideoFrame.stVFrame.u64PhyAddr[i],
                          stVideoFrame.stVFrame.pu8VirAddr[i],
                          stVideoFrame.stVFrame.u32Length[i]);
  }
  fclose(fp);
  return CVI_SUCCESS;
}
/* SAMPLE_COMM_VPSS_SendFrame:
 *   send frame, whose data loaded from given filename.
 *
 * VpssGrp: the VPSS Grp to control.
 * stSize: size of image.
 * enPixelFormat: format of image
 * filename: file to read.
 */
CVI_S32 SAMPLE_COMM_VPSS_SendFrame(VPSS_GRP VpssGrp, SIZE_S *stSize,
                                   PIXEL_FORMAT_E enPixelFormat,
                                   const char *filename) {
  VIDEO_FRAME_INFO_S stVideoFrame;
  VB_BLK blk;
  FILE *fp;
  CVI_U32 u32len;
  VB_CAL_CONFIG_S stVbCalConfig;

  COMMON_GetPicBufferConfig(stSize->u32Width, stSize->u32Height, enPixelFormat,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN,
                            &stVbCalConfig);

  memset(&stVideoFrame, 0, sizeof(stVideoFrame));
  stVideoFrame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
  stVideoFrame.stVFrame.enPixelFormat = enPixelFormat;
  stVideoFrame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVideoFrame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
  stVideoFrame.stVFrame.u32Width = stSize->u32Width;
  stVideoFrame.stVFrame.u32Height = stSize->u32Height;
  stVideoFrame.stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;
  stVideoFrame.stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32TimeRef = 0;
  stVideoFrame.stVFrame.u64PTS = 0;
  stVideoFrame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;

  blk = CVI_VB_GetBlock(VB_INVALID_POOLID, stVbCalConfig.u32VBSize);
  if (blk == VB_INVALID_HANDLE) {
    VPSS_UT_PRT("SAMPLE_COMM_VPSS_SendFrame: Can't acquire vb block\n");
    return CVI_FAILURE;
  }

  // open data file & fread into the mmap address
  fp = fopen(filename, "r");
  if (fp == CVI_NULL) {
    VPSS_UT_PRT("open data file error\n");
    CVI_VB_ReleaseBlock(blk);
    return CVI_FAILURE;
  }

  stVideoFrame.u32PoolId = CVI_VB_Handle2PoolId(blk);
  stVideoFrame.stVFrame.u32Length[0] = stVbCalConfig.u32MainYSize;
  stVideoFrame.stVFrame.u32Length[1] = stVbCalConfig.u32MainCSize;
  stVideoFrame.stVFrame.u64PhyAddr[0] = CVI_VB_Handle2PhysAddr(blk);
  stVideoFrame.stVFrame.u64PhyAddr[1] =
      stVideoFrame.stVFrame.u64PhyAddr[0] +
      ALIGN(stVbCalConfig.u32MainYSize, stVbCalConfig.u16AddrAlign);
  if (stVbCalConfig.plane_num == 3) {
    stVideoFrame.stVFrame.u32Length[2] = stVbCalConfig.u32MainCSize;
    stVideoFrame.stVFrame.u64PhyAddr[2] =
        stVideoFrame.stVFrame.u64PhyAddr[1] +
        ALIGN(stVbCalConfig.u32MainCSize, stVbCalConfig.u16AddrAlign);
  }

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    stVideoFrame.stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[i],
                               stVideoFrame.stVFrame.u32Length[i]);
  }

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    stVideoFrame.stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[i],
                               stVideoFrame.stVFrame.u32Length[i]);

    u32len = fread(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i], 1, fp);
    if (u32len <= 0) {
      VPSS_UT_PRT("vpss send frame: fread plane%d error\n", i);
      fclose(fp);
      CVI_VB_ReleaseBlock(blk);
      return CVI_FAILURE;
    }
    CVI_SYS_IonInvalidateCache(stVideoFrame.stVFrame.u64PhyAddr[i],
                               stVideoFrame.stVFrame.pu8VirAddr[i],
                               stVideoFrame.stVFrame.u32Length[i]);
  }

  VPSS_UT_PRT(
      "length of buffer(%d, %d, %d)\n", stVideoFrame.stVFrame.u32Length[0],
      stVideoFrame.stVFrame.u32Length[1], stVideoFrame.stVFrame.u32Length[2]);
  VPSS_UT_PRT(
      "phy addr(%#lx, %#lx, %#lx)\n", stVideoFrame.stVFrame.u64PhyAddr[0],
      stVideoFrame.stVFrame.u64PhyAddr[1], stVideoFrame.stVFrame.u64PhyAddr[2]);
  VPSS_UT_PRT("vir addr(%p, %p, %p)\n", stVideoFrame.stVFrame.pu8VirAddr[0],
              stVideoFrame.stVFrame.pu8VirAddr[1],
              stVideoFrame.stVFrame.pu8VirAddr[2]);

  fclose(fp);

  VPSS_UT_PRT("read file done and send out frame.\n");
  CVI_VPSS_SendFrame(VpssGrp, &stVideoFrame, -1);
  CVI_VB_ReleaseBlock(blk);

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    CVI_SYS_Munmap(stVideoFrame.stVFrame.pu8VirAddr[i],
                   stVideoFrame.stVFrame.u32Length[i]);
  }
  return CVI_SUCCESS;
}

/* SAMPLE_COMM_FRAME_SaveToFile:
 *   Save videoframe to the file
 *
 * [in]filename: char string of the file to save data.
 * [IN]pstVideoFrame: the videoframe whose data will be saved to file.
 * return: CVI_SUCCESS if no problem.
 */
CVI_S32 SAMPLE_COMM_FRAME_SaveToFile(const char *filename,
                                     VIDEO_FRAME_INFO_S *pstVideoFrame) {
  FILE *fp;
  CVI_U32 u32len, u32DataLen;

  fp = fopen(filename, "w");
  if (fp == CVI_NULL) {
    CVI_TRACE_LOG(CVI_DBG_ERR, "open data file error\n");
    return CVI_FAILURE;
  }
  printf("save to file: %s,width: %d,height: %d\n", filename,
         pstVideoFrame->stVFrame.u32Width, pstVideoFrame->stVFrame.u32Height);
  for (int i = 0; i < 3; ++i) {
    u32DataLen = pstVideoFrame->stVFrame.u32Stride[i] *
                 pstVideoFrame->stVFrame.u32Height;
    if (u32DataLen == 0) continue;
    if (i > 0 &&
        ((pstVideoFrame->stVFrame.enPixelFormat ==
          PIXEL_FORMAT_YUV_PLANAR_420) ||
         (pstVideoFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_NV12) ||
         (pstVideoFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_NV21)))
      u32DataLen >>= 1;

    pstVideoFrame->stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(pstVideoFrame->stVFrame.u64PhyAddr[i],
                               pstVideoFrame->stVFrame.u32Length[i]);

    CVI_TRACE_LOG(CVI_DBG_INFO, "plane(%d): paddr(%#lx) vaddr(%p) stride(%d)\n",
                  i, pstVideoFrame->stVFrame.u64PhyAddr[i],
                  pstVideoFrame->stVFrame.pu8VirAddr[i],
                  pstVideoFrame->stVFrame.u32Stride[i]);
    CVI_TRACE_LOG(CVI_DBG_INFO, " data_len(%d) plane_len(%d)\n", u32DataLen,
                  pstVideoFrame->stVFrame.u32Length[i]);
    u32len = fwrite(pstVideoFrame->stVFrame.pu8VirAddr[i], u32DataLen, 1, fp);
    if (u32len <= 0) {
      CVI_TRACE_LOG(CVI_DBG_ERR, "fwrite data(%d) error\n", i);
      break;
    }
    CVI_SYS_Munmap(pstVideoFrame->stVFrame.pu8VirAddr[i],
                   pstVideoFrame->stVFrame.u32Length[i]);
  }

  fclose(fp);
  return CVI_SUCCESS;
}

/* SAMPLE_COMM_VPSS_Stop: stop vpss grp
 *
 * VpssGrp: the VPSS Grp to control
 * pabChnEnable: array of VPSS CHN, stop if true.
 */
CVI_S32 SAMPLE_COMM_VPSS_Stop(VPSS_GRP VpssGrp, CVI_BOOL *pabChnEnable) {
  CVI_S32 j;
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_CHN VpssChn;

  for (j = 0; j < VPSS_MAX_PHY_CHN_NUM; j++) {
    if (pabChnEnable[j]) {
      VpssChn = j;
      s32Ret = CVI_VPSS_DisableChn(VpssGrp, VpssChn);
      if (s32Ret != CVI_SUCCESS) {
        VPSS_UT_PRT("Vpss stop Grp %d channel %d failed! Please check param\n",
                    VpssGrp, VpssChn);
        return CVI_FAILURE;
      }
    }
  }

  s32Ret = CVI_VPSS_StopGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("Vpss Stop Grp %d failed! Please check param\n", VpssGrp);
    return CVI_FAILURE;
  }

  s32Ret = CVI_VPSS_DestroyGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("Vpss Destroy Grp %d failed! Please check\n", VpssGrp);
    return CVI_FAILURE;
  }

  return CVI_SUCCESS;
}

CVI_S32 vpss_send_chn_frm_test(const char *src_file, const char *dst_file) {
  SIZE_S stSize = {.u32Width = 1920, .u32Height = 1080};
  PIXEL_FORMAT_E enPixelFormat =
      PIXEL_FORMAT_YUV_PLANAR_422 /*PIXEL_FORMAT_YUV_PLANAR_420*/;
  PIXEL_FORMAT_E enPixelFormatOut = enPixelFormat;
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSize;
  CVI_S32 s32Ret = CVI_SUCCESS;

  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  stVbConf.u32MaxPoolCnt = 1;

  u32BlkSize = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height,
                                       enPixelFormat, DATA_BITWIDTH_8,
                                       COMPRESS_MODE_NONE, DEFAULT_ALIGN);
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[0].u32BlkCnt = 1;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSize);

  // s32Ret = CVI_VB_SetConfig(&stVbConf);
  // if (s32Ret != CVI_SUCCESS) {
  //   VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
  //   return s32Ret;
  // }

  // s32Ret = CVI_VB_Init();
  // if (s32Ret != CVI_SUCCESS) {
  //   VPSS_UT_PRT("CVI_VB_Init failed!\n");
  //   return s32Ret;
  // }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    return s32Ret;
  }

  PIXEL_FORMAT_E output_format = PIXEL_FORMAT_UINT8_C3_PLANAR;
  /************************************************
   * step2:  Init VPSS
   ************************************************/
  VPSS_GRP VpssGrp = 0;
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  VPSS_CHN VpssChn = VPSS_CHN0;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S astVpssChnAttr[VPSS_MAX_PHY_CHN_NUM] = {0};

  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enPixelFormat;
  stVpssGrpAttr.u32MaxW = stSize.u32Width;
  stVpssGrpAttr.u32MaxH = stSize.u32Height;

  SIZE_S dstSize = {.u32Width = 640, .u32Height = 480};
  astVpssChnAttr[VpssChn].u32Width = dstSize.u32Width;
  astVpssChnAttr[VpssChn].u32Height = dstSize.u32Height;
  astVpssChnAttr[VpssChn].enVideoFormat = VIDEO_FORMAT_LINEAR;
  astVpssChnAttr[VpssChn].enPixelFormat = output_format;
  astVpssChnAttr[VpssChn].stFrameRate.s32SrcFrameRate = 30;
  astVpssChnAttr[VpssChn].stFrameRate.s32DstFrameRate = 30;
  astVpssChnAttr[VpssChn].u32Depth = 1;
  astVpssChnAttr[VpssChn].bMirror = CVI_FALSE;
  astVpssChnAttr[VpssChn].bFlip = CVI_FALSE;
  astVpssChnAttr[VpssChn].stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
  astVpssChnAttr[VpssChn].stAspectRatio.bEnableBgColor = CVI_FALSE;
  astVpssChnAttr[VpssChn].stAspectRatio.stVideoRect.s32X = 0;
  astVpssChnAttr[VpssChn].stAspectRatio.stVideoRect.s32Y = 0;
  astVpssChnAttr[VpssChn].stAspectRatio.stVideoRect.u32Width = dstSize.u32Width;
  astVpssChnAttr[VpssChn].stAspectRatio.stVideoRect.u32Height =
      dstSize.u32Height;
  astVpssChnAttr[VpssChn].stNormalize.bEnable = CVI_TRUE;
  astVpssChnAttr[VpssChn].stNormalize.factor[0] = 1.0f;
  astVpssChnAttr[VpssChn].stNormalize.factor[1] = 1.0f;
  astVpssChnAttr[VpssChn].stNormalize.factor[2] = 1.0f;
  astVpssChnAttr[VpssChn].stNormalize.mean[0] = 0.0f;
  astVpssChnAttr[VpssChn].stNormalize.mean[1] = 0.0f;
  astVpssChnAttr[VpssChn].stNormalize.mean[2] = 0.0f;
  /*start vpss*/
  abChnEnable[0] = CVI_TRUE;
  s32Ret = SAMPLE_COMM_VPSS_Init(VpssGrp, abChnEnable, &stVpssGrpAttr,
                                 astVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr,
                                  astVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  /************************************************
   * step3:  VPSS work
   ************************************************/
  VIDEO_FRAME_INFO_S stVideoFrame;
  VB_CAL_CONFIG_S stVbCalConfig;

  COMMON_GetPicBufferConfig(dstSize.u32Width, dstSize.u32Height, output_format,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN,
                            &stVbCalConfig);

  memset(&stVideoFrame, 0, sizeof(stVideoFrame));
  stVideoFrame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
  stVideoFrame.stVFrame.enPixelFormat = output_format;
  stVideoFrame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVideoFrame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
  stVideoFrame.stVFrame.u32Width = dstSize.u32Width;
  stVideoFrame.stVFrame.u32Height = dstSize.u32Height;
  stVideoFrame.stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;
  stVideoFrame.stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
  stVideoFrame.stVFrame.u32TimeRef = 0;
  stVideoFrame.stVFrame.u64PTS = 0;
  stVideoFrame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
  stVideoFrame.stVFrame.u32Length[0] = stVbCalConfig.u32MainYSize;
  stVideoFrame.stVFrame.u32Length[1] = stVbCalConfig.u32MainCSize;
  if (stVbCalConfig.plane_num == 3)
    stVideoFrame.stVFrame.u32Length[2] = stVbCalConfig.u32MainCSize;

  VIDEO_FRAME_S *vFrame = &(stVideoFrame.stVFrame);
  CVI_U32 u32MapSize =
      vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2];
  CVI_SYS_IonAlloc(&vFrame->u64PhyAddr[0], (CVI_VOID **)&vFrame->pu8VirAddr[0],
                   "alloc_name", u32MapSize);
  stVideoFrame.stVFrame.u64PhyAddr[1] =
      stVideoFrame.stVFrame.u64PhyAddr[0] +
      ALIGN(stVbCalConfig.u32MainYSize, stVbCalConfig.u16AddrAlign);
  if (stVbCalConfig.plane_num == 3) {
    stVideoFrame.stVFrame.u64PhyAddr[2] =
        stVideoFrame.stVFrame.u64PhyAddr[1] +
        ALIGN(stVbCalConfig.u32MainCSize, stVbCalConfig.u16AddrAlign);
  }

  for (int i = 0; i < stVbCalConfig.plane_num; ++i) {
    if (stVideoFrame.stVFrame.u32Length[i] == 0) continue;
    stVideoFrame.stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[i],
                               stVideoFrame.stVFrame.u32Length[i]);
  }

  s32Ret = CVI_VPSS_SendChnFrame(0, 0, &stVideoFrame, -1);
  if (s32Ret != CVI_SUCCESS) {
    CVI_TRACE_LOG(CVI_DBG_ERR,
                  "CVI_VPSS_SendChnFrame for grp1 chn0. s32Ret: 0x%x !\n",
                  s32Ret);
    goto ERR_VPSS_COMBINE;
  }
  VIDEO_FRAME_INFO_S stSrcFrame;
  s32Ret = SAMPLE_COMM_ReadFrame(&stSize, enPixelFormat, src_file, stSrcFrame);

  CVI_VPSS_SendFrame(VpssGrp, &stSrcFrame, -1);
  if (s32Ret != CVI_SUCCESS) {
    CVI_TRACE_LOG(CVI_DBG_ERR,
                  "CVI_VPSS_SendFrame for grp1 chn0. s32Ret: 0x%x !\n", s32Ret);
    goto ERR_VPSS_COMBINE;
  }

  s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stVideoFrame, 10000 /*100*/);
  if (s32Ret != CVI_SUCCESS) {
    CVI_TRACE_LOG(CVI_DBG_ERR,
                  "CVI_VPSS_GetChnFrame for grp1 chn0. s32Ret: 0x%x !\n",
                  s32Ret);
    goto ERR_VPSS_COMBINE;
  }

  SAMPLE_COMM_FRAME_SaveToFile(dst_file, &stVideoFrame);
  CVI_VPSS_ReleaseChnFrame(0, 0, &stVideoFrame);
  CVI_VPSS_ReleaseChnFrame(0, 0, &stSrcFrame);

ERR_VPSS_COMBINE:
  /*vpss exit fot next case*/
  for (int grp_num = 0; grp_num <= VpssGrp; grp_num++) {
    SAMPLE_COMM_VPSS_Stop(grp_num, abChnEnable);
  }
  CVI_SYS_Exit();
  // CVI_VB_Exit();

  VPSS_UT_PRT("Send channel frame test %s\n",
              s32Ret == CVI_SUCCESS ? "pass" : "fail");
  return s32Ret;
}

static CVI_VOID *multi_thread_run(CVI_VOID *arg) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 i, s32Repeat = TEST_CNT0, s32AvgFrameRate;
  VPSS_GRP VpssGrp;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VIDEO_FRAME_INFO_S stVideoFrameOut, stVideoFrameIn;
  PIXEL_FORMAT_E enFormatIn = PIXEL_FORMAT_YUV_PLANAR_420;
  PIXEL_FORMAT_E enFormatOut = PIXEL_FORMAT_NV21;
  SIZE_S stSizeIn = {DEFAULT_W, DEFAULT_H};
  CVI_CHAR *pFileNameIn = VPSS_DEFAULT_FILE_IN;
  CVI_U64 u64PTS, u64StartPTS, u64EndPTS, u64CurPTS1, u64CurPTS2, u64CostTime;
  CVI_U32 u32FrameCnt;
  CVI_U64 u64MinCostTime = 10000, u64MaxCostTime = 0;

  arg = arg;

  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enFormatIn;
  stVpssGrpAttr.u32MaxW = stSizeIn.u32Width;
  stVpssGrpAttr.u32MaxH = stSizeIn.u32Height;

  stVpssChnAttr.u32Width = DEFAULT_W;
  stVpssChnAttr.u32Height = DEFAULT_H;
  stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr.enPixelFormat = enFormatOut;
  stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssChnAttr.u32Depth = 1;
  stVpssChnAttr.bMirror = CVI_FALSE;
  stVpssChnAttr.bFlip = CVI_FALSE;
  stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

  VpssGrp = CVI_VPSS_GetAvailableGrp();
  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    return NULL;
  }

  s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    goto exit0;
  }

  s32Ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, 1);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_AttachVbPool failed with %#x\n", s32Ret);
    goto exit0;
  }

  s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    goto exit0;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit1;
  }

  // send frame
  s32Ret = FileToFrame(&stSizeIn, enFormatIn, pFileNameIn, &stVideoFrameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileToFrame fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit2;
  }

  CVI_SYS_GetCurPTS(&u64PTS);
  u32FrameCnt = 0;
  u64StartPTS = u64PTS;

  for (i = 0; i < s32Repeat; i++) {
    CVI_SYS_GetCurPTS(&u64CurPTS1);
    s32Ret = CVI_VPSS_SendFrame(VpssGrp, &stVideoFrameIn, 1000);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SendFrame fail.\n");
      goto exit3;
    }

    s32Ret =
        CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrameOut, UT_TIMEOUT_MS);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n", s32Ret);
      goto exit3;
    }
    CVI_SYS_GetCurPTS(&u64CurPTS2);

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrameOut);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_ReleaseChnFrame for grp0 chn0. s32Ret: 0x%x !\n",
                  s32Ret);
      goto exit3;
    }

    u32FrameCnt++;
    u64CostTime = u64CurPTS2 - u64CurPTS1;
    u64MinCostTime =
        u64CostTime < u64MinCostTime ? u64CostTime : u64MinCostTime;
    u64MaxCostTime =
        u64CostTime > u64MaxCostTime ? u64CostTime : u64MaxCostTime;

    if ((u64CurPTS2 - u64PTS) >= 1000000) {
      VPSS_UT_PRT("[VpssGrp%d] FrameRate:%d fps\n", VpssGrp, u32FrameCnt);
      u32FrameCnt = 0;
      u64PTS = u64CurPTS2;
    }
  }
  CVI_SYS_GetCurPTS(&u64EndPTS);
  s32AvgFrameRate = s32Repeat / ((u64EndPTS - u64StartPTS) / 1000000);
#if 0
	if ((u64MaxCostTime - u64MinCostTime) > 4000)
		s32Ret = -1;
	if (s32AvgFrameRate < (2400/THREAD_CNT - 5))
		s32Ret = -1;
#endif
  VPSS_UT_PRT(
      "\n[VpssGrp%d] 1080P cost time: Min-Max: (%ld, %ld)us, offset=%ld\n",
      VpssGrp, u64MinCostTime, u64MaxCostTime, u64MaxCostTime - u64MinCostTime);
  VPSS_UT_PRT("\n[VpssGrp%d] average FrameRate:%d fps\n", VpssGrp,
              s32AvgFrameRate);

exit3:
  CVI_VB_ReleaseBlock(
      CVI_VB_PhysAddr2Handle(stVideoFrameIn.stVFrame.u64PhyAddr[0]));
exit2:
  CVI_VPSS_StopGrp(VpssGrp);
exit1:
  CVI_VPSS_DisableChn(VpssGrp, VpssChn);
exit0:
  CVI_VPSS_DestroyGrp(VpssGrp);

  if (s32Ret == CVI_SUCCESS) {
    pthread_mutex_lock(&s_SyncMutex);
    s_u32Flag |= BIT(VpssGrp);
    pthread_mutex_unlock(&s_SyncMutex);
  }

  return NULL;
}

static CVI_S32 vpss_test_resize(CVI_VOID) {
  CVI_S32 i, s32Ret = CVI_SUCCESS;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSizeIn, u32BlkSizeOut;
  VIDEO_FRAME_INFO_S stVideoFrameIn, stVideoFrameOut;
  SIZE_S stSizeIn = {1920, 1080};
  PIXEL_FORMAT_E enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  CVI_CHAR *pstFileNameIn = VPSS_DEFAULT_FILE_IN;
  CVI_S32 s32Min = 16;
  CVI_S32 s32Max = 8192;
#ifndef FPGA_PORTING
  CVI_S32 s32Step = 32;
#else
  CVI_S32 s32Step = 256;
#endif
  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  u32BlkSizeIn = COMMON_GetPicBufferSize(stSizeIn.u32Width, stSizeIn.u32Height,
                                         enPixelFormat, DATA_BITWIDTH_8,
                                         COMPRESS_MODE_NONE, DEFAULT_ALIGN);
  u32BlkSizeOut =
      COMMON_GetPicBufferSize(8192, 8192, enPixelFormat, DATA_BITWIDTH_8,
                              COMPRESS_MODE_NONE, DEFAULT_ALIGN);

  stVbConf.u32MaxPoolCnt = 2;
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSizeIn;
  stVbConf.astCommPool[0].u32BlkCnt = 1;
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
  stVbConf.astCommPool[1].u32BlkSize = u32BlkSizeOut;
  stVbConf.astCommPool[1].u32BlkCnt = 1;
  stVbConf.astCommPool[1].enRemapMode = VB_REMAP_MODE_CACHED;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSizeIn);
  VPSS_UT_PRT("common pool[1] BlkSize %d\n", u32BlkSizeOut);

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_Init failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    goto exit0;
  }

  /************************************************
   * step2:  Init VPSS
   ************************************************/
  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enPixelFormat;
  stVpssGrpAttr.u32MaxW = stSizeIn.u32Width;
  stVpssGrpAttr.u32MaxH = stSizeIn.u32Height;

  stVpssChnAttr.u32Width = stSizeIn.u32Width;
  stVpssChnAttr.u32Height = stSizeIn.u32Height;
  stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr.enPixelFormat = enPixelFormat;
  stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssChnAttr.u32Depth = 1;
  stVpssChnAttr.bMirror = CVI_FALSE;
  stVpssChnAttr.bFlip = CVI_FALSE;
  stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    goto exit1;
  }

  s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, 1);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_AttachVbPool failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    goto exit2;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit3;
  }

  // send frame
  s32Ret =
      FileToFrame(&stSizeIn, enPixelFormat, pstFileNameIn, &stVideoFrameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileToFrame fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit4;
  }

  for (i = s32Min; i <= s32Max; i += s32Step) {
    stVpssChnAttr.u32Width = i;
    stVpssChnAttr.u32Height = i;
    s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
      goto exit5;
    }
    s32Ret = CVI_VPSS_SendFrame(VpssGrp, &stVideoFrameIn, 1000);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SendFrame fail.\n");
      goto exit5;
    }
    s32Ret =
        CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrameOut, UT_TIMEOUT_MS);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n", s32Ret);
      VPSS_UT_PRT("output: w=%d h=%d fail\n", i, i);
      goto exit5;
    }
    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrameOut);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_ReleaseChnFrame for grp0 chn0. s32Ret: 0x%x !\n",
                  s32Ret);
      goto exit5;
    }
  }

exit5:
  CVI_VB_ReleaseBlock(
      CVI_VB_PhysAddr2Handle(stVideoFrameIn.stVFrame.u64PhyAddr[0]));
exit4:
  CVI_VPSS_StopGrp(VpssGrp);
exit3:
  CVI_VPSS_DisableChn(VpssGrp, VpssChn);
exit2:
  CVI_VPSS_DestroyGrp(VpssGrp);
exit1:
  CVI_SYS_Exit();
exit0:
  CVI_VB_Exit();

  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

static CVI_S32 test_crop(CVI_BOOL isChn) {
  CVI_S32 i, s32Ret = CVI_SUCCESS;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSizeIn, u32BlkSizeOut;
  VIDEO_FRAME_INFO_S stVideoFrameIn, stVideoFrameOut;
  SIZE_S stSizeIn = {1920, 1080};
  SIZE_S stSizeOut = {640, 480};
  PIXEL_FORMAT_E enPixelFormatIn = PIXEL_FORMAT_YUV_PLANAR_420;
  PIXEL_FORMAT_E enPixelFormatOut = PIXEL_FORMAT_RGB_888;
  CVI_CHAR *pstFileNameIn = VPSS_DEFAULT_FILE_IN;
  VPSS_CROP_INFO_S stCropInfo;

  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  u32BlkSizeIn = COMMON_GetPicBufferSize(stSizeIn.u32Width, stSizeIn.u32Height,
                                         enPixelFormatIn, DATA_BITWIDTH_8,
                                         COMPRESS_MODE_NONE, DEFAULT_ALIGN);
  u32BlkSizeOut = COMMON_GetPicBufferSize(
      stSizeOut.u32Width, stSizeOut.u32Height, enPixelFormatOut,
      DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);

  stVbConf.u32MaxPoolCnt = 2;
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSizeIn;
  stVbConf.astCommPool[0].u32BlkCnt = 1;
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
  stVbConf.astCommPool[1].u32BlkSize = u32BlkSizeOut;
  stVbConf.astCommPool[1].u32BlkCnt = 1;
  stVbConf.astCommPool[1].enRemapMode = VB_REMAP_MODE_CACHED;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSizeIn);
  VPSS_UT_PRT("common pool[1] BlkSize %d\n", u32BlkSizeOut);

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_Init failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    goto exit0;
  }

  /************************************************
   * step2:  Init VPSS
   ************************************************/
  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enPixelFormatIn;
  stVpssGrpAttr.u32MaxW = stSizeIn.u32Width;
  stVpssGrpAttr.u32MaxH = stSizeIn.u32Height;

  stVpssChnAttr.u32Width = stSizeOut.u32Width;
  stVpssChnAttr.u32Height = stSizeOut.u32Height;
  stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr.enPixelFormat = enPixelFormatOut;
  stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssChnAttr.u32Depth = 1;
  stVpssChnAttr.bMirror = CVI_FALSE;
  stVpssChnAttr.bFlip = CVI_FALSE;
  stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    goto exit1;
  }

  s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, 1);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_AttachVbPool failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    goto exit2;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit3;
  }

  // send frame
  s32Ret =
      FileToFrame(&stSizeIn, enPixelFormatIn, pstFileNameIn, &stVideoFrameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileToFrame fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit4;
  }

  stCropInfo.bEnable = CVI_TRUE;
  stCropInfo.enCropCoordinate = VPSS_CROP_ABS_COOR;
  srand((unsigned)time(NULL));

  for (i = 0; i <= 50; i++) {
    stCropInfo.stCropRect.s32X = RANDOM(0, stSizeIn.u32Width - 16);
    stCropInfo.stCropRect.s32Y = RANDOM(0, stSizeIn.u32Height - 16);
    stCropInfo.stCropRect.u32Width = RANDOM(16, stSizeIn.u32Width) & ~(0x1);
    stCropInfo.stCropRect.u32Height = RANDOM(16, stSizeIn.u32Height) & ~(0x1);
    if (((stCropInfo.stCropRect.s32X + stCropInfo.stCropRect.u32Width) >
         stSizeIn.u32Width) ||
        ((stCropInfo.stCropRect.s32Y + stCropInfo.stCropRect.u32Height) >
         stSizeIn.u32Height)) {
      i--;
      continue;
    }

    if (isChn)
      s32Ret = CVI_VPSS_SetChnCrop(VpssGrp, VpssChn, &stCropInfo);
    else
      s32Ret = CVI_VPSS_SetGrpCrop(VpssGrp, &stCropInfo);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
      goto exit5;
    }
    s32Ret = CVI_VPSS_SendFrame(VpssGrp, &stVideoFrameIn, 1000);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SendFrame fail.\n");
      goto exit5;
    }
    s32Ret =
        CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrameOut, UT_TIMEOUT_MS);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n", s32Ret);
      VPSS_UT_PRT("crop fail,x=%d y=%d w=%d h=%d\n", stCropInfo.stCropRect.s32X,
                  stCropInfo.stCropRect.s32Y, stCropInfo.stCropRect.u32Width,
                  stCropInfo.stCropRect.u32Height);
      goto exit5;
    }
    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrameOut);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_ReleaseChnFrame for grp0 chn0. s32Ret: 0x%x !\n",
                  s32Ret);
      goto exit5;
    }
  }

exit5:
  CVI_VB_ReleaseBlock(
      CVI_VB_PhysAddr2Handle(stVideoFrameIn.stVFrame.u64PhyAddr[0]));
exit4:
  CVI_VPSS_StopGrp(VpssGrp);
exit3:
  CVI_VPSS_DisableChn(VpssGrp, VpssChn);
exit2:
  CVI_VPSS_DestroyGrp(VpssGrp);
exit1:
  CVI_SYS_Exit();
exit0:
  CVI_VB_Exit();

  return s32Ret;
}

static CVI_S32 vpss_test_grp_crop(CVI_VOID) {
  CVI_S32 s32Ret = test_crop(CVI_FALSE);

  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

static CVI_S32 vpss_test_chn_crop(CVI_VOID) {
  CVI_S32 s32Ret = test_crop(CVI_TRUE);

  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

// normalize: x*factor - mean
static CVI_S32 vpss_test_normalize(CVI_VOID) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_BASIC_TEST_PARAM stTestParam;

  memset(&stTestParam, 0, sizeof(stTestParam));
  stTestParam.VpssGrp = 0;
  stTestParam.stSizeIn.u32Width = DEFAULT_W;
  stTestParam.stSizeIn.u32Height = DEFAULT_H;
  stTestParam.stSizeOut.u32Width = DEFAULT_W;
  stTestParam.stSizeOut.u32Height = DEFAULT_H;
  stTestParam.bMirror = CVI_FALSE;
  stTestParam.bFlip = CVI_FALSE;
  stTestParam.enFormatIn = PIXEL_FORMAT_RGB_888;
  stTestParam.enFormatOut = PIXEL_FORMAT_RGB_888;
  stTestParam.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stTestParam.stNormalize.bEnable = CVI_TRUE;
  stTestParam.stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
  stTestParam.stNormalize.factor[0] = 0.5;
  stTestParam.stNormalize.factor[1] = 0.5;
  stTestParam.stNormalize.factor[2] = 0.5;
  stTestParam.stNormalize.mean[0] = 10;
  stTestParam.stNormalize.mean[1] = 10;
  stTestParam.stNormalize.mean[2] = 10;
  stTestParam.u32CheckSum = 0x891e2d0b;
  strncpy(stTestParam.aszMD5Sum, MD5_NORMALIZE, sizeof(stTestParam.aszMD5Sum));
  strncpy(stTestParam.aszFileNameIn, VPSS_RGB_FILE_IN,
          sizeof(stTestParam.aszFileNameIn));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret = basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

// convert: ax + b
static CVI_S32 vpss_test_convert(CVI_VOID) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_BASIC_TEST_PARAM stTestParam;

  memset(&stTestParam, 0, sizeof(stTestParam));
  stTestParam.VpssGrp = 0;
  stTestParam.stSizeIn.u32Width = DEFAULT_W;
  stTestParam.stSizeIn.u32Height = DEFAULT_H;
  stTestParam.stSizeOut.u32Width = DEFAULT_W;
  stTestParam.stSizeOut.u32Height = DEFAULT_H;
  stTestParam.bMirror = CVI_FALSE;
  stTestParam.bFlip = CVI_FALSE;
  stTestParam.enFormatIn = PIXEL_FORMAT_RGB_888;
  stTestParam.enFormatOut = PIXEL_FORMAT_UINT8_C3_PLANAR;
  stTestParam.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stTestParam.stNormalize.bEnable = CVI_FALSE;
  stTestParam.stConvert.bEnable = CVI_TRUE;
  stTestParam.stConvert.u32aFactor[0] = 2 * 8192;
  stTestParam.stConvert.u32aFactor[1] = 2 * 8192;
  stTestParam.stConvert.u32aFactor[2] = 2 * 8192;
  stTestParam.stConvert.u32bFactor[0] = 5 * 8192;
  stTestParam.stConvert.u32bFactor[1] = 5 * 8192;
  stTestParam.stConvert.u32bFactor[2] = 5 * 8192;
  stTestParam.u32CheckSum = 0xda899972;
  strncpy(stTestParam.aszMD5Sum, MD5_CONVERT, sizeof(stTestParam.aszMD5Sum));
  strncpy(stTestParam.aszFileNameIn, VPSS_RGB_FILE_IN,
          sizeof(stTestParam.aszFileNameIn));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret = basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

static CVI_S32 vpss_test_scale_coef(CVI_VOID) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_BASIC_TEST_PARAM stTestParam;

  memset(&stTestParam, 0, sizeof(stTestParam));
  stTestParam.VpssGrp = 0;
  stTestParam.stSizeIn.u32Width = DEFAULT_W;
  stTestParam.stSizeIn.u32Height = DEFAULT_H;
  stTestParam.stSizeOut.u32Width = 1280;
  stTestParam.stSizeOut.u32Height = 720;
  stTestParam.bMirror = CVI_FALSE;
  stTestParam.bFlip = CVI_FALSE;
  stTestParam.enFormatIn = PIXEL_FORMAT_YUV_PLANAR_420;
  stTestParam.enFormatOut = PIXEL_FORMAT_YUV_PLANAR_420;
  stTestParam.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stTestParam.stNormalize.bEnable = CVI_FALSE;
  stTestParam.enCoef = VPSS_SCALE_COEF_BICUBIC;
  stTestParam.u32CheckSum = 0xc3cf194b;
  strncpy(stTestParam.aszMD5Sum, MD5_SCALE_COEF1,
          sizeof(stTestParam.aszMD5Sum));
  strncpy(stTestParam.aszFileNameIn, VPSS_DEFAULT_FILE_IN,
          sizeof(stTestParam.aszFileNameIn));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  // bilinear
  stTestParam.enCoef = VPSS_SCALE_COEF_BILINEAR;
  stTestParam.u32CheckSum = 0x5583d343;
  strncpy(stTestParam.aszMD5Sum, MD5_SCALE_COEF2,
          sizeof(stTestParam.aszMD5Sum));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  // nearest
  stTestParam.enCoef = VPSS_SCALE_COEF_NEAREST;
  stTestParam.u32CheckSum = 0x39768945;
  strncpy(stTestParam.aszMD5Sum, MD5_SCALE_COEF3,
          sizeof(stTestParam.aszMD5Sum));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  // opencv bicubic
  stTestParam.enCoef = VPSS_SCALE_COEF_BICUBIC_OPENCV;
  stTestParam.u32CheckSum = 0xbdc97502;
  strncpy(stTestParam.aszMD5Sum, MD5_SCALE_COEF4,
          sizeof(stTestParam.aszMD5Sum));
  snprintf(stTestParam.aszFileNameOut, 64, "%s/%s_%d_%d.bin", OUT_FILE_PREFIX,
           __func__, stTestParam.stSizeOut.u32Width,
           stTestParam.stSizeOut.u32Height);

  s32Ret |= basic(&stTestParam);
  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

static CVI_VOID *pressure_thread_run(CVI_VOID *arg) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 i, j, s32Repeat = TEST_CNT0;
  VPSS_GRP VpssGrp;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VIDEO_FRAME_INFO_S stVideoFrameOut, stVideoFrameIn;
  PIXEL_FORMAT_E enFormatIn = PIXEL_FORMAT_YUV_PLANAR_420;
  PIXEL_FORMAT_E enFormatOut = PIXEL_FORMAT_NV21;
  SIZE_S stSizeIn = {DEFAULT_W, DEFAULT_H};
  SIZE_S astSizeOut[VPSS_MAX_CHN_NUM] = {
      {DEFAULT_W, DEFAULT_H}, {1280, 720}, {640, 360}, {320, 180}};
  CVI_BOOL abChnEnable[VPSS_MAX_CHN_NUM];
  CVI_CHAR *pFileNameIn = VPSS_DEFAULT_FILE_IN;
  CVI_U32 u32ChnMask;

  arg = arg;

  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enFormatIn;
  stVpssGrpAttr.u32MaxW = stSizeIn.u32Width;
  stVpssGrpAttr.u32MaxH = stSizeIn.u32Height;

  VpssGrp = CVI_VPSS_GetAvailableGrp();
  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    return NULL;
  }

  for (i = 0; i < VPSS_MAX_CHN_NUM; i++) {
    VpssChn = i;
    stVpssChnAttr.u32Width = astSizeOut[i].u32Width;
    stVpssChnAttr.u32Height = astSizeOut[i].u32Height;
    stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    stVpssChnAttr.enPixelFormat = enFormatOut;
    stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssChnAttr.u32Depth = 1;
    stVpssChnAttr.bMirror = CVI_FALSE;
    stVpssChnAttr.bFlip = CVI_FALSE;
    stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

    s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
      goto exit0;
    }

    s32Ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, 1 + i);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_AttachVbPool failed with %#x\n", s32Ret);
      goto exit0;
    }

    s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
      goto exit0;
    }
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit1;
  }

  // send frame
  s32Ret = FileToFrame(&stSizeIn, enFormatIn, pFileNameIn, &stVideoFrameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileToFrame fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit2;
  }

  for (i = 0; i < s32Repeat; i++) {
    u32ChnMask = 0;

    for (j = 0; j < VPSS_MAX_CHN_NUM; j++) {
      abChnEnable[j] = rand() % 2 ? CVI_TRUE : CVI_FALSE;
      if (abChnEnable[j]) {
        CVI_VPSS_EnableChn(VpssGrp, j);
        u32ChnMask |= BIT(j);
      } else
        CVI_VPSS_DisableChn(VpssGrp, j);
    }

    if (!u32ChnMask) continue;

    s32Ret = CVI_VPSS_SendFrame(VpssGrp, &stVideoFrameIn, 1000);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SendFrame fail.\n");
      goto exit3;
    }

    for (j = 0; j < VPSS_MAX_CHN_NUM; j++) {
      if (!abChnEnable[j]) continue;
      VpssChn = j;
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrameOut,
                                    UT_TIMEOUT_MS);
      if (s32Ret != CVI_SUCCESS) {
        VPSS_UT_PRT("CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n", s32Ret);
        goto exit3;
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrameOut);
      if (s32Ret != CVI_SUCCESS) {
        VPSS_UT_PRT("CVI_VPSS_ReleaseChnFrame for grp0 chn0. s32Ret: 0x%x !\n",
                    s32Ret);
        goto exit3;
      }
    }
  }

exit3:
  CVI_VB_ReleaseBlock(
      CVI_VB_PhysAddr2Handle(stVideoFrameIn.stVFrame.u64PhyAddr[0]));
exit2:
  CVI_VPSS_StopGrp(VpssGrp);
exit1:
  for (j = 0; j < VPSS_MAX_CHN_NUM; j++) CVI_VPSS_DisableChn(VpssGrp, j);
exit0:
  CVI_VPSS_DestroyGrp(VpssGrp);

  if (s32Ret == CVI_SUCCESS) {
    pthread_mutex_lock(&s_SyncMutex);
    s_u32Flag |= BIT(VpssGrp);
    pthread_mutex_unlock(&s_SyncMutex);
  }

  return NULL;
}

static CVI_S32 vpss_test_perf(CVI_VOID) {
  CVI_S32 i, s32Ret = CVI_SUCCESS;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSize;
  VIDEO_FRAME_INFO_S stVideoFrameIn, stVideoFrameOut;
  SIZE_S stSize = {1920, 1080};
  PIXEL_FORMAT_E enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  CVI_CHAR *pstFileNameIn = VPSS_DEFAULT_FILE_IN;
  CVI_U64 u64CurPTS1, u64CurPTS2, u64CostTime;
  CVI_U64 u64MinCostTime = 10000, u64MaxCostTime = 0;

  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  u32BlkSize = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height,
                                       enPixelFormat, DATA_BITWIDTH_8,
                                       COMPRESS_MODE_NONE, DEFAULT_ALIGN);

  stVbConf.u32MaxPoolCnt = 1;
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[0].u32BlkCnt = 2;
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSize);

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_Init failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    goto exit0;
  }

  /************************************************
   * step2:  Init VPSS
   ************************************************/
  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enPixelFormat;
  stVpssGrpAttr.u32MaxW = stSize.u32Width;
  stVpssGrpAttr.u32MaxH = stSize.u32Height;

  stVpssChnAttr.u32Width = stSize.u32Width;
  stVpssChnAttr.u32Height = stSize.u32Height;
  stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr.enPixelFormat = enPixelFormat;
  stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssChnAttr.u32Depth = 1;
  stVpssChnAttr.bMirror = CVI_FALSE;
  stVpssChnAttr.bFlip = CVI_FALSE;
  stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    goto exit1;
  }

  s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    goto exit2;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit3;
  }

  // send frame
  s32Ret = FileToFrame(&stSize, enPixelFormat, pstFileNameIn, &stVideoFrameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileToFrame fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit4;
  }

  for (i = 0; i < TEST_CNT0; i++) {
    CVI_SYS_GetCurPTS(&u64CurPTS1);
    s32Ret = CVI_VPSS_SendFrame(VpssGrp, &stVideoFrameIn, 1000);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_SendFrame fail.\n");
      goto exit5;
    }
    s32Ret =
        CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVideoFrameOut, UT_TIMEOUT_MS);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_GetChnFrame fail. s32Ret: 0x%x !\n", s32Ret);
      goto exit5;
    }
    CVI_SYS_GetCurPTS(&u64CurPTS2);
    u64CostTime = u64CurPTS2 - u64CurPTS1;
    u64MinCostTime =
        u64CostTime < u64MinCostTime ? u64CostTime : u64MinCostTime;
    u64MaxCostTime =
        u64CostTime > u64MaxCostTime ? u64CostTime : u64MaxCostTime;

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVideoFrameOut);
    if (s32Ret != CVI_SUCCESS) {
      VPSS_UT_PRT("CVI_VPSS_ReleaseChnFrame for grp0 chn0. s32Ret: 0x%x !\n",
                  s32Ret);
      goto exit5;
    }
    // system("cat /proc/soph/vpss");
  }
  if ((u64MaxCostTime - u64MinCostTime) > 500) {
    s32Ret = -1;
    VPSS_UT_PRT("Time fluctuation anomaly !!!\n");
  }
  VPSS_UT_PRT("1080P cost time: Min-Max: (%ld, %ld)us, offset=%ld\n",
              u64MinCostTime, u64MaxCostTime, u64MaxCostTime - u64MinCostTime);

exit5:
  CVI_VB_ReleaseBlock(
      CVI_VB_PhysAddr2Handle(stVideoFrameIn.stVFrame.u64PhyAddr[0]));
exit4:
  CVI_VPSS_StopGrp(VpssGrp);
exit3:
  CVI_VPSS_DisableChn(VpssGrp, VpssChn);
exit2:
  CVI_VPSS_DestroyGrp(VpssGrp);
exit1:
  CVI_SYS_Exit();
exit0:
  CVI_VB_Exit();

  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

static CVI_S32 vpss_mp_get_chn_frm_test(CVI_VOID) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr = {0};
  VB_CONFIG_S stVbConf;
  CVI_U32 u32BlkSize;
  SIZE_S stSize = {DEFAULT_W, DEFAULT_H};
  PIXEL_FORMAT_E enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  CVI_CHAR *pstFileNameIn = VPSS_DEFAULT_FILE_IN;

  /************************************************
   * step1:  Init SYS and common VB
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));

  u32BlkSize = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height,
                                       enPixelFormat, DATA_BITWIDTH_8,
                                       COMPRESS_MODE_NONE, DEFAULT_ALIGN);

  stVbConf.u32MaxPoolCnt = 1;
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[0].u32BlkCnt = 2;
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
  VPSS_UT_PRT("common pool[0] BlkSize %d\n", u32BlkSize);

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VB_Init failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_SYS_Init failed!\n");
    goto exit0;
  }

  /************************************************
   * step2:  Init VPSS
   ************************************************/
  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = enPixelFormat;
  stVpssGrpAttr.u32MaxW = stSize.u32Width;
  stVpssGrpAttr.u32MaxH = stSize.u32Height;

  stVpssChnAttr.u32Width = stSize.u32Width;
  stVpssChnAttr.u32Height = stSize.u32Height;
  stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr.enPixelFormat = enPixelFormat;
  stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssChnAttr.u32Depth = 1;
  stVpssChnAttr.bMirror = CVI_FALSE;
  stVpssChnAttr.bFlip = CVI_FALSE;
  stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

  s32Ret = CVI_VPSS_CreateGrp(VpssGrp, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrp,
                s32Ret);
    goto exit1;
  }

  s32Ret = CVI_VPSS_SetChnAttr(VpssGrp, VpssChn, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    goto exit2;
  }

  s32Ret = CVI_VPSS_EnableChn(VpssGrp, VpssChn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    goto exit2;
  }

  /*start vpss*/
  s32Ret = CVI_VPSS_StartGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    goto exit3;
  }

  // send frame
  s32Ret = FileSendToVpss(VpssGrp, &stSize, enPixelFormat, pstFileNameIn);
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("FileToFrame fail, s32Ret: 0x%x !\n", s32Ret);
    goto exit4;
  }

  s32Ret = system("./vpss_ut_client 0");
  if (s32Ret != CVI_SUCCESS) {
    VPSS_UT_PRT("client fail\n");
  }

exit4:
  CVI_VPSS_StopGrp(VpssGrp);
exit3:
  CVI_VPSS_DisableChn(VpssGrp, VpssChn);
exit2:
  CVI_VPSS_DestroyGrp(VpssGrp);
exit1:
  CVI_SYS_Exit();
exit0:
  CVI_VB_Exit();

  TEST_CHECK_RET(s32Ret);

  return s32Ret;
}

int main(int argc, char **argv) {
  CVI_S32 s32Ret;
  CVI_S32 op = 255;
  char mkdir_cmd[64] = {0};

  VPSS_UT_PRT("Create Output Directory %s !\n", OUT_FILE_PREFIX);
  snprintf(mkdir_cmd, 63, "mkdir -p %s", OUT_FILE_PREFIX);
  system(mkdir_cmd);

  system("stty erase ^H");

  signal(SIGINT, vpss_ut_HandleSig);
  signal(SIGTERM, vpss_ut_HandleSig);

  return vpss_send_chn_frm_test(argv[1], argv[2]);
}
