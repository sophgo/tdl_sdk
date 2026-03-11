#include "utils/frame_dump.hpp"
#include "utils/tdl_log.hpp"

int32_t FrameDump::saveFrame(char *filename,
                             VIDEO_FRAME_INFO_S *pstVideoFrame) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  FILE *fp;
  CVI_U32 u32len, u32DataLen;

  fp = fopen(filename, "w");
  if (fp == CVI_NULL) {
    LOGI("open data file(%s) error\n", filename);
    return CVI_FAILURE;
  }

  for (int i = 0; i < 3; ++i) {
    u32DataLen = pstVideoFrame->stVFrame.u32Stride[i] *
                 pstVideoFrame->stVFrame.u32Height;
    if (u32DataLen == 0) {
      continue;
    }
    if (i > 0 &&
        ((pstVideoFrame->stVFrame.enPixelFormat ==
          PIXEL_FORMAT_YUV_PLANAR_420) ||
         (pstVideoFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_NV12) ||
         (pstVideoFrame->stVFrame.enPixelFormat == PIXEL_FORMAT_NV21))) {
      u32DataLen >>= 1;
    }

    pstVideoFrame->stVFrame.pu8VirAddr[i] =
        (CVI_U8 *)CVI_SYS_Mmap(pstVideoFrame->stVFrame.u64PhyAddr[i],
                               pstVideoFrame->stVFrame.u32Length[i]);
    CVI_SYS_IonFlushCache(pstVideoFrame->stVFrame.u64PhyAddr[i],
                          pstVideoFrame->stVFrame.pu8VirAddr[i],
                          pstVideoFrame->stVFrame.u32Length[i]);
    LOGI("plane(%d): paddr(%#" PRIx64 ") vaddr(%p) stride(%d)\n", i,
         pstVideoFrame->stVFrame.u64PhyAddr[i],
         pstVideoFrame->stVFrame.pu8VirAddr[i],
         pstVideoFrame->stVFrame.u32Stride[i]);
    LOGI(" data_len(%d) plane_len(%d)\n", u32DataLen,
         pstVideoFrame->stVFrame.u32Length[i]);
    u32len = fwrite(pstVideoFrame->stVFrame.pu8VirAddr[i], u32DataLen, 1, fp);
    if (u32len <= 0) {
      LOGI("fwrite data(%d) error\n", i);
      s32Ret = CVI_FAILURE;
      break;
    }
    CVI_SYS_Munmap(pstVideoFrame->stVFrame.pu8VirAddr[i],
                   pstVideoFrame->stVFrame.u32Length[i]);
  }

  fclose(fp);
  return s32Ret;
}