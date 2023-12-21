#include "soc_process.hpp"

void SocProcessor::process_init(int vpssgrp_width, int vpssgrp_height, PIXEL_FORMAT_E enSrcFormat,
                                int BlkCount, int BlkCount) {
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, enSrcFormat, BlkCount,
                                 vpssgrp_width, vpssgrp_height, enSrcFormat, BlkCount);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
}