#ifndef VI_VO_UTILS_H_
#define VI_VO_UTILS_H_

#include <cvi_sys.h>
#include <cvi_vi.h>
#include "sample_comm.h"

CVI_S32 InitVI(SAMPLE_VI_CONFIG_S *pstViConfig, CVI_U32 *devNum);

CVI_S32 InitVO(const CVI_U32 width, const CVI_U32 height, SAMPLE_VO_CONFIG_S *stVoConfig);

CVI_S32 InitVPSS(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn, const VPSS_CHN vpssChnVO,
                 const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                 const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened);
#endif