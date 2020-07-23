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
#include "utils/vpss_helper.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(void) {
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, SAMPLE_PIXEL_FORMAT, vpssgrp_width,
                        vpssgrp_height, SAMPLE_PIXEL_FORMAT);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                            "/mnt/data/retina_face.cvimodel");
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                            "/mnt/data/bmface.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  VB_BLK blk;
  VIDEO_FRAME_INFO_S stfdFrame;
  ret = CVI_AI_ReadImage("/mnt/data/rgb_frame.jpg", &blk, &stfdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("Read image failed with %#x!\n", ret);
    return ret;
  }

  VB_BLK blk_fr;
  VIDEO_FRAME_INFO_S frFrame;
  ret = CVI_AI_ReadImage("/mnt/data/rgb_frame.jpg", &blk_fr, &frFrame, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("Read image failed with %#x!\n", ret);
    return ret;
  }

  int face_count = 0;
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_RetinaFace(facelib_handle, &stfdFrame, &face, &face_count);
  printf("face_count %d\n", face.size);
  CVI_AI_FaceAttribute(facelib_handle, &frFrame, &face);

  CVI_AI_Free(&face);
  CVI_VB_ReleaseBlock(blk);
  CVI_VB_ReleaseBlock(blk_fr);
  CVI_AI_DestroyHandle(facelib_handle);
}
