#include "core/utils/vpss_helper.h"
#include "cviai.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <retina_model_path> <attribute_model_path> <image>.\n", argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  // Setup model path and model config.
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, false);

  VB_BLK blk;
  VIDEO_FRAME_INFO_S fdFrame;
  CVI_AI_ReadImage(argv[3], &blk, &fdFrame, PIXEL_FORMAT_RGB_888);

  // Run inference and print result.
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  CVI_AI_RetinaFace(ai_handle, &fdFrame, &face);
  printf("Face found %x.\n", face.size);
  CVI_AI_FaceAttribute(ai_handle, &fdFrame, &face);
  CVI_AI_Free(&face);

  // Free image and handles.
  CVI_VB_ReleaseBlock(blk);
  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
