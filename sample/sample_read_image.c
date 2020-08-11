#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: sample_read_image <retina_model_path> <image>.\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR);
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

  // Read image using IVE.
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  IVE_IMAGE_S image = CVI_IVE_ReadImage(ive_handle, argv[2], IVE_IMAGE_TYPE_U8C3_PACKAGE);
  if (image.u16Width == 0) {
    printf("Read image failed with %x!\n", ret);
    return ret;
  }
  // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
  VIDEO_FRAME_INFO_S fdFrame;
  ret = CVI_IVE_Image2VideoFrameInfo(&image, &fdFrame, false);
  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }

  // Run inference and print result.
  int face_count = 0;
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  CVI_AI_RetinaFace(ai_handle, &fdFrame, &face, &face_count);
  printf("Face found %x.\n", face_count);
  CVI_AI_FaceAttribute(ai_handle, &fdFrame, &face);
  CVI_AI_Free(&face);

  // Free image and handles.
  CVI_SYS_FreeI(ive_handle, &image);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_IVE_DestroyHandle(ive_handle);
  return ret;
}
