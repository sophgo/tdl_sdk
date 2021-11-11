#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"
#include "sample_utils.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <retina_model_path> <fr_model_path> <image>.\n", argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  // Create vb pools for reading images and vpss preprocessing.
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);

  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  GOTO_IF_FAILED(CVI_AI_CreateHandle(&ai_handle), ret, create_ai_fail);

  // Setup model path and model config.
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]), ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, argv[2]), ret,
                 setup_ai_fail);

  // Read image using IVE.
  IVE_IMAGE_S image = CVI_IVE_ReadImage(ive_handle, argv[3], IVE_IMAGE_TYPE_U8C3_PACKAGE);
  if (image.u16Width == 0) {
    printf("Read image failed with %x!\n", ret);
    goto setup_ai_fail;
  }

  // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
  VIDEO_FRAME_INFO_S fdFrame;
  ret = CVI_IVE_Image2VideoFrameInfo(&image, &fdFrame, false);
  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    goto read_image_fail;
  }

  // Run inference and print result.
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  GOTO_IF_FAILED(CVI_AI_RetinaFace(ai_handle, &fdFrame, &face), ret, ai_fail);
  printf("Face found: %d.\n", face.size);

  GOTO_IF_FAILED(CVI_AI_FaceRecognition(ai_handle, &fdFrame, &face), ret, ai_fail);

  // Write all face feature to binary file.
  for (uint32_t fid = 0; fid < face.size; fid++) {
    FILE *feature_file;
    char output_path[512];
    snprintf(output_path, sizeof(output_path), "face_feature-%d.bin", fid);
    feature_file = fopen(output_path, "wb");

    // write face feature according to it's type
    fwrite(face.info[fid].feature.ptr, getFeatureTypeSize(face.info[fid].feature.type),
           face.info[fid].feature.size, feature_file);
    printf("write feature to %s\n", output_path);
    fclose(feature_file);
  }

ai_fail:
  CVI_AI_Free(&face);
read_image_fail:
  CVI_SYS_FreeI(ive_handle, &image);
setup_ai_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  CVI_IVE_DestroyHandle(ive_handle);
  CVI_VB_Exit();
  CVI_SYS_Exit();
  return ret;
}
