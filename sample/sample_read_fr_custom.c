#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

#include <string.h>

// This function is a wrapper of how you init a custom AI.
CVI_S32 CVI_AI_CustomInit(cviai_handle_t handle, const char *filepath, uint32_t *id);

// This function is a wrapper of how you run your model.
CVI_S32 CVI_AI_CustomFaceAttribute(cviai_handle_t handle, const uint32_t id,
                                   VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <retina_model_path> <attribute_model_path> <image>.\n", argv[0]);
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
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE, false);

  // Init custom model
  uint32_t custom_id;
  CVI_AI_CustomInit(ai_handle, argv[2], &custom_id);

  // Read image using IVE.
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  IVE_IMAGE_S image = CVI_IVE_ReadImage(ive_handle, argv[3], IVE_IMAGE_TYPE_U8C3_PACKAGE);
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

  // Run retina face to get face first.
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  CVI_AI_RetinaFace(ai_handle, &fdFrame, &face);
  printf("Face found %x.\n", face.size);

  printf("Run Face Attribute from custom AI framework.\n");
  if (CVI_AI_CustomFaceAttribute(ai_handle, custom_id, &fdFrame, &face) == CVI_SUCCESS) {
    // Save a copy of custom attribute result (only feature).
    cvai_face_t custom_face;
    custom_face.size = face.size;
    custom_face.info = (cvai_face_info_t *)malloc(custom_face.size * sizeof(cvai_face_info_t));
    memset(custom_face.info, 0, custom_face.size * sizeof(cvai_face_info_t));
    for (uint32_t i = 0; i < face.size; i++) {
      custom_face.info[i].feature.ptr = (int8_t *)malloc(face.info[i].feature.size);
      custom_face.info[i].feature.size = face.info[i].feature.size;
      custom_face.info[i].feature.type = face.info[i].feature.type;
      memcpy(custom_face.info[i].feature.ptr, face.info[i].feature.ptr, face.info[i].feature.size);
    }

    printf("Run original Face Attribute.\n");
    CVI_AI_FaceAttribute(ai_handle, &fdFrame, &face);

    bool is_same = true;
    for (uint32_t i = 0; i < face.size; i++) {
      for (uint32_t j = 0; j < face.info[i].feature.size; j++) {
        if (custom_face.info[i].feature.ptr[j] != face.info[i].feature.ptr[j]) {
          printf("[Face %u] At feature index %u is not the same %d != %d.\n", i, j,
                 (int32_t)custom_face.info[i].feature.ptr[j], (int32_t)face.info[i].feature.ptr[j]);
          is_same = false;
          break;
        }
      }
    }
    printf("Verify answer: %s\n", is_same ? "SAME" : "NOT SAME");
    CVI_AI_Free(&custom_face);
  }
  CVI_AI_Free(&face);

  // Free image and handles.
  CVI_SYS_FreeI(ive_handle, &image);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_IVE_DestroyHandle(ive_handle);
  return ret;
}

// Some model defines
#define FACE_ATTRIBUTE_FACTOR (1 / 128.f)
#define FACE_ATTRIBUTE_MEAN (0.99609375)
#define FACE_ATTRIBUTE_TENSORNAME "BMFace_dense_MatMul_folded"

// This is a function pointer that will be inserted into the custom AI framework.
static void PreProcessing(VIDEO_FRAME_INFO_S *stInFrame, VIDEO_FRAME_INFO_S *stOutFrame) {
  printf("This is a function passed into custom framework.\n");
  *stOutFrame = *stInFrame;
}

CVI_S32 CVI_AI_CustomInit(cviai_handle_t handle, const char *filepath, uint32_t *id) {
  // Add inference instance.
  CVI_AI_Custom_AddInference(handle, id);
  // Set model path.
  CVI_AI_Custom_SetModelPath(handle, *id, filepath);
  // (Optional) Get set model path.
  char *savedFilePath = NULL;
  CVI_AI_Custom_GetModelPath(handle, *id, &savedFilePath);
  printf("Init id %u, model %s.\n", *id, savedFilePath);
  free(savedFilePath);
  // Must set if you want to use VPSS in custom AI framework.
  const float factor = FACE_ATTRIBUTE_FACTOR;
  const float mean = FACE_ATTRIBUTE_MEAN;
  CVI_AI_Custom_SetVpssPreprocessParam(handle, *id, &factor, &mean, 1, false);
  // Optional in this case. You can pass function pointer into the framework if you like.
  CVI_AI_Custom_SetPreprocessFuncPtr(handle, *id, PreProcessing, false, true);
  // Don't do dequantization if choose to skip.
  CVI_AI_Custom_SetSkipPostProcess(handle, *id, true);
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_CustomFaceAttribute(cviai_handle_t handle, const uint32_t id,
                                   VIDEO_FRAME_INFO_S *frame, cvai_face_t *faces) {
  VB_BLK blk;
  VIDEO_FRAME_INFO_S outFrame;
  uint32_t n = 0, c = 0, h = 0, w = 0;
  // Get the input tensor size.
  if (CVI_AI_Custom_GetInputTensorNCHW(handle, id, NULL, &n, &c, &h, &w) != CVI_SUCCESS) {
    printf("Cannot get NCHW.\n");
    return CVI_FAILURE;
  }
  // Create VIDEO_FRAME_INFO_S and preallocate for CVI_AI_FaceAlignment.
  if (CREATE_VBFRAME_HELPER(&blk, &outFrame, w, h, PIXEL_FORMAT_RGB_888) != CVI_SUCCESS) {
    return CVI_FAILURE;
  }
  for (uint32_t i = 0; i < faces->size; ++i) {
    // This is the preprocessing of the model.
    CVI_AI_FaceAlignment(frame, faces->width, faces->height, &faces->info[i], &outFrame, false);
    // Inference.
    CVI_AI_Custom_RunInference(handle, id, &outFrame);
    // Post-processing.
    int8_t *tensor = NULL;
    uint32_t tensorCount = 0;
    uint16_t unitSize = 0;
    if (CVI_AI_Custom_GetOutputTensor(handle, id, FACE_ATTRIBUTE_TENSORNAME, &tensor, &tensorCount,
                                      &unitSize) != CVI_SUCCESS) {
      printf("Failed to get tensor %s.\n", FACE_ATTRIBUTE_TENSORNAME);
      return CVI_FAILURE;
    }
    cvai_feature_t *feature = &faces->info[i].feature;
    if (feature->size != tensorCount) {
      free(feature->ptr);
      feature->ptr = (int8_t *)malloc(unitSize * tensorCount);
      feature->size = tensorCount;
      feature->type = TYPE_INT8;
    }
    memcpy(feature->ptr, tensor, feature->size);
  }
  // Release created VIDEO_FRAME_INFO_S.
  CVI_VB_ReleaseBlock(blk);
  return CVI_SUCCESS;
}
