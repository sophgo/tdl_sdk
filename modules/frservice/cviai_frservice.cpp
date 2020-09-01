#include "frservice/cviai_frservice.h"

#include "cviai_core_internal.hpp"
#include "digital_tracking/digital_tracking.hpp"
#include "draw_rect/draw_rect.hpp"

#include <cvimath/cvimath.h>

typedef struct {
  cvai_frservice_feature_array_t feature_array;
  float *feature_unit_length = nullptr;
  float *feature_array_buffer = nullptr;
} cvai_frservice_feature_array_ext_t;

typedef struct {
  cvai_frservice_feature_array_ext_t feature_array_ext;
  cviai_handle_t ai_handle = NULL;
  cviai::service::DigitalTracking *m_dt = nullptr;
} cviai_frservice_context_t;

inline void FreeFeatureArrayExt(cvai_frservice_feature_array_ext_t *feature_array_ext) {
  if (feature_array_ext->feature_unit_length != nullptr) {
    delete feature_array_ext->feature_unit_length;
    feature_array_ext->feature_unit_length = nullptr;
  }
  if (feature_array_ext->feature_array_buffer != nullptr) {
    delete feature_array_ext->feature_array_buffer;
    feature_array_ext->feature_array_buffer = nullptr;
  }
  if (feature_array_ext->feature_array.ptr != NULL) {
    free(feature_array_ext->feature_array.ptr);
    feature_array_ext->feature_array.ptr = NULL;
  }
}

CVI_S32 CVI_AI_FRService_CreateHandle(cviai_frservice_handle_t *handle, cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
    return CVI_FAILURE;
  }
  cviai_frservice_context_t *ctx = new cviai_frservice_context_t;
  memset(&ctx->feature_array_ext.feature_array, 0, sizeof(cvai_frservice_feature_array_t));
  ctx->ai_handle = ai_handle;
  *handle = ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_DestroyHandle(cviai_frservice_handle_t handle) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  FreeFeatureArrayExt(&ctx->feature_array_ext);
  delete ctx->m_dt;
  delete ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_RegisterFeatureArray(cviai_frservice_handle_t handle,
                                              const cvai_frservice_feature_array_t featureArray) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  float *unit_length = new float[featureArray.feature_length * featureArray.data_num];
  switch (featureArray.type) {
    case TYPE_INT8: {
      cvm_gen_precached_i8_unit_length((int8_t *)featureArray.ptr, unit_length,
                                       featureArray.feature_length, featureArray.data_num);
    } break;
    case TYPE_UINT8: {
      cvm_gen_precached_u8_unit_length((uint8_t *)featureArray.ptr, unit_length,
                                       featureArray.feature_length, featureArray.data_num);
    } break;
    default: {
      printf("Unsupported register data type %x.\n", featureArray.type);
      delete[] unit_length;
      return CVI_FAILURE;
    } break;
  }
  FreeFeatureArrayExt(&ctx->feature_array_ext);
  ctx->feature_array_ext.feature_array = featureArray;
  ctx->feature_array_ext.feature_unit_length = unit_length;
  ctx->feature_array_ext.feature_array_buffer =
      new float[featureArray.feature_length * featureArray.data_num];
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_FaceInfoMatching(cviai_frservice_handle_t handle, const cvai_face_t *face,
                                          const uint32_t k, uint32_t **index) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  if (ctx->feature_array_ext.feature_array_buffer == nullptr) {
    printf("Feature array not registered yet.\n");
    return CVI_FAILURE;
  }
  if (face->info->face_feature.ptr == NULL) {
    printf("No feature in face.\n");
    return CVI_FAILURE;
  }
  if (ctx->feature_array_ext.feature_array.type != face->info->face_feature.type) {
    printf("The registered feature array type %x is not the same as the input type %x.\n",
           ctx->feature_array_ext.feature_array.type, face->info->face_feature.type);
    return CVI_FAILURE;
  }
  uint32_t *k_index = (uint32_t *)malloc(sizeof(uint32_t) * k);
  float *k_value = (float *)malloc(sizeof(float) * k);
  switch (ctx->feature_array_ext.feature_array.type) {
    case TYPE_INT8: {
      cvm_cpu_i8data_ip_match((int8_t *)face->info->face_feature.ptr,
                              (int8_t *)ctx->feature_array_ext.feature_array.ptr,
                              ctx->feature_array_ext.feature_unit_length, k_index, k_value,
                              ctx->feature_array_ext.feature_array_buffer,
                              ctx->feature_array_ext.feature_array.feature_length,
                              ctx->feature_array_ext.feature_array.data_num, k);
    } break;
    case TYPE_UINT8: {
      cvm_cpu_u8data_ip_match((uint8_t *)face->info->face_feature.ptr,
                              (uint8_t *)ctx->feature_array_ext.feature_array.ptr,
                              ctx->feature_array_ext.feature_unit_length, k_index, k_value,
                              ctx->feature_array_ext.feature_array_buffer,
                              ctx->feature_array_ext.feature_array.feature_length,
                              ctx->feature_array_ext.feature_array.data_num, k);
    } break;
    default: {
      printf("Unsupported register data type %x.\n", ctx->feature_array_ext.feature_array.type);
      free(k_index);
      free(k_value);
      return CVI_FAILURE;
    } break;
  }
  *index = k_index;
  free(k_value);

  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_RawMatching(cviai_frservice_handle_t handle, const uint8_t *feature,
                                     const feature_type_e type, const uint32_t k,
                                     uint32_t **index) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  if (ctx->feature_array_ext.feature_array_buffer == nullptr) {
    printf("Feature array not registered yet.\n");
    return CVI_FAILURE;
  }
  if (ctx->feature_array_ext.feature_array.type != type) {
    printf("The registered feature array type %x is not the same as the input type %x.\n",
           ctx->feature_array_ext.feature_array.type, type);
    return CVI_FAILURE;
  }
  uint32_t *k_index = (uint32_t *)malloc(sizeof(uint32_t) * k);
  float *k_value = (float *)malloc(sizeof(float) * k);
  switch (ctx->feature_array_ext.feature_array.type) {
    case TYPE_INT8: {
      cvm_cpu_i8data_ip_match((int8_t *)feature, (int8_t *)ctx->feature_array_ext.feature_array.ptr,
                              ctx->feature_array_ext.feature_unit_length, k_index, k_value,
                              ctx->feature_array_ext.feature_array_buffer,
                              ctx->feature_array_ext.feature_array.feature_length,
                              ctx->feature_array_ext.feature_array.data_num, k);
    } break;
    case TYPE_UINT8: {
      cvm_cpu_u8data_ip_match((uint8_t *)feature,
                              (uint8_t *)ctx->feature_array_ext.feature_array.ptr,
                              ctx->feature_array_ext.feature_unit_length, k_index, k_value,
                              ctx->feature_array_ext.feature_array_buffer,
                              ctx->feature_array_ext.feature_array.feature_length,
                              ctx->feature_array_ext.feature_array.data_num, k);
    } break;
    default: {
      printf("Unsupported register data type %x.\n", ctx->feature_array_ext.feature_array.type);
      free(k_index);
      free(k_value);
      return CVI_FAILURE;
    } break;
  }
  *index = k_index;
  free(k_value);

  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_DigitalZoom(cviai_frservice_handle_t handle,
                                     const VIDEO_FRAME_INFO_S *inFrame, const cvai_face_t *meta,
                                     const float face_skip_ratio, const float trans_ratio,
                                     VIDEO_FRAME_INFO_S *outFrame) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  if (ctx->m_dt == nullptr) {
    ctx->m_dt = new cviai::service::DigitalTracking();
  }

  ctx->m_dt->setVpssEngine(CVI_AI_GetVpssEngine(ctx->ai_handle, 0));
  return ctx->m_dt->run(inFrame, meta, outFrame, face_skip_ratio, trans_ratio);
}

CVI_S32 CVI_AI_FRService_DrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame) {
  return cviai::service::DrawMeta(meta, frame);
}
