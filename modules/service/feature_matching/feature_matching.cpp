#include "feature_matching.hpp"

#include <cvimath/cvimath_internal.h>
#include <cviruntime.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <cmath>

namespace cviai {
namespace service {

inline void __attribute__((always_inline))
FreeFeatureArrayExt(cvai_service_feature_array_ext_t *feature_array_ext) {
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

inline void __attribute__((always_inline))
FreeFeatureArrayTpuExt(CVI_RT_HANDLE rt_handle,
                       cvai_service_feature_array_tpu_ext_t *feature_array_ext) {
  if (feature_array_ext->feature_input.rtmem != NULL) {
    CVI_RT_MemFree(rt_handle, feature_array_ext->feature_input.rtmem);
    feature_array_ext->feature_input.rtmem = NULL;
  }
  if (feature_array_ext->feature_array.rtmem != NULL) {
    CVI_RT_MemFree(rt_handle, feature_array_ext->feature_array.rtmem);
    feature_array_ext->feature_array.rtmem = NULL;
  }
  if (feature_array_ext->buffer_array.rtmem != NULL) {
    CVI_RT_MemFree(rt_handle, feature_array_ext->buffer_array.rtmem);
    feature_array_ext->buffer_array.rtmem = NULL;
  }
  if (feature_array_ext->slice_num != nullptr) {
    delete feature_array_ext->slice_num;
    feature_array_ext->slice_num = nullptr;
  }
  if (feature_array_ext->feature_unit_length != nullptr) {
    delete feature_array_ext->feature_unit_length;
    feature_array_ext->feature_unit_length = nullptr;
  }
  if (feature_array_ext->array_buffer_32 != nullptr) {
    delete feature_array_ext->array_buffer_32;
    feature_array_ext->array_buffer_32 = nullptr;
  }
  if (feature_array_ext->array_buffer_f != nullptr) {
    delete feature_array_ext->array_buffer_f;
    feature_array_ext->array_buffer_f = nullptr;
  }
}

FeatureMatching::~FeatureMatching() {
  FreeFeatureArrayExt(&m_cpu_ipfeature);
  FreeFeatureArrayTpuExt(m_rt_handle, &m_tpu_ipfeature);
  destroyHandle(m_rt_handle, m_cvk_ctx);
}

int FeatureMatching::init() { return createHandle(&m_rt_handle, &m_cvk_ctx); }

int FeatureMatching::createHandle(CVI_RT_HANDLE *rt_handle, cvk_context_t **cvk_ctx) {
  if (CVI_RT_Init(rt_handle) != CVI_SUCCESS) {
    LOGE("Runtime init failed.\n");
    return CVI_FAILURE;
  }
  struct sysinfo info;
  if (sysinfo(&info) < 0) {
    return CVI_FAILURE;
  }
  // FIXME: Rewrite command buffer to fit feature matching size.
  uint64_t mem = 50000;
  if (info.freeram <= mem) {
    LOGE("Memory insufficient.\n");
    return CVI_FAILURE;
  }
  *cvk_ctx = (cvk_context_t *)CVI_RT_RegisterKernel(*rt_handle, mem);
  return CVI_SUCCESS;
}

int FeatureMatching::destroyHandle(CVI_RT_HANDLE rt_handle, cvk_context_t *cvk_ctx) {
  CVI_RT_UnRegisterKernel(cvk_ctx);
  CVI_RT_DeInit(rt_handle);
  return CVI_SUCCESS;
}

int FeatureMatching::registerData(const cvai_service_feature_array_t &feature_array,
                                  const cvai_service_feature_matching_e &matching_method) {
  int ret = CVI_SUCCESS;
  m_matching_method = matching_method;
  switch (m_matching_method) {
    case INNER_PRODUCT: {
      ret = innerProductRegister(feature_array);
    } break;
    default:
      LOGE("Unsupported mathinc method %u\n", m_matching_method);
      ret = CVI_FAILURE;
      break;
  }
  return ret;
}

int FeatureMatching::run(const uint8_t *feature, const feature_type_e &type, const uint32_t k,
                         uint32_t **index) {
  int ret = CVI_SUCCESS;
  uint32_t *k_index = (uint32_t *)malloc(sizeof(uint32_t) * k);
  switch (m_matching_method) {
    case INNER_PRODUCT: {
      if ((ret = innerProductRun(feature, type, k, k_index)) != CVI_SUCCESS) {
        free(k_index);
      }
      *index = k_index;
    } break;
    default:
      LOGE("Unsupported mathinc method %u\n", m_matching_method);
      ret = CVI_FAILURE;
      break;
  }
  return ret;
}

int FeatureMatching::innerProductRegister(const cvai_service_feature_array_t &feature_array) {
  const uint32_t total_length = feature_array.feature_length * feature_array.data_num;
  float *unit_length = new float[total_length];
  switch (feature_array.type) {
    case TYPE_INT8: {
      cvm_gen_precached_i8_unit_length((int8_t *)feature_array.ptr, unit_length,
                                       feature_array.feature_length, feature_array.data_num);
    } break;
    default: {
      LOGE("Unsupported register data type %x.\n", feature_array.type);
      delete[] unit_length;
      return CVI_FAILURE;
    } break;
  }
  FreeFeatureArrayExt(&m_cpu_ipfeature);
  FreeFeatureArrayTpuExt(m_rt_handle, &m_tpu_ipfeature);
  if (feature_array.data_num < 1000) {
    m_is_cpu = true;
    m_cpu_ipfeature.feature_array = feature_array;
    m_cpu_ipfeature.feature_unit_length = unit_length;
    m_cpu_ipfeature.feature_array_buffer = new float[total_length];
  } else {
    m_is_cpu = false;
    m_tpu_ipfeature.feature_length = feature_array.feature_length;
    m_tpu_ipfeature.data_num = feature_array.data_num;
    m_tpu_ipfeature.feature_unit_length = unit_length;
    m_tpu_ipfeature.array_buffer_32 = new uint32_t[feature_array.data_num];
    m_tpu_ipfeature.array_buffer_f = new float[feature_array.data_num];
    // Clear buffer first
    // Gen cmd buffer here
    // Create buffer for input
    rtinfo &input = m_tpu_ipfeature.feature_input;
    input.rtmem = CVI_RT_MemAlloc(m_rt_handle, feature_array.feature_length);
    input.paddr = CVI_RT_MemGetPAddr(input.rtmem);
    input.vaddr = CVI_RT_MemGetVAddr(input.rtmem);
    // Create buffer for array
    rtinfo &info = m_tpu_ipfeature.feature_array;
    info.rtmem = CVI_RT_MemAlloc(m_rt_handle, total_length);
    info.paddr = CVI_RT_MemGetPAddr(info.rtmem);
    info.vaddr = CVI_RT_MemGetVAddr(info.rtmem);
    // Create buffer for array
    rtinfo &buffer = m_tpu_ipfeature.buffer_array;
    buffer.rtmem = CVI_RT_MemAlloc(m_rt_handle, feature_array.data_num * sizeof(uint32_t));
    buffer.paddr = CVI_RT_MemGetPAddr(buffer.rtmem);
    buffer.vaddr = CVI_RT_MemGetVAddr(buffer.rtmem);
    // Copy feature array to ion
    memcpy(info.vaddr, feature_array.ptr, total_length);
    for (uint32_t n = 0; n < total_length; n++) {
      int i = n / feature_array.data_num;
      int j = n % feature_array.data_num;
      ((int8_t *)info.vaddr)[n] = feature_array.ptr[feature_array.feature_length * j + i];
    }
    CVI_RT_MemFlush(m_rt_handle, info.rtmem);
  }
  return CVI_SUCCESS;
}

int FeatureMatching::innerProductRun(const uint8_t *feature, const feature_type_e &type,
                                     const uint32_t k, uint32_t *k_index) {
  int ret = CVI_SUCCESS;
  float *k_value = (float *)malloc(sizeof(float) * k);
  switch (type) {
    case TYPE_INT8: {
      if (m_is_cpu) {
        cvm_cpu_i8data_ip_match((int8_t *)feature, (int8_t *)m_cpu_ipfeature.feature_array.ptr,
                                m_cpu_ipfeature.feature_unit_length, k_index, k_value,
                                m_cpu_ipfeature.feature_array_buffer,
                                m_cpu_ipfeature.feature_array.feature_length,
                                m_cpu_ipfeature.feature_array.data_num, k);
      } else {
        int8_t *i8_feature = (int8_t *)feature;
        memcpy(m_tpu_ipfeature.feature_input.vaddr, i8_feature, m_tpu_ipfeature.feature_length);
        CVI_RT_MemFlush(m_rt_handle, m_tpu_ipfeature.feature_input.rtmem);
        // Submit command buffer without erasing it.
        size_t *slice_num =
            cvm_gemm(m_cvk_ctx, m_tpu_ipfeature.feature_input.paddr,
                     m_tpu_ipfeature.feature_array.paddr, m_tpu_ipfeature.buffer_array.paddr, 1,
                     m_tpu_ipfeature.feature_length, m_tpu_ipfeature.data_num, CVK_FMT_I8);
        CVI_RT_Submit(m_cvk_ctx);
        CVI_RT_MemInvld(m_rt_handle, m_tpu_ipfeature.buffer_array.rtmem);
        cvm_combin_gemm_i8(slice_num, m_tpu_ipfeature.buffer_array.vaddr,
                           m_tpu_ipfeature.array_buffer_32, 1, m_tpu_ipfeature.data_num);
        free(slice_num);
        // Get a length
        int32_t dot_result = 0;
        for (uint32_t i = 0; i < m_tpu_ipfeature.feature_length; i++) {
          dot_result += ((short)i8_feature[i] * i8_feature[i]);
        }
        float unit_i8 = sqrt(dot_result);
        // Get a length end

        for (uint32_t i = 0; i < m_tpu_ipfeature.data_num; i++) {
          m_tpu_ipfeature.array_buffer_f[i] = ((int32_t *)m_tpu_ipfeature.array_buffer_32)[i] /
                                              (unit_i8 * m_tpu_ipfeature.feature_unit_length[i]);
        }
        // Get k result
        for (uint32_t i = 0; i < k; i++) {
          uint32_t largest = 0;
          for (uint32_t j = 0; j < m_tpu_ipfeature.data_num; j++) {
            if (m_tpu_ipfeature.array_buffer_f[j] > m_tpu_ipfeature.array_buffer_f[largest]) {
              largest = j;
            }
          }
          k_value[i] = m_tpu_ipfeature.array_buffer_f[largest];
          k_index[i] = largest;
          m_tpu_ipfeature.array_buffer_f[largest] = 0;
        }
      }
    } break;
    default: {
      LOGE("Unsupported register data type %x.\n", type);
      ret = CVI_FAILURE;
    } break;
  }
  free(k_value);
  return ret;
}
}  // namespace service
}  // namespace cviai