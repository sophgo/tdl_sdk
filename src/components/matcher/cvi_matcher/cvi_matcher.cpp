#include "cvi_matcher.hpp"
#include <string.h>
#include "utils/cvimath_internal.h"
#ifndef CONFIG_ALIOS
#include <sys/sysinfo.h>
#else
#include <yoc/sysinfo.h>
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

inline void __attribute__((always_inline)) FreeFeatureArrayExt(
    CPUFeatureArrayInfo *feature_array_ext) {
  if (feature_array_ext->feature_unit_length != nullptr) {
    delete feature_array_ext->feature_unit_length;
    feature_array_ext->feature_unit_length = nullptr;
  }
  if (feature_array_ext->feature_array_buffer != nullptr) {
    delete feature_array_ext->feature_array_buffer;
    feature_array_ext->feature_array_buffer = nullptr;
  }
}

inline void __attribute__((always_inline)) FreeFeatureArrayTpuExt(
    CviRtHandle rt_handle, TPUFeatureArrayInfo *feature_array_ext) {
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

// 构造函数
CviMatcher::CviMatcher(int device_id) { init(device_id); }

// 析构函数
CviMatcher::~CviMatcher() {
  FreeFeatureArrayExt(&cpu_feature_info_);
  FreeFeatureArrayTpuExt(rt_handle_, &tpu_feature_info_);
  destroyHandle(rt_handle_, kernel_context_);
}

// BaseMatcher的init方法实现
void CviMatcher::init(int device_id) {
  tpu_feature_info_.array_buffer_32 = NULL;
  tpu_feature_info_.array_buffer_f = NULL;
  tpu_feature_info_.data_num = 0;
  tpu_feature_info_.feature_length = 0;
  tpu_feature_info_.feature_unit_length = 0;
  tpu_feature_info_.slice_num = NULL;

  cpu_feature_info_.feature_array.data_num = 0;
  cpu_feature_info_.feature_array.feature_length = 0;
  cpu_feature_info_.feature_array.ptr = NULL;
  cpu_feature_info_.feature_array_buffer = NULL;
  cpu_feature_info_.feature_unit_length = 0;
  use_cpu_ = true;
  int ret = createHandle(&rt_handle_, &kernel_context_);
  if (ret != 0) {
    std::cout << "createHandle failed." << std::endl;
    return;
  }
}

// 创建handle
int CviMatcher::createHandle(CviRtHandle *rt_handle, KernelContext **cvk_ctx) {
  if (CVI_RT_Init(rt_handle) != 0) {
    std::cout << "Runtime init failed." << std::endl;
    return -1;
  }
  struct sysinfo info;
  if (sysinfo(&info) < 0) {
    std::cout << "sysinfo failed." << std::endl;
    return -1;
  }
  uint64_t mem = 50000;
  if (info.freeram <= mem) {
    std::cout << "Memory insufficient." << std::endl;
    return -1;
  }
  *cvk_ctx = (KernelContext *)CVI_RT_RegisterKernel(*rt_handle, mem);
  return 0;
}

// 销毁handle
int CviMatcher::destroyHandle(CviRtHandle rt_handle, KernelContext *cvk_ctx) {
  CVI_RT_UnRegisterKernel(cvk_ctx);
  CVI_RT_DeInit(rt_handle);
  return 0;
}

// BaseMatcher的loadGallery方法实现
int32_t CviMatcher::loadGallery(
    const std::vector<std::shared_ptr<ModelFeatureInfo>> &gallery_features) {
  gallery_features_ = &gallery_features;
  gallery_features_num_ = gallery_features.size();

  if (gallery_features_num_ <= 0) {
    return -1;
  }

  feature_dim_ = (*gallery_features_)[0]->embedding_num;

  // 保存特征数据类型
  feature_data_type_ = (*gallery_features_)[0]->embedding_type;

  // 创建feature_array
  FeatureArray feature_array;
  feature_array.ptr =
      new int8_t[gallery_features_num_ * feature_dim_ *
                 (feature_data_type_ == TDLDataType::FP32 ? sizeof(float) : 1)];
  feature_array.feature_length = feature_dim_;
  feature_array.data_num = gallery_features_num_;

  for (int i = 0; i < gallery_features_num_; i++) {
    size_t feature_size =
        feature_dim_ *
        (feature_data_type_ == TDLDataType::FP32 ? sizeof(float) : 1);
    memcpy((int8_t *)feature_array.ptr + i * feature_size,
           (*gallery_features_)[i]->embedding, feature_size);
  }

  int ret = cosSimilarityRegister(feature_array);
  if (ret != 0) {
    std::cout << "Cosine similarity register failed." << std::endl;
    delete[] (int8_t *)feature_array.ptr;
    return -1;
  }

  is_loaded_ = true;
  delete[] (int8_t *)feature_array.ptr;

  return 0;
}

int32_t CviMatcher::queryWithTopK(
    const std::vector<std::shared_ptr<ModelFeatureInfo>> &query_features,
    int32_t topk, MatchResult &results) {
  if (!is_loaded_) {
    std::cout << "Gallery not loaded yet, please call loadGallery first."
              << std::endl;
    return -1;
  }

  query_features_ = &query_features;
  query_features_num_ = query_features.size();
  if ((*query_features_)[0]->embedding_num != feature_dim_) {
    std::cout << "Query feature dimension mismatch." << std::endl;
    return -1;
  }

  for (int i = 0; i < query_features_num_; i++) {
    uint32_t size = 0;

    // 分配临时内存
    uint32_t *indices = new uint32_t[topk];
    float *scores = new float[topk];

    int ret = cosSimilarityRun((*query_features_)[i]->embedding, topk, indices,
                               scores, &size);
    if (ret != 0) {
      std::cout << "Cosine similarity run failed." << std::endl;
      delete[] indices;
      delete[] scores;
      return -1;
    }

    // 填充结果
    results.indices.push_back(std::vector<int>(indices, indices + size));
    results.scores.push_back(std::vector<float>(scores, scores + size));

    // 释放临时内存
    delete[] indices;
    delete[] scores;
  }

  return 0;
}

int32_t CviMatcher::updateGalleryCol(void *p_data, int col) {
  if (!is_loaded_ || col < 0 || col >= gallery_features_num_) {
    return -1;
  }

  // 根据数据类型确定要复制的字节数
  size_t feature_size =
      feature_dim_ *
      (feature_data_type_ == TDLDataType::FP32 ? sizeof(float) : 1);

  memcpy((*gallery_features_)[col]->embedding, p_data, feature_size);

  // 重新加载特征库
  return loadGallery(*gallery_features_);
}

// 余弦相似度特征注册
int CviMatcher::cosSimilarityRegister(const FeatureArray &feature_array) {
  const uint32_t total_length =
      feature_array.feature_length * feature_array.data_num;
  float *unit_length = new float[total_length];

  // 根据特征数据类型选择适当的单位长度计算函数
  if (feature_data_type_ == TDLDataType::UINT8) {
    cvmGenPrecachedU8UnitLength((uint8_t *)feature_array.ptr, unit_length,
                                feature_array.feature_length,
                                feature_array.data_num);
  } else if (feature_data_type_ == TDLDataType::INT8) {
    cvmGenPrecachedI8UnitLength((int8_t *)feature_array.ptr, unit_length,
                                feature_array.feature_length,
                                feature_array.data_num);
  } else if (feature_data_type_ == TDLDataType::FP32) {
    cvmGenPrecachedFP32UnitLength((float *)feature_array.ptr, unit_length,
                                  feature_array.feature_length,
                                  feature_array.data_num);
  }

  FreeFeatureArrayExt(&cpu_feature_info_);
  FreeFeatureArrayTpuExt(rt_handle_, &tpu_feature_info_);
  if (feature_array.data_num < 1000) {
    use_cpu_ = true;
    cpu_feature_info_.feature_array = feature_array;
    cpu_feature_info_.feature_unit_length = unit_length;
    cpu_feature_info_.feature_array_buffer = new float[total_length];
  } else {
    use_cpu_ = false;
    tpu_feature_info_.feature_length = feature_array.feature_length;
    tpu_feature_info_.data_num = feature_array.data_num;
    tpu_feature_info_.feature_unit_length = unit_length;
    tpu_feature_info_.array_buffer_32 = new uint32_t[feature_array.data_num];
    tpu_feature_info_.array_buffer_f = new float[feature_array.data_num];
    // 生成cmd缓冲区
    Rinfo &input = tpu_feature_info_.feature_input;
    input.rtmem = CVI_RT_MemAlloc(
        rt_handle_,
        feature_array.feature_length *
            (feature_data_type_ == TDLDataType::FP32 ? sizeof(float) : 1));
    input.paddr = CVI_RT_MemGetPAddr(input.rtmem);
    input.vaddr = CVI_RT_MemGetVAddr(input.rtmem);
    // 为数组创建缓冲区
    Rinfo &info = tpu_feature_info_.feature_array;
    info.rtmem = CVI_RT_MemAlloc(
        rt_handle_,
        total_length *
            (feature_data_type_ == TDLDataType::FP32 ? sizeof(float) : 1));
    info.paddr = CVI_RT_MemGetPAddr(info.rtmem);
    info.vaddr = CVI_RT_MemGetVAddr(info.rtmem);
    // 为数组创建缓冲区
    Rinfo &buffer = tpu_feature_info_.buffer_array;
    buffer.rtmem =
        CVI_RT_MemAlloc(rt_handle_, feature_array.data_num * sizeof(uint32_t));
    buffer.paddr = CVI_RT_MemGetPAddr(buffer.rtmem);
    buffer.vaddr = CVI_RT_MemGetVAddr(buffer.rtmem);

    // 将特征数组复制到ion
    if (feature_data_type_ == TDLDataType::FP32) {
      // 对于FP32类型，直接复制内存
      memcpy(info.vaddr, feature_array.ptr, total_length * sizeof(float));
    } else {
      // 对于INT8和UINT8类型，使用原来的转置逻辑
      memcpy(info.vaddr, feature_array.ptr, total_length);
      for (uint32_t n = 0; n < total_length; n++) {
        int i = n / feature_array.data_num;
        int j = n % feature_array.data_num;
        ((int8_t *)info.vaddr)[n] =
            ((int8_t *)feature_array.ptr)[feature_array.feature_length * j + i];
      }
    }
    CVI_RT_MemFlush(rt_handle_, info.rtmem);
  }

  return 0;
}

template <typename T>
static std::vector<size_t> sort_indexes(const std::vector<T> &v) {
  // 初始化原始索引位置
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // 根据v中的值比较对索引进行排序
  // 使用std::stable_sort而不是std::sort
  // 以避免在v包含相等值的元素时进行不必要的索引重新排序
  std::stable_sort(idx.begin(), idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

template <typename T>
static void k_selection_sort(T *array, uint32_t *index, T *value,
                             const uint32_t array_size, const uint32_t k) {
  // 创建数组的副本，防止修改原始数据
  T *array_copy = new T[array_size];
  memcpy(array_copy, array, array_size * sizeof(T));

  for (uint32_t i = 0; i < k; i++) {
    int largest = 0;
    for (uint32_t j = 0; j < array_size; j++) {
      if (array_copy[j] > array_copy[largest]) {
        largest = j;
      }
    }
    value[i] = array_copy[largest];
    index[i] = largest;
    array_copy[largest] = 0;
  }

  delete[] array_copy;
}

// 余弦相似度查询
int CviMatcher::cosSimilarityRun(const void *feature, const uint32_t topk,
                                 uint32_t *k_index, float *k_value,
                                 uint32_t *size) {
  if (topk == 0) {
    std::cout << "topk是无效值" << std::endl;
    *size = 0;
    return -1;
  }

  if (cpu_feature_info_.feature_array.data_num == 0 &&
      tpu_feature_info_.data_num == 0) {
    std::cout << "尚未注册特征，请调用loadGallery注册特征。" << std::endl;
    return -1;
  }

  *size = std::min<uint32_t>(gallery_features_num_, topk);

  float *scores = new float[*size];
  uint32_t *indices = new uint32_t[*size];

  if (use_cpu_) {
    // 根据数据类型使用不同的计算函数
    if (feature_data_type_ == TDLDataType::UINT8) {
      cvmCpuU8dataIpMatch((uint8_t *)feature,
                          (uint8_t *)cpu_feature_info_.feature_array.ptr,
                          cpu_feature_info_.feature_unit_length, indices,
                          scores, cpu_feature_info_.feature_array_buffer,
                          cpu_feature_info_.feature_array.feature_length,
                          cpu_feature_info_.feature_array.data_num, *size);
    } else if (feature_data_type_ == TDLDataType::INT8) {
      cvmCpuI8dataIpMatch((int8_t *)feature,
                          (int8_t *)cpu_feature_info_.feature_array.ptr,
                          cpu_feature_info_.feature_unit_length, indices,
                          scores, cpu_feature_info_.feature_array_buffer,
                          cpu_feature_info_.feature_array.feature_length,
                          cpu_feature_info_.feature_array.data_num, *size);
    } else if (feature_data_type_ == TDLDataType::FP32) {
      cvmCpuFP32dataIpMatch((float *)feature,
                            (float *)cpu_feature_info_.feature_array.ptr,
                            cpu_feature_info_.feature_unit_length, indices,
                            scores, cpu_feature_info_.feature_array_buffer,
                            cpu_feature_info_.feature_array.feature_length,
                            cpu_feature_info_.feature_array.data_num, *size);
    }
  } else {
    if (feature_data_type_ == TDLDataType::UINT8) {
      uint8_t *u8_feature = (uint8_t *)feature;
      memcpy(tpu_feature_info_.feature_input.vaddr, u8_feature,
             tpu_feature_info_.feature_length);
      CVI_RT_MemFlush(rt_handle_, tpu_feature_info_.feature_input.rtmem);

      // 提交命令缓冲区而不擦除它
      size_t *slice_num =
          cvmGemm(kernel_context_, tpu_feature_info_.feature_input.paddr,
                  tpu_feature_info_.feature_array.paddr,
                  tpu_feature_info_.buffer_array.paddr, 1,
                  tpu_feature_info_.feature_length, tpu_feature_info_.data_num,
                  CVK_FMT_U8);

      CVI_RT_Submit(kernel_context_);
      CVI_RT_MemInvld(rt_handle_, tpu_feature_info_.buffer_array.rtmem);
      cvmCombinGemmI8(slice_num, tpu_feature_info_.buffer_array.vaddr,
                      tpu_feature_info_.array_buffer_32, 1,
                      tpu_feature_info_.data_num);
      free(slice_num);

      // 获取长度
      int32_t dot_result = 0;
      for (uint32_t i = 0; i < tpu_feature_info_.feature_length; i++) {
        dot_result += ((short)u8_feature[i] * u8_feature[i]);
      }
      float unit_u8 = sqrt(dot_result);

      // 计算最终相似度
      for (uint32_t i = 0; i < tpu_feature_info_.data_num; i++) {
        tpu_feature_info_.array_buffer_f[i] =
            ((int32_t *)tpu_feature_info_.array_buffer_32)[i] /
            (unit_u8 * tpu_feature_info_.feature_unit_length[i]);
      }
    } else if (feature_data_type_ == TDLDataType::INT8) {
      // INT8类型处理
      int8_t *i8_feature = (int8_t *)feature;
      memcpy(tpu_feature_info_.feature_input.vaddr, i8_feature,
             tpu_feature_info_.feature_length);
      CVI_RT_MemFlush(rt_handle_, tpu_feature_info_.feature_input.rtmem);

      // 提交命令缓冲区而不擦除它
      size_t *slice_num =
          cvmGemm(kernel_context_, tpu_feature_info_.feature_input.paddr,
                  tpu_feature_info_.feature_array.paddr,
                  tpu_feature_info_.buffer_array.paddr, 1,
                  tpu_feature_info_.feature_length, tpu_feature_info_.data_num,
                  CVK_FMT_I8);

      CVI_RT_Submit(kernel_context_);
      CVI_RT_MemInvld(rt_handle_, tpu_feature_info_.buffer_array.rtmem);
      cvmCombinGemmI8(slice_num, tpu_feature_info_.buffer_array.vaddr,
                      tpu_feature_info_.array_buffer_32, 1,
                      tpu_feature_info_.data_num);
      free(slice_num);

      // 获取长度
      int32_t dot_result = 0;
      for (uint32_t i = 0; i < tpu_feature_info_.feature_length; i++) {
        dot_result += ((short)i8_feature[i] * i8_feature[i]);
      }
      float unit_i8 = sqrt(dot_result);

      // 计算最终相似度
      for (uint32_t i = 0; i < tpu_feature_info_.data_num; i++) {
        tpu_feature_info_.array_buffer_f[i] =
            ((int32_t *)tpu_feature_info_.array_buffer_32)[i] /
            (unit_i8 * tpu_feature_info_.feature_unit_length[i]);
      }
    } else if (feature_data_type_ == TDLDataType::FP32) {
      float *f32_feature = (float *)feature;

      // 计算查询特征的模长
      float query_norm = 0.0f;
      for (uint32_t i = 0; i < tpu_feature_info_.feature_length; i++) {
        query_norm += f32_feature[i] * f32_feature[i];
      }
      query_norm = sqrt(query_norm);

      // 从设备内存获取库特征
      float *gallery_features = new float[tpu_feature_info_.data_num *
                                          tpu_feature_info_.feature_length];
      memcpy(gallery_features, tpu_feature_info_.feature_array.vaddr,
             tpu_feature_info_.data_num * tpu_feature_info_.feature_length *
                 sizeof(float));

      // 计算余弦相似度
      for (uint32_t i = 0; i < tpu_feature_info_.data_num; i++) {
        float dot_product = 0.0f;
        for (uint32_t j = 0; j < tpu_feature_info_.feature_length; j++) {
          dot_product +=
              f32_feature[j] *
              gallery_features[i * tpu_feature_info_.feature_length + j];
        }
        tpu_feature_info_.array_buffer_f[i] =
            dot_product /
            (query_norm * tpu_feature_info_.feature_unit_length[i]);
      }

      delete[] gallery_features;
    }

    // 获取k个结果 (对所有类型的计算结果进行排序)
    if (*size == tpu_feature_info_.data_num) {
      std::vector<float> scores_v = std::vector<float>(
          tpu_feature_info_.array_buffer_f,
          tpu_feature_info_.array_buffer_f + tpu_feature_info_.data_num);
      std::vector<size_t> sorted_indices = sort_indexes(scores_v);

      for (uint32_t i = 0; i < *size; i++) {
        indices[i] = sorted_indices[i];
        scores[i] = tpu_feature_info_.array_buffer_f[indices[i]];
      }
    } else {
      for (uint32_t i = 0; i < *size; i++) {
        uint32_t largest = 0;
        for (uint32_t j = 0; j < tpu_feature_info_.data_num; j++) {
          if (tpu_feature_info_.array_buffer_f[j] >
              tpu_feature_info_.array_buffer_f[largest]) {
            largest = j;
          }
        }
        scores[i] = tpu_feature_info_.array_buffer_f[largest];
        indices[i] = largest;
        tpu_feature_info_.array_buffer_f[largest] = 0;
      }
    }
  }

  uint32_t j = 0;
  for (uint32_t i = 0; i < *size; i++) {
    k_value[j] = scores[i];
    k_index[j] = indices[i];
    j++;
  }
  *size = j;

  delete[] scores;
  delete[] indices;

  return 0;
}
