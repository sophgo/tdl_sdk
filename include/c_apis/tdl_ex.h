#ifndef TDL_EX_H
#define TDL_EX_H

#include <stdio.h>
#include "tdl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)

/**
 * @brief 创建一个 TDLHandleEx 对象
 *
 * @param tpu_device_id 指定 TPU 设备的 ID
 * @return  返回创建的 TDLHandleEx 对象, 如果失败返回 NULL
 */
TDLHandleEx TDL_CreateHandleEx(const int32_t tpu_device_id);

/**
 * @brief 调用API服务
 *
 * @param handle TDLHandleEx 对象
 * @param client_type 客户端类型（如"sophnet"）
 * @param method_name 方法名（如"chat"）
 * @param params_json 参数JSON字符串
 * @param result_buf 用于接收结果的缓冲区
 * @param buf_size 结果缓冲区大小
 * @return 0成功，非0失败
 */
int32_t TDL_LLMApiCall(TDLHandleEx handle, const char *client_type,
                       const char *method_name, const char *params_json,
                       char *result_buf, size_t buf_size);

/**
 * @brief 销毁一个 TDLHandleEx 对象
 *
 * @param handle 需要销毁的 TDLHandleEx 对象
 */
int32_t TDL_DestroyHandleEx(TDLHandleEx handle);
#endif

#ifdef __cplusplus
}
#endif
#endif
