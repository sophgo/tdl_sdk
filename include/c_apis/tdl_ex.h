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

/**
 * @brief 初始化MediaAnalysisServer
 *
 * @param config_path 配置文件路径
 * @return 0成功，非0失败
 */
int32_t TDL_MediaAnalysisServer_Init(const char *config_path);

/**
 * @brief 停止MediaAnalysisServer
 *
 * @return 0成功，非0失败
 */
int32_t TDL_MediaAnalysisServer_Stop();

/**
 * @brief 发送图像数据到MediaAnalysisServer
 *
 * @param image_data 图像数据指针 (JPEG/PNG encoded data)
 * @param size 图像数据大小
 * @param timestamp 时间戳
 * @param channel_id 通道ID
 * @param frame_id 帧ID
 * @param metadata_json 附加的元数据 JSON
 * 字符串，可以包含检测框等信息，无则传NULL
 * @return 0成功，非0失败
 */
int32_t TDL_MediaAnalysisServer_SendImage(const uint8_t *image_data,
                                          size_t size, uint64_t timestamp,
                                          int channel_id, uint64_t frame_id,
                                          const char *metadata_json);

/**
 * @brief 向face_matching任务注入registered_id与face_track_id映射
 *
 * @param registered_id 注册ID
 * @param face_track_id 人脸track_id
 * @return 0成功，非0失败
 */
int32_t TDL_MediaAnalysisServer_AddFaceInfo(int registered_id,
                                            int face_track_id);

/**
 * @brief 提交行为分析视频到cloud_client进行LLM分析
 *
 * @param video_path 视频文件路径 (.h264)
 * @param person_name 人员名称
 * @param person_id 人员ID
 * @param appearance_id appearance ID
 * @param duration_sec 视频片段时长(秒)
 * @return 0成功，非0失败
 */
int32_t TDL_MediaAnalysisServer_SubmitBehaviorVideo(const char *video_path,
                                                    const char *person_name,
                                                    int person_id,
                                                    uint64_t appearance_id,
                                                    uint32_t duration_sec);

#endif

#ifdef __cplusplus
}
#endif
#endif
