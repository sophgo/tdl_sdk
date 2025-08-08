#ifndef _RTSP_UTILS_H_
#define _RTSP_UTILS_H_

#include <cvi_comm_video.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "tdl_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QUEUE_SIZE 8

typedef struct {
  int32_t chn;
  PAYLOAD_TYPE_E pay_load_type;
  int32_t frame_width;
  int32_t frame_height;
} TDLRTSPContext;

typedef enum {
  TDL_IMAGE_GRAY = 0,
  TDL_IMAGE_RGB_PLANAR,
  TDL_IMAGE_RGB_PACKED,
  TDL_IMAGE_BGR_PLANAR,
  TDL_IMAGE_BGR_PACKED,
  TDL_IMAGE_YUV420SP_UV,  // NV12,semi-planar,one Y plane,one interleaved UV
                          // plane,size = width * height * 1.5
  TDL_IMAGE_YUV420SP_VU,  // NV21,semi-planar,one Y plane,one interleaved VU
                          // plane,size = width * height * 1.5
  TDL_IMAGE_YUV420P_UV,   // I420,planar,one Y plane(w*h),one U
                          // plane(w/2*h/2),one V plane(w/2*h/2),size = width *
                          // height * 1.5
  TDL_IMAGE_YUV420P_VU,   // YV12,size = width * height * 1.5
  TDL_IMAGE_YUV422P_UV,   // I422_16,size = width * height * 2
  TDL_IMAGE_YUV422P_VU,   // YV12_16,size = width * height * 2
  TDL_IMAGE_YUV422SP_UV,  // NV16,size = width * height * 2
  TDL_IMAGE_YUV422SP_VU,  // NV61,size = width * height * 2

  TDL_IMAGE_UNKOWN
} TDLImageFormatE;

typedef struct {
  TDLImage queue[QUEUE_SIZE];
  int front;
  int rear;
  int count;
  pthread_mutex_t mutex;
} TDLImageQueue;

#if !defined(__BM168X__) && !defined(__CMODEL_CV181X__)
/**
 * @brief 初始化Camera，板端的/mnt/data路径下需要有sensor_cfg.ini
 * @param handle 已初始化的 TDLHandle 对象，通过 TDL_CreateHandle 创建
 * @param w Camera图像的输出长度
 * @param h Camera图像的输出宽度
 * @param image_fmt Camera图像的输出格式
 * @param vb_buffer_num Camera模块使用的vb buffer数量
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_InitCamera(TDLHandle handle, int w, int h,
                       TDLImageFormatE image_fmt, int vb_buffer_num);

/**
 * @brief 获取camera的一帧图像
 *
 * @param handle 已初始化的 TDLHandle 对象，通过 TDL_CreateHandle 创建
 * @param chn Camera图像的chn通道
 * @return 返回包装的TDLImageHandle对象, 如果失败返回 NULL
 */
TDLImage TDL_GetCameraFrame(TDLHandle handle, int chn);

/**
 * @brief 释放图像资源
 *
 * @param handle 已初始化的 TDLHandle 对象，通过 TDL_CreateHandle 创建
 * @param chn Camera图像的chn通道
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_ReleaseCameraFrame(TDLHandle handle, int chn);

/**
 * @brief 销毁Camera
 *
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_DestoryCamera(TDLHandle handle);
#endif

/**
 * @brief 发送图像到RTSP服务器
 *
 * @param frame 图像数据
 * @param rtsp_context RTSP上下文
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_SendFrameRTSP(VIDEO_FRAME_INFO_S *frame,
                          TDLRTSPContext *rtsp_context);

#ifdef __cplusplus
}
#endif

#endif  // _RTSP_UTILS_H_