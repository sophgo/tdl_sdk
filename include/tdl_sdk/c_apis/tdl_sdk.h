#ifndef TDL_SDK_H
#define TDL_SDK_H

#include "tdl_model_def.h"
#include "tdl_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建一个 TDLContextHandle 对象
 *
 * @param tpu_device_id 指定 TPU 设备的 ID
 * @return  返回创建的 TDLContextHandle 对象, 如果失败返回 NULL
 */
cvtdl_handle_t CVI_TDL_CreateHandle(const int32_t tpu_device_id);

/**
 * @brief 销毁一个 TDLContextHandle 对象
 *
 * @param context_handle 需要销毁的 TDLContextHandle 对象
 */
int32_t CVI_TDL_DestroyHandle(cvtdl_handle_t handle);

/**
 * @brief 包装一个 VPSS 帧为 TDLImageHandle 对象
 *
 * @param vpss_frame 需要包装的 VPSS 帧
 * @param own_memory 是否拥有内存所有权
 * @return  返回包装的 TDLImageHandle 对象, 如果失败返回 NULL
 */
cvtdl_image_t CVI_TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory);

/**
 * @brief 读取一张图片为 TDLImageHandle 对象
 *
 * @param path 图片路径
 * @return  返回读取的 TDLImageHandle 对象, 如果失败返回 NULL
 */
cvtdl_image_t CVI_TDL_ReadImage(const char *path);

/**
 * @brief 销毁一个 TDLImageHandle 对象
 *
 * @param image_handle 需要销毁的 TDLImageHandle 对象
 */
int32_t CVI_TDL_DestroyImage(cvtdl_image_t image_handle);

int32_t CVI_TDL_OpenModel(cvtdl_handle_t handle, const cvtdl_model_e model_id,
                          const char *model_path);

int32_t CVI_TDL_CloseModel(cvtdl_handle_t handle, const cvtdl_model_e model_id);

int32_t CVI_TDL_Detection(cvtdl_handle_t handle, const cvtdl_model_e model_id,
                          cvtdl_image_t image_handle,
                          cvtdl_object_t *object_meta);

int32_t CVI_TDL_FaceDetection(cvtdl_handle_t handle,
                              const cvtdl_model_e model_id,
                              cvtdl_image_t image_handle,
                              cvtdl_face_t *face_meta);
int32_t CVI_TDL_FaceLandmark(cvtdl_handle_t handle,
                               const cvtdl_model_e model_id,
                               cvtdl_image_t image_handle,
                               cvtdl_face_t *face_meta);

int32_t CVI_TDL_Classfification(cvtdl_handle_t handle,
                                const cvtdl_model_e model_id,
                                cvtdl_image_t image_handle,
                                cvtdl_class_info_t *class_info);

int32_t CVI_TDL_ObjectClassification(cvtdl_handle_t handle,
                                     const cvtdl_model_e model_id,
                                     cvtdl_image_t image_handle,
                                     cvtdl_object_t *object_meta,
                                     cvtdl_class_t *class_info);

int32_t CVI_TDL_FaceAttribute(cvtdl_handle_t handle,
                              const cvtdl_model_e model_id,
                              cvtdl_image_t image_handle,
                              cvtdl_face_t *face_meta);

int32_t CVI_TDL_KeypointDetection(cvtdl_handle_t handle,
                                  const cvtdl_model_e model_id,
                                  cvtdl_image_t image_handle,
                                  cvtdl_keypoint_t *keypoint_meta);

int32_t CVI_TDL_InstanceSeg(cvtdl_handle_t handle, 
                            const cvtdl_model_e model_id,
                            cvtdl_image_t image_handle,
                            cvtdl_object_t *object_meta);

int32_t CVI_TDL_SemanticSeg(cvtdl_handle_t handle,
                             const cvtdl_model_e model_id,
                             cvtdl_image_t image_handle,
                             cvtdl_seg_t *seg_meta);

int32_t CVI_TDL_FeatureExtraction(cvtdl_handle_t handle,
                                  const cvtdl_model_e model_id,
                                  cvtdl_image_t image_handle,
                                  cvtdl_feature_t *feature_meta);
int32_t CVI_TDL_LaneDetection(cvtdl_handle_t handle,
                              const cvtdl_model_e model_id,
                              cvtdl_image_t image_handle,
                              cvtdl_lane_t *lane_meta);

#ifdef __cplusplus
}
#endif

#endif
