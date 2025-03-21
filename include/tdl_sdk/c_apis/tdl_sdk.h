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
tdl_handle_t TDL_CreateHandle(const int32_t tpu_device_id);

/**
 * @brief 销毁一个 TDLContextHandle 对象
 *
 * @param context_handle 需要销毁的 TDLContextHandle 对象
 */
int32_t TDL_DestroyHandle(tdl_handle_t handle);

/**
 * @brief 包装一个 VPSS 帧为 TDLImageHandle 对象
 *
 * @param vpss_frame 需要包装的 VPSS 帧
 * @param own_memory 是否拥有内存所有权
 * @return  返回包装的 TDLImageHandle 对象, 如果失败返回 NULL
 */
tdl_image_t TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory);

/**
 * @brief 读取一张图片为 TDLImageHandle 对象
 *
 * @param path 图片路径
 * @return  返回读取的 TDLImageHandle 对象, 如果失败返回 NULL
 */
tdl_image_t TDL_ReadImage(const char *path);

/**
 * @brief 销毁一个 TDLImageHandle 对象
 *
 * @param image_handle 需要销毁的 TDLImageHandle 对象
 */
int32_t TDL_DestroyImage(tdl_image_t image_handle);

int32_t TDL_OpenModel(tdl_handle_t handle,
                      const tdl_model_e model_id,
                      const char *model_path);

int32_t TDL_CloseModel(tdl_handle_t handle,
                       const tdl_model_e model_id);

int32_t TDL_Detection(tdl_handle_t handle,
                      const tdl_model_e model_id,
                      tdl_image_t image_handle,
                      tdl_object_t *object_meta);

int32_t TDL_FaceDetection(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_face_t *face_meta);

int32_t TDL_FaceAttribute(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_face_t *face_meta);

int32_t TDL_FaceLandmark(tdl_handle_t handle,
                         const tdl_model_e model_id,
                         tdl_image_t image_handle,
                         tdl_face_t *face_meta);

int32_t TDL_Classfification(tdl_handle_t handle,
                            const tdl_model_e model_id,
                            tdl_image_t image_handle,
                            tdl_class_info_t *class_info);

int32_t TDL_ObjectClassification(tdl_handle_t handle,
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_object_t *object_meta,
                                 tdl_class_t *class_info);

int32_t TDL_KeypointDetection(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_keypoint_t *keypoint_meta);

int32_t TDL_InstanceSegmentation(tdl_handle_t handle, 
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_instance_seg_t *inst_seg_meta);

int32_t TDL_SemanticSegmentation(tdl_handle_t handle,
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_seg_t *seg_meta);

int32_t TDL_FeatureExtraction(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_feature_t *feature_meta);

int32_t TDL_LaneDetection(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_lane_t *lane_meta);

int32_t TDL_DepthStereo(tdl_handle_t handle,
                        const tdl_model_e model_id,
                        tdl_image_t image_handle,
                        tdl_depth_logits_t *depth_logist);

int32_t TDL_Tracking(tdl_handle_t handle,
                     const tdl_model_e model_id,
                     tdl_image_t image_handle,
                     tdl_object_t *object_meta,
                     tdl_tracker_t *tracker_meta);

#ifdef __cplusplus
}
#endif

#endif
