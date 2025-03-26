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

/**
 * @brief 加载指定模型到 TDLContextHandle 对象
 *
 * @param handle 已初始化的 TDLContextHandle 对象，通过 TDL_CreateHandle 创建
 * @param model_id 要加载的模型类型枚举值
 * @param model_path 模型文件路径，绝对路径或相对路径
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_OpenModel(tdl_handle_t handle,
                      const tdl_model_e model_id,
                      const char *model_path);

/**
 * @brief 卸载指定模型并释放相关资源
 *
 * @param handle 已初始化的 TDLContextHandle 对象
 * @param model_id 要卸载的模型类型枚举值
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_CloseModel(tdl_handle_t handle,
                       const tdl_model_e model_id);

/**
 * @brief 执行通用目标检测
 *
 * @param handle 已加载模型的 TDLContextHandle 对象
 * @param model_id 使用的检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param object_meta 输出目标检测结果元数据
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Detection(tdl_handle_t handle,
                      const tdl_model_e model_id,
                      tdl_image_t image_handle,
                      tdl_object_t *object_meta);

/**
 * @brief 执行人脸检测
 *
 * @param handle 已加载人脸模型的 TDLContextHandle 对象
 * @param model_id 使用的人脸检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param face_meta 输出人脸检测结果元数据
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FaceDetection(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_face_t *face_meta);

/**
 * @brief 执行人脸属性分析
 *
 * @param handle 已加载属性模型的 TDLContextHandle 对象
 * @param model_id 使用的人脸属性模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param face_meta 输入/输出人脸数据，包含检测结果和属性信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FaceAttribute(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_face_t *face_meta);

/**
 * @brief 执行人脸关键点检测
 *
 * @param handle 已加载关键点模型的 TDLContextHandle 对象
 * @param model_id 使用的人脸关键点模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param face_meta 输入/输出人脸数据，包含检测结果和关键点信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FaceLandmark(tdl_handle_t handle,
                         const tdl_model_e model_id,
                         tdl_image_t image_handle,
                         tdl_face_t *face_meta);

/**
 * @brief 执行图像分类任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定使用的模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param class_info 输出参数，存储分类结果，类别置信度、标签等
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Classfification(tdl_handle_t handle,
                            const tdl_model_e model_id,
                            tdl_image_t image_handle,
                            tdl_class_info_t *class_info);

/**
 * @brief 对检测到的目标进行细粒度分类
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定目标分类模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param object_meta 输入参数，包含待分类的目标检测框信息
 * @param class_info 输出参数，存储目标分类结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_ObjectClassification(tdl_handle_t handle,
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_object_t *object_meta,
                                 tdl_class_t *class_info);

/**
 * @brief 执行关键点检测任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定关键点检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param keypoint_meta 输出参数，存储检测到的关键点坐标及置信度
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_KeypointDetection(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_keypoint_t *keypoint_meta);

/**
 * @brief 执行实例分割任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定实例分割模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param inst_seg_meta 输出参数，存储实例分割结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_InstanceSegmentation(tdl_handle_t handle, 
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_instance_seg_t *inst_seg_meta);

/**
 * @brief 执行语义分割任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定语义分割模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param seg_meta 输出参数，存储语义分割结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_SemanticSegmentation(tdl_handle_t handle,
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_seg_t *seg_meta);

/**
 * @brief 执行特征提取任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定使用的特征提取模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param feature_meta 输出参数，存储提取的特征向量
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FeatureExtraction(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_feature_t *feature_meta);

/**
 * @brief 执行车道线检测任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定车道线检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param lane_meta 输出参数，存储检测到的车道线信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_LaneDetection(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_lane_t *lane_meta);

/**
 * @brief 执行立体视觉深度估计任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定深度估计模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param depth_logist 输出参数，存储深度估计结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_DepthStereo(tdl_handle_t handle,
                        const tdl_model_e model_id,
                        tdl_image_t image_handle,
                        tdl_depth_logits_t *depth_logist);

/**
 * @brief 执行目标跟踪任务
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定跟踪模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param object_meta 输入/输出参数，包含待跟踪目标信息并更新跟踪状态
 * @param tracker_meta 输出参数，存储跟踪器状态信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Tracking(tdl_handle_t handle,
                     const tdl_model_e model_id,
                     tdl_image_t image_handle,
                     tdl_object_t *object_meta,
                     tdl_tracker_t *tracker_meta);

/**
 * @brief 执行字符识别任务（OCR）
 *
 * @param handle TDLContextHandle 对象
 * @param model_id 指定字符识别模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param char_meta 输出参数，存储识别结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_CharacterRecognition(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_ocr_t *char_meta);
#ifdef __cplusplus
}
#endif

#endif
