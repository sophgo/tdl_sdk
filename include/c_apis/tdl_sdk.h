#ifndef TDL_SDK_H
#define TDL_SDK_H

#include <getopt.h>
#include "tdl_model_def.h"
#include "tdl_types.h"
#include "tdl_utils.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建一个 TDLHandle 对象
 *
 * @param tpu_device_id 指定 TPU 设备的 ID
 * @return  返回创建的 TDLHandle 对象, 如果失败返回 NULL
 */
TDLHandle TDL_CreateHandle(const int32_t tpu_device_id);

/**
 * @brief 销毁一个 TDLHandle 对象
 *
 * @param context_handle 需要销毁的 TDLHandle 对象
 */
int32_t TDL_DestroyHandle(TDLHandle handle);

/**
 * @brief 包装一帧图像信息为 TDLImageHandle 对象
 *
 * @param frame 需要包装的帧图像信息
 * @param own_memory 是否拥有内存所有权
 * @return  返回包装的 TDLImageHandle 对象, 如果失败返回 NULL
 */
TDLImage TDL_WrapFrame(void *frame, bool own_memory);

#if !defined(__BM168X__) && !defined(__CMODEL_CV181X__)
/**
 * @brief 初始化Camera，板端的/mnt/data路径下需要有sensor_cfg.ini
 *
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_InitCamera(TDLHandle handle);

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

/**
 * @brief 初始化移动侦测任务
 *
 * @param handle TDLHandle 对象
 * @return 成功返回 0，失败返回-1
 */
#endif

/**
 * @brief 读取一张图片为 TDLImageHandle 对象
 *
 * @param path 图片路径
 * @return  返回读取的 TDLImageHandle 对象, 如果失败返回 NULL
 */
TDLImage TDL_ReadImage(const char *path);

/**
 * @brief 读取文件内容为 TDLImageHandle 对象
 *
 * @param path 文件路径
 * @param data_type 文件数据类型
 * @return  返回读取的 TDLImageHandle 对象, 如果失败返回 NULL
 */
TDLImage TDL_ReadBin(const char *path, TDLDataTypeE data_type);

/**
 * @brief 销毁一个 TDLImageHandle 对象
 *
 * @param image_handle 需要销毁的 TDLImageHandle 对象
 */
int32_t TDL_DestroyImage(TDLImage image_handle);

/**
 * @brief 加载模型配置信息，加载后可以仅通过模型id去打开模型
 *
 * @param handle 已初始化的 TDLHandle 对象，通过 TDL_CreateHandle 创建
 * @param model_config_json_path
 * 模型配置文件路径，如果为NULL，默认使用configs/model/model_config.json
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_LoadModelConfig(TDLHandle handle,
                            const char *model_config_json_path);
/**
 * @brief 设置模型文件夹路径
 *
 * @param handle 已初始化的 TDLHandle 对象，通过 TDL_CreateHandle 创建
 * @param model_dir 为tdl_models仓库路径(下面各平台的子文件夹)
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_SetModelDir(TDLHandle handle, const char *model_dir);
/**
 * @brief 加载指定模型到 TDLHandle 对象
 *
 * @param handle 已初始化的 TDLHandle 对象，通过 TDL_CreateHandle 创建
 * @param model_id 要加载的模型类型枚举值
 * @param model_path
 * 模型文件路径，绝对路径，假如使用配置信息里的路径，则可以传入NULL
 * @param model_config_json 模型配置信息
 * 1）使用TDL_LoadModelConfig加载后可以传入NULL,
 * 2）不使用TDL_LoadModelConfig加载，大部分专有模型也可以传入NULL，此时会使用算法类内部的默认配置，
 *    部分通用模型如特征提取、声音指令等需要传入模型配置信息，可以参考configs/model/model_config.json
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_OpenModel(TDLHandle handle, const TDLModel model_id,
                      const char *model_path, const char *model_config_json);

/**
 * @brief 卸载指定模型并释放相关资源
 *
 * @param handle 已初始化的 TDLHandle 对象
 * @param model_id 要卸载的模型类型枚举值
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_CloseModel(TDLHandle handle, const TDLModel model_id);

/**
 * @brief 设置模型threshold
 *
 * @param handle 已初始化的 TDLHandle 对象
 * @param model_id 要设置的模型类型枚举值
 * @param threshold 模型的阈值
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_SetModelThreshold(TDLHandle handle, const TDLModel model_id,
                              float threshold);

/**
 * @brief 执行通用目标检测
 *
 * @param handle 已加载模型的 TDLHandle 对象
 * @param model_id 使用的检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param object_meta 输出目标检测结果元数据
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Detection(TDLHandle handle, const TDLModel model_id,
                      TDLImage image_handle, TDLObject *object_meta);

/**
 * @brief 执行人脸检测
 *
 * @param handle 已加载人脸模型的 TDLHandle 对象
 * @param model_id 使用的人脸检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param face_meta 输出人脸检测结果元数据
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FaceDetection(TDLHandle handle, const TDLModel model_id,
                          TDLImage image_handle, TDLFace *face_meta);

/**
 * @brief 执行人脸属性分析
 *
 * @param handle 已加载属性模型的 TDLHandle 对象
 * @param model_id 使用的人脸属性模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param face_meta 输入/输出人脸数据，包含检测结果和属性信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FaceAttribute(TDLHandle handle, const TDLModel model_id,
                          TDLImage image_handle, TDLFace *face_meta);

/**
 * @brief 执行人脸关键点检测
 *
 * @param handle 已加载关键点模型的 TDLHandle 对象
 * @param model_id 使用的人脸关键点模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param face_meta 输入/输出人脸数据，包含检测结果和关键点信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FaceLandmark(TDLHandle handle, const TDLModel model_id,
                         TDLImage image_handle, TDLFace *face_meta);

/**
 * @brief 执行图像分类任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定使用的模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param class_info 输出参数，存储分类结果，类别置信度、标签等
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Classfification(TDLHandle handle, const TDLModel model_id,
                            TDLImage image_handle, TDLClassInfo *class_info);

/**
 * @brief 对检测到的目标进行细粒度分类
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定目标分类模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param object_meta 输入参数，包含待分类的目标检测框信息
 * @param class_info 输出参数，存储目标分类结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_ObjectClassification(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle, TDLObject *object_meta,
                                 TDLClass *class_info);

/**
 * @brief 执行ISP图像分类任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定目标分类模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param isp_meta 输入参数，包含isp相关的数据
 * @param class_info 输出参数，存储目标分类结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_IspClassification(TDLHandle handle, const TDLModel model_id,
                              TDLImage image_handle, TDLIspMeta *isp_meta,
                              TDLClass *class_info);

/**
 * @brief 执行关键点检测任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定关键点检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param keypoint_meta 输出参数，存储检测到的关键点坐标及置信度
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Keypoint(TDLHandle handle, const TDLModel model_id,
                     TDLImage image_handle, TDLKeypoint *keypoint_meta);

/**
 * @brief 执行关键点检测任务（根据目标的坐标进行裁剪后再执行关键点检测）
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定关键点检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param object_meta 输出参数，存储检测到的关键点坐标及置信度
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_DetectionKeypoint(TDLHandle handle, const TDLModel model_id,
                              TDLImage image_handle, TDLObject *object_meta);

/**
 * @brief 执行实例分割任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定实例分割模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param inst_seg_meta 输出参数，存储实例分割结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_InstanceSegmentation(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle,
                                 TDLInstanceSeg *inst_seg_meta);

/**
 * @brief 执行语义分割任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定语义分割模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param seg_meta 输出参数，存储语义分割结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_SemanticSegmentation(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle,
                                 TDLSegmentation *seg_meta);

/**
 * @brief 执行特征提取任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定使用的特征提取模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param feature_meta 输出参数，存储提取的特征向量
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_FeatureExtraction(TDLHandle handle, const TDLModel model_id,
                              TDLImage image_handle, TDLFeature *feature_meta);

/**
 * @brief 执行车道线检测任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定车道线检测模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param lane_meta 输出参数，存储检测到的车道线信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_LaneDetection(TDLHandle handle, const TDLModel model_id,
                          TDLImage image_handle, TDLLane *lane_meta);

/**
 * @brief 执行字符识别任务（OCR）
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定字符识别模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param char_meta 输出参数，存储识别结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_CharacterRecognition(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle, TDLOcr *char_meta);

/**
 * @brief 执行立体视觉深度估计任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定深度估计模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param depth_logist 输出参数，存储深度估计结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_DepthStereo(TDLHandle handle, const TDLModel model_id,
                        TDLImage image_handle, TDLDepthLogits *depth_logist);

/**
 * @brief 执行目标跟踪任务
 *
 * @param handle TDLHandle 对象
 * @param model_id 指定跟踪模型类型枚举值
 * @param image_handle TDLImageHandle 对象
 * @param obj_meta 输入/输出参数，包含待跟踪目标信息并更新跟踪状态
 * @param tracker_meta 输出参数，存储跟踪器状态信息
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_Tracking(TDLHandle handle, int frame_id, TDLFace *face_meta,
                     TDLObject *obj_meta, TDLTracker *track_meta);

/**
 * @brief 执行入侵检测
 *
 * @param regions 背景区域点集数组
 * @param box 检测区域bbox
 * @param is_intrusion 输出参数，存储入侵检测结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_IntrusionDetection(TDLHandle handle, TDLPoints *regions,
                               TDLBox *box, bool *is_intrusion);

#if defined(__CV181X__) || defined(__CV184X__)
/**
 * @brief 执行移动侦测任务
 *
 * @param handle TDLHandle 对象
 * @param background 背景图像
 * @param detect_image 检测图像
 * @param roi 检测区域
 * @param threshold 阈值
 * @param min_area 最小面积
 * @param obj_meta 输出参数，存储检测结果
 * @param background_update_interval 背景更新间隔
 * @return 成功返回 0，失败返回-1
 */

int32_t TDL_MotionDetection(TDLHandle handle, TDLImage background,
                            TDLImage detect_image, TDLObject *roi,
                            uint8_t threshold, double min_area,
                            TDLObject *obj_meta,
                            uint32_t background_update_interval);
#endif

/*******************************************
 *              APP API
 * 以下接口提供一个多模型封装的简化接口
 * 主要用于一些复杂的任务场景
 * 如人脸抓拍、宠物检测、辅助驾驶等
 * 这些任务通常需要多个模型的协同工作
 * 通过APP API可以简化调用流程
 * 使得用户可以更专注于任务逻辑而非模型细节
 *******************************************/

/**
 * @brief 初始化APP任务
 *
 * @param handle TDLHandle 对象
 * @param task APP任务名称
 * @param config_file APP的json配置文件路径
 * @param channel_names 每一路视频流的名称信息
 * @param channel_size 视频流的路数
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_APP_Init(TDLHandle handle, const char *task,
                     const char *config_file, char ***channel_names,
                     uint8_t *channel_size);

/**
 * @brief 往APP送帧
 *
 * @param handle TDLHandle 对象
 * @param channel_name 当前channel的名称
 * @param image_handle TDLImageHandle 对象
 * @param frame_id 当前TDLImageHandle 对象的frame id
 * @param buffer_size 推理线程缓存的帧数
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_APP_SetFrame(TDLHandle handle, const char *channel_name,
                         TDLImage image_handle, uint64_t frame_id,
                         int buffer_size);

/**
 * @brief 执行人脸抓拍任务
 *
 * @param handle TDLHandle 对象
 * @param channel_name 当前channel的名称
 * @param capture_info 抓拍结果
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_APP_Capture(TDLHandle handle, const char *channel_name,
                        TDLCaptureInfo *capture_info);

/**
 * @brief 执行客流统计任务
 *
 * @param handle TDLHandle 对象
 * @param channel_name 当前channel的名称
 * @param object_meta 检测结果
 * @param enter_num 进入人数
 * @param miss_num 离开人数
 * @return 成功返回 0，失败返回-1
 */
int32_t TDL_APP_ConsumerCounting(TDLHandle handle, const char *channel_name,
                                 TDLObject *object_meta, uint32_t *enter_num,
                                 uint32_t *miss_num);

#ifdef __cplusplus
}
#endif

#endif
