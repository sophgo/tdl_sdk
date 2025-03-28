.. vim: syntax=rst

API参考
================

句柄
~~~~~~~~~~~~~~~

【语法】

.. code-block:: c
  
  typedef void *tdl_handle_t;
  typedef void *tdl_image_t;

【描述】

TDL SDK句柄，tdl_handle_t是核心操作句柄，tdl_image_t是图像数据抽象句柄。

TDL_Core
~~~~~~~~~~~~~~~

通用
^^^^^^^^^

TDL_CreateHandle
--------------------

【语法】

.. code-block:: c

  tdl_handle_t TDL_CreateHandle(const int32_t tpu_device_id);

【描述】

创建一个 TDLContextHandle 对象。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const int32_t
     - tpu_device_id
     - 指定 TPU 设备的 ID

TDL_DestroyHandle
---------------------

【语法】

.. code-block:: c

  int32_t TDL_DestroyHandle(tdl_handle_t handle);

【描述】

销毁一个 TDLContextHandle 对象。

【参数】

.. list-table::
   :widths: 1 2 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - 需要销毁的 TDLContextHandle 对象

TDL_WrapVPSSFrame
---------------------

【语法】

.. code-block:: c

  tdl_image_t TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory);

【描述】

包装一个 VPSS 帧为 TDLImageHandle 对象。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - void\*
     - vpss_frame
     - 需要包装的 VPSS 帧

   * - 输入
     - bool
     - own_memory
     - 是否拥有内存所有权

TDL_ReadImage
--------------------

.. code-block:: c

  tdl_image_t TDL_ReadImage(const char *path);

【描述】

读取一张图片为 TDLImageHandle 对象。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const char\*
     - path
     - 图片路径

TDL_ReadAudio
-----------------

【语法】

.. code-block:: c

  tdl_image_t TDL_ReadAudio(const char *path, int frame_size);

【描述】

读取一段音频为 TDLImageHandle 对象。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - const char\*
     - path
     - 音频路径

   * - 输入
     - int
     - frame_size
     - 音频数据大小

TDL_DestroyImage
-----------------------------

【语法】

.. code-block:: c

  int32_t TDL_DestroyImage(tdl_image_t image_handle);

【描述】

销毁一个 TDLImageHandle 对象。

【参数】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_image_t
     - image_handle
     - 需要销毁的 TDLImageHandle 对象

TDL_OpenModel
-----------------------------

【语法】

.. code-block:: c

  int32_t TDL_OpenModel(tdl_handle_t handle,
                      const tdl_model_e model_id,
                      const char *model_path);

【描述】

加载指定类型的模型到 TDLContextHandle 对象中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - const char\*
     - model_path
     - 模型路径

TDL_CloseModel
---------------------

【语法】

.. code-block:: c

  int32_t TDL_CloseModel(tdl_handle_t handle,
                       const tdl_model_e model_id);

【描述】

卸载指定类型的模型并释放相关资源。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

TDL_Detection
----------------------

【语法】

.. code-block:: c

  int32_t TDL_Detection(tdl_handle_t handle,
                      const tdl_model_e model_id,
                      tdl_image_t image_handle,
                      tdl_object_t *object_meta);

【描述】

执行指定模型的推理检测，并返回检测结果元数据。

【参数】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_object_t\*
     - object_meta
     - 输出检测结果元数据

TDL_FaceDetection
---------------------

【语法】

.. code-block:: c

  int32_t TDL_FaceDetection(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_face_t *face_meta);

【描述】

执行人脸检测并返回人脸检测结果元数据。

【参数】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_face_t\*
     - face_meta
     - 输出人脸检测结果元数据

TDL_FaceAttribute
-----------------------------

【语法】

.. code-block:: c

  int32_t TDL_FaceAttribute(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_face_t *face_meta);

【描述】

执行人脸属性分析，需配合已检测到的人脸框进行特征分析。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输入/输出
     - tdl_face_t\*
     - face_meta
     - 输入人脸检测结果，输出补充属性信息

TDL_FaceLandmark
----------------------

【语法】

.. code-block:: c

  int32_t TDL_FaceLandmark(tdl_handle_t handle,
                         const tdl_model_e model_id,
                         tdl_image_t image_handle,
                         tdl_face_t *face_meta);

【描述】

执行人脸关键点检测，在已有的人脸检测结果上补充关键点坐标。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输入/输出
     - tdl_face_t\*
     - face_meta
     - 输入人脸检测结果，输出补充关键点坐标

TDL_Classfification
--------------------

【语法】

.. code-block:: c

  int32_t TDL_Classfification(tdl_handle_t handle,
                            const tdl_model_e model_id,
                            tdl_image_t image_handle,
                            tdl_class_info_t *class_info);

【描述】

执行通用分类识别。

【参数】

.. list-table::
   :widths: 1 2 1 3
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_class_info_t\*
     - class_info
     - 输出分类结果

TDL_ObjectClassification
--------------------------

【语法】

.. code-block:: c

  int32_t TDL_ObjectClassification(tdl_handle_t handle,
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_object_t *object_meta,
                                 tdl_class_t *class_info);

【描述】

对检测到的目标进行细粒度分类。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输入
     - tdl_object_t\*
     - object_meta
     - 已检测到的目标信息

   * - 输出
     - tdl_class_t\*
     - class_info
     - 输出目标分类结果

TDL_KeypointDetection
---------------------

【语法】

.. code-block:: c

  int32_t TDL_KeypointDetection(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_keypoint_t *keypoint_meta);

【描述】

执行人体/物体关键点检测。

【参数】

.. list-table::
   :widths: 1 4 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_keypoint_t\*
     - keypoint_meta
     - 输出关键点坐标及置信度

TDL_InstanceSegmentation
--------------------------

【语法】

.. code-block:: c

  int32_t TDL_InstanceSegmentation(tdl_handle_t handle, 
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_instance_seg_t *inst_seg_meta);

【描述】

执行实例分割（Instance Segmentation），检测图像中每个独立目标的像素级轮廓。

【参数】

.. list-table::
   :widths: 1 5 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_instance_seg_t\*
     - inst_seg_meta
     - 输出实例分割结果（包含mask和bbox）

TDL_SemanticSegmentation
--------------------------

【语法】

.. code-block:: c

  int32_t TDL_SemanticSegmentation(tdl_handle_t handle,
                                 const tdl_model_e model_id,
                                 tdl_image_t image_handle,
                                 tdl_seg_t *seg_meta);

【描述】

执行语义分割（Semantic Segmentation），对图像进行像素级分类。

【参数】

.. list-table::
   :widths: 1 2 2 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_seg_t\*
     - seg_meta
     - 输出分割结果（类别标签图）

TDL_FeatureExtraction
----------------------

【语法】

.. code-block:: c

  int32_t TDL_FeatureExtraction(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_feature_t *feature_meta);

【描述】

提取图像的深度特征向量。

【参数】

.. list-table::
   :widths: 1 2 1 3
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_feature_t\*
     - feature_meta
     - 输出特征向量

TDL_LaneDetection
------------------

【语法】

.. code-block:: c

  int32_t TDL_LaneDetection(tdl_handle_t handle,
                          const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_lane_t *lane_meta);

【描述】

检测道路车道线及其属性。

【参数】

.. list-table::
   :widths: 1 2 1 3
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_lane_t\*
     - lane_meta
     - 输出车道线坐标及属性

TDL_DepthStereo
---------------------

【语法】

.. code-block:: c

  int32_t TDL_DepthStereo(tdl_handle_t handle,
                        const tdl_model_e model_id,
                        tdl_image_t image_handle,
                        tdl_depth_logits_t *depth_logist);

【描述】

基于双目立体视觉的深度估计，输出深度置信度图。

【参数】

.. list-table::
   :widths: 1 3 2 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_depth_logits_t\*
     - depth_logist
     - 输出深度置信度数据

TDL_Tracking
-----------------

【语法】

.. code-block:: c

  int32_t TDL_Tracking(tdl_handle_t handle,
                     const tdl_model_e model_id,
                     tdl_image_t image_handle,
                     tdl_object_t *object_meta,
                     tdl_tracker_t *tracker_meta);


【描述】

多目标跟踪，基于检测结果进行跨帧目标关联。

【参数】

.. list-table::
   :widths: 1 3 2 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输入/输出
     - tdl_object_t\*
     - object_meta
     - 输入检测结果，输出补充跟踪ID

   * - 输出
     - tdl_tracker_t\*
     - tracker_meta
     - 输出跟踪器状态信息

TDL_CharacterRecognition
-------------------------

【语法】

.. code-block:: c

  Cint32_t TDL_CharacterRecognition(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_ocr_t *char_meta);

【描述】

字符识别，支持文本检测与识别。

【参数】

.. list-table::
   :widths: 1 3 2 3
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - tdl_handle_t
     - handle
     - TDLContextHandle 对象

   * - 输入
     - const tdl_model_e
     - model_id
     - 模型类型枚举

   * - 输入
     - tdl_image_t
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - tdl_ocr_t\*
     - char_meta
     - 输出识别结果（文本内容和位置）
