.. vim: syntax=rst

模型列表
================

.. list-table::
   :widths: 1 1 

   * - 模型名称
     - 注释

   * - TDL_MODEL_MBV2_DET_PERSON
     - 人体检测模型

   * - TDL_MODEL_YOLOV5_DET_COCO80
     - YOLOv5 COCO80检测模型

   * - TDL_MODEL_YOLOV8_DET_COCO80
     - YOLOv8 COCO80检测模型

   * - TDL_MODEL_YOLOV10_DET_COCO80
     - YOLOv10 COCO80检测模型

   * - TDL_MODEL_YOLOV8N_DET_HAND
     - 手部检测模型

   * - TDL_MODEL_YOLOV8N_DET_PET_PERSON
     - 宠物与人检测模型 (0:猫, 1:狗, 2:人)

   * - TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE
     - 人与车辆检测模型 (0:车, 1:公交, 2:卡车, 3:骑摩托车者, 4:人, 5:自行车, 6:摩托车)

   * - TDL_MODEL_YOLOV8N_DET_HAND_FACE_PERSON
     - 手、脸与人检测模型 (0:手, 1:脸, 2:人)

   * - TDL_MODEL_YOLOV8N_DET_HEAD_PERSON
     - 人头检测模型 (0:人, 1:头)

   * - TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT
     - 头部与安全帽检测模型 (0:头, 1:安全帽)

   * - TDL_MODEL_YOLOV8N_DET_FIRE_SMOKE
     - 火与烟检测模型 (0:火, 1:烟)

   * - TDL_MODEL_YOLOV8N_DET_FIRE
     - 火检测模型 (0:火)

   * - TDL_MODEL_YOLOV8N_DET_HEAD_SHOULDER
     - 头肩检测模型 (0:头肩)

   * - TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE
     - 车牌检测模型 (0:车牌)

   * - TDL_MODEL_YOLOV8N_DET_TRAFFIC_LIGHT
     - 交通信号灯检测模型 (0:红, 1:黄, 2:绿, 3:关闭, 4:等待)

   * - TDL_MODEL_SCRFD_DET_FACE
     - 人脸检测模型 (0:人脸 + 关键点)

   * - TDL_MODEL_RETINA_DET_FACE
     - 人脸检测模型

   * - TDL_MODEL_RETINA_DET_FACE_IR
     - 红外人脸检测模型

   * - TDL_MODEL_KEYPOINT_FACE_V2
     - 5个关键点 + 模糊评分的人脸检测模型

   * - TDL_MODEL_CLS_ATTRIBUTE_FACE
     - 人脸属性分类模型 (年龄, 性别, 眼镜, 面具)

   * - TDL_MODEL_FEATURE_BMFACER34
     - ResNet34 512维特征提取模型

   * - TDL_MODEL_FEATURE_BMFACER50
     - ResNet50 512维特征提取模型

   * - TDL_MODEL_CLS_MASK
     - 口罩检测模型 (0:戴口罩, 1:不戴口罩)

   * - TDL_MODEL_CLS_RGBLIVENESS
     - 活体检测模型 (0:活体, 1:伪造)

   * - TDL_MODEL_CLS_ISP_SCENE
     - ISP场景分类模型

   * - TDL_MODEL_CLS_HAND_GESTURE
     - 手势分类模型 (0:拳头, 1:五指, 2:无, 3:二)

   * - TDL_MODEL_CLS_KEYPOINT_HAND_GESTURE
     - 手势关键点分类模型 (0:拳头, 1:五指, 2:四指, 3:无, 4:好, 5:一, 6:三, 7:三2, 8:二)

   * - TDL_MODEL_CLS_SOUND_BABAY_CRY
     - 婴儿哭声分类模型 (0:背景, 1:哭声)

   * - TDL_MODEL_CLS_SOUND_COMMAND
     - 命令声音分类模型 

   * - TDL_MODEL_KEYPOINT_LICENSE_PLATE
     - 车牌关键点检测模型

   * - TDL_MODEL_KEYPOINT_HAND
     - 手部关键点检测模型

   * - TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17
     - 人体17个关键点检测模型

   * - TDL_MODEL_KEYPOINT_SIMCC_PERSON17
     - SIMCC 17个关键点检测模型

   * - TDL_MODEL_LSTR_DET_LANE
     - 车道检测模型

   * - TDL_MODEL_RECOGNITION_LICENSE_PLATE
     - 车牌识别模型

   * - TDL_MODEL_YOLOV8_SEG_COCO80
     - YOLOv8 COCO80分割模型

   * - TDL_MODEL_SEG_PERSON_FACE_VEHICLE \
       _VEHICLE
     - 人、脸与车辆分割模型 (0:背景, 1:人, 2:脸, 3:车辆, 4:车牌)

   * - TDL_MODEL_SEG_MOTION
     - 动作分割模型 (0:静态, 2:过渡, 3:运动)

   * - TDL_MODEL_IMG_FEATURE_CLIP
     - 图像特征提取模型

   * - TDL_MODELTEXT_FEATURE_CLIP
     - 文本特征提取模型

结构体参考
================

TDLDataTypeE
~~~~~~~~~~~~~~~

【说明】

数据类型枚举类

【定义】

.. code-block:: c

  typedef enum {
    TDL_TYPE_INT8 = 0, /**< Equals to int8_t. */
    TDL_TYPE_UINT8,    /**< Equals to uint8_t. */
    TDL_TYPE_INT16,    /**< Equals to int16_t. */
    TDL_TYPE_UINT16,   /**< Equals to uint16_t. */
    TDL_TYPE_INT32,    /**< Equals to int32_t. */
    TDL_TYPE_UINT32,   /**< Equals to uint32_t. */
    TDL_TYPE_BF16,     /**< Equals to bf17. */
    TDL_TYPE_FP16,     /**< Equals to fp16. */
    TDL_TYPE_FP32,     /**< Equals to fp32. */
    TDL_TYPE_UNKOWN    /**< Equals to unkown. */
  } TDLDataTypeE;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - TDL_TYPE_INT8
     - 有符号8位整数

   * - TDL_TYPE_UINT8
     - 无符号8位整数

   * - TDL_TYPE_INT16
     - 有符号16位整数

   * - TDL_TYPE_UINT16
     - 无符号16位整数

   * - TDL_TYPE_INT32
     - 有符号32位整数

   * - TDL_TYPE_UINT32
     - 无符号32位整数

   * - TDL_TYPE_BF16
     - 16位浮点数 (1 位符号, 8 位指数和 7 位尾数)

   * - TDL_TYPE_FP16
     - 16位浮点数 (1 位符号, 5 位指数和 10 位尾数)

   * - FTDL_TYPE_FP32
     - 32位浮点数 

TDLBox
~~~~~~~~~~~~~~~

【说明】

box的坐标数据

【定义】

.. code-block:: c

  typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
  } TDLBox;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - x1
     - box左上角x的坐标

   * - y1
     - box左上角y的坐标

   * - x2
     - box右下角x的坐标

   * - y2
     - box右下角y的坐标


TDLFeature
~~~~~~~~~~~~~~~

【说明】

特征值数据

【定义】

.. code-block:: c

  typedef struct {
    int8_t *ptr;
    uint32_t size;
    TDLDataTypeE type;
  } TDLFeature;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - ptr
     - 特征值数据

   * - size
     - 数据大小

   * - type
     - 数据类型


TDLPoints
~~~~~~~~~~~~~~~

【说明】

坐标队列数据

【定义】

.. code-block:: c

  typedef struct {
    float *x;
    float *y;
    uint32_t size;
    float score;
  } TDLPoints;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - x
     - 坐标队列的x数据

   * - y
     - 坐标队列的y数据

   * - size
     - 坐标队列的大小
  
   * - score
     - 分数

TDLLandmarkInfo
~~~~~~~~~~~~~~~~~~~~~~

【说明】

特征点信息

【定义】

.. code-block:: c

  typedef struct {
    float x;
    float y;
    float score;
  } TDLLandmarkInfo;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - x
     - 特征点的x坐标

   * - y
     - 特征点的y坐标
  
   * - score
     - 分数

TDLObjectInfo
~~~~~~~~~~~~~~~~~~~~~~

【说明】

目标检测信息

【定义】

.. code-block:: c

  typedef struct {
    TDLBox box;
    float score;
    int class_id;
    uint32_t landmark_size;
    TDLLandmarkInfo *landmark_properity;
    TDLObjectTypeE obj_type;
  } TDLObjectInfo;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - score
     - 目标检测的分数

   * - class_id
     - 目标检测的类别id
  
   * - landmark_size
     - 目标检测的特征点大小

   * - TDLLandmarkInfo
     - 目标检测的特征点信息

   * - obj_type
     - 目标检测的类型

TDLObject
~~~~~~~~~~~~~~~

【说明】

目标检测数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;

    TDLObjectInfo *info;
  } TDLObject;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 目标检测的个数

   * - width
     - 目标检测图像的宽度
  
   * - height
     - 目标检测图像的高度

   * - info
     - 目标检测信息

TDLFaceInfo
~~~~~~~~~~~~~~~~~~~~~~

【说明】

人脸信息

【定义】

.. code-block:: c

  typedef struct {
    char name[128];
    float score;
    uint64_t track_id;
    TDLBox box;
    TDLPoints landmarks;
    TDLFeature feature;

    float gender_score;
    float glass_score;
    float age;
    float liveness_score;
    float hardhat_score;
    float mask_score;

    float recog_score;
    float face_quality;
    float pose_score;
    float blurness;
  } TDLFaceInfo;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - name
     - 人脸的姓名

   * - score
     - 人脸的分数
  
   * - track_id
     - 人脸的追踪id

   * - box
     - 人脸的box信息

   * - landmarks
     - 人脸的特征点

   * - feature
     - 人脸的特征值
  
   * - gender_score
     - 人脸的性别分数

   * - glass_score
     - 人脸是否带眼镜

   * - age
     - 人脸的年龄

   * - liveness_score
     - 人脸的活体分数
  
   * - hardhat_score
     - 人脸的是否带安全帽分数

   * - recog_score
     - 人脸的识别罩分数

   * - face_quality
     - 人脸的质量分数

   * - pose_score
     - 人脸的姿态分数
  
   * - blurness
     - 人脸的模糊度

TDLFace
~~~~~~~~~~~~~~~

【说明】

人脸数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    TDLFaceInfo *info;
  } TDLFace;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 人脸的个数

   * - width
     - 人脸图像的宽度
  
   * - height
     - 人脸图像的高度

   * - info
     - 人脸信息

TDLClassInfo
~~~~~~~~~~~~~~~~~~~~~~

【说明】

分类信息

【定义】

.. code-block:: c

  typedef struct {
    int32_t class_id;
    float score;
  } TDLClassInfo;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - class_id
     - 分类的类别

   * - score
     - 分类的分数
  
TDLClass
~~~~~~~~~~~~~~~

【说明】

分类数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    TDLClassInfo *info;
  } TDLClass;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 分类的个数

   * - info
     - 分类信息

TDLKeypointInfo
~~~~~~~~~~~~~~~~~~~~~~

【说明】

关键点信息

【定义】

.. code-block:: c

  typedef struct {
    float x;
    float y;
    float score;
  } TDLKeypointInfo;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - x
     - 关键点的x坐标

   * - y
     - 关键点的y坐标

   * - score
     - 关键点的分数

TDLKeypoint
~~~~~~~~~~~~~~~

【说明】

关键点数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    TDLKeypointInfo *info;
  } TDLKeypoint;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 关键点的个数

   * - width
     - 图像的宽度
  
   * - height
     - 图像的高度

   * - info
     - 关键点信息

TDLSegmentation
~~~~~~~~~~~~~~~

【说明】

语义分割数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t output_width;
    uint32_t output_height;
    uint8_t *class_id;
    uint8_t *class_conf;
  } TDLSegmentation;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - width
     - 图像的宽度
  
   * - height
     - 图像的高度

   * - output_width
     - 输出图像的宽度
  
   * - output_height
     - 输出图像的高度

   * - class_id
     - 分类的类别

   * - class_conf
     - 分类的坐标信息

TDLInstanceSegInfo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

【说明】

实例分割信息

【定义】

.. code-block:: c

  typedef struct {
    uint8_t *mask;
    float *mask_point;
    uint32_t mask_point_size;
    TDLObjectInfo *obj_info;
  } TDLInstanceSegInfo;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - mask
     - 实例分割的mask队列
  
   * - mask_point
     - 实例分割的mask_point队列

   * - mask_point_size
     - 实例分割的point个数
  
   * - output_height
     - 输出图像的高度

   * - obj_info
     - 实例分割的目标检测信息

TDLInstanceSeg
~~~~~~~~~~~~~~~~~~~~~~

【说明】

实例分割数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    uint32_t mask_width;
    uint32_t mask_height;
    TDLInstanceSegInfo *info;
  } TDLInstanceSeg;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 实例分割的个数

   * - width
     - 图像的宽度
  
   * - height
     - 图像的高度

   * - mask_width
     - mask的宽度
  
   * - mask_height
     - mask的高度

   * - info
     - 实例分割信息

TDLLanePoint
~~~~~~~~~~~~~~~~~~~~~~

【说明】

线检测的坐标点

【定义】

.. code-block:: c

  typedef struct {
    float x[2];
    float y[2];
    float score;
  } TDLLanePoint;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - x
     - x坐标队列

   * - y
     - y坐标队列
  
   * - score
     - 线检测的分数

TDLLane
~~~~~~~~~~~~~~~

【说明】

线检测数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    TDLLanePoint *lane;
    int lane_state;
  } TDLLane;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 线检测的个数

   * - width
     - 图像的宽度
  
   * - height
     - 图像的高度

   * - lane
     - 线检测坐标点
  
   * - lane_state
     - 线条状态

TDLDepthLogits
~~~~~~~~~~~~~~~~~~~~~~

【说明】

深度估计数据

【定义】

.. code-block:: c

  typedef struct {
    int w;
    int h;
    int8_t *int_logits;
  } TDLDepthLogits;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - w
     - 图像的宽度
  
   * - h
     - 图像的高度

   * - int_logits
     - 深度估计信息
  
TDLTracker
~~~~~~~~~~~~~~~

【说明】

追踪数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    uint64_t id;
    TDLBox bbox;
    int out_num;
  } TDLTracker;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 追踪目标的个数
  
   * - id
     - 追踪目标的id

   * - bbox
     - 追踪目标的box

   * - out_num
     - 追踪目标的小时次数

TDLOcr
~~~~~~~~~~~~~~~

【说明】

文本识别数据

【定义】

.. code-block:: c

  typedef struct {
    uint32_t size;
    char* text_info;
  } TDLOcr;

【成员】

.. list-table::
   :widths: 1 1

   * - 数据类型枚举类
     - 注释

   * - size
     - 文本识别的个数
  
   * - text_info
     - 文本识别的信息

API参考
================

句柄
~~~~~~~~~~~~~~~

【语法】

.. code-block:: c
  
  typedef void *TDLHandle;
  typedef void *TDLImage;

【描述】

TDL SDK句柄，TDLHandle是核心操作句柄，TDLImage是图像数据抽象句柄。

TDL_CreateHandle
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  TDLHandle TDL_CreateHandle(const int32_t tpu_device_id);

【描述】

创建一个 TDLHandle 对象。

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
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_DestroyHandle(TDLHandle handle);

【描述】

销毁一个 TDLHandle 对象。

【参数】

.. list-table::
   :widths: 1 2 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - TDLHandle
     - handle
     - 需要销毁的 TDLHandle 对象

TDL_WrapVPSSFrame
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  TDLImage TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory);

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
~~~~~~~~~~~~~~~~~~

.. code-block:: c

  TDLImage TDL_ReadImage(const char *path);

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
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  TDLImage TDL_ReadAudio(const char *path, int frame_size);

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
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_DestroyImage(TDLImage image_handle);

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
     - TDLImage
     - image_handle
     - 需要销毁的 TDLImageHandle 对象

TDL_OpenModel
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_OpenModel(TDLHandle handle,
                        const TDLModel model_id,
                        const char *model_path);

【描述】

加载指定类型的模型到 TDLHandle 对象中。

【参数】

.. list-table::
   :widths: 1 3 1 2
   :header-rows: 1

   * -
     - 数据型态
     - 参数名称
     - 说明

   * - 输入
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - const char\*
     - model_path
     - 模型路径

TDL_CloseModel
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_CloseModel(TDLHandle handle,
                         const TDLModel model_id);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

TDL_Detection
~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_Detection(TDLHandle handle,
                        const TDLModel model_id,
                        TDLImage image_handle,
                        TDLObject *object_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLObject\*
     - object_meta
     - 输出检测结果元数据

TDL_FaceDetection
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_FaceDetection(TDLHandle handle,
                            const TDLModel model_id,
                            TDLImage image_handle,
                            TDLFace *face_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLFace\*
     - face_meta
     - 输出人脸检测结果元数据

TDL_FaceAttribute
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_FaceAttribute(TDLHandle handle,
                            const TDLModel model_id,
                            TDLImage image_handle,
                            TDLFace *face_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输入/输出
     - TDLFace\*
     - face_meta
     - 输入人脸检测结果，输出补充属性信息

TDL_FaceLandmark
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_FaceLandmark(TDLHandle handle,
                           const TDLModel model_id,
                           TDLImage image_handle,
                           TDLFace *face_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输入/输出
     - TDLFace\*
     - face_meta
     - 输入人脸检测结果，输出补充关键点坐标

TDL_Classfification
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_Classfification(TDLHandle handle,
                              const TDLModel model_id,
                              TDLImage image_handle,
                              TDLClassInfo *class_info);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLClassInfo\*
     - class_info
     - 输出分类结果

TDL_ObjectClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_ObjectClassification(TDLHandle handle,
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLObject *object_meta,
                                   TDLClass *class_info);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输入
     - TDLObject\*
     - object_meta
     - 已检测到的目标信息

   * - 输出
     - TDLClass\*
     - class_info
     - 输出目标分类结果

TDL_KeypointDetection
~~~~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_KeypointDetection(TDLHandle handle,
                                const TDLModel model_id,
                                TDLImage image_handle,
                                TDLKeypoint *keypoint_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLKeypoint\*
     - keypoint_meta
     - 输出关键点坐标及置信度

TDL_InstanceSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_InstanceSegmentation(TDLHandle handle, 
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLInstanceSeg *inst_seg_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLInstanceSeg\*
     - inst_seg_meta
     - 输出实例分割结果（包含mask和bbox）

TDL_SemanticSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_SemanticSegmentation(TDLHandle handle,
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLSegmentation *seg_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLSegmentation\*
     - seg_meta
     - 输出分割结果（类别标签图）

TDL_FeatureExtraction
~~~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_FeatureExtraction(TDLHandle handle,
                                const TDLModel model_id,
                                TDLImage image_handle,
                                TDLFeature *feature_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLFeature\*
     - feature_meta
     - 输出特征向量

TDL_LaneDetection
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_LaneDetection(TDLHandle handle,
                            const TDLModel model_id,
                            TDLImage image_handle,
                            TDLLane *lane_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLLane\*
     - lane_meta
     - 输出车道线坐标及属性

TDL_DepthStereo
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_DepthStereo(TDLHandle handle,
                          const TDLModel model_id,
                          TDLImage image_handle,
                          TDLDepthLogits *depth_logist);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLDepthLogits\*
     - depth_logist
     - 输出深度置信度数据

TDL_Tracking
~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_Tracking(TDLHandle handle,
                       const TDLModel model_id,
                       TDLImage image_handle,
                       TDLObject *object_meta,
                       TDLTracker *tracker_meta);


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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输入/输出
     - TDLObject\*
     - object_meta
     - 输入检测结果，输出补充跟踪ID

   * - 输出
     - TDLTracker\*
     - tracker_meta
     - 输出跟踪器状态信息

TDL_CharacterRecognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~

【语法】

.. code-block:: c

  int32_t TDL_CharacterRecognition(TDLHandle handle,
                                   const TDLModel model_id,
                                   TDLImage image_handle,
                                   TDLOcr *char_meta);

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
     - TDLHandle
     - handle
     - TDLHandle 对象

   * - 输入
     - const TDLModel
     - model_id
     - 模型类型枚举

   * - 输入
     - TDLImage
     - image_handle
     - TDLImageHandle 对象

   * - 输出
     - TDLOcr\*
     - char_meta
     - 输出识别结果（文本内容和位置）
