===================
TDL SDK Python API
===================

图像处理模块 (tdl.image)
=========================

枚举类型
---------

ImageFormat
~~~~~~~~~~~~

.. list-table::
   :widths: 1 1

   * - RGB_PLANAR
     - RGB平面格式

   * - BGR_PLANAR
     - BGR平面格式

   * - RGB_PACKED
     - RGB包装格式

   * - BGR_PACKED
     - BGR包装格式

   * - GRAY
     - 灰度图像

   * - YUV420SP_UV
     - YUV420SP_UV格式

   * - YUV420SP_VU
     - YUV420SP_VU格式

   * - YUV420P_UV
     - YUV420P_UV格式

   * - YUV420P_VU
     - YUV420P_VU格式

   * - YUV422P_UV
     - YUV422P_UV格式

   * - YUV422P_VU
     - YUV422P_VU格式

   * - YUV422SP_UV
     - YUV422SP_UV格式

   * - YUV422SP_VU
     - YUV422SP_VU格式

TDLDataType
~~~~~~~~~~~~

.. list-table::
   :widths: 1 1 

   * - 数据类型枚举类
     - 注释

   * - UINT8
     - 无符号8位整数

   * - INT8
     - 有符号8位整数

   * - UINT16
     - 无符号16位整数

   * - INT16
     - 有符号16位整数

   * - UINT32
     - 无符号32位整数

   * - INT32
     - 有符号32位整数

   * - FP32
     - 32位浮点数  


图像处理函数
-------------
.. list-table::
   :widths: 2 2 1 1

   * - 函数名
     - 输入
     - 输出
     - 说明

   * - read(path)
     - path: 图像文件路径
     - Image对象
     - 读取图像文件

   * - write(image, path)
     - image: Image对象

       path: 保存路径
     - -
     - 保存图像到文件

   * - resize(src, width, height)
     - src: 源图像

       width: 目标宽度

       height: 目标高度
     - 调整大小后的Image对象
     - 调整图像大小

   * - crop(src, roi)
     - src: 源图像

       roi: (x, y, width, height)元组
     - 裁剪后的Image对象
     - 调裁剪图像

   * - crop_resize(src, roi, width, height)
     - src: 源图像

       roi: (x, y, width, height)元组

       width: 目标宽度

       height: 目标高度
     - 处理后的Image对象
     - 裁剪并调整图像大小

神经网络模块 (tdl.nn)
======================

枚举类型
--------

.. list-table::
   :widths: 1 1 

   * - 模型名称
     - 注释

   * - MBV2_DET_PERSON_256_448
     - 人体检测模型

   * - YOLOV5_DET_COCO80
     - YOLOv5 COCO80检测模型

   * - YOLOV6_DET_COCO80
     - YOLOv6 COCO80检测模型

   * - YOLOV8_DET_COCO80
     - YOLOv8 COCO80检测模型

   * - YOLOV10_DET_COCO80
     - YOLOv10 COCO80检测模型

   * - YOLOV8N_DET_HAND
     - 手部检测模型

   * - YOLOV8N_DET_PET_PERSON
     - 宠物与人检测模型 (0:猫, 1:狗, 2:人)

   * - YOLOV8N_DET_PERSON_VEHICLE
     - 人与车辆检测模型 (0:车, 1:公交, 2:卡车, 3:骑摩托车者, 4:人, 5:自行车, 6:摩托车)

   * - YOLOV8N_DET_HAND_FACE_PERSON
     - 手、脸与人检测模型 (0:手, 1:脸, 2:人)

   * - YOLOV8N_DET_HEAD_PERSON
     - 人头检测模型 (0:人, 1:头)

   * - YOLOV8N_DET_HEAD_HARDHAT
     - 头部与安全帽检测模型 (0:头, 1:安全帽)

   * - YOLOV8N_DET_FIRE_SMOKE
     - 火与烟检测模型 (0:火, 1:烟)

   * - YOLOV8N_DET_FIRE
     - 火检测模型 (0:火)

   * - YOLOV8N_DET_HEAD_SHOULDER
     - 头肩检测模型 (0:头肩)

   * - YOLOV8N_DET_LICENSE_PLATE
     - 车牌检测模型 (0:车牌)

   * - YOLOV8N_DET_TRAFFIC_LIGHT
     - 交通信号灯检测模型 (0:红, 1:黄, 2:绿, 3:关闭, 4:等待)

   * - SCRFD_DET_FACE
     - 人脸检测模型 (0:人脸 + 关键点)

   * - RETINA_DET_FACE
     - 人脸检测模型

   * - RETINA_DET_FACE_IR
     - 红外人脸检测模型

   * - KEYPOINT_FACE_V2
     - 5个关键点 + 模糊评分的人脸检测模型

   * - CLS_ATTRIBUTE_FACE
     - 人脸属性分类模型 (年龄, 性别, 眼镜, 面具)

   * - RESNET_FEATURE_BMFACE_R34
     - ResNet34 512维特征提取模型

   * - RESNET_FEATURE_BMFACE_R50
     - ResNet50 512维特征提取模型

   * - CLS_MASK
     - 口罩检测模型 (0:戴口罩, 1:不戴口罩)

   * - CLS_RGBLIVENESS
     - 活体检测模型 (0:活体, 1:伪造)

   * - CLS_ISP_SCENE
     - ISP场景分类模型

   * - CLS_HAND_GESTURE
     - 手势分类模型 (0:拳头, 1:五指, 2:无, 3:二)

   * - CLS_KEYPOINT_HAND_GESTURE
     - 手势关键点分类模型 (0:拳头, 1:五指, 2:四指, 3:无, 4:好, 5:一, 6:三, 7:三2, 8:二)

   * - CLS_SOUND_BABAY_CRY
     - 婴儿哭声分类模型 (0:背景, 1:哭声)

   * - CLS_SOUND_COMMAND
     - 命令声音分类模型 

   * - KEYPOINT_LICENSE_PLATE
     - 车牌关键点检测模型

   * - KEYPOINT_HAND
     - 手部关键点检测模型

   * - KEYPOINT_YOLOV8POSE_PERSON17
     - 人体17个关键点检测模型

   * - KEYPOINT_SIMCC_PERSON17
     - SIMCC 17个关键点检测模型

   * - LSTR_DET_LANE
     - 车道检测模型

   * - RECOGNITION_LICENSE_PLATE
     - 车牌识别模型

   * - YOLOV8_SEG_COCO80
     - YOLOv8 COCO80分割模型

   * - TOPFORMER_SEG_PERSON_FACE \
       _VEHICLE
     - 人、脸与车辆分割模型 (0:背景, 1:人, 2:脸, 3:车辆, 4:车牌)

   * - TOPFORMER_SEG_MOTION
     - 动作分割模型 (0:静态, 2:过渡, 3:运动)

   * - CLIP_FEATURE_IMG
     - 图像特征提取模型

   * - CLIP_FEATURE_TEXT
     - 文本特征提取模型



模型类
--------

.. list-table::
   :widths: 2 1 4

   * - 模型名称
     - 功能
     - 模型id

   * - ObjectDetector

     - 目标检测

     - MBV2_DET_PERSON_256_448

       YOLOV5_DET_COCO80

       YOLOV6_DET_COCO80

       YOLOV8_DET_COCO80

       YOLOV10_DET_COCO80

       YOLOV8N_DET_HAND

       YOLOV8N_DET_PET_PERSON

       YOLOV8N_DET_PERSON_VEHICLE

       YOLOV8N_DET_HAND_FACE_PERSON

       YOLOV8N_DET_HEAD_PERSON

       YOLOV8N_DET_HEAD_HARDHAT

       YOLOV8N_DET_FIRE_SMOKE

       YOLOV8N_DET_FIRE

       YOLOV8N_DET_HEAD_SHOULDER

       YOLOV8N_DET_LICENSE_PLATE

       YOLOV8N_DET_TRAFFIC_LIGHT

       YOLOV5

       YOLOV6

       YOLOV8

       YOLOV10


   * - FaceDetector

     - 人脸检测

     - SCRFD_DET_FACE

       RETINA_DET_FACE

       RETINA_DET_FACE_IR

       KEYPOINT_FACE_V2 

   * - Classifier

     - 分类

     - CLS_MASK

       CLS_RGBLIVENESS

       CLS_ISP_SCENE

       CLS_HAND_GESTURE

       CLS_KEYPOINT_HAND_GESTURE

       CLS_SOUND_BABAY_CRY

       CLS_SOUND_COMMAND 

   * - KeyPointDetector

     - 关键点检测
    
     - KEYPOINT_LICENSE_PLATE

       KEYPOINT_HAND

       KEYPOINT_YOLOV8POSE_PERSON17

       KEYPOINT_SIMCC_PERSON17 

   * - SemanticSegmentation

     - 语义分割
     
     - TOPFORMER_SEG_PERSON_FACE_VEHICLE

       TOPFORMER_SEG_MOTION 

   * - InstanceSegmentation

     - 实例分割

     - YOLOV8_SEG_COCO80

   * - LaneDetection

     - 车道线检测

     - LSTR_DET_LANE

   * - AttributeExtractor

     - 属性提取

     - CLS_ATTRIBUTE_FACE

   * - FeatureExtractor

     - 特征提取

     - RESNET_FEATURE_BMFACE_R34
     
       RESNET_FEATURE_BMFACE_R50

       CLIP_FEATURE_IMG

       CLIP_FEATURE_TEXT 

   * - CharacterRecognitor

     - 字符提取

     - RECOGNITION_LICENSE_PLATE
    
.. note::
    ObjectDetector类中YOLOV5、YOLOV6、YOLOV8、YOLOV10模型id，可供用户使用自主开发模型。

使用示例
---------

【使用tdl.image预处理】

.. code-block:: python

   from tdl import nn, image

   # 人脸检测示例
   face_detector = nn.FaceDetector(nn.ModelType.FD_SCRFD, "model_path")
   img = image.read("image_path")
   bboxes = face_detector.inference(img)

【使用opencv预处理】

.. code-block:: python
  
   import cv2 
   from tdl import nn, image

   # 人脸检测示例
   face_detector = nn.FaceDetector(nn.ModelType.FD_SCRFD, "model_path")
   img = cv2.imread(img_path)
   bboxes = face_detector.inference(img)

.. note::
   每个模型类的具体参数和返回值可能不同。