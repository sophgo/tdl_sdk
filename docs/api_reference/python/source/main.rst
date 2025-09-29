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

   * - MBV2_DET_PERSON
     - 人体检测模型

   * - YOLOV5_DET_COCO80
     - YOLOv5 COCO80检测模型

   * - YOLOV6_DET_COCO80
     - YOLOv6 COCO80检测模型

    * - YOLOV7_DET_COCO80
      - YOLOv7 COCO80检测模型

   * - YOLOV8_DET_COCO80
     - YOLOv8 COCO80检测模型

   * - YOLOV10_DET_COCO80
     - YOLOv10 COCO80检测模型
  
   * - PPYOLOE_DET_COCO80
     - PPYOLOE COCO80检测模型

   * - YOLOX_DET_COCO80
     - YOLOX COCO80检测模型

   * - YOLOV8N_DET_HAND
     - 手部检测模型

   * - YOLOV8N_DET_PET_PERSON
     - 宠物与人检测模型 (0:猫, 1:狗, 2:人)

   * - YOLOV8N_DET_PERSON_VEHICLE
     - 人与车辆检测模型 (0:车, 1:公交, 2:卡车, 3:骑摩托车者, 4:人, 5:自行车, 6:摩托车)

   * - YOLOV8N_DET_HAND_FACE_PERSON
     - 手、脸与人检测模型 (0:手, 1:脸, 2:人)

   * - YOLOV8N_DET_FACE_HEAD_PERSON_HEAD
     - 人脸、头部、人、宠物检测模型 (0:人脸, 1:头部, 2:人， 3:宠物)

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

   * - CLS_ATTRIBUTE_GENDER_AGE_GLASS
     - 人脸属性分类模型 (年龄, 性别, 眼镜)

   * - CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK
     - 人脸属性分类模型 (年龄, 性别, 眼镜, 口罩)

   * - CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION
     - 人脸属性分类模型 (年龄, 性别, 眼镜, 情绪)

   * - FEATURE_CVIFACE
     - CV人脸特征提取模型

   * - FEATURE_BMFACE_R34
     - ResNet34 512维特征提取模型

   * - FEATURE_BMFACE_R50
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
   
   * - CLS_SOUND_COMMAND_NIHAOSHIYUN
     - 命令声音分类模型 (0:背景, 1:你好视云)

   * - CLS_SOUND_COMMAND_NIHAOSUANNENG
     - 命令声音分类模型 (0:背景, 1:你好算能)

   * - CLS_SOUND_COMMAND_XIAOAIXIAOAI
     - 命令声音分类模型 (0:背景, 1:小爱小爱)

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

   * - TOPFORMER_SEG_PERSON_FACE_VEHICLE
     - 人、脸与车辆分割模型 (0:背景, 1:人, 2:脸, 3:车辆, 4:车牌)

   * - TOPFORMER_SEG_MOTION
     - 动作分割模型 (0:静态, 2:过渡, 3:运动)

   * - FEATURE_CLIP_IMG
     - Clip图像特征提取模型

   * - FEATURE_CLIP_TEXT
     - Clip文本特征提取模型

   * - FEATURE_MOBILECLIP2_IMG
     - MobileClip2图像特征提取模型

   * - FEATURE_MOBILECLIP2_TEXT
     - MobileClip2文本特征提取模型


模型创建函数
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 2 2 1 1

   * - 函数名
     - 输入
     - 输出
     - 说明

   * - get_model(model_type, model_path, model_config={}, device_id=0)
     - model_type: 模型类型枚举值

       model_path: 模型路径

       model_config: 预处理参数字典

       包含以下可选字段:

       - mean: tuple(float,float,float)
       - scale: tuple(float,float,float)
       - rgb_order: 【可选】str, 通道顺序(bgr, rgb, gray)，默认rgb
       - types: 【可选】list[str], 类别列表
       - comment: 【可选】str, 注释说明
       - custom_config_str: 【可选】dict[str,str], 字符串配置

       device_id: 设备ID，默认为0
     - PyModel
     - 创建模型实例，可指定配置参数

   * - get_model_from_dir(model_type, model_dir="", device_id=0)
     - model_type: 模型类型枚举值

       model_dir: 模型文件夹路径，下面包含各平台模型文件夹(bm1688,bm1684x,cv181x,cv184x,...)

       device_id: 设备ID，默认为0
     - PyModel
     - 创建模型实例，使用默认配置



PyModel 类接口说明
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 2 2 1 1

   * - 接口名
     - 输入
     - 输出
     - 说明

   * - inference(image)
     - image: 输入图像
       
       支持 PyImage 对象或 numpy 数组
     - list
     - 执行模型推理，返回推理结果列表

   * - getPreprocessParameters()
     - -
     - dict
     - 获取模型预处理参数，返回包含均值(mean)和缩放(scale)的字典

使用示例
---------

【使用tdl.image预处理】

.. code-block:: python

   from tdl import nn, image

   # 人脸检测示例
   face_detector = nn.get_model(nn.ModelType.SCRFD_DET_FACE, "model_path")
   img = image.read("image_path")
   bboxes = face_detector.inference(img)

【使用opencv预处理】

.. code-block:: python
  
   import cv2 
   from tdl import nn, image

   # 人脸检测示例
   face_detector = nn.get_model(nn.ModelType.SCRFD_DET_FACE, "model_path")
   img = cv2.imread(img_path)
   bboxes = face_detector.inference(img)

.. note::
   每个模型类的具体参数和返回值可能不同。