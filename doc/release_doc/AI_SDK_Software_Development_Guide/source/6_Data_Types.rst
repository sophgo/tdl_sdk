.. vim: syntax=rst

数据类型
=======================

CVI_AI_Core
~~~~~~~~~~~~~~~~~~~~~~~~~

CVI_AI_SUPPORTED_MODEL_E
------------------------

【描述】

此enum定义AI SDK中所有Deep Learning Model。下表为每个模型Id和其模型功能说明。

.. list-table::
   :widths: 2 1
   :header-rows: 1


   * - 模型ID
     - 说明

   * - CVI_AI_SUPPORTED_MODEL_RETINAFACE
     - 人脸侦测

   * - CVI_AI_SUPPORTED_MODEL_THERMALFACE
     - 热显人脸侦测    

   * - CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE
     - 人脸属性和人脸识别

   * - CVI_AI_SUPPORTED_MODEL_FACERECOGNITION
     - 人脸识别

   * - CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION
     - 戴口罩人脸识别  

   * - CVI_AI_SUPPORTED_MODEL_FACEQUALITY
     - 人脸质量

   * - CVI_AI_SUPPORTED_MODEL_LIVENESS
     - 双目活体识别    

   * - CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION
     - 人脸口罩识别    

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE
     - 交通工具侦测    

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN
     - 行人侦测

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS
     - 猫狗及人型侦测  

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80
     - 80类对象侦测    

   * - CVI_AI_SUPPORTED_MODEL_YOLOV3
     - 80类对象侦测    

   * - CVI_AI_SUPPORTED_MODEL_OSNET
     - 行人重识别

   * - CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION
     - 声音识别

   * - CVI_AI_SUPPORTED_MODEL_WPODNET
     - 车牌侦测

   * - CVI_AI_SUPPORTED_MODEL_LPRNET_TW
     - 台湾地区车牌识别

   * - CVI_AI_SUPPORTED_MODEL_LPRNET_CN
     - 大陆地区车牌识别

   * - CVI_AI_SUPPORTED_MODEL_DEEPLABV3
     - 语意分割

   * - CVI_AI_SUPPORTED_MODEL_ALPHAPOSE
     - 人体关键点侦测  

   * - CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION
     - 闭眼识别

   * - CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION
     - 打哈欠识别

   * - CVI_AI_SUPPORTED_MODEL_FACELANDMARKER
     - 人脸关键点侦测  

   * - CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION
     - 车内对象识别    

   * - CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION
     - 抽烟识别

   * - CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION
     - 口罩人脸侦测    


下表为每个模型Id对应的模型档案及推理使用的function：

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 模型ID
     - Inference  Function
     - 模型档案   

   * - CVI_AI_SUPPORTED_MODEL_RETINAFACE
     - CVI_AI_RetinaFace
     - retinaface_mnet0.25_342_608.cvimodel

       retinaface_mnet0.25_608_342.cvimodel

       retinaface_mnet0.25_608.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_THERMALFACE
     - CVI_AI_ThermalFace
     - thermalfd-v1.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE
     - CVI_AI_FaceAttribute  CVI_AI_FaceAttributeOne
     - cviface-v3-attribute.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACERECOGNITION
     - CVI_AI_FaceRecognition  CVI_AI_FaceRecognitionOne
     - cviface-v4.cvimodel

       cviface-v5-m.cvimodel

       cviface-v5-s.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION
     - CVI_AI_MaskFaceRecognition
     - masked-fr-v1-m.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACEQUALITY
     - CVI_AI_FaceQuality
     - fqnet-v5_shufflenetv2-softmax.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_LIVENESS
     - CVI_AI_Liveness
     - liveness-rgb-ir.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION
     - CVI_AI_MaskClassification
     - mask_classifier.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE
     - CVI_AI_MobileDetV2_Vehicle
     - mobil edetv2-vehicle-d0-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN
     - CVI_AI_MobileDetV2_Pedestrian
     - mobiledetv2-pedestrian-d0-ls-384.cvimodel

       mobiledetv2-pedestrian-d0-ls-640.cvimodel

       mobiledetv2-pedestrian-d0-ls-768.cvimodel

       mobileDetV2-pedestrian-d1-ls.cvimodel

       mobiledetv2-pedestrian-d1-ls-1024.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE
     - CVI_AI_MobileDetV2_Per son_Vehicle
     - mobiledetv2-person-vehicle-ls-768.cvimodel

       mobiledetv2-person-vehicle-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS
     - CVI_AI_MobileDetV2_Person_Pets
     - mobiledetv2-lite-person-pets-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80
     - CVI_AI_MobileD etV2_COCO80
     - mobiledetv2-d0-ls.cvimodel

       mobiledetv2-d1-ls.cvimodel

       mobiledetv2-d2-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YOLOV3
     - CVI_AI_Yolov3
     - yolo_v3_416.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_OSNET
     - CVI_AI_OSNet  CVI_AI_OSNetOne
     - person-reid-v1.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION
     - CVI_AI_SoundClassification
     - es_classification.cvimodel

       soundcmd_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_WPODNET
     - CVI_AI_LicensePlateDetection
     - wpodnet_v0_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_LPRNET_TW
     - CVI_AI_LicensePlateRecognition_TW
     - lprnet_v0_tw_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_LPRNET_CN
     - CVI_AI_LicensePlateRecognition_CN
     - lprnet_v1_cn_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_DEEPLABV3
     - CVI_AI_DeeplabV3
     - deeplabv3_mobilenetv2_640x360.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_ALPHAPOSE
     - CVI_AI_AlphaPose
     - alphapose.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_EYECLASSIFICATION
     - CVI_AI_EyeClassification
     - eye_v1_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION
     - CVI_AI_YawnClassification
     - yawn_v1_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACELANDMARKER
     - CVI_AI_FaceLandmarker
     - face_landmark_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION
     - CVI_AI_IncarObje ctDetection
     - incar_od_v0_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION
     - CVI_AI_SmokeClassification
     - N/A

   * - CVI_AI_SUPPORTED_MODEL_FDMASKDETECTION
     - CVI_AI_FaceMaskDetection
     - yolox_RetinafaceMask_mosaic1_lrelu_wmp_addinoc25_occlude_432_768_int8.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YOLOX
     - CVI_AI_YoloX
     - yolox_nano.cvimodel

       yolox_tiny.cvimodel


cvai_obj_class_id_e
-------------------

【描述】

此enum定义对象侦测类别。每一类别归属于一个类别群组。

.. list-table::
   :widths: 2 1
   :header-rows: 1


   * - 类别
     - 类别群组      

   * - CVI_AI_DET_TYPE_PERSON
     - CVI_AI_DET_GROUP_PERSON

   * - CVI_AI_DET_TYPE_BICYCLE
     - CVI_AI_DET_GROUP_VEHICLE

   * - CVI_AI_DET_TYPE_CAR
     -

   * - CVI_AI_DET_TYPE_MOTORBIKE
     -

   * - CVI_AI_DET_TYPE_AEROPLANE
     -

   * - CVI_AI_DET_TYPE_BUS
     -

   * - CVI_AI_DET_TYPE_TRAIN
     -

   * - CVI_AI_DET_TYPE_TRUCK
     -

   * - CVI_AI_DET_TYPE_BOAT
     -

   * - CVI_AI_DET_TYPE_TRAFFIC_LIGHT
     - CVI_AI_DET_GROUP_OUTDOOR

   * - CVI_AI_DET_TYPE_FIRE_HYDRANT
     -

   * - CVI_AI_DET_TYPE_STREET_SIGN
     -

   * - CVI_AI_DET_TYPE_STOP_SIGN
     -

   * - CVI_AI_DET_TYPE_PARKING_METER
     -

   * - CVI_AI_DET_TYPE_BENCH
     -

   * - CVI_AI_DET_TYPE_BIRD
     - CVI_AI_DET_GROUP_ANIMAL

   * - CVI_AI_DET_TYPE_CAT
     -

   * - CVI_AI_DET_TYPE_DOG
     -

   * - CVI_AI_DET_TYPE_HORSE
     -

   * - CVI_AI_DET_TYPE_SHEEP
     -

   * - CVI_AI_DET_TYPE_COW
     -

   * - CVI_AI_DET_TYPE_ELEPHANT
     -

   * - CVI_AI_DET_TYPE_BEAR
     -

   * - CVI_AI_DET_TYPE_ZEBRA
     -

   * - CVI_AI_DET_TYPE_GIRAFFE
     -

   * - CVI_AI_DET_TYPE_HAT
     - CVI_AI_DET_GROUP_ACCESSORY

   * - CVI_AI_DET_TYPE_BACKPACK
     -

   * - CVI_AI_DET_TYPE_UMBRELLA
     -

   * - CVI_AI_DET_TYPE_SHOE
     -

   * - CVI_AI_DET_TYPE_EYE_GLASSES
     -

   * - CVI_AI_DET_TYPE_HANDBAG
     -

   * - CVI_AI_DET_TYPE_TIE
     -

   * - CVI_AI_DET_TYPE_SUITCASE
     -

   * - CVI_AI_DET_TYPE_FRISBEE
     - CVI_AI_DET_GROUP_SPORTS

   * - CVI_AI_DET_TYPE_SKIS
     -

   * - CVI_AI_DET_TYPE_SNOWBOARD
     -

   * - CVI_AI_DET_TYPE_SPORTS_BALL
     -

   * - CVI_AI_DET_TYPE_KITE
     -

   * - CVI_AI_DET_TYPE_BASEBALL_BAT
     -

   * - CVI_AI_DET_TYPE_BASEBALL_GLOVE
     -

   * - CVI_AI_DET_TYPE_SKATEBOARD
     -

   * - CVI_AI_DET_TYPE_SURFBOARD
     -

   * - CVI_AI_DET_TYPE_TENNIS_RACKET
     -

   * - CVI_AI_DET_TYPE_BOTTLE
     - CVI_AI_DET_GROUP_KITCHEN

   * - CVI_AI_DET_TYPE_PLATE
     -

   * - CVI_AI_DET_TYPE_WINE_GLASS
     -

   * - CVI_AI_DET_TYPE_CUP
     -

   * - CVI_AI_DET_TYPE_FORK
     -

   * - CVI_AI_DET_TYPE_KNIFE
     -

   * - CVI_AI_DET_TYPE_SPOON
     -

   * - CVI_AI_DET_TYPE_BOWL
     -

   * - CVI_AI_DET_TYPE_BANANA
     - CVI_AI_DET_GROUP_FOOD

   * - CVI_AI_DET_TYPE_APPLE
     -

   * - CVI_AI_DET_TYPE_SANDWICH
     -

   * - CVI_AI_DET_TYPE_ORANGE
     -

   * - CVI_AI_DET_TYPE_BROCCOLI
     -

   * - CVI_AI_DET_TYPE_CARROT
     -

   * - CVI_AI_DET_TYPE_HOT_DOG
     -

   * - CVI_AI_DET_TYPE_PIZZA
     -

   * - CVI_AI_DET_TYPE_DONUT
     -

   * - CVI_AI_DET_TYPE_CAKE
     -

   * - CVI_AI_DET_TYPE_CHAIR
     - CVI_AI_DET_GROUP_FURNITURE

   * - CVI_AI_DET_TYPE_SOFA
     -

   * - CVI_AI_DET_TYPE_POTTED_PLANT
     -

   * - CVI_AI_DET_TYPE_BED
     -

   * - CVI_AI_DET_TYPE_MIRROR
     -

   * - CVI_AI_DET_TYPE_DINING_TABLE
     -

   * - CVI_AI_DET_TYPE_WINDOW
     -

   * - CVI_AI_DET_TYPE_DESK
     -

   * - CVI_AI_DET_TYPE_TOILET
     -

   * - CVI_AI_DET_TYPE_DOOR
     -

   * - CVI_AI_DET_TYPE_TV_MONITOR
     - CVI_AI_DET_GROUP_ELECTRONIC

   * - CVI_AI_DET_TYPE_LAPTOP
     -

   * - CVI_AI_DET_TYPE_MOUSE
     -

   * - CVI_AI_DET_TYPE_REMOTE
     -

   * - CVI_AI_DET_TYPE_KEYBOARD
     -

   * - CVI_AI_DET_TYPE_CELL_PHONE
     -

   * - CVI_AI_DET_TYPE_MICROWAVE
     - CVI_AI_DET_GROUP_APPLIANCE

   * - CVI_AI_DET_TYPE_OVEN
     -

   * - CVI_AI_DET_TYPE_TOASTER
     -

   * - CVI_AI_DET_TYPE_SINK
     -

   * - CVI_AI_DET_TYPE_REFRIGERATOR
     -

   * - CVI_AI_DET_TYPE_BLENDER
     -

   * - CVI_AI_DET_TYPE_BOOK
     - CVI_AI_DET_GROUP_INDOOR

   * - CVI_AI_DET_TYPE_CLOCK
     -

   * - CVI_AI_DET_TYPE_VASE
     -

   * - CVI_AI_DET_TYPE_SCISSORS
     -

   * - CVI_AI_DET_TYPE_TEDDY_BEAR
     -

   * - CVI_AI_DET_TYPE_HAIR_DRIER
     -

   * - CVI_AI_DET_TYPE_TOOTHBRUSH
     -

   * - CVI_AI_DET_TYPE_HAIR_BRUSH
     -


cvai_obj_det_group_type_e
-------------------------

【描述】

此enum定义对象类别群组。

.. list-table::
   :widths: 2 1
   :header-rows: 1


   * - 类别群组
     - 描述

   * - CVI_AI_DET_GROUP_ALL
     - 全部类别      

   * - CVI_AI_DET_GROUP_PERSON
     - 人形   

   * - CVI_AI_DET_GROUP_VEHICLE
     - 交通工具      

   * - CVI_AI_DET_GROUP_OUTDOOR
     - 户外   

   * - CVI_AI_DET_GROUP_ANIMAL
     - 动物   

   * - CVI_AI_DET_GROUP_ACCESSORY
     - 配件   

   * - CVI_AI_DET_GROUP_SPORTS
     - 运动   

   * - CVI_AI_DET_GROUP_KITCHEN
     - 厨房   

   * - CVI_AI_DET_GROUP_FOOD
     - 食物   

   * - CVI_AI_DET_GROUP_FURNITURE
     - 家具   

   * - CVI_AI_DET_GROUP_ELECTRONIC
     - 电子设备      

   * - CVI_AI_DET_GROUP_APPLIANCE
     - 器具   

   * - CVI_AI_DET_GROUP_INDOOR
     - 室内用品      


feature_type_e
--------------

【enum】

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数值
     - 参数名称
     - 描述

   * - 0
     - TYPE_INT8
     - int8_t特征类型

   * - 1
     - TYPE_UINT8
     - uint8_t特征类型      

   * - 2
     - TYPE_INT16
     - int16_t特征类型      

   * - 3
     - TYPE_UINT16
     - uint16_t特征类型     

   * - 4
     - TYPE_INT32
     - int32_t特征类型      

   * - 5
     - TYPE_UINT32
     - uint32_t特征类型     

   * - 6
     - TYPE_BF16
     - bf16特征类型  

   * - 7
     - TYPE_FLOAT
     - float特征类型 


meta_rescale_type_e
-------------------

【enum】

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数值
     - 参数名称
     - 描述

   * - 0
     - RESCALE_UNKNOWN
     - 未知   

   * - 1
     - RESCALE_NOASPECT
     - 不依比例直接调整     

   * - 2
     - RESCALE_CENTER
     - 在四周进行padding    

   * - 3
     - RESCALE_RB
     - 在右下进行padding    


cvai_bbox_t
-----------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - x1
     - 侦测框左上点坐标之 x 值

   * - float
     - y1
     - 侦测框左上点坐标之 y 值

   * - float
     - x2
     - 侦测框右下点坐标之 x 值

   * - float
     - y2
     - 侦测框右下点坐标之 y 值

   * - float
     - score
     - 侦测框之信心程度     


cvai_feature_t
--------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - int8_t\*
     - ptr
     - 地址   

   * - uint32_t
     - size
     - 特征维度      

   * - feature_type_e
     - type
     - 特征型态      


cvai_pts_t
----------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float\*
     - x
     - 坐标x  

   * - float\*
     - y
     - 坐标y  

   * - uint32_t
     - size
     - 坐标点个数    


cvai_4_pts_t
------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - x[4]
     - 4个坐标点之x坐标值   

   * - float
     - y[4]
     - 4个坐标点之y坐标值   


cvai_vpssconfig_t
-----------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - VPSS_SCALE_COEF_E
     - chn_coeff
     - Rescale方式  

   * - VPSS_CHN_ATTR_S
     - chn_attr
     - VPSS属性数据 


cvai_tracker_t
--------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t
     - size
     - 追踪讯息数量  

   * - cvai_tracker_info_t\*
     - info
     - 追踪讯息结构  


cvai_tracker_info_t
-------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - cvai_trk_state_type_t
     - state
     - 追踪状态     

   * - cvai_bbox_t
     - bbox
     - 追踪预测之Bounding Box


cvai_trk_state_type_t
---------------------

【enum】

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数值
     - 参数名称
     - 描述

   * - 0
     - CVI_TRACKER_NEW
     - 追踪状态为新增

   * - 1
     - CVI_TRACKER_UNSTABLE
     - 追踪状态为不稳定     

   * - 2
     - CVI_TRACKER_STABLE
     - 追踪状态为稳定


cvai_deepsort_config_t
----------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - max_distance_iou
     - 进行BB   ox匹配时最大IOU距离

   * - float
     - ma x_distance_consine
     - 进行Feature匹    配时最大consine距离

   * - int
     - max_unmatched_times_for_bbox_matching
     - 参与BBox匹配的目标最 大未匹配次数之数量

   * - cvai_kalman_filter_config_t
     - kfilter_conf
     - Kalman Filter设定 

   * - cvai_kalman_tracker_config_t
     - ktracker_conf
     - Kalman Tracker 设定


cvai_kalman_filter_config_t
---------------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float[8]
     - Q_std_alpha
     - Process Noise 参数 

   * - float[8]
     - Q_std_beta
     - Process Noise 参数 

   * - int[8]
     - Q_std_x_idx
     - Process Noise 参数 

   * - float[4]
     - R_std_alpha
     - Measurement Noise 参数    

   * - float[4]
     - R_std_beta
     - Measurement Noise 参数    

   * - int[4]
     - R_std_x_idx
     - Measurement Noise 参数    


【描述】

对于追踪目标运动状态X

Process Nose (运动偏差), Q, 其中 

:math:`Q\lbrack i\rbrack = \left( {Alpha}_{Q}\lbrack i\rbrack \bullet X\left\lbrack {Idx}_{Q}\lbrack i\rbrack \right\rbrack + {Beta}_{Q}\lbrack i\rbrack \right)^{2}`

Measurement Nose (量测偏差), R, 同理运动偏差公式

cvai_kalman_tracker_config_t
----------------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - int
     - max_unmatched_num
     - 追踪目标最大遗失数   

   * - int
     - acc reditation_threshold
     - 追踪状态转为稳定之阀值 

   * - int
     - feature_budget_size
     - 保存追踪目标feature之最大数量

   * - int
     - fea ture_update_interval
     - 更新feature之时间间距

   * - float[8]
     - P_std_alpha
     - Initial Covariance 参数

   * - float[8]
     - P_std_beta
     - Initial Covariance 参数

   * - int[8]
     - P_std_x_idx
     - Initial Covariance 参数


【描述】

Initial Covariance (初始运动状态偏差), P, 同理运动偏差公式

cvai_liveness_ir_position_e
---------------------------

【enum】

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数值
     - 参数名称
     - 描述

   * - 0
     - LIVENESS_IR_LEFT
     - IR镜头在RGB镜头左侧  

   * - 1
     - LIVENESS_IR_RIGHT
     - IR镜头在RGB镜头右侧  


cvai_head_pose_t
----------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - yaw
     - 偏摆角

   * - float
     - pitch
     - 俯仰角

   * - float
     - roll
     - 翻滚角

   * - float[3]
     - facialUnitNormalVector
     - 脸部之面向方位      


cvai_face_info_t
----------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - char
     - name[128]
     - 人脸名

   * - uint64_t
     - unique_id
     - 人脸ID

   * - cvai_bbox_t
     - bbox
     - 人脸侦测框   

   * - cvai_pts_t
     - pts
     - 人脸特征点   

   * - cvai_feature_t
     - feature
     - 人脸特征     

   * - cvai_face_emotion_e
     - emotion
     - 表情  

   * - cvai_face_gender_e
     - gender
     - 性别  

   * - cvai_face_race_e
     - race
     - 种族  

   * - float
     - age
     - 年龄  

   * - float
     - liveness_score
     - 活体机率值   

   * - float
     - mask_score
     - 人脸戴口罩机率值    

   * - float
     - face_quality
     - 人脸品质     

   * - cvai_head_pose_t
     - head_pose
     - 人脸角度信息 


cvai_face_t
-----------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t
     - size
     - 人脸个数     

   * - uint32_t
     - width
     - 原始图片之宽 

   * - uint32_t
     - height
     - 原始图片之高 

   * - cvai_face_info_t\*
     - info
     - 人脸综合信息 


cvai_pose17_meta_t
------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - x[17]
     - 17个骨骼关键点的x坐标

   * - float
     - y[17]
     - 17个骨骼关键点的y坐标

   * - float
     - score[17]
     - 17个骨骼关键点的预测信心值   


cvai_vehicle_meta
-----------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - cvai_4_pts_t
     - license_pts
     - 车牌4个角坐标

   * - cvai_bbox_t
     - license_bbox
     - 车牌Bounding Box    

   * - char[255]
     - license_char
     - 车牌号码     


【描述】

车牌4个角坐标依序为左上、右上、右下至左下。

cvai_class_filter_t
-------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t\*
     - preserved_class_ids
     - 要保留的类别id      

   * - uint32_t
     - num_preserved_classes
     - 要保留的类别id个数  


cvai_dms_t
----------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - reye_score
     - 右眼开合分数 

   * - float
     - leye_score
     - 左眼开合分数 

   * - float
     - yawn_score
     - 嘴巴闭合分数 

   * - float
     - phone_score
     - 讲电话分数   

   * - float
     - smoke_score
     - 抽烟分数     

   * - cvai_pts_t
     - landmarks_106
     - 106个特征点  

   * - cvai_pts_t
     - landmarks_5
     - 5个特征点    

   * - cvai_head_pose_t
     - head_pose
     - 透过106个特征点算出来的人脸角度

   * - cvai_dms_od_t
     - dms_od
     - 车内的物件侦测结果  


cvai_dms_od_t
-------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t
     - size
     - 有几个物件   

   * - uint32_t
     - width
     - 宽度  

   * - uint32_t
     - height
     - 长度  

   * - meta_rescale_type_e
     - rescale_type
     - rescale的形态

   * - cvai_dms_od_info_t\*
     - info
     - 物件的资讯   


cvai_dms_od_info_t
------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - char[128]
     - name
     - 物体名称     

   * - int
     - classes
     - 物体类别     

   * - cvai_bbox_t
     - bbox
     - 物体Bounding Box    


cvai_face_emotion_e
-------------------

【描述】

人脸表情Enmu

.. list-table::
   :widths: 2 1
   :header-rows: 1


   * - 表情    
     - 描述

   * - EMOTION_UNKNOWN
     - 未知    

   * - EMOTION_HAPPY
     - 高兴    

   * - EMOTION_SURPRISE
     - 惊讶    

   * - EMOTION_FEAR
     - 恐惧    

   * - EMOTION_DISGUST
     - 厌恶    

   * - EMOTION_SAD
     - 伤心    

   * - EMOTION_ANGER
     - 生气    

   * - EMOTION_NEUTRAL
     - 自然    


cvai_face_race_e
----------------

.. list-table::
   :widths: 2 1
   :header-rows: 1


   * - 种族    
     - 描述

   * - RACE_UNKNOWN
     - 未知    

   * - RACE_CAUCASIAN
     - 高加索人

   * - RACE_BLACK
     - 黑人    

   * - RACE_ASIAN
     - 亚洲人  


cvai_pedestrian_meta
--------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - cvai_pose17_meta_t
     - pose17
     - 人体17关键点   

   * - bool
     - fall
     - 受否跌倒


cvai_object_info_t
------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - char
     - name
     - 对象类别名    

   * - uint64_t
     - unique_id
     - Unique id

   * - cvai_box_t
     - bbox
     - Bounding box

   * - cvai_feature_t
     - feature
     - 对象特征      

   * - int
     - classes
     - 类别ID 

   * - cvai_vehicle_meta
     - vehicle_property
     - 车辆属性      

   * - cvai_pedestrian_meta
     - pedestrian_property
     - 行人属性      


cvai_object_t
-------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t
     - size
     - info所含物件个数    

   * - uint32_t
     - width
     - 原始图片之宽

   * - uint32_t
     - height
     - 原始图片之高

   * - meta_rescale_type_e
     - rescale_type
     - 模型  前处理采用的resize方式

   * - cvai_object_info_t\*
     - info
     - 物件信息    


CVI_AI_Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cvai_service_feature_matching_e
-------------------------------

【描述】

特征比对计算方法，目前仅支持Cosine Similarity。

【定义】

.. list-table::
   :widths: 2 1
   :header-rows: 1


   * - 参数名称 
     - 描述

   * - COS_SIMILARITY
     - Cosine similarity


cvai_service_feature_array_t
----------------------------

【描述】

特征数组，此结构包含了特征数组指针, 长度, 特征个数, 及特征类型等信息。在注册特征库时需要传入此结构。

【定义】

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - int8_t\*
     - ptr
     - 特征数组指针 

   * - uint32_t
     - feature_length
     - 单一特征长度 

   * - uint32_t
     - data_num
     - 特征个数   

   * - feature_type_e
     - type
     - 特征类型   


cvai_service_brush_t
--------------------

【描述】

绘图笔刷结构，可指定欲使用之RGB及笔刷大小。

【定义】

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - Inner structure
     - color
     - 欲使用的RGB值

   * - uint32_t
     - size
     - 笔刷大小   

