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
     - 人脸侦测(RetinaFace)

   * - CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR
     - 红外线人脸侦测(RetinaFace)

   * - CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT
     - 安全帽人脸检测(RetinaFace)

   * - CVI_AI_SUPPORTED_MODEL_SCRFDFACE
     - 人脸侦测(ScrFD Face)

   * - CVI_AI_SUPPORTED_MODEL_THERMALFACE
     - 热显人脸侦测    

   * - CVI_AI_SUPPORTED_MODEL_THERMALPERSON
     - 热显人体侦测   

   * - CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE
     - 人脸属性和人脸识别

   * - CVI_AI_SUPPORTED_MODEL_FACERECOGNITION
     - 人脸识别

   * - CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION
     - 戴口罩人脸识别  

   * - CVI_AI_SUPPORTED_MODEL_FACEQUALITY
     - 人脸质量

   * - CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION
     - 人脸口罩识别

   * - CVI_AI_SUPPORTED_MODEL_HANDCLASSIFICATION
     - 手势识别

   * - CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT
     - 手势关键点侦测

   * - CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION
     - 手势关键点识别

   * - CVI_AI_SUPPORTED_MODEL_LIVENESS
     - 双目活体识别    

   * - CVI_AI_SUPPORTED_MODEL_HAND_DETECTION
     - 手部侦测

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE
     - 人形及交通工具侦测    

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

   * - CVI_AI_SUPPORTED_MODEL_YOLOV5
     - 80类对象侦测    

   * - CVI_AI_SUPPORTED_MODEL_YOLOX
     - 80类对象侦测    

   * - CVI_AI_SUPPORTED_MODEL_OSNET
     - 行人重识别

   * - CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION
     - 声音识别

   * - CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2
     - 声音识别 V2

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

   * - CVI_AI_SUPPORTED_MODEL_FACELANDMARKERDET2
     - 人脸关键点侦测2

   * - CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION
     - 车内对象识别    

   * - CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION
     - 抽烟识别

   * - CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION
     - 口罩人脸侦测 

   * - CVI_AI_SUPPORTED_MODEL_IRLIVENESS
     - 红外线活体侦测

   * - CVI_AI_SUPPORTED_MODEL_PERSON_PETS_DETECTION
     - 人形及猫狗侦测

   * - CVI_AI_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION
     - 人形及车辆侦测

   * - CVI_AI_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION
     - 手部、脸及人型侦测

   * - CVI_AI_SUPPORTED_MODEL_HEAD_PERSON_DETECTION
     - 手部及人型侦测

   * - CVI_AI_SUPPORTED_MODEL_YOLOV8POSE
     - 姿态侦测

   * - CVI_AI_SUPPORTED_MODEL_SIMCC_POSE
     - 姿态侦测

   * - CVI_AI_SUPPORTED_MODEL_LANDMARK_DET3
     - 人脸关键点侦测

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

   * - CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR
     - CVI_AI_RetinaFace_IR
     - retinafaceIR_mnet0.25_342_608.cvimodel

       retinafaceIR_mnet0.25_608_342.cvimodel

       retinafaceIR_mnet0.25_608_608.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT
     - CVI_AI_RetinaFace_Hardhat
     - hardhat_720_1280.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SCRFDFACE
     - CVI_AI_ScrFDFace
     - scrfd_320_256_ir.cvimodel

       scrfd_480_270_int8.cvimodel

       scrfd_480_360_int8.cvimodel

       scrfd_500m_bnkps_432_768.cvimodel

       scrfd_768_432_int8_1x.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_THERMALFACE
     - CVI_AI_ThermalFace
     - thermalfd-v1.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_THERMALPERSON
     - CVI_AI_ThermalPerson
     - thermal_person_detection.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE
     - CVI_AI_FaceAttribute  CVI_AI_FaceAttributeOne
     - cviface-v3-attribute.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACERECOGNITION
     - CVI_AI_FaceRecognition  CVI_AI_FaceRecognitionOne
     - cviface-v4.cvimodel

       cviface-v5-m.cvimodel

       cviface-v5-s.cvimodel

       cviface-v6-s.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION
     - CVI_AI_MaskFaceRecognition
     - masked-fr-v1-m.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_FACEQUALITY
     - CVI_AI_FaceQuality
     - fqnet-v5_shufflenetv2-softmax.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION
     - CVI_AI_MaskClassification
     - mask_classifier.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_HANDCLASSIFICATION
     - CVI_AI_HandClassification
     - hand_cls_128x128.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT
     - CVI_AI_HandKeypoint
     - hand_kpt_128x128.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION
     - CVI_AI_HandKeypointClassification
     - hand_kpt_cls9.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_LIVENESS
     - CVI_AI_Liveness
     - liveness-rgb-ir.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_HAND_DETECTION
     - CVI_AI_Hand_Detection
     - hand_det_qat_640x384.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE
     - CVI_AI_MobileDetV2_Vehicle
     - mobiledetv2-vehicle-d0-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN
     - CVI_AI_MobileDetV2_Pedestrian
     - mobiledetv2-pedestrian-d0-ls-384.cvimodel

       mobiledetv2-pedestrian-d0-ls-640.cvimodel

       mobiledetv2-pedestrian-d0-ls-768.cvimodel

       mobileDetV2-pedestrian-d1-ls.cvimodel

       mobiledetv2-pedestrian-d1-ls-1024.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE
     - CVI_AI_MobileDetV2_Person_Vehicle
     - mobiledetv2-person-vehicle-ls-768.cvimodel

       mobiledetv2-person-vehicle-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS
     - CVI_AI_MobileDetV2_Person_Pets
     - mobiledetv2-lite-person-pets-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80
     - CVI_AI_MobileDetV2_COCO80
     - mobiledetv2-d0-ls.cvimodel

       mobiledetv2-d1-ls.cvimodel

       mobiledetv2-d2-ls.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YOLOV3
     - CVI_AI_Yolov3
     - yolo_v3_416.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YOLOV5
     - CVI_AI_Yolov5
     - yolov5s_3_branch_int8.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YOLOX
     - CVI_AI_YoloX
     - yolox_nano.cvimodel

       yolox_tiny.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_OSNET
     - CVI_AI_OSNet  CVI_AI_OSNetOne
     - person-reid-v1.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION
     - CVI_AI_SoundClassification
     - es_classification.cvimodel

       soundcmd_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SOUNDCLASSIFICATION_V2
     - CVI_AI_SoundClassification_V2
     - c10_lightv2_mse40_mix.cvimodel

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

   * - CVI_AI_SUPPORTED_MODEL_FACELANDMARKERDET2 
     - CVI_AI_FaceLandmarkerDet2
     - pipnet_blurness_v5_64_retinaface_50ep.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION
     - CVI_AI_IncarObjectDetection
     - incar_od_v0_bf16.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SMOKECLASSIFICATION
     - CVI_AI_SmokeClassification
     - N/A

   * - CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION
     - CVI_AI_FaceMaskDetection
     - retinaface_yolox_fdmask.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_IRLIVENESS
     - CVI_AI_IrLiveness
     - liveness-rgb-ir.cvimodel 

       liveness-rgb-ir-3d.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_PERSON_PETS_DETECTION
     - CVI_AI_PersonPet_Detection
     - pet_det_640x384.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION
     - CVI_AI_PersonVehicle_Detection
     - yolov8n_384_640_person_vehicle.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION
     - CVI_AI_HandFacePerson_Detection
     - meeting_det_640x384.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_HEAD_PERSON_DETECTION
     - CVI_AI_HeadPerson_Detection
     - yolov8n_headperson.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_YOLOV8POSE
     - CVI_AI_Yolov8_Pose
     - yolov8n_pose_384_640.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_SIMCC_POSE
     - CVI_AI_Simcc_Pose
     - simcc_mv2_pose.cvimodel

   * - CVI_AI_SUPPORTED_MODEL_LANDMARK_DET3
     - CVI_AI_FLDet3
     - onet_int8.cvimodel


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

   * - CVI_AI_DET_GROUP_MASK_HEAD
     - 自订类别

   * - CVI_AI_DET_GROUP_MASK_START
     - 自订类别开始

   * - CVI_AI_DET_GROUP_MASK_END
     - 自订类别结束


feature_type_e
--------------

【enum】

.. list-table::
   :widths: 1 1 2
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
   :widths: 1 1 2
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
     - 追踪预测之边界框


cvai_trk_state_type_t
---------------------

【enum】

.. list-table::
   :widths: 1 1 2
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
   :widths: 1 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - max_distance_iou
     - 进行BBox匹配时最大IOU距离

   * - float
     - ma x_distance_consine
     - 进行Feature匹配时最大consine距离

   * - int
     - max_unmatched_times_for_bbox_matching
     - 参与BBox匹配的目标最大未匹配次数之数量

   * - bool
     - enable_internal_FQ
     - 启用内部特征品质

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

   * - bool
     - enable_X_constraint_0
     - 启用第 0 个 X 约束

   * - bool
     - enable_X_constraint_1
     - 启用第 1 个 X 约束

   * - float
     - X_constraint_min[8]
     - X 约束下限

   * - float
     - X_constraint_max[8]
     - X 约束上限

   * - bool
     - enable_bounding_stay
     - 保留边界

   * - mahalanobis_confidence_e
     - confidence_level
     - 马氏距离信心度

   * - float
     - chi2_threshold
     - 卡方阈值

   * - float
     - Q_std_alpha[8]
     - Process Noise 参数 

   * - float
     - Q_std_beta[8]
     - Process Noise 参数 

   * - int
     - Q_std_x_idx[8]
     - Process Noise 参数 

   * - float
     - R_std_alpha[4]
     - Measurement Noise 参数    

   * - float
     - R_std_beta[4]
     - Measurement Noise 参数    

   * - int
     - R_std_x_idx[4]
     - Measurement Noise 参数    


【描述】

对于追踪目标运动状态X

Process Nose (运动偏差), Q, 其中 

:math:`Q\lbrack i\rbrack = \left( {Alpha}_{Q}\lbrack i\rbrack \bullet X\left\lbrack {Idx}_{Q}\lbrack i\rbrack \right\rbrack + {Beta}_{Q}\lbrack i\rbrack \right)^{2}`

Measurement Nose (量测偏差), R, 同理运动偏差公式

cvai_kalman_tracker_config_t
----------------------------

.. list-table::
   :widths: 1 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - int
     - max_unmatched_num
     - 追踪目标最大遗失数   

   * - int
     - accreditation_threshold
     - 追踪状态转为稳定之阀值 

   * - int
     - feature_budget_size
     - 保存追踪目标feature之最大数量

   * - int
     - feature_update_interval
     - 更新feature之时间间距

   * - bool
     - enable_QA_feature_init
     - 启用 QA 特征初始化

   * - bool
     - enable_QA_feature_update
     - 启用 QA 特征更新

   * - float
     - feature_init_quality_threshold
     - 特征初始化品质阈值

   * - float
     - feature_update_quality_threshold
     - 特征更新品质阈值

   * - float
     - P_std_alpha[8]
     - Initial Covariance 参数

   * - float
     - P_std_beta[8]
     - Initial Covariance 参数

   * - int
     - P_std_x_idx[8]
     - Initial Covariance 参数


【描述】

Initial Covariance (初始运动状态偏差), P, 同理运动偏差公式

cvai_liveness_ir_position_e
---------------------------

【enum】

.. list-table::
   :widths: 1 1 2
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

   * - float
     - facialUnitNormalVector[3]
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
     - score
     - 分数

   * - float
     - age
     - 年龄  

   * - float
     - liveness_score
     - 活体机率值   

   * - float
     - hardhat_score
     - 安全帽机率值   

   * - float
     - mask_score
     - 人脸戴口罩机率值

   * - float
     - recog_score
     - 识别分数

   * - float
     - face_quality
     - 人脸品质

   * - float
     - pose_score
     - 姿势分数

   * - float
     - pose_score1
     - 姿势分数

   * - float
     - sharpness_score
     - 清晰度分数

   * - float
     - blurness
     - 模糊性

   * - cvai_head_pose_t
     - head_pose
     - 人脸角度信息

   * - int
     - track_state
     - 追踪状态


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

   * - meta_rescale_type_e\*
     - rescale_type
     - rescale的形态

   * - cvai_face_info_t\*
     - info
     - 人脸综合信息 

   * - cvai_dms_t\*
     - dms
     - 駕駛综合信息 


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
     - 车牌边界框   

   * - char[125]
     - license_char
     - 车牌号码     


【描述】

车牌4个角坐标依序为左上、右上、右下至左下。

cvai_class_filter_t
-------------------

.. list-table::
   :widths: 1 1 2
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
     - 物体边界框


cvai_face_emotion_e
-------------------

【描述】

人脸表情

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
     - 唯一 id

   * - cvai_box_t
     - bbox
     - 框的边界讯息

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

   * - int
     - track_state
     - 追踪状态


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

   * - uint32_t
     - entry_num
     - entry数量

   * - uint32_t
     - miss_num
     - miss数量

   * - meta_rescale_type_e
     - rescale_type
     - 模型前处理采用的resize方式

   * - cvai_object_info_t\*
     - info
     - 物件信息    


cvai_handpose21_meta_t
----------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - xn[21]
     - 归一化 x 点 

   * - float
     - x[21]
     - x 点

   * - float
     - yn[21]
     - 归一化 y 点

   * - float
     - y[21]
     - y 点

   * - float
     - bbox_x
     - 框的x 座标

   * - float
     - bbox_y
     - 框的y 座标

   * - float
     - bbox_w
     - 框的宽

   * - float
     - bbox_h
     - 框的高

   * - int
     - label
     - 手势类别

   * - float
     - score
     - 手势分数


cvai_handpose21_meta_ts
-----------------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t
     - size
     - 侦测到手的数量

   * - uint32_t
     - width
     - 图片宽

   * - uint32_t
     - height
     - 图片高

   * - cvai_handpose21_meta_t\*
     - info
     - 手部关键点


Yolov5PreParam
--------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - float
     - factor[3]
     - 缩放因子

   * - float
     - mean[3]
     - 图像均值

   * - meta_rescale_type_e
     - rescale_type
     - 缩放模式

   * - bool\*
     - pad_reverse
     - 反向填充

   * - bool\*
     - keep_aspect_ratio
     - 保持宽高比例缩放

   * - bool\*
     - use_quantize_scale
     - 量化缩放

   * - bool\*
     - use_crop
     - 裁剪调整图像大小

   * - VPSS_SCALE_COEF_E\*
     - resize_method
     - 缩放方法

   * - PIXEL_FORMAT_E\*
     - format
     - 图像格式


YOLOV5AlgParam
--------------

.. list-table::
   :widths: 2 1 2
   :header-rows: 1


   * - 数据类型
     - 参数名称
     - 描述

   * - uint32_t
     - anchors[3][3][2]
     - 模型錨點

   * - float
     - conf_thresh
     - 信心度阀值

   * - float
     - nms_thresh
     - 均方根阀值


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


cvai_area_detect_e
------------------

【enum】

.. list-table::
   :widths: 1 1 2
   :header-rows: 1


   * - 数值
     - 参数名称
     - 描述

   * - 0
     - UNKNOWN
     - int8_t特征类型

   * - 1
     - NO_INTERSECT
     - 不相交

   * - 2
     - ON_LINE
     - 在线上

   * - 3
     - CROSS_LINE_POS
     - 正向交叉

   * - 4
     - CROSS_LINE_NEG
     - 负向交叉

   * - 5
     - INSIDE_POLYGON
     - 在多边形内部

   * - 6
     - OUTSIDE_POLYGON
     - 在多边形外部
