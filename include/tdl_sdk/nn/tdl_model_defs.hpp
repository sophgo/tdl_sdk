#ifndef TDL_MODEL_DEFS_H
#define TDL_MODEL_DEFS_H

enum class ModelType {
  INVALID = -1,

  // object detection models
  MBV2_PERSON,
  YOLOV5_COCO80,
  YOLOV6_COCO80,
  YOLOV8_COCO80,
  YOLOV10_COCO80,
  YOLOV8N_HAND,
  YOLOV8N_PET_PERSON,        // 0:cat,1:dog,2:person
  YOLOV8N_PERSON_VEHICLE,    // 0:car,1:bus,2:truck,3:rider with
                             // motorcycle,4:person,5:bike,6:motorcycle
  YOLOV8N_HAND_FACE_PERSON,  // 0:hand,1:face,2:person
  YOLOV8N_HEAD_PERSON,       // 0:person,1:head
  YOLOV8N_HEAD_HARDHAT,      // 0:head,1:hardhat
  YOLOV8N_FIRRE_SMOKE,       // 0:fire,1:smoke
  YOLOV8N_FIRE,              // 0:fire
  YOLOV8N_HEAD_SHOULDER,     // 0:head shoulder
  YOLOV8N_LICENSE_PLATE,     // 0:license plate
  YOLOV8N_TRAFFIC_LIGHT,     // 0:red,1:yellow,2:green,3:off,4:wait on

  // face detection model
  SCRFD_FACE,  // 0:face + landm
  RETINA_FACE,
  RETINA_FACE_IR,
  KEYPOINT_FACE_V2,  // 5 landmarks + blurness score
  ATTRIBUTE_FACE,    // age,gener,glass,mask
  FEATURE_BMFACER34,  // resnet34 512 dim feature
  FEATURE_BMFACER50,  // resnet50 512 dim feature

  // image classification models
  CLS_MASK,         // 0:mask,1:no mask
  CLS_RGBLIVENESS,  // 0:live,1:fake
  CLS_ISP_SCENE,
  CLS_HAND_GESTURE,  // 0:fist,1:five,2:none,3:two
  KEYPOINT_CLS_HAND_GESTURE,  // 0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two

  // sound classification models
  SOUND_CLS_BABAY_CRY,  // 0:cry
  SOUND_CLS_COMMAND,
  SOUND_CLS_NI,  // 0:nihao suanneng,1:清空缓存

  // image keypoint models
  KEYPOINT_LICENSE_PLATE,
  KEYPOINT_HAND,
  KEYPOINT_YOLOV8N_POSE_V1,
  YOLOV8_POSE_PERSON17,  // 17 keypoints for person
  KEYPOINT_SIMCC,

  // lane detection models
  LANE_DETECTION_LSTR,

  // license plate recognition models
  LICENSE_PLATE_RECOGNITION,

  // image segmentation models
  SEG_YOLOV8_COCO80,
  SEG_PERSON_FACE_VEHICLE,  // 0:background,1:person,2:face,3:vehicle,4:license plate
  SEG_MOTION,  // 0:static,2:transsition 3:motion

  // LLM models
  IMG_FEATURE_CLIP,
  TEXT_FEATURE_CLIP,

  // custom models,user should specify necessary params like preprocess and
  // input/output and parse parameters

  // object detection models
  YOLOV8,          // custom model,should specify num_cls(number of classes)
  YOLOV10,         // custom model,should specify num_cls(number of classes)
  YOLOV3,          // custom model,should specify num_cls(number of classes)
  YOLOV5,          // custom model,should specify num_cls(number of classes)
  YOLOV6,          // custom model,should specify num_cls(number of classes)

  IMG_CLS,    // should specify rgb order and rgb mean /rgb std
  SOUND_CLS,  // should specify sample rate and channel num

};

#endif
