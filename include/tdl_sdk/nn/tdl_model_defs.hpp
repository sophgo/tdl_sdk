#ifndef TDL_MODEL_DEFS_H
#define TDL_MODEL_DEFS_H

enum class ModelType {
  INVALID = 0,

  // object detection models
  MBV2_PERSON,
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

  // image classification models
  CLS_MASK,         // 0:mask,1:no mask
  CLS_RGBLIVENESS,  // 0:live,1:fake
  CLS_ISPSCENE,
  CLS_HAND_GESTURE,  // 0:fist,1:five,2:none,3:two

  KEYPOINT_CLS_HAND_GESTURE,  // 0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two

  // image attribute models
  ATTRIBUTE_FACE,  // age,gener,glass,mask

  // sound classification models
  SOUND_CLS_BABAY_CRY,  // 0:cry
  SOUND_CLS_COMMAND,

  // image keypoint models
  KEYPOINT_FACE_V2,  // 5 landmarks + blurness score

  KEYPOINT_HAND,
  KEYPOINT_YOLOV8N_POSE_V1,
  KEYPOINT_SIMCC,

  // lane detection models
  LANE_DETECTION_LSTR,

  // image feature models
  FEATURE_BMFACER34,
  IMG_FEATURE_CLIP,
  TEXT_FEATURE_CLIP,

  // image segmentation models
  SEG_TOP_FROMER_PERSON_FACE_VEHICLE,  // 0:background,1:person,2:face,3:vehicle,4:license
                                       // plate
  SEG_MOTION,  // 0:static,2:transsition 3:motion

  // custom models,user should specify necessary params like preprocess and
  // input/output and parse parameters

  // object detection models
  YOLOV8,          // custom model,should specify num_cls(number of classes)
  YOLOV8_COCO80,   // number of classes is 80
  YOLOV10,         // custom model,should specify num_cls(number of classes)
  YOLOV10_COCO80,  // number of classes is 80
  YOLOV3,          // custom model,should specify num_cls(number of classes)
  YOLOV5,          // custom model,should specify num_cls(number of classes)
  YOLOV6,          // custom model,should specify num_cls(number of classes)
  YOLOV6_COCO80,   // number of classes is 80

  IMG_CLS,    // should specify rgb order and rgb mean /rgb std
  SOUND_CLS,  // should specify sample rate and channel num

  // image keypoint models
  YOLOV8_POSE,
  YOLOV8_POSE_PERSON17,  // 17 keypoints for person
  // image feature models
  IMG_FEATURE,  // should specify rgb order and rgb mean /rgb std

  TEXT_FEATURE,  // should specify text encoding type

  // segmentation models
  YOLOV8_SEG,
  YOLOV8_SEG_COCO80,
};

#endif
