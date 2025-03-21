#ifndef TDL_MODEL_DEFS_H
#define TDL_MODEL_DEFS_H

enum class ModelType {
  INVALID = -1,

  // object detection models
  MBV2_DET_PERSON,
  YOLOV5_DET_COCO80,
  YOLOV6_DET_COCO80,
  YOLOV8_DET_COCO80,
  YOLOV10_DET_COCO80,
  YOLOV8N_DET_HAND,
  YOLOV8N_DET_PET_PERSON,        // 0:cat,1:dog,2:person
  YOLOV8N_DET_PERSON_VEHICLE,    // 0:car,1:bus,2:truck,3:rider with
                             // motorcycle,4:person,5:bike,6:motorcycle
  YOLOV8N_DET_HAND_FACE_PERSON,  // 0:hand,1:face,2:person
  YOLOV8N_DET_HEAD_PERSON,       // 0:person,1:head
  YOLOV8N_DET_HEAD_HARDHAT,      // 0:head,1:hardhat
  YOLOV8N_DET_FIRRE_SMOKE,       // 0:fire,1:smoke
  YOLOV8N_DET_FIRE,              // 0:fire
  YOLOV8N_DET_HEAD_SHOULDER,     // 0:head shoulder
  YOLOV8N_DET_LICENSE_PLATE,     // 0:license plate
  YOLOV8N_DET_TRAFFIC_LIGHT,     // 0:red,1:yellow,2:green,3:off,4:wait on

  // face detection model
  SCRFD_DET_FACE,  // 0:face + landm
  RETINA_DET_FACE,
  RETINA_DET_FACE_IR,
  KEYPOINT_FACE_V2,  // 5 landmarks + blurness score
  CLS_ATTRIBUTE_FACE,    // age,gener,glass,mask
  FEATURE_BMFACER34,  // resnet34 512 dim feature
  FEATURE_BMFACER50,  // resnet50 512 dim feature

  // image classification models
  CLS_MASK,         // 0:mask,1:no mask
  CLS_RGBLIVENESS,  // 0:live,1:fake
  CLS_ISP_SCENE,
  CLS_HAND_GESTURE,  // 0:fist,1:five,2:none,3:two
  CLS_KEYPOINT_HAND_GESTURE,  // 0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two

  // sound classification models
  CLS_SOUND_BABAY_CRY,  // 0:background,1:cry
  CLS_SOUND_COMMAND,     // 0:background,1:command1,2:command2 ...

  // image keypoint models
  KEYPOINT_LICENSE_PLATE,
  KEYPOINT_HAND,
  KEYPOINT_YOLOV8POSE_PERSON17,  // 17 keypoints for person
  KEYPOINT_SIMCC_PERSON17,

  // lane detection models
  LSTR_DET_LANE,

  // license plate recognition models
  RECOGNITION_LICENSE_PLATE,

  // image segmentation models
  YOLOV8_SEG_COCO80,
  TOPFORMER_SEG_PERSON_FACE_VEHICLE,  // 0:background,1:person,2:face,3:vehicle,4:license plate
  TOPFORMER_SEG_MOTION,  // 0:static,2:transsition 3:motion

  // LLM models
  CLIP_FEATURE_IMG,
  CLIP_FEATURE_TEXT,

  // custom models,user should specify necessary params like preprocess and
  // input/output and parse parameters

  // object detection models
  YOLOV8,          // custom model,should specify num_cls(number of classes)
  YOLOV10,         // custom model,should specify num_cls(number of classes)
  YOLOV3,          // custom model,should specify num_cls(number of classes)
  YOLOV5,          // custom model,should specify num_cls(number of classes)
  YOLOV6,          // custom model,should specify num_cls(number of classes)

  CLS_IMG,    // should specify rgb order and rgb mean /rgb std
  CLS_SOUND,  // should specify sample rate and channel num

};

#endif
