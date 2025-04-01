#ifndef TDL_MODEL_DEF_H
#define TDL_MODEL_DEF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {

  // detection model list
  TDL_DETECTION_MODEL = 0,
  TDL_MODEL_MBV2_DET_PERSON,  // 0:person
  TDL_MODEL_YOLOV5_DET_COCO80,
  TDL_MODEL_YOLOV8_DET_COCO80,
  TDL_MODEL_YOLOV10_DET_COCO80,
  TDL_MODEL_YOLOV8N_DET_HAND,            // 0:hand
  TDL_MODEL_YOLOV8N_DET_PET_PERSON,      // 0:cat,1:dog,2:person
  TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE,  // 0:car,1:bus,2:truck,3:rider with
                                         // motorcycle,4:person,5:bike,6:motorcycle
  TDL_MODEL_YOLOV8N_DET_HAND_FACE_PERSON,  // 0:hand,1:face,2:person
  TDL_MODEL_YOLOV8N_DET_HEAD_PERSON,       // 0:person,1:head
  TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT,      // 0:head,1:hardhat
  TDL_MODEL_YOLOV8N_DET_FIRE_SMOKE,        // 0:fire,1:smoke
  TDL_MODEL_YOLOV8N_DET_FIRE,              // 0:fire
  TDL_MODEL_YOLOV8N_DET_HEAD_SHOULDER,     // 0:head shoulder
  TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE,     // 0:license plate
  TDL_MODEL_YOLOV8N_DET_TRAFFIC_LIGHT,   // 0:red,1:yellow,2:green,3:off,4:wait
                                         // on
  TDL_MODEL_YOLOV8N_DET_MONITOR_PERSON,  // 0:person
  TDL_DETECTION_MODEL_MAX,

  // face model list
  TDL_FACE_MODEL = 100,
  TDL_MODEL_SCRFD_DET_FACE,  // 0:face + landm
  TDL_MODEL_RETINA_DET_FACE,
  TDL_MODEL_RETINA_DET_FACE_IR,
  TDL_MODEL_KEYPOINT_FACE_V2,    // 5 landmarks + blurness score
  TDL_MODEL_CLS_ATTRIBUTE_FACE,  // age,gener,glass,mask
  TDL_MODEL_FEATURE_BMFACER34,   // resnet34 512 dim feature
  TDL_MODEL_FEATURE_BMFACER50,   // resnet50 512 dim feature
  TDL_FACE_MODEL_MAX,

  // classification model list
  TDL_CLS_MODEL = 200,
  TDL_MODEL_CLS_MASK,         // 0:mask,1:no mask
  TDL_MODEL_CLS_RGBLIVENESS,  // 0:fake,1:live
  TDL_MODEL_CLS_ISP_SCENE,
  TDL_MODEL_CLS_HAND_GESTURE,  // 0:fist,1:five,2:none,3:two
  TDL_MODEL_CLS_KEYPOINT_HAND_GESTURE,  // 0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two
  TDL_MODEL_CLS_BABAY_CRY,  // 0:cry
  TDL_MODEL_CLS_SOUND_COMMAND,
  TDL_MODEL_CLS_NI,  // 0:nihao suanneng,1:清空缓存
  TDL_CLS_MODEL_MAX,

  // keypoint model list
  TDL_KEYPOINT_MODEL = 300,
  TDL_MODEL_KEYPOINT_LICENSE_PLATE,
  TDL_MODEL_KEYPOINT_HAND,
  TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17,
  TDL_MODEL_KEYPOINT_SIMICC,
  TDL_MODEL_LSTR_DET_LANE,
  TDL_KEYPOINT_MODEL_MAX,

  // recogntion model list
  TDL_RECOGNITION_MODEL = 400,
  TDL_MODEL_RECOGNITION_LICENSE_PLATE,
  TDL_REGOGNITION_MODEL_MAX,

  // segmentation model list
  TDL_SEGMENTATION_MODEL = 500,
  TDL_MODEL_SEG_YOLOV8_COCO80,
  TDL_MODEL_SEG_PERSON_FACE_VEHICLE,  // 0:background,1:person,2:face,3:vehicle,4:license
                                      // plate
  TDL_MODEL_SEG_MOTION,  // 0:static,2:transsition 3:motion
  TDL_SEGMENTATION_MODEL_MAX,

  // depth estimation model list
  TDL_DEPTH_MODEL = 600,
  TDL_MODEL_DEPTHANYTHING,  // todo
  TDL_DEPTH_MODEL_MAX,

  // LLM model list
  TDL_LLM_MODEL = 700,
  TDL_MODEL_IMG_FEATURE_CLIP,
  TDL_MODEL_TEXT_FEATURE_CLIP,
  TDL_LLM_MODEL_MAX,


} TDLModel;

#ifdef __cplusplus
}
#endif

#endif
