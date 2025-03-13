#ifndef TDL_MODEL_DEF_H
#define TDL_MODEL_DEF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TDL_MODEL_UNKOWN = 0,
  TDL_MODEL_MBV2_PERSON,             // 0:person
  TDL_MODEL_YOLOV8N_HAND,            // 0:hand
  TDL_MODEL_YOLOV8N_PET_PERSON,      // 0:cat,1:dog,2:person
  TDL_MODEL_YOLOV8N_PERSON_VEHICLE,  // 0:car,1:bus,2:truck,3:rider with
                                     // motorcycle,4:person,5:bike,6:motorcycle
  TDL_MODEL_YOLOV8N_HAND_FACE_PERSON,  // 0:hand,1:face,2:person
  TDL_MODEL_YOLOV8N_HEAD_PERSON,       // 0:person,1:head
  TDL_MODEL_YOLOV8N_HEAD_HARDHAT,      // 0:head,1:hardhat
  TDL_MODEL_YOLOV8N_FIRRE_SMOKE,       // 0:fire,1:smoke
  TDL_MODEL_YOLOV8N_FIRE,              // 0:fire
  TDL_MODEL_YOLOV8N_HEAD_SHOULDER,     // 0:head shoulder
  TDL_MODEL_YOLOV8N_LICENSE_PLATE,     // 0:license plate
  TDL_MODEL_YOLOV8N_TRAFFIC_LIGHT,     // 0:red,1:yellow,2:green,3:off,4:wait on

  TDL_MODEL_SCRFD_FACE,  // 0:face + landm
  TDL_MODEL_RETINA_FACE,
  TDL_MODEL_RETINA_FACE_IR,

  TDL_MODEL_CLS_MASK,         // 0:mask,1:no mask
  TDL_MODEL_CLS_RGBLIVENESS,  // 0:live,1:fake
  TDL_MODEL_CLS_ISPSCENE,
  TDL_MODEL_CLS_HAND_GESTURE,  // 0:fist,1:five,2:none,3:two
  TDL_MODEL_KEYPOINT_CLS_HAND_GESTURE,  // 0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two
  TDL_MODEL_ATTRIBUTE_FACE,  // age,gener,glass,mask
  TDL_MODEL_CLS_BABAY_CRY,   // 0:cry
  TDL_MODEL_CLS_NI,          // 0:nihao suanneng,1:清空缓存

  TDL_MODEL_KEYPOINT_FACE_V2 = 100,  // 5 landmarks + blurness score
  TDL_MODEL_KEYPOINT_HAND,
  TDL_MODEL_KEYPOINT_YOLOV8N_POSE_V1,
  TDL_MODEL_FEATURE_BMFACER34,  // resnet34 512 dim feature
  TDL_MODEL_IMG_FEATURE_CLIP,
  TDL_MODEL_TEXT_FEATURE_CLIP,

  TDL_MODEL_SEG_PERSON_FACE_VEHICLE =
      130,  // 0:background,1:person,2:face,3:vehicle,4:license plate
  TDL_MODEL_SEG_MOTION,  // 0:static,2:transsition 3:motion

} cvtdl_model_e;

#ifdef __cplusplus
}
#endif

#endif
