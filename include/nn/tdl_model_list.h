#ifndef TDL_MODEL_LIST_H
#define TDL_MODEL_LIST_H

#define MODEL_TYPE_LIST                                                       \
  X(MBV2_DET_PERSON, "0:person")                                              \
  X(YOLOV8N_DET_HAND, "")                                                     \
  X(YOLOV8N_DET_PET_PERSON, "0:cat,1:dog,2:person")                           \
  X(YOLOV8N_DET_PERSON_VEHICLE,                                               \
    "0:car,1:bus,2:truck,3:rider with "                                       \
    "motorcycle,4:person,5:bike,6:motorcycle")                                \
  X(YOLOV8N_DET_HAND_FACE_PERSON, "0:hand,1:face,2:person")                   \
  X(YOLOV8N_DET_FACE_HEAD_PERSON_PET, "0:face,1:head,2:person,3:pet")         \
  X(YOLOV8N_DET_HEAD_PERSON, "0:person,1:head")                               \
  X(YOLOV8N_DET_HEAD_HARDHAT, "0:head,1:hardhat")                             \
  X(YOLOV8N_DET_FIRE_SMOKE, "0:fire,1:smoke")                                 \
  X(YOLOV8N_DET_FIRE, "0:fire")                                               \
  X(YOLOV8N_DET_HEAD_SHOULDER, "0:head shoulder")                             \
  X(YOLOV8N_DET_LICENSE_PLATE, "0:license plate")                             \
  X(YOLOV8N_DET_TRAFFIC_LIGHT, "0:red,1:yellow,2:green,3:off,4:wait on")      \
  X(YOLOV8N_DET_MONITOR_PERSON, "0:person")                                   \
  /* coco 80 classes object detection models */                               \
  X(YOLOV5_DET_COCO80, "")                                                    \
                                                                              \
  X(YOLOV6_DET_COCO80, "")                                                    \
  X(YOLOV7_DET_COCO80, "")                                                    \
  X(YOLOV8_DET_COCO80, "")                                                    \
  X(YOLOV10_DET_COCO80, "")                                                   \
  X(PPYOLOE_DET_COCO80, "")                                                   \
  X(YOLOX_DET_COCO80, "")                                                     \
  /* custom object detection (need set num_cls) */                            \
  X(YOLOV5, "custom model, specify num_cls")                                  \
  X(YOLOV6, "custom model, specify num_cls")                                  \
  X(YOLOV7, "custom model, specify num_cls")                                  \
  X(YOLOV8, "custom model, specify num_cls")                                  \
  X(YOLOV10, "custom model, specify num_cls")                                 \
  X(PPYOLOE, "custom model, specify num_cls")                                 \
  X(YOLOX, "custom model, specify num_cls")                                   \
                                                                              \
  /* face detection */                                                        \
  X(SCRFD_DET_FACE, "0:face")                                                 \
  X(RETINA_DET_FACE, "")                                                      \
  X(RETINA_DET_FACE_IR, "")                                                   \
                                                                              \
  /* face attribute & keypoint */                                             \
  X(KEYPOINT_FACE_V2, "0:face")                                               \
  X(CLS_ATTRIBUTE_FACE, "0:age,1:gender,2:glass")                             \
                                                                              \
  /* image classification */                                                  \
  X(CLS_MASK, "0:mask,1:no mask")                                             \
  X(CLS_RGBLIVENESS, "0:fake,1:live")                                         \
  X(CLS_ISP_SCENE, "0:snow, 1:fog, 2:backlight, 3:grass, 4:common")           \
  X(CLS_HAND_GESTURE, "0:fist,1:five,2:none,3:two")                           \
  X(CLS_KEYPOINT_HAND_GESTURE,                                                \
    "0:fist,1:five,2:four,3:none,4:ok,5:one,6:three,7:three2,8:two")          \
                                                                              \
  /* sound classification */                                                  \
  X(CLS_SOUND_BABAY_CRY, "0:background,1:cry")                                \
  X(CLS_SOUND_COMMAND_NIHAOSHIYUN, "0:background,1:nihaoshiyun")              \
  X(CLS_SOUND_COMMAND_NIHAOSUANNENG, "0:background,1:nihaosuanneng")          \
  X(CLS_SOUND_COMMAND_XIAOAIXIAOAI, "0:background,1:nihaosuanneng")           \
  X(CLS_SOUND_COMMAND, "0:background,1:command1,2:command2")                  \
                                                                              \
  /* custom classification */                                                 \
  X(CLS_IMG, "custom classification, specify rgb order and rgb mean/std")     \
                                                                              \
  /* keypoint models */                                                       \
  X(KEYPOINT_LICENSE_PLATE, "output 4 license plate keypoints")               \
  X(KEYPOINT_HAND, "output 21 hand keypoints")                                \
  X(KEYPOINT_YOLOV8POSE_PERSON17, "output 17 person keypoints and box")       \
  X(KEYPOINT_SIMCC_PERSON17, "output 17 person keypoints from cropped image") \
                                                                              \
  /* lane detection */                                                        \
  X(LSTR_DET_LANE, "output lane keypoints")                                   \
                                                                              \
  /* license plate recognition */                                             \
  X(RECOGNITION_LICENSE_PLATE, "output 7 license plate characters")           \
                                                                              \
  /* segmentation models */                                                   \
  X(YOLOV8_SEG, "custom segmentation")                                        \
  X(YOLOV8_SEG_COCO80, "output 80 segmentation mask")                         \
  X(TOPFORMER_SEG_PERSON_FACE_VEHICLE,                                        \
    "0:background,1:person,2:face,3:vehicle,4:license plate")                 \
  X(TOPFORMER_SEG_MOTION, "0:static,2:transition,3:motion")                   \
                                                                              \
  /* image feature & multimodal */                                            \
  X(FEATURE_IMG, "output image feature vector")                               \
  X(FEATURE_CLIP_IMG, "output image clip feature vector")                     \
  X(FEATURE_CLIP_TEXT, "output text clip feature vector")                     \
                                                                              \
  /* face feature extraction */                                               \
  X(FEATURE_CVIFACE, "cviface 256-dimensional feature")                       \
  X(FEATURE_BMFACE_R34, "resnet34 512-dimensional BMFace feature")            \
  X(FEATURE_BMFACE_R50, "resnet50 512-dimensional BMFace feature")

#endif
