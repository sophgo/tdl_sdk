#include "tdl_model_factory.hpp"

#include <fstream>
#include "audio_classification/audio_classification.hpp"
#include "audio_classification/fsmn_vad.hpp"
#include "face_attribute/face_attribute_cls.hpp"
#include "face_detection/scrfd.hpp"
#include "face_landmark/face_landmark_det2.hpp"
#include "feature_extract/clip_image.hpp"
#include "feature_extract/clip_text.hpp"
#include "feature_extract/feature_extraction.hpp"
#include "image_classification/hand_keypopint_classification.hpp"
#include "image_classification/isp_image_classification.hpp"
#include "image_classification/rgb_image_classification.hpp"
#include "keypoints_detection/hand_keypoint.hpp"
#include "keypoints_detection/license_plate_keypoint.hpp"
#include "keypoints_detection/lstr_lane.hpp"
#include "keypoints_detection/simcc_pose.hpp"
#include "keypoints_detection/yolov8_pose.hpp"
#include "license_plate_recognition/license_plate_recognition.hpp"
#include "object_detection/mobiledet.hpp"
#include "object_detection/ppyoloe.hpp"
#include "object_detection/yolov10.hpp"
#include "object_detection/yolov5.hpp"
#include "object_detection/yolov6.hpp"
#include "object_detection/yolov7.hpp"
#include "object_detection/yolov8.hpp"
#include "object_detection/yolox.hpp"
#include "object_tracking/feartrack.hpp"
#include "segmentation/topformer_seg.hpp"
#include "segmentation/yolov8_seg.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
#ifndef DISABLE_SPEECH_RECOGNITION
#include "speech_recognition/zipformer_decoder.hpp"
#include "speech_recognition/zipformer_encoder.hpp"
#include "speech_recognition/zipformer_joiner.hpp"
#endif

std::shared_ptr<BaseModel> TDLModelFactory::getModelInstance(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
  LOGIP("getModelInstance model_type:%s",
        modelTypeToString(model_type).c_str());
  // 1. 目标检测模型（YOLO和MobileNet系列）
  if (isObjectDetectionModel(model_type)) {
    model = createObjectDetectionModel(model_type);
  }
  // 2. 人脸检测模型
  else if (isFaceDetectionModel(model_type)) {
    model = createFaceDetectionModel(model_type);
  }
  // 3. 车道线检测模型
  else if (isLaneDetectionModel(model_type)) {
    model = createLaneDetectionModel(model_type);
  }
  // 4. 关键点检测模型
  else if (isKeypointDetectionModel(model_type)) {
    model = createKeypointDetectionModel(model_type);
  }
  // 5. 分类模型
  else if (isClassificationModel(model_type)) {
    model = createClassificationModel(model_type);
  }
  // 6. 分割模型
  else if (isSegmentationModel(model_type)) {
    model = createSegmentationModel(model_type);
  }
  // 7. 特征提取模型
  else if (isFeatureExtractionModel(model_type)) {
    model = createFeatureExtractionModel(model_type);
  }
  // 8. 目标跟踪模型
  else if (isObjectTrackingModel(model_type)) {
    model = createObjectTrackingModel(model_type);
  }
  // 9. OCR模型
  else if (isOCRModel(model_type)) {
    model = createOCRModel(model_type);
    // 10. 语音识别模型
  } else if (isSpeechRecognitionModel(model_type)) {
    model = createSpeechRecognitionModel(model_type);
    // 11. 人声检测模型
  } else if (isVoiceActivityDetectionModel(model_type)) {
    model = createVoiceActivityDetectionModel(model_type);
  } else {
    LOGE("model type %s not supported", modelTypeToString(model_type).c_str());
    return nullptr;
  }
  return model;
}

// 判断函数实现
bool TDLModelFactory::isObjectDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::YOLOV8N_DET_PERSON_VEHICLE ||
          model_type == ModelType::YOLOV8N_DET_LICENSE_PLATE ||
          model_type == ModelType::YOLOV8N_DET_HAND ||
          model_type == ModelType::YOLOV8N_DET_PET_PERSON ||
          model_type == ModelType::YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE ||
          model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON ||
          model_type == ModelType::YOLOV8N_DET_FACE_HEAD_PERSON_PET ||
          model_type == ModelType::YOLOV8N_DET_HEAD_PERSON ||
          model_type == ModelType::YOLOV8N_DET_FIRE_SMOKE ||
          model_type == ModelType::YOLOV8N_DET_FIRE ||
          model_type == ModelType::YOLOV8N_DET_HEAD_SHOULDER ||
          model_type == ModelType::YOLOV8N_DET_TRAFFIC_LIGHT ||
          model_type == ModelType::YOLOV8N_DET_HEAD_HARDHAT ||
          model_type == ModelType::YOLOV8N_DET_MONITOR_PERSON ||
          model_type == ModelType::YOLOV11N_DET_MONITOR_PERSON ||
          model_type == ModelType::YOLOV11N_DET_BICYCLE_MOTOR_EBICYCLE ||
          model_type == ModelType::YOLOV11N_DET_COCO80 ||
          model_type == ModelType::YOLOV8_DET_COCO80 ||
          model_type == ModelType::YOLOV10_DET_COCO80 ||
          model_type == ModelType::YOLOV7_DET_COCO80 ||
          model_type == ModelType::YOLOV6_DET_COCO80 ||
          model_type == ModelType::YOLOV5_DET_COCO80 ||
          model_type == ModelType::PPYOLOE_DET_COCO80 ||
          model_type == ModelType::YOLOX_DET_COCO80 ||
          model_type == ModelType::YOLOV8 || model_type == ModelType::YOLOV10 ||
          model_type == ModelType::YOLOV6 || model_type == ModelType::PPYOLOE ||
          model_type == ModelType::YOLOV5 || model_type == ModelType::YOLOX ||
          model_type == ModelType::YOLOV7 ||
          model_type == ModelType::MBV2_DET_PERSON);
}

bool TDLModelFactory::isFaceDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::SCRFD_DET_FACE);
}

bool TDLModelFactory::isLaneDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::LSTR_DET_LANE);
}

bool TDLModelFactory::isKeypointDetectionModel(const ModelType model_type) {
  return (model_type == ModelType::KEYPOINT_SIMCC_PERSON17 ||
          model_type == ModelType::KEYPOINT_HAND ||
          model_type == ModelType::KEYPOINT_LICENSE_PLATE ||
          model_type == ModelType::KEYPOINT_YOLOV8POSE_PERSON17 ||
          model_type == ModelType::KEYPOINT_FACE_V2);
}

bool TDLModelFactory::isClassificationModel(const ModelType model_type) {
  return (model_type == ModelType::CLS_HAND_GESTURE ||
          model_type == ModelType::CLS_KEYPOINT_HAND_GESTURE ||
          model_type == ModelType::CLS_SOUND_BABAY_CRY ||
          model_type == ModelType::CLS_SOUND_COMMAND ||
          model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSHIYUN ||
          model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSUANNENG ||
          model_type == ModelType::CLS_SOUND_COMMAND_XIAOAIXIAOAI ||
          model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS ||
          model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK ||
          model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION ||
          model_type == ModelType::CLS_RGBLIVENESS ||
          model_type == ModelType::CLS_YOLOV8 ||
          model_type == ModelType::CLS_ISP_SCENE ||
          model_type == ModelType::CLS_IMG);
}

bool TDLModelFactory::isSegmentationModel(const ModelType model_type) {
  return (model_type == ModelType::YOLOV8_SEG_COCO80 ||
          model_type == ModelType::YOLOV8_SEG ||
          model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE);
}

bool TDLModelFactory::isFeatureExtractionModel(const ModelType model_type) {
  return (model_type == ModelType::FEATURE_CLIP_IMG ||
          model_type == ModelType::FEATURE_CLIP_TEXT ||
          model_type == ModelType::FEATURE_MOBILECLIP2_IMG ||
          model_type == ModelType::FEATURE_MOBILECLIP2_TEXT ||
          model_type == ModelType::FEATURE_BMFACE_R34 ||
          model_type == ModelType::FEATURE_BMFACE_R50 ||
          model_type == ModelType::FEATURE_CVIFACE ||
          model_type == ModelType::FEATURE_IMG);
}

bool TDLModelFactory::isOCRModel(const ModelType model_type) {
  return (model_type == ModelType::RECOGNITION_LICENSE_PLATE);
}

bool TDLModelFactory::isObjectTrackingModel(const ModelType model_type) {
  return (model_type == ModelType::TRACKING_FEARTRACK);
}

bool TDLModelFactory::isSpeechRecognitionModel(const ModelType model_type) {
  return (model_type == ModelType::RECOGNITION_SPEECH_ZIPFORMER_ENCODER ||
          model_type == ModelType::RECOGNITION_SPEECH_ZIPFORMER_DECODER ||
          model_type == ModelType::RECOGNITION_SPEECH_ZIPFORMER_JOINER);
}

bool TDLModelFactory::isVoiceActivityDetectionModel(
    const ModelType model_type) {
  return (model_type == ModelType::VAD_FSMN);
}
// 创建函数实现
std::shared_ptr<BaseModel> TDLModelFactory::createObjectDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
  int num_classes = 0;
  std::map<int, TDLObjectType> model_type_mapping;
  int model_category =
      0;  // 0:yolov8,1:yolov10,2:yolov6,3:yolov3,4:yolov5,5:yolov6,6:mbv2,7:ppyoloe,8:yolox,9:yolov7

#ifndef DISABLE_OBJECT_DETECTION
  if (model_type == ModelType::YOLOV8N_DET_PERSON_VEHICLE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_CAR;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_BUS;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_TRUCK;
    model_type_mapping[3] = TDLObjectType::OBJECT_RIDER_WITH_MOTORCYCLE;
    model_type_mapping[4] = TDLObjectType::OBJECT_TYPE_PERSON;
    model_type_mapping[5] = TDLObjectType::OBJECT_TYPE_BICYCLE;
    model_type_mapping[6] = TDLObjectType::OBJECT_TYPE_MOTORBIKE;
    num_classes = 7;
  } else if (model_type == ModelType::YOLOV8N_DET_LICENSE_PLATE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_LICENSE_PLATE;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV8N_DET_HAND) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HAND;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV8N_DET_PET_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_CAT;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_DOG;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_BICYCLE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_MOTORBIKE;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_EBICYCLE;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_HAND_FACE_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HAND;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_FACE;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 3;
  } else if (model_type == ModelType::YOLOV8N_DET_FACE_HEAD_PERSON_PET) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_FACE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_PERSON;
    model_type_mapping[3] = TDLObjectType::OBJECT_TYPE_PET;
    num_classes = 4;
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 2;
  } else if (model_type == ModelType::YOLOV8N_DET_FIRE_SMOKE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_FIRE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_SMOKE;
    num_classes = 2;
  } else if (model_type == ModelType::YOLOV8N_DET_FIRE) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_FIRE;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_SHOULDER) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD_SHOULDER;

  } else if (model_type == ModelType::YOLOV8N_DET_TRAFFIC_LIGHT) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_RED;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_YELLOW;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_GREEN;
    model_type_mapping[3] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_OFF;
    model_type_mapping[4] = TDLObjectType::OBJECT_TYPE_TRAFFIC_LIGHT_WAIT_ON;
    num_classes = 5;
  } else if (model_type == ModelType::YOLOV8N_DET_HEAD_HARDHAT) {
    num_classes = 2;
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_HEAD;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_HARD_HAT;
  } else if (model_type == ModelType::YOLOV8N_DET_MONITOR_PERSON) {
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV11N_DET_MONITOR_PERSON) {
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;
    num_classes = 1;
  } else if (model_type == ModelType::YOLOV11N_DET_BICYCLE_MOTOR_EBICYCLE) {
    num_classes = 3;
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_BICYCLE;
    model_type_mapping[1] = TDLObjectType::OBJECT_TYPE_MOTORBIKE;
    model_type_mapping[2] = TDLObjectType::OBJECT_TYPE_EBICYCLE;
  } else if (model_type == ModelType::YOLOV8_DET_COCO80) {
    model_category = 0;  // YOLOV8
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV11N_DET_COCO80) {
    model_category = 0;  // YOLOV8
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV10_DET_COCO80) {
    model_category = 1;  // YOLOV10
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV6_DET_COCO80) {
    model_category = 2;  // YOLOV6
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV5_DET_COCO80) {
    model_category = 4;  // YOLOV5
    num_classes = 80;
  } else if (model_type == ModelType::PPYOLOE_DET_COCO80) {
    model_category = 7;  // PPYOLOE
    num_classes = 80;
  } else if (model_type == ModelType::YOLOX_DET_COCO80) {
    model_category = 8;  // YOLOX
    num_classes = 80;
  } else if (model_type == ModelType::YOLOV7_DET_COCO80) {
    model_category = 9;  // YOLOX
    num_classes = 80;
  } else if (model_type == ModelType::MBV2_DET_PERSON) {
    model_category = 6;  // MobileDetV2
    model_type_mapping[0] = TDLObjectType::OBJECT_TYPE_PERSON;
  } else if (model_type == ModelType::YOLOV8) {
    model_category = 0;  // YOLOV8
  } else if (model_type == ModelType::YOLOV10) {
    model_category = 1;  // YOLOV10
  } else if (model_type == ModelType::YOLOV6) {
    model_category = 2;  // YOLOV6
  } else if (model_type == ModelType::PPYOLOE) {
    model_category = 7;  // PPYOLOE
  } else if (model_type == ModelType::YOLOX) {
    model_category = 8;  // YOLOX
  } else if (model_type == ModelType::YOLOV7) {
    model_category = 9;  // YOLOX
  } else {
    LOGE("model type not supported: %d", static_cast<int>(model_type));
    return nullptr;
  }

  if (model_category == 0) {
    model = std::make_shared<YoloV8Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 1) {
    model = std::make_shared<YoloV10Detection>(std::make_pair(64, num_classes));
  } else if (model_category == 2) {
    model = std::make_shared<YoloV6Detection>(std::make_pair(4, num_classes));
  } else if (model_category == 4) {
    model = std::make_shared<YoloV5Detection>(std::make_pair(4, num_classes));
  } else if (model_category == 6) {
    model = std::make_shared<MobileDetV2Detection>(
        MobileDetV2Detection::Category::pedestrian);
  } else if (model_category == 7) {
    model = std::make_shared<PPYoloEDetection>(std::make_pair(4, num_classes));
  } else if (model_category == 8) {
    model = std::make_shared<YoloXDetection>();
  } else if (model_category == 9) {
    model = std::make_shared<YoloV7Detection>(std::make_pair(4, num_classes));
  } else {
    LOGE("createObjectDetectionModel failed,model type not supported: %d",
         static_cast<int>(model_type));
    return nullptr;
  }
  LOGIP("createObjectDetectionModel success,model type:%d,category:%d",
        static_cast<int>(model_type), model_category);
  model->setTypeMapping(model_type_mapping);

#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createFaceDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

#ifndef DISABLE_FACE_DETECTION
  if (model_type == ModelType::SCRFD_DET_FACE) {
    model = std::make_shared<SCRFD>();
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createLaneDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
#ifndef DISABLE_KEYPOINTS_DETECTION
  if (model_type == ModelType::LSTR_DET_LANE) {
    model = std::make_shared<LstrLane>();
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createSegmentationModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

#ifndef DISABLE_SEGMENTATION
  if (model_type == ModelType::YOLOV8_SEG_COCO80) {
    model = std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, 80));
  } else if (model_type == ModelType::YOLOV8_SEG) {
    int num_cls = 0;
    model =
        std::make_shared<YoloV8Segmentation>(std::make_tuple(64, 32, num_cls));
  } else if (model_type == ModelType::TOPFORMER_SEG_PERSON_FACE_VEHICLE) {
    model = std::make_shared<TopformerSeg>(16);  // Downsampling ratio
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createFeatureExtractionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

#ifndef DISABLE_FEATURE_EXTRACT
  if (model_type == ModelType::FEATURE_CLIP_IMG ||
      model_type == ModelType::FEATURE_MOBILECLIP2_IMG) {
    model = std::make_shared<Clip_Image>();
  } else if (model_type == ModelType::FEATURE_CLIP_TEXT ||
             model_type == ModelType::FEATURE_MOBILECLIP2_TEXT) {
    model = std::make_shared<Clip_Text>();
  } else if (model_type == ModelType::FEATURE_BMFACE_R34 ||
             model_type == ModelType::FEATURE_BMFACE_R50 ||
             model_type == ModelType::FEATURE_CVIFACE ||
             model_type == ModelType::FEATURE_IMG) {
    model = std::make_shared<FeatureExtraction>();
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createKeypointDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
#ifndef DISABLE_KEYPOINTS_DETECTION
  if (model_type == ModelType::KEYPOINT_SIMCC_PERSON17) {
    model = std::make_shared<SimccPose>();
  } else if (model_type == ModelType::KEYPOINT_HAND) {
    model = std::make_shared<HandKeypoint>();
  } else if (model_type == ModelType::KEYPOINT_YOLOV8POSE_PERSON17) {
    model = std::make_shared<YoloV8Pose>(std::make_tuple(64, 17, 1));
  } else if (model_type == ModelType::KEYPOINT_LICENSE_PLATE) {
    model = std::make_shared<LicensePlateKeypoint>();
  }
#endif

#ifndef DISABLE_FACE_LANDMARK
  if (model_type == ModelType::KEYPOINT_FACE_V2) {
    model = std::make_shared<FaceLandmarkerDet2>();
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createClassificationModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

#ifndef DISABLE_IMAGE_CLASSIFICATION
  if (model_type == ModelType::CLS_HAND_GESTURE ||
      model_type == ModelType::CLS_RGBLIVENESS ||
      model_type == ModelType::CLS_YOLOV8 || model_type == ModelType::CLS_IMG) {
    model = std::make_shared<RgbImageClassification>();
  } else if (model_type == ModelType::CLS_KEYPOINT_HAND_GESTURE) {
    model = std::make_shared<HandKeypointClassification>();
  } else if (model_type == ModelType::CLS_ISP_SCENE) {
    model = std::make_shared<IspImageClassification>();
  }
#endif

#ifndef DISABLE_FACE_ATTRIBUTE
  if (model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS) {
    model = std::make_shared<FaceAttribute_CLS>(
        FaceAttributeModel::GENDER_AGE_GLASS);
  } else if (model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK) {
    model = std::make_shared<FaceAttribute_CLS>(
        FaceAttributeModel::GENDER_AGE_GLASS_MASK);
  } else if (model_type == ModelType::CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION) {
    model = std::make_shared<FaceAttribute_CLS>(
        FaceAttributeModel::GENDER_AGE_GLASS_EMOTION);
  }
#endif

#ifndef DISABLE_AUDIO_CLASSIFICATION
  if (model_type == ModelType::CLS_SOUND_BABAY_CRY) {
    model = std::make_shared<AudioClassification>();
  } else if (model_type == ModelType::CLS_SOUND_COMMAND ||
             model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSHIYUN ||
             model_type == ModelType::CLS_SOUND_COMMAND_NIHAOSUANNENG ||
             model_type == ModelType::CLS_SOUND_COMMAND_XIAOAIXIAOAI) {
    model = std::make_shared<AudioClassification>();
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createOCRModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;

#ifndef DISABLE_LICENSE_PLATE_RECOGNITION
  if (model_type == ModelType::RECOGNITION_LICENSE_PLATE) {
    model = std::make_shared<LicensePlateRecognition>();
  }
#endif

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createObjectTrackingModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
#ifndef DISABLE_OBJECT_TRACKING
  if (model_type == ModelType::TRACKING_FEARTRACK) {
    model = std::make_shared<FearTrack>();
  }
#endif

  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createSpeechRecognitionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
#ifndef DISABLE_SPEECH_RECOGNITION
  if (model_type == ModelType::RECOGNITION_SPEECH_ZIPFORMER_ENCODER) {
    model = std::make_shared<ZipformerEncoder>();
  } else if (model_type == ModelType::RECOGNITION_SPEECH_ZIPFORMER_DECODER) {
    model = std::make_shared<ZipformerDecoder>();
  } else if (model_type == ModelType::RECOGNITION_SPEECH_ZIPFORMER_JOINER) {
    model = std::make_shared<ZipformerJoiner>();
  }
#endif
  return model;
}

std::shared_ptr<BaseModel> TDLModelFactory::createVoiceActivityDetectionModel(
    const ModelType model_type) {
  std::shared_ptr<BaseModel> model = nullptr;
#ifndef DISABLE_VOICE_ACTIVITY_DETECTION
  if (model_type == ModelType::VAD_FSMN) {
    model = std::make_shared<FsmnVad>();
  }
#endif

  return model;
}
