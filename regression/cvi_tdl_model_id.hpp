#include <unordered_map>
#include "tdl_model_defs.hpp"

// json文件中model_name与ModelType枚举的映射关系。
const std::unordered_map<std::string, ModelType> model_type_map = {
    {"MBV2_DET_PERSON", ModelType::MBV2_DET_PERSON},
    {"YOLOV8N_DET_HAND", ModelType::YOLOV8N_DET_HAND},
    {"YOLOV8N_DET_PET_PERSON", ModelType::YOLOV8N_DET_PET_PERSON},
    {"YOLOV8N_DET_PERSON_VEHICLE", ModelType::YOLOV8N_DET_PERSON_VEHICLE},
    {"YOLOV8N_DET_HAND_FACE_PERSON", ModelType::YOLOV8N_DET_HAND_FACE_PERSON},
    {"YOLOV8N_DET_HEAD_PERSON", ModelType::YOLOV8N_DET_HEAD_PERSON},
    {"YOLOV8N_DET_HEAD_HARDHAT", ModelType::YOLOV8N_DET_HEAD_HARDHAT},
    {"YOLOV8N_DET_FIRE_SMOKE", ModelType::YOLOV8N_DET_FIRE_SMOKE},
    {"YOLOV8N_DET_FIRE", ModelType::YOLOV8N_DET_FIRE},
    {"YOLOV8N_DET_HEAD_SHOULDER", ModelType::YOLOV8N_DET_HEAD_SHOULDER},
    {"YOLOV8N_DET_LICENSE_PLATE", ModelType::YOLOV8N_DET_LICENSE_PLATE},
    {"YOLOV8N_DET_TRAFFIC_LIGHT", ModelType::YOLOV8N_DET_TRAFFIC_LIGHT},
    {"YOLOV8N_DET_MONITOR_PERSON", ModelType::YOLOV8N_DET_MONITOR_PERSON},
    {"SCRFD_DET_FACE", ModelType::SCRFD_DET_FACE},
    {"CLS_RGBLIVENESS", ModelType::CLS_RGBLIVENESS},
    {"CLS_SOUND_BABAY_CRY", ModelType::CLS_SOUND_BABAY_CRY},
    {"CLS_SOUND_COMMAND", ModelType::CLS_SOUND_COMMAND},
    // keypoint models
    {"KEYPOINT_HAND", ModelType::KEYPOINT_HAND},
    {"KEYPOINT_LICENSE_PLATE", ModelType::KEYPOINT_LICENSE_PLATE},
    {"KEYPOINT_YOLOV8POSE_PERSON17", ModelType::KEYPOINT_YOLOV8POSE_PERSON17},
    {"KEYPOINT_SIMCC_PERSON17", ModelType::KEYPOINT_SIMCC_PERSON17},
    // 添加其他映射
};