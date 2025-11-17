#include "tdl_model_factory.hpp"

#include <fstream>
#include "audio_classification/audio_classification.hpp"
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

TDLModelFactory::TDLModelFactory() {
  coco_types_ = {"person",        "bicycle",      "car",
                 "motorcycle",    "airplane",     "bus",
                 "train",         "truck",        "boat",
                 "traffic light", "fire hydrant", "stop sign",
                 "parking meter", "bench",        "bird",
                 "cat",           "dog",          "horse",
                 "sheep",         "cow",          "elephant",
                 "bear",          "zebra",        "giraffe",
                 "backpack",      "umbrella",     "handbag",
                 "tie",           "suitcase",     "frisbee",
                 "skis",          "snowboard",    "sports ball",
                 "kite",          "baseball bat", "baseball glove",
                 "skateboard",    "surfboard",    "tennis racket",
                 "bottle",        "wine glass",   "cup",
                 "fork",          "knife",        "spoon",
                 "bowl",          "banana",       "apple",
                 "sandwich",      "orange",       "broccoli",
                 "carrot",        "hot dog",      "pizza",
                 "donut",         "cake",         "chair",
                 "couch",         "potted plant", "bed",
                 "dining table",  "toilet",       "tv",
                 "laptop",        "mouse",        "remote",
                 "keyboard",      "cell phone",   "microwave",
                 "oven",          "toaster",      "sink",
                 "refrigerator",  "book",         "clock",
                 "vase",          "scissors",     "teddy bear",
                 "hair drier",    "toothbrush"};
  // loadModelConfig();
}

TDLModelFactory::~TDLModelFactory() {}

int32_t TDLModelFactory::loadModelConfig(const std::string &model_config_file) {
  std::string config_file = model_config_file;
  std::string parent_dir;
  if (config_file.empty()) {
    std::string so_dir = CommonUtils::getLibraryDir();
    std::string exe_dir = CommonUtils::getExecutableDir();
    if (so_dir != exe_dir) {
      parent_dir = CommonUtils::getParentDir(so_dir);
    } else {
      parent_dir =
          CommonUtils::getParentDir(CommonUtils::getParentDir(exe_dir));
    }
    config_file = parent_dir + "/configs/model/model_factory.json";
    LOGIP("input model config file is empty, load model config from %s",
          config_file.c_str());
  }
  std::ifstream inf(config_file);
  nlohmann::json json_config;
  model_config_map_.clear();
  if (!inf.is_open()) {
    LOGE("model config file not found: %s", config_file.c_str());
    return -1;
  }

  try {
    inf >> json_config;
  } catch (const nlohmann::json::parse_error &e) {
    LOGE("model config file %s parse error: %s", config_file.c_str(), e.what());
    return -1;
  }

  const auto &model_list = json_config.at("model_list");
  for (auto it = model_list.begin(); it != model_list.end(); ++it) {
    const std::string model_name = it.key();
    const nlohmann::json &info_json = it.value();
    model_config_map_[model_name] = info_json;
  }
  LOGIP("load model config from %s done,model size:%ld", config_file.c_str(),
        model_config_map_.size());
  return 0;
}
TDLModelFactory &TDLModelFactory::getInstance() {
  static TDLModelFactory instance;
  return instance;
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(const ModelType model_type,
                                                     const int device_id) {
  if (model_type == ModelType::INVALID) {
    LOGE("model type not found for model type: %d",
         static_cast<int>(model_type));
    return nullptr;
  }
  std::string model_name = modelTypeToString(model_type);
  if (model_config_map_.find(model_name) == model_config_map_.end()) {
    LOGE("model path not found for model type: %s,model size:%ld",
         model_name.c_str(), model_config_map_.size());
    return nullptr;
  }

  std::string model_path = getModelPath(model_type);
  if (model_path.empty()) {
    LOGE("model path not found for model type: %s",
         modelTypeToString(model_type).c_str());
    return nullptr;
  }
  ModelConfig model_config = parseModelConfig(model_config_map_[model_name]);
  return getModel(model_type, model_path, model_config, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const int device_id) {
  ModelType model_type_enum = modelTypeFromString(model_type);
  return getModel(model_type_enum, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const int device_id) {
  ModelConfig model_config;
  std::string model_name = modelTypeToString(model_type);
  if (model_config_map_.find(model_name) != model_config_map_.end()) {
    model_config = parseModelConfig(model_config_map_[model_name]);
  }
  return getModel(model_type, model_path, model_config, device_id);
}
std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const std::string &model_path,
    const int device_id) {
  ModelType model_type_enum = modelTypeFromString(model_type);
  return getModel(model_type_enum, model_path, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const ModelConfig &model_config, const int device_id) {
  return getModelImpl(model_type, model_path, nullptr, 0, model_config, {}, {},
                      device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const std::string &model_path,
    const std::vector<uint64_t> &mem_addrs,
    const std::vector<uint32_t> &mem_sizes, const int device_id) {
  ModelConfig model_config;
  std::string model_name = modelTypeToString(model_type);
  if (model_config_map_.find(model_name) != model_config_map_.end()) {
    model_config = parseModelConfig(model_config_map_[model_name]);
  }
  return getModelImpl(model_type, model_path, nullptr, 0, model_config,
                      mem_addrs, mem_sizes, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const uint8_t *model_buffer,
    const uint32_t model_buffer_size, const std::vector<uint64_t> &mem_addrs,
    const std::vector<uint32_t> &mem_sizes, const int device_id) {
  if (model_buffer == nullptr || model_buffer_size == 0) {
    LOGE("model buffer is nullptr or model buffer size is 0");
    return nullptr;
  }
  ModelConfig model_config;
  std::string model_name = modelTypeToString(model_type);
  if (model_config_map_.find(model_name) != model_config_map_.end()) {
    model_config = parseModelConfig(model_config_map_[model_name]);
  }
  return getModelImpl(model_type, "", model_buffer, model_buffer_size,
                      model_config, mem_addrs, mem_sizes, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const std::string &model_type, const uint8_t *model_buffer,
    const uint32_t model_buffer_size, const std::vector<uint64_t> &mem_addrs,
    const std::vector<uint32_t> &mem_sizes, const int device_id) {
  ModelType model_type_enum = modelTypeFromString(model_type);
  return getModel(model_type_enum, model_buffer, model_buffer_size, mem_addrs,
                  mem_sizes, device_id);
}

std::shared_ptr<BaseModel> TDLModelFactory::getModel(
    const ModelType model_type, const uint8_t *model_buffer,
    const uint32_t model_buffer_size, const ModelConfig &model_config,
    const std::vector<uint64_t> &mem_addrs,
    const std::vector<uint32_t> &mem_sizes, const int device_id) {
  return getModelImpl(model_type, "", model_buffer, model_buffer_size,
                      model_config, mem_addrs, mem_sizes, device_id);
}
ModelConfig TDLModelFactory::getModelConfig(const ModelType model_type) {
  std::string model_type_str = modelTypeToString(model_type);
  if (model_config_map_.find(model_type_str) == model_config_map_.end()) {
    LOGW("model config not found for model type: %s", model_type_str.c_str());
    return ModelConfig();
  }
  auto json_config = model_config_map_[model_type_str];
  ModelConfig model_config = parseModelConfig(json_config);
  return model_config;
}

void TDLModelFactory::setModelDir(const std::string &model_dir) {
  model_dir_ = model_dir;
  LOGIP("setModelDir success,model_dir:%s", model_dir.c_str());
}

std::string TDLModelFactory::getModelPath(const ModelType model_type) {
  if (model_dir_ == "") {
    LOGE("model_dir not set");
    return "";
  }
  std::string model_type_str = modelTypeToString(model_type);
  if (model_config_map_.find(model_type_str) == model_config_map_.end()) {
    LOGE("model config not found for model type: %s", model_type_str.c_str());
    return "";
  }
  nlohmann::json model_config = model_config_map_[model_type_str];
  std::string model_file_name = model_config.at("file_name").get<std::string>();
  std::string model_path;
  std::string platform;
  std::string model_extension;
  getPlatformAndModelExtension(platform, model_extension);
  if (model_file_name.find(model_extension) != std::string::npos) {
    if ('/' == model_file_name[0]) {
      model_path = model_file_name;
    } else {
      model_path = model_dir_ + std::string("/") + platform + std::string("/") +
                   model_file_name;
    }
  } else {
    model_path = model_dir_ + std::string("/") + platform + std::string("/") +
                 model_file_name + std::string("_") + platform +
                 model_extension;
  }

  return model_path;
}
bool TDLModelFactory::parseModelConfig(const std::string &model_config_json,
                                       ModelConfig &model_config) {
  try {
    nlohmann::json json_config = nlohmann::json::parse(model_config_json);
    model_config = parseModelConfig(json_config);
  } catch (const std::exception &e) {
    LOGE("Failed to parse model config: %s", e.what());
    return false;
  }
  return true;
}
ModelConfig TDLModelFactory::parseModelConfig(
    const nlohmann::json &json_config) {
  ModelConfig model_config;

  for (auto it = json_config.begin(); it != json_config.end(); ++it) {
    const std::string key = it.key();
    const nlohmann::json &val = it.value();

    if (key == "_comment") {
      model_config.comment = val.get<std::string>();
      continue;
    } else if (key == "is_coco_types") {
      bool is_coco_types = val.get<bool>();
      if (is_coco_types) {
        model_config.types = coco_types_;
      }
    } else if (key == "types") {
      model_config.types = val.get<std::vector<std::string>>();
    } else if (key == "rgb_order") {
      model_config.rgb_order = val.get<std::string>();
    } else if (key == "mean") {
      model_config.mean = val.get<std::vector<float>>();
    } else if (key == "std") {
      model_config.std = val.get<std::vector<float>>();
    } else if (val.is_number_integer()) {
      model_config.custom_config_i[key] = val.get<int>();
      LOGI("model config parse int,key:%s,value:%d", key.c_str(),
           val.get<int>());
    } else if (val.is_number_float()) {
      model_config.custom_config_f[key] = val.get<float>();
      LOGI("model config parse float,key:%s,value:%f", key.c_str(),
           val.get<float>());
    } else if (val.is_string()) {
      model_config.custom_config_str[key] = val.get<std::string>();
      LOGI("model config parse string,key:%s,value:%s", key.c_str(),
           val.get<std::string>().c_str());
    } else {
      // 修复 dump() 的临时对象悬空问题：先把它存到局部变量里
      std::string dumped = val.dump();
      LOGW("model config %s : %s not supported", key.c_str(), dumped.c_str());
    }
  }

  return model_config;
}

std::vector<std::string> TDLModelFactory::getModelList() {
  std::vector<std::string> model_list;
  for (auto &item : model_config_map_) {
    model_list.push_back(item.first);
  }
  return model_list;
}

void TDLModelFactory::getPlatformAndModelExtension(
    std::string &platform, std::string &model_extension) {
#if defined(__CV180X__)
  platform = "cv180x";
  model_extension = ".cvimodel";
#elif defined(__CV181X__)
  platform = "cv181x";
  model_extension = ".cvimodel";
#elif defined(__CV182X__)
  platform = "cv182x";
  model_extension = ".cvimodel";
#elif defined(__CV184X__)
  platform = "cv184x";
  model_extension = ".bmodel";
#elif defined(__CV186X__)
  platform = "cv186x";
  model_extension = ".bmodel";
#elif defined(__BM1684__)
  platform = "bm1684";
  model_extension = ".bmodel";
#elif defined(__BM1688__)
  platform = "bm1688";
  model_extension = ".bmodel";
#elif defined(__BM1684X__)
  platform = "bm1684x";
  model_extension = ".bmodel";
#elif defined(__CMODEL_CV181X__)
  platform = "cv181x";
  model_extension = ".cvimodel";
#elif defined(__CMODEL_CV184X__)
  platform = "cv184x";
  model_extension = ".bmodel";
#else
  LOGE("platform not supported");
  assert(false);
#endif
}

std::shared_ptr<BaseModel> TDLModelFactory::getModelImpl(
    ModelType model_type, const std::string &model_path,
    const uint8_t *model_buffer, const uint32_t model_buffer_size,
    const ModelConfig &model_config, const std::vector<uint64_t> &mem_addrs,
    const std::vector<uint32_t> &mem_sizes, const int device_id) {
  if (model_path.empty() && model_buffer == nullptr && model_buffer_size == 0) {
    LOGE("model path or model buffer is empty");
    return nullptr;
  } else if (!model_path.empty() && model_buffer != nullptr &&
             model_buffer_size != 0) {
    LOGE("model path and model buffer are not empty");
    return nullptr;
  }

  if (model_type == ModelType::INVALID) {
    LOGE("model type not found for model type: %d",
         static_cast<int>(model_type));
    return nullptr;
  }
  if (mem_addrs.size() != mem_addrs.size()) {
    LOGE("mem_addrs size is not equal to mem_sizes size");
    return nullptr;
  }
  if (mem_addrs.size() != 0 && mem_addrs.size() != 5) {
    LOGE("mem_addrs size is not equal to 5 or 0,current size:%d",
         mem_addrs.size());
    return nullptr;
  }

  std::shared_ptr<BaseModel> model = getModelInstance(model_type);
  if (model == nullptr) {
    LOGE("model not found for model type: %d", static_cast<int>(model_type));
    return nullptr;
  }
  model->setModelType(model_type);
  NetParam net_param_default = model->getNetParam();
  net_param_default.device_id = device_id;
  // merge net_param_default into  model_config
  ModelConfig model_config_merged = model_config;
  if (model_config_merged.rgb_order.empty()) {
    model_config_merged.rgb_order = net_param_default.model_config.rgb_order;
  }
  if (model_config_merged.mean.empty()) {
    model_config_merged.mean = net_param_default.model_config.mean;
  }
  if (model_config_merged.std.empty()) {
    model_config_merged.std = net_param_default.model_config.std;
  }
  if (!model_path.empty()) {
    net_param_default.model_file_path = model_path;
    LOGIP("model_path: %s", model_path.c_str());
  } else if (model_buffer != nullptr && model_buffer_size != 0) {
    net_param_default.model_buffer = const_cast<uint8_t *>(model_buffer);
    net_param_default.model_buffer_size = model_buffer_size;
  }
  net_param_default.model_config = model_config_merged;
  net_param_default.runtime_mem_addrs = mem_addrs;
  net_param_default.runtime_mem_sizes = mem_sizes;

  model->setNetParam(net_param_default);

  int ret = model->modelOpen(device_id);
  if (ret != 0) {
    return nullptr;
  }
  return model;
}
