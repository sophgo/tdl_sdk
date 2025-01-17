#ifndef FACTORY_MODEL_HPP__
#define FACTORY_MODEL_HPP__

#include "detection/yolov5.hpp"
#include "face/face_cssd.hpp"
#include "face/face_ddfa.hpp"
#include "face/face_landmark.hpp"

#include "face/face_spoof.hpp"
#include "face/face_scrfd.hpp"
#include "face/face_util.hpp"
#include "framework/base_model.hpp"
#include "reid/feature_extract.hpp"
#include "detection/obj_landmark.hpp"
#include "classify/carplate_ocr.hpp"

// to make npuvideo interface compatial
typedef FeatureExtract FaceExtract;
#define DEVICE_ID 0

using NNBaseModel::ModelType;

class NNFactory {
 public:
  explicit NNFactory(const std::string &model_dir)
      : model_dir_(model_dir + "/") {}

  ~NNFactory() = default;

  BaseModel *get_model(ModelType type, int device_id = 0) const;

  BaseModel *get_face_detector(ModelType type, int device_id = 0) const;

  BaseModel *get_face_landmark(ModelType type, int device_id = 0) const;

  BaseModel *get_face_quality(ModelType type, int device_id = 0) const;

  BaseModel *get_extractor(ModelType type, int device_id = 0) const;
  BaseModel *get_yolo_v5_detector(ModelType type, int device_id = 0) const;

  BaseModel *get_carplate_landmark(ModelType type, int device_id = 0) const;
  BaseModel *get_carplate_ocr(ModelType type, int device_id = 0) const;
  
  void set_data_dir(const std::string &data_dir) { data_dir_ = data_dir; }

 private:
  std::string model_dir_;
  std::string data_dir_ = "data/";
};

using NNBaseModel::ModelType::AGE;
using NNBaseModel::ModelType::BLACKLIST;
using NNBaseModel::ModelType::BMFACER18;
using NNBaseModel::ModelType::BMFACER34;
using NNBaseModel::ModelType::BMFACER34_V2;
using NNBaseModel::ModelType::BMFACER34_V2_FP32;
using NNBaseModel::ModelType::BMFACEV03M;
using NNBaseModel::ModelType::BMMASKFACER34;
using NNBaseModel::ModelType::CSSD;
using NNBaseModel::ModelType::DDFA;
using NNBaseModel::ModelType::DET3;
using NNBaseModel::ModelType::EXPRESSION;
using NNBaseModel::ModelType::FACERETINA;
using NNBaseModel::ModelType::FACERETINA_IR;
using NNBaseModel::ModelType::FACESPOOF;
using NNBaseModel::ModelType::GENDER;
using NNBaseModel::ModelType::GENDERAGE;
using NNBaseModel::ModelType::GENDERGLASSAGE;
using NNBaseModel::ModelType::GENDERHAT;
using NNBaseModel::ModelType::GLASSESMASK;
using NNBaseModel::ModelType::MOBILESSD;
using NNBaseModel::ModelType::MTCNN;
using NNBaseModel::ModelType::MULTITASK;
using NNBaseModel::ModelType::OCCLUSION;
using NNBaseModel::ModelType::OCCLUSION_V2;
using NNBaseModel::ModelType::REID;
using NNBaseModel::ModelType::RETINA;
using NNBaseModel::ModelType::SSH;
using NNBaseModel::ModelType::SCRFD;
using NNBaseModel::ModelType::BMFACER50_V1;
#endif
