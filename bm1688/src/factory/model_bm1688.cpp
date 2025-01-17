
#include "common/common.hpp"
#include "factory/model.hpp"

BaseModel* NNFactory::get_model(ModelType type, int device_id /*=0*/) const {
  LOG(INFO) << "to get model:" << type << "device:" << device_id;
  switch (type) {
    case CSSD:
    case NNBaseModel::SCRFD:
      return get_face_detector(type, device_id);
      break;
    case DET3:
    case NNBaseModel::BMMARK:
      return get_face_landmark(type, device_id);
      break;
    case BMFACEV03M:
    case BMFACER18:
    case BMFACER34:
    case BMFACER34_V2:
    case BMFACER34_V2_FP32:
    case BMMASKFACER34:
    case NNBaseModel::BMFACER34_V3:
    case NNBaseModel::BMVEHICLE_R18:
    case NNBaseModel::BMPERSON_R18:
    case NNBaseModel::BMFACER50_V1:
      return get_extractor(type, device_id);
      break;

    case DDFA:
    case OCCLUSION_V2:
    case OCCLUSION:
    case NNBaseModel::SOPHONFOD:
      return get_face_quality(type, device_id);
      break;

    case NNBaseModel::YOLO_V5_VEHICLE:
    case NNBaseModel::YOLO_V5_VEHICLEX2:
    case NNBaseModel::YOLOX_INT8:
      return get_yolo_v5_detector(type, device_id);
    case NNBaseModel::CARPLATE_LANDMARK:
      return get_carplate_landmark(type, device_id);
    case NNBaseModel::CARPLATE_OCR:
      return get_carplate_ocr(type, device_id);

    default:
      break;
  }
  return nullptr;
}

BaseModel* NNFactory::get_face_detector(ModelType type,
                                        int device_id /*=0*/) const {
  BaseModel* model_ptr = nullptr;
  if (type == CSSD) {
    stNetParam param;
    param.device_id = device_id;
    param.net_name = "cssd";
    param.model_file = model_dir_ + std::string("/cssd_int8.bmodel");
    param.output_names = {"mbox_loc", "mbox_conf"};
    param.input_names = {"data"};
    model_ptr = (BaseModel*)new FaceCSSD(param);
    model_ptr->model_dir_ = model_dir_;
  } else if (type == NNBaseModel::SCRFD) {
    stNetParam param;
    param.device_id = device_id;
    param.net_name = "scrfd_500m_bnkps_432_768";
    param.model_file =
        model_dir_ + std::string("/scrfd_500m_bnkps_432_768_cv186x.bmodel");
    param.mean.push_back(127.5 / 128.0);
    param.scale.push_back(1.0 / 128.0);
    param.use_rgb = true;

    model_ptr = (BaseModel*)new FaceSCRFD(param);
    model_ptr->model_dir_ = model_dir_;
  }
  if (model_ptr != nullptr) model_ptr->setup();
  return model_ptr;
}

BaseModel* NNFactory::get_face_landmark(ModelType type,
                                        int device_id /*=0*/) const {
  BaseModel* model_ptr = nullptr;
  stNetParam param;
  if (type == DET3) {
    param.device_id = device_id;
    param.model_file = model_dir_ + std::string("/landmark_det3.bmodel");
    param.net_name = "det3";
    param.input_names = {"data"};
    param.output_names = {"prob1__softmax_reshape", "conv6-3"};
    model_ptr = (BaseModel*)new FaceLandmark(param);
  } else if (type == NNBaseModel::BMMARK) {
    param.device_id = device_id;
    param.model_file =
        model_dir_ + std::string("/bm1684_bmmark_float32_combine_6.bmodel");
    // param.net_name = "BMMark";
    model_ptr = (BaseModel*)new BMMark(param);
  } else {
    LOG(ERROR) << "Unsupported model type\n";
  }
  if (model_ptr != nullptr) {
    model_ptr->setup();
  }
  return model_ptr;
}

BaseModel* NNFactory::get_extractor(ModelType type,
                                    int device_id /*=0*/) const {
  BaseModel* model_ptr = nullptr;
  stNetParam param;
  param.device_id = device_id;
  if (type == NNBaseModel::BMFACER34) {
    // param.net_name = "bmface_bj_res34";
    param.model_file = model_dir_ + std::string("/bmface_r34.bmodel");

  } else if (type == NNBaseModel::BMFACER18) {
    // param.net_name = "bmface_resnet18";
    param.model_file =
        model_dir_ + std::string("/compilation_resnet18_1248.bmodel");
    param.resize_mode = IMG_STRETCH_RESIZE;

  } else if (type == NNBaseModel::BMFACER34_V3) {
    // param.net_name = "bmface_r34_v3";
    param.model_file = model_dir_ + std::string("/bmface_r34_v3.bmodel");
    param.mean = {1};
    param.scale = {0.0078431372549};
    param.use_rgb = true;

  } else if (type == NNBaseModel::BMVEHICLE_R18) {
    // param.net_name = "vehicle_r18";
    param.model_file = model_dir_ + std::string("/vehicle_reid_r18.bmodel");
    param.mean = {2.11790393, 2.03571429, 1.80444444};
    param.scale = {0.01712475, 0.017507, 0.01742919};
    param.use_rgb = true;
    param.resize_mode = IMG_STRETCH_RESIZE;
  } else if (type == NNBaseModel::BMPERSON_R18) {
    // param.net_name = "person_r18";
    param.model_file = model_dir_ + std::string("/person_reid_r18.bmodel");
    param.mean = {2.11790393, 2.03571429, 1.80444444};
    param.scale = {0.01712475, 0.017507, 0.01742919};
    param.use_rgb = true;
    param.resize_mode = IMG_STRETCH_RESIZE;
  } else if (type == NNBaseModel::BMFACER50_V1) {
    param.model_file = model_dir_ + std::string("/bmface_r50_v1_bmnetp.bmodel");
    param.mean = {2.11790393, 2.03571429, 1.80444444};
    param.scale = {0.01712475, 0.017507, 0.01742919};
    param.use_rgb = true;
    param.resize_mode = IMG_STRETCH_RESIZE;
  } else {
    LOG(FATAL) << "not supported model:" << type;
  }

  model_ptr = (BaseModel*)new FeatureExtract(param);
  if (model_ptr != nullptr) {
    model_ptr->setup();
  }
  return model_ptr;
}

BaseModel* NNFactory::get_face_quality(ModelType type,
                                       int device_id /*=0*/) const {
  BaseModel* model_ptr = nullptr;
  stNetParam param;
  param.device_id = device_id;
  if (type == NNBaseModel::DDFA) {
    // param.net_name = "dwnet_1";
    param.model_file =
        model_dir_ + std::string("/compilation_3ddfa_float32_1.bmodel");
    model_ptr = (BaseModel*)new FaceDDFA(param);
  } else {
    LOG(ERROR) << "Unsupported model type\n";
  }
  if (model_ptr != nullptr) {
    model_ptr->setup();
  }
  return model_ptr;
}

BaseModel* NNFactory::get_yolo_v5_detector(ModelType type,
                                           int device_id /*=0*/) const {
  BaseModel* model_ptr = nullptr;
  stNetParam param;
  param.device_id = device_id;

  if (type == NNBaseModel::YOLO_V5_VEHICLE) {
    param.model_file = model_dir_ + std::string("/yoloxcls7.bmodel");
    param.scale = {1.0 / 255};

    model_ptr = (BaseModel*)new YOLOV5(param);
    model_ptr->set_pad_value(114);
  }

  else if (type == NNBaseModel::YOLOX_INT8) {
    param.model_file = model_dir_ + std::string("/yolox_int8_cls7.bmodel");
    param.scale = {1.0 / 255};
    model_ptr = (BaseModel*)new YOLOV5(param);
    model_ptr->set_pad_value(114);
  }
  if (model_ptr != nullptr) {
    model_ptr->setup();
  }
  return model_ptr;
}

BaseModel* NNFactory::get_carplate_landmark(ModelType type,
                                            int device_id /*= 0*/) const {
  BaseModel* model_ptr = nullptr;
  if (type == NNBaseModel::CARPLATE_LANDMARK) {
    stNetParam param;
    param.device_id = device_id;
    // param.net_name = "carplate_landmark";
    param.model_file = model_dir_ + std::string("/carplate_landmark.bmodel");
    param.scale = {1.0 / 255};
    param.resize_mode = IMG_STRETCH_RESIZE;
    param.output_names = {"20", "17"};
    model_ptr = (BaseModel*)new ObjectLandmark(param);
  }
  if (model_ptr != nullptr) {
    model_ptr->setup();
  }
  return model_ptr;
}
BaseModel* NNFactory::get_carplate_ocr(ModelType type,
                                       int device_id /*= 0*/) const {
  BaseModel* model_ptr = nullptr;
  stNetParam param;
  if (type == NNBaseModel::CARPLATE_OCR) {
    param.device_id = device_id;
    // param.net_name = "carplate_ocr";
    param.model_file = model_dir_ + std::string("/carplate_ocr.bmodel");
    param.scale = {0.0078125};
    param.mean = {0.99609375};
    param.output_names = {"192", "195", "198", "189"};
    param.resize_mode = IMG_STRETCH_RESIZE;
    model_ptr = (BaseModel*)new CarplateOCR(param);
  }
  if (model_ptr != nullptr) {
    model_ptr->setup();
  }
  return model_ptr;
}
