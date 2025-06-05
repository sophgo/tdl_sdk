#include "face_pet_capture_app.hpp"
#include <json.hpp>
#include "app/app_data_types.hpp"
#include "components/snapshot/object_quality.hpp"
#include "components/snapshot/object_snapshot.hpp"
#include "components/tracker/tracker_types.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "utils/tdl_log.hpp"

typedef enum {
  OBJECT_FACE = 0,
  OBJECT_HEAD = 1,
  OBJECT_PERSON = 2,
  OBJECT_PET = 3,
} FaceCapObjectClass;

template <typename T>
T getNodeData(const std::string &node_name, PtrFrameInfo &frame_info) {
  if (frame_info->node_data_.find(node_name) == frame_info->node_data_.end()) {
    printf("node %s not found\n", node_name.c_str());
    assert(false);
  }
  return frame_info->node_data_[node_name].get<T>();
}

FacePetCaptureApp::FacePetCaptureApp(const std::string &task_name,
                                     const std::string &json_config)
    : AppTask(task_name, json_config) {}

int32_t FacePetCaptureApp::init() {
  std::string model_dir = json_config_.at("model_dir").get<std::string>();
  int32_t frame_buffer_size =
      json_config_.at("frame_buffer_size").get<int32_t>();
  TDLModelFactory::getInstance().loadModelConfig();
  TDLModelFactory::getInstance().setModelDir(model_dir);
  for (const auto &pl : json_config_.at("pipelines")) {
    // a) 名称
    std::string pipeline_name = pl.at("name").get<std::string>();
    std::cout << "pipeline: " << pipeline_name << "\n";
    nlohmann::json nodes_cfg = pl.at("nodes");
    addPipeline(pipeline_name, frame_buffer_size, nodes_cfg);
  }
  return 0;
}

int32_t FacePetCaptureApp::addPipeline(const std::string &pipeline_name,
                                       int32_t frame_buffer_size,
                                       const nlohmann::json &nodes_cfg) {
  std::shared_ptr<PipelineChannel> face_capture_channel =
      std::make_shared<PipelineChannel>(pipeline_name, frame_buffer_size);
  auto get_config = [](const std::string &key, const nlohmann::json &node_cfg) {
    if (node_cfg.contains(key)) {
      return node_cfg.at(key);
    }
    return nlohmann::json();
  };

  face_capture_channel->addNode(
      getVideoNode(get_config("video_node", nodes_cfg)));
  face_capture_channel->addNode(
      getObjectDetectionNode(get_config("object_detection_node", nodes_cfg)));
  face_capture_channel->addNode(
      getTrackNode(get_config("track_node", nodes_cfg)));
  face_capture_channel->addNode(getLandmarkDetectionNode(
      get_config("landmark_detection_node", nodes_cfg)));
  face_capture_channel->addNode(
      getSnapshotNode(get_config("snapshot_node", nodes_cfg)));
  face_capture_channel->addNode(getFeatureExtractionNode(
      get_config("feature_extraction_node", nodes_cfg)));
  face_capture_channel->start();
  pipeline_channels_[pipeline_name] = face_capture_channel;

  auto lambda_clear_func = [](PtrFrameInfo &frame_info) {
    frame_info->frame_id_ = 0;
    // TO
    auto iter = frame_info->node_data_.begin();
    while (iter != frame_info->node_data_.end()) {
      if (iter->first != "image") {
        iter = frame_info->node_data_.erase(iter);
      } else {
        iter++;
      }
    }
  };
  face_capture_channel->setClearFrameFunc(lambda_clear_func);
  LOGI("add pipeline %s\n", pipeline_name.c_str());
  return 0;
}

int32_t FacePetCaptureApp::getResult(const std::string &pipeline_name,
                                     Packet &result) {
  std::shared_ptr<FacePetCaptureResult> face_pet_capture_result =
      std::make_shared<FacePetCaptureResult>();
  PtrFrameInfo frame_info =
      pipeline_channels_[pipeline_name]->getProcessedFrame(0);

  auto image =
      frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
  if (image == nullptr) {
    std::cout << "image is nullptr" << std::endl;
    return -1;
  }
  face_pet_capture_result->image = image;
  face_pet_capture_result->frame_id = frame_info->frame_id_;
  face_pet_capture_result->frame_width = image->getWidth();
  face_pet_capture_result->frame_height = image->getHeight();

  // face_pet_capture_result->person_boxes =
  //     getNodeData<std::vector<ObjectBoxInfo>>("object_meta", frame_info);

  face_pet_capture_result->face_boxes =
      getNodeData<std::vector<ObjectBoxLandmarkInfo>>("face_meta", frame_info);
  face_pet_capture_result->person_boxes =
      getNodeData<std::vector<ObjectBoxInfo>>("person_meta", frame_info);
  face_pet_capture_result->pet_boxes =
      getNodeData<std::vector<ObjectBoxInfo>>("pet_meta", frame_info);
  face_pet_capture_result->track_results =
      getNodeData<std::vector<TrackerInfo>>("track_results", frame_info);
  face_pet_capture_result->face_snapshots =
      getNodeData<std::vector<ObjectSnapshotInfo>>("snapshots", frame_info);
  face_pet_capture_result->face_features =
      getNodeData<std::map<uint64_t, std::vector<float>>>("face_features",
                                                          frame_info);
  pipeline_channels_[pipeline_name]->addFreeFrame(std::move(frame_info));
  result = Packet::make(face_pet_capture_result);
  return 0;
}

int32_t FacePetCaptureApp::release() {
  for (auto &channel : pipeline_channels_) {
    channel.second->stop();
  }
  pipeline_channels_.clear();
  return 0;
}

std::shared_ptr<PipelineNode> FacePetCaptureApp::getVideoNode(
    const nlohmann::json &node_config) {
  std::string video_type = node_config.at("video_type");
  std::string video_path = node_config.at("video_path");
  VideoDecoderType decoder_type = VideoDecoderType::UNKNOWN;
  if (video_type == "image_folder") {
    decoder_type = VideoDecoderType::IMAGE_FOLDER;
  } else if (video_type == "vi") {
    decoder_type = VideoDecoderType::VI;
  } else if (video_type == "opencv") {
    decoder_type = VideoDecoderType::OPENCV;
  } else {
    LOGE("Unsupported video type: %s\n", video_type.c_str());
    assert(false);
  }
  std::shared_ptr<VideoDecoder> video_decoder =
      VideoDecoderFactory::createVideoDecoder(decoder_type);
  int32_t ret = video_decoder->init(video_path);
  if (ret != 0) {
    LOGE("video_decoder init failed\n");
    assert(false);
  }
  std::shared_ptr<PipelineNode> video_node =
      std::make_shared<PipelineNode>(Packet::make(video_decoder));
  video_node->setName("video_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<VideoDecoder> video_decoder =
        packet.get<std::shared_ptr<VideoDecoder>>();
    std::shared_ptr<BaseImage> image = nullptr;

    int ret = video_decoder->read(image);

    if (ret != 0) {
      std::cout << "video_decoder read failed" << std::endl;
      // assert(false);
    }
    frame_info->node_data_["image"] = Packet::make(image);
    frame_info->frame_id_ = video_decoder->getFrameId();
    return 0;
  };
  video_node->setProcessFunc(lambda_func);
  video_node->setFristNode(true);

  return video_node;
}

std::shared_ptr<PipelineNode> FacePetCaptureApp::getObjectDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> object_detection_model = nullptr;
  if (model_map_.count("object_detection")) {
    object_detection_model = model_map_["object_detection"];
  } else {
    object_detection_model = TDLModelFactory::getInstance().getModel(
        ModelType::YOLOV8N_DET_FACE_HEAD_PERSON_PET);
    model_map_["object_detection"] = object_detection_model;
  }
  std::shared_ptr<PipelineNode> object_detection_node =
      node_factory_.createModelNode(object_detection_model);
  object_detection_node->setName("object_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<BaseModel> object_detection_model =
        packet.get<std::shared_ptr<BaseModel>>();
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }

    std::shared_ptr<ModelOutputInfo> out_data = nullptr;
    int32_t ret = object_detection_model->inference(image, out_data);
    if (ret != 0) {
      std::cout << "object_detection_model process failed" << std::endl;
      assert(false);
    }

    std::shared_ptr<ModelBoxInfo> object_meta =
        std::dynamic_pointer_cast<ModelBoxInfo>(out_data);

    frame_info->node_data_["object_meta"] = Packet::make(object_meta->bboxes);

    LOGI("frame id:%d, detecte object size: %d\n", frame_info->frame_id_,
         object_meta->bboxes.size());

    return 0;
  };
  object_detection_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    object_detection_model->setModelThreshold(thresh);
  }

  return object_detection_node;
}

std::shared_ptr<PipelineNode> FacePetCaptureApp::getTrackNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  std::shared_ptr<PipelineNode> track_node =
      std::make_shared<PipelineNode>(Packet::make(tracker));
  track_node->setName("track_node");
  std::map<TDLObjectType, TDLObjectType> object_pair_config;
  object_pair_config[OBJECT_TYPE_FACE] = OBJECT_TYPE_PERSON;
  tracker->setPairConfig(object_pair_config);

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }
    std::shared_ptr<Tracker> tracker = packet.get<std::shared_ptr<Tracker>>();
    tracker->setImgSize(image->getWidth(), image->getHeight());
    std::vector<ObjectBoxInfo> object_infos =
        frame_info->node_data_["object_meta"].get<std::vector<ObjectBoxInfo>>();

    std::vector<ObjectBoxInfo> bbox_infos;
    std::vector<ObjectBoxLandmarkInfo> face_landmark_bbox;
    std::vector<ObjectBoxInfo> face_bbox;
    std::vector<ObjectBoxInfo> head_bbox;

    std::vector<ObjectBoxInfo> person_bbox;
    std::vector<ObjectBoxInfo> pet_bbox;

    for (auto &object_info : object_infos) {
      if (object_info.class_id == FaceCapObjectClass::OBJECT_FACE) {
        object_info.object_type = TDLObjectType::OBJECT_TYPE_FACE;
        bbox_infos.push_back(object_info);

        ObjectBoxLandmarkInfo box;
        box.class_id = 0;
        box.object_type = TDLObjectType::OBJECT_TYPE_FACE;
        box.score = object_info.score;
        box.x1 = object_info.x1;
        box.y1 = object_info.y1;
        box.x2 = object_info.x2;
        box.y2 = object_info.y2;
        face_landmark_bbox.push_back(box);

        face_bbox.push_back(object_info);

      } else if (object_info.class_id == FaceCapObjectClass::OBJECT_PERSON) {
        object_info.object_type = TDLObjectType::OBJECT_TYPE_PERSON;
        bbox_infos.push_back(object_info);
        person_bbox.push_back(object_info);

      } else if (object_info.class_id == FaceCapObjectClass::OBJECT_HEAD) {
        head_bbox.push_back(object_info);
      } else if (object_info.class_id == FaceCapObjectClass::OBJECT_PET) {
        pet_bbox.push_back(object_info);
      } else {
        continue;
      }
    }

    std::vector<float> face_head_quality;
    ObjectQualityHelper::getFaceQuality(face_bbox, head_bbox,
                                        face_head_quality);

    frame_info->node_data_["face_head_quality"] =
        Packet::make(face_head_quality);
    frame_info->node_data_["face_meta"] =
        Packet::make(face_landmark_bbox);  // face_meta without landmarks
    frame_info->node_data_["person_meta"] = Packet::make(person_bbox);
    frame_info->node_data_["pet_meta"] = Packet::make(pet_bbox);

    std::vector<TrackerInfo> track_results;
    tracker->track(bbox_infos, frame_info->frame_id_, track_results);
    frame_info->node_data_["track_results"] = Packet::make(track_results);

    LOGI("frame id:%d, track size: %d\n", frame_info->frame_id_,
         track_results.size());

    return 0;
  };
  track_node->setProcessFunc(lambda_func);

  return track_node;
}

std::shared_ptr<PipelineNode> FacePetCaptureApp::getLandmarkDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> landmark_detection_model = nullptr;
  if (model_map_.count("landmark_detection")) {
    landmark_detection_model = model_map_["landmark_detection"];
  } else {
    landmark_detection_model =
        TDLModelFactory::getInstance().getModel(ModelType::KEYPOINT_FACE_V2);
    model_map_["landmark_detection"] = landmark_detection_model;
  }
  std::shared_ptr<PipelineNode> landmark_detection_node =
      node_factory_.createModelNode(landmark_detection_model);
  landmark_detection_node->setName("landmark_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<BaseModel> landmark_detection_model =
        packet.get<std::shared_ptr<BaseModel>>();
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }

    std::vector<TrackerInfo> track_results =
        frame_info->node_data_["track_results"].get<std::vector<TrackerInfo>>();

    std::vector<ObjectBoxLandmarkInfo> face_infos =
        frame_info->node_data_["face_meta"]
            .get<std::vector<ObjectBoxLandmarkInfo>>();

    const std::vector<float> &face_head_quality =
        frame_info->node_data_["face_head_quality"].get<std::vector<float>>();

    if (face_head_quality.size() != face_infos.size()) {
      std::cout << "face_head_quality.size() != face_infos.size()" << std::endl;
      assert(false);
    }

    // float quality_threshold = 0.5;

    int img_width = (int)image->getWidth();
    int img_height = (int)image->getHeight();

    std::map<uint64_t, std::shared_ptr<BaseImage>> crop_face_imgs = {};

    for (auto &t : track_results) {
      if (t.box_info_.object_type == TDLObjectType::OBJECT_TYPE_FACE) {
        if (t.obj_idx_ != -1 && face_head_quality[t.obj_idx_] > 0.4) {
          ObjectBoxLandmarkInfo face_info = face_infos[t.obj_idx_];
          // 1. 计算原始框的宽高和中心
          float orig_w = face_info.x2 - face_info.x1;
          float orig_h = face_info.y2 - face_info.y1;
          float cx = face_info.x1 + orig_w * 0.5f;
          float cy = face_info.y1 + orig_h * 0.5f;

          // 2. 决定裁剪区域的方形区域
          int crop_w = int(std::max(orig_w, orig_h) + 0.5f);
          int crop_h = crop_w;

          // 3. 如果有最小裁剪限度，也可以在这里保证不小于它
          float scale = crop_w < 128 ? 128.0f / crop_w : 256.0f / crop_w;

          int dst_width = int(crop_w * scale + 0.5f);
          int dst_height = int(crop_h * scale + 0.5f);

          // 4. 以中心为基准，计算左上角
          int crop_x = int(cx - crop_w * 0.5f + 0.5f);
          int crop_y = int(cy - crop_h * 0.5f + 0.5f);

          // 5. 边界控制：保证整个裁剪框在图像内部
          if (crop_x < 0) crop_x = 0;
          if (crop_y < 0) crop_y = 0;
          if (crop_x + crop_w > img_width)
            crop_x = std::max(0, img_width - crop_w);
          if (crop_y + crop_h > img_height)
            crop_y = std::max(0, img_height - crop_h);

          face_infos[t.obj_idx_].x1 =
              (face_infos[t.obj_idx_].x1 - crop_x) * scale;
          face_infos[t.obj_idx_].y1 =
              (face_infos[t.obj_idx_].y1 - crop_y) * scale;
          face_infos[t.obj_idx_].x2 =
              (face_infos[t.obj_idx_].x2 - crop_x) * scale;
          face_infos[t.obj_idx_].y2 =
              (face_infos[t.obj_idx_].y2 - crop_y) * scale;

          std::shared_ptr<BasePreprocessor> preprocessor =
              landmark_detection_model->getPreprocessor();

          std::shared_ptr<BaseImage> crop_face_img = preprocessor->cropResize(
              image, crop_x, crop_y, crop_w, crop_w, dst_width, dst_height);

          crop_face_imgs[t.track_id_] = crop_face_img;

          std::shared_ptr<ModelOutputInfo> out_data = nullptr;
          int32_t ret =
              landmark_detection_model->inference(crop_face_img, out_data);
          if (ret != 0) {
            std::cout << "landmark_detection_model process failed" << std::endl;
            assert(false);
          }
          std::shared_ptr<ModelLandmarksInfo> landmark_meta =
              std::dynamic_pointer_cast<ModelLandmarksInfo>(out_data);

          t.blurness =
              landmark_meta->attributes
                  [TDLObjectAttributeType::OBJECT_CLS_ATTRIBUTE_FACE_BLURNESS];

          if (landmark_meta->landmarks_x.size() != 5) {
            LOGE("landmark_detection_model predict failed!\n");
            LOGE("landmark_meta->landmarks_x.size(): %ld\n",
                 landmark_meta->landmarks_x.size());
            assert(false);
          }

          face_infos[t.obj_idx_].landmarks_x = landmark_meta->landmarks_x;
          face_infos[t.obj_idx_].landmarks_y = landmark_meta->landmarks_y;
        }
      }
    }

    frame_info->node_data_["track_results"] =
        Packet::make(track_results);  // to update track_results (blurness)
    frame_info->node_data_["rescale_face_meta"] = Packet::make(face_infos);
    frame_info->node_data_["crop_face_imgs"] = Packet::make(crop_face_imgs);

    LOGI("frame id:%d, crop_face_imgs size: %d\n", frame_info->frame_id_,
         crop_face_imgs.size());
    return 0;
  };
  landmark_detection_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    landmark_detection_model->setModelThreshold(thresh);
  }

  return landmark_detection_node;
}

std::shared_ptr<PipelineNode> FacePetCaptureApp::getSnapshotNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<ObjectSnapshot> snapshot = std::make_shared<ObjectSnapshot>();
  snapshot->updateConfig(node_config);
  std::shared_ptr<PipelineNode> snapshot_node =
      std::make_shared<PipelineNode>(Packet::make(snapshot));
  snapshot_node->setName("snapshot_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    std::shared_ptr<ObjectSnapshot> snapshot =
        packet.get<std::shared_ptr<ObjectSnapshot>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      std::vector<ObjectSnapshotInfo> snapshots;
      snapshot->getSnapshotData(snapshots, true);
      frame_info->node_data_["snapshots"] = Packet::make(snapshots);
      return -1;
    }

    std::map<uint64_t, ObjectBoxInfo> face_track_boxes;
    std::vector<TrackerInfo> face_track_results;
    std::map<uint64_t, float> face_qaulity_scores;
    std::vector<float> face_landmarks;
    const std::vector<ObjectBoxLandmarkInfo> &face_infos =
        frame_info->node_data_["rescale_face_meta"]
            .get<std::vector<ObjectBoxLandmarkInfo>>();
    const std::vector<TrackerInfo> &track_results =
        frame_info->node_data_["track_results"].get<std::vector<TrackerInfo>>();
    const std::vector<float> &face_head_quality =
        frame_info->node_data_["face_head_quality"].get<std::vector<float>>();

    for (auto &t : track_results) {
      if (t.box_info_.object_type == OBJECT_TYPE_FACE) {
        face_track_results.push_back(t);

        if (t.obj_idx_ != -1) {
          ObjectBoxLandmarkInfo face_info = face_infos[t.obj_idx_];
          face_track_boxes[t.track_id_] =
              ObjectBoxInfo(face_info.class_id, face_info.score, face_info.x1,
                            face_info.y1, face_info.x2, face_info.y2);
          float vel = std::hypot(t.velocity_x_, t.velocity_y_);

          if (face_head_quality[t.obj_idx_] < 0.4) {
            face_qaulity_scores[t.track_id_] = -1.0f;  // skip directly
          } else {
            std::map<std::string, float> other_info = {{"vel", vel},
                                                       {"blr", t.blurness}};

            float face_quality = ObjectQualityHelper::getFaceQuality(
                face_info, image->getWidth(), image->getHeight(), other_info);
            LOGI("track_id:%lu,frame_id:%lu,face_quality:%f\n", t.track_id_,
                 frame_info->frame_id_, face_quality);
            face_qaulity_scores[t.track_id_] = face_quality;
          }
        }
      }
    }

    const std::map<uint64_t, std::shared_ptr<BaseImage>> &crop_face_imgs =
        frame_info->node_data_["crop_face_imgs"]
            .get<std::map<uint64_t, std::shared_ptr<BaseImage>>>();

    std::map<std::string, Packet> other_info;
    other_info["face_landmark"] = Packet::make(face_infos);
    LOGI("to update snapshot,frame_id:%lu,face_track_boxes.size:%zu\n",
         frame_info->frame_id_, face_track_boxes.size());
    snapshot->updateSnapshot(image, frame_info->frame_id_, face_track_boxes,
                             face_track_results, face_qaulity_scores,
                             other_info, crop_face_imgs);

    std::vector<ObjectSnapshotInfo> snapshots;
    snapshot->getSnapshotData(snapshots);

    for (auto &snapshot : snapshots) {
      LOGI("snapshot_node,frame_id:%lu,track_id:%lu,quality:%f\n",
           snapshot.snapshot_frame_id, snapshot.track_id, snapshot.quality);
    }

    frame_info->node_data_["snapshots"] = Packet::make(snapshots);

    LOGI("snapshot_node,frame_id:%lu,snapshots.size:%zu update done\n",
         frame_info->frame_id_, snapshots.size());
    return 0;
  };
  snapshot_node->setProcessFunc(lambda_func);
  return snapshot_node;
}

std::shared_ptr<PipelineNode> FacePetCaptureApp::getFeatureExtractionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> feature_extraction_model = nullptr;
  if (model_map_.count("feature_extraction")) {
    feature_extraction_model = model_map_["feature_extraction"];
  } else {
    feature_extraction_model =
        TDLModelFactory::getInstance().getModel(ModelType::RECOGNITION_CVIFACE);
    model_map_["feature_extraction"] = feature_extraction_model;
  }
  std::shared_ptr<PipelineNode> feature_extraction_node =
      node_factory_.createModelNode(feature_extraction_model);
  feature_extraction_node->setName("feature_extraction_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<BaseModel> feature_extraction_model =
        packet.get<std::shared_ptr<BaseModel>>();

    const std::vector<ObjectSnapshotInfo> &snapshots_data =
        frame_info->node_data_["snapshots"]
            .get<std::vector<ObjectSnapshotInfo>>();

    std::vector<std::shared_ptr<BaseImage>> images = {};

    for (size_t i = 0; i < snapshots_data.size(); i++) {
      ObjectSnapshotInfo snapshot_data = snapshots_data[i];
      std::shared_ptr<BaseImage> object_image = snapshot_data.object_image;

      if (snapshot_data.other_info.count("landmarks") == 0) {
        printf("failed to get landmarks\n");
        assert(false);
      }

      std::vector<float> landmarks =
          snapshot_data.other_info.at("landmarks").get<std::vector<float>>();

      float dst_landmarks[10];
      for (int i = 0; i < 5; i++) {
        dst_landmarks[2 * i] = landmarks[2 * i];
        dst_landmarks[2 * i + 1] = landmarks[2 * i + 1];
      }

      std::shared_ptr<BaseImage> face_crop = ImageFactory::alignFace(
          object_image, dst_landmarks, nullptr, 5, nullptr);
      images.push_back(face_crop);
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    if (images.size() > 0) {
      int32_t ret = feature_extraction_model->inference(images, out_data);
      if (ret != 0) {
        std::cout << "feature_extraction_model process failed" << std::endl;
        assert(false);
      }
    }

    std::map<uint64_t, std::vector<float>> face_features = {};

    for (size_t i = 0; i < snapshots_data.size(); i++) {
      ObjectSnapshotInfo snapshot_data = snapshots_data[i];

      std::shared_ptr<ModelFeatureInfo> feature_meta =
          std::dynamic_pointer_cast<ModelFeatureInfo>(out_data[i]);

      std::vector<float> feature_vec(feature_meta->embedding_num);
      switch (feature_meta->embedding_type) {
        case TDLDataType::INT8: {
          int8_t *feature = reinterpret_cast<int8_t *>(feature_meta->embedding);
          for (int32_t i = 0; i < feature_meta->embedding_num; ++i) {
            feature_vec[i] = (float)feature[i];
          }
          break;
        }
        case TDLDataType::UINT8: {
          uint8_t *feature =
              reinterpret_cast<uint8_t *>(feature_meta->embedding);
          for (int32_t i = 0; i < feature_meta->embedding_num; ++i) {
            feature_vec[i] = (float)feature[i];
          }
          break;
        }
        case TDLDataType::FP32: {
          float *feature = reinterpret_cast<float *>(feature_meta->embedding);
          std::memcpy(feature_vec.data(), feature,
                      feature_meta->embedding_num * sizeof(float));
          break;
        }
        default:
          assert(false && "Unsupported embedding_type");
      }

      face_features[snapshot_data.track_id] = feature_vec;
    }

    frame_info->node_data_["face_features"] = Packet::make(face_features);
    LOGI("frame_id:%lu,face_features size:%zu\n", frame_info->frame_id_,
         face_features.size());

    return 0;
  };
  feature_extraction_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    feature_extraction_model->setModelThreshold(thresh);
  }

  return feature_extraction_node;
}