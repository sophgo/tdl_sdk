#include "face_capture_app.hpp"
#include <json.hpp>
#include "app/app_data_types.hpp"
#include "components/tracker/tracker_types.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "utils/tdl_log.hpp"

FaceCaptureApp::FaceCaptureApp(const std::string &task_name,
                               const std::string &json_config)
    : AppTask(task_name, json_config) {}

int32_t FaceCaptureApp::init() {
  std::string model_dir = json_config_.at("model_dir").get<std::string>();
  int32_t frame_buffer_size =
      json_config_.at("frame_buffer_size").get<int32_t>();

  model_factory_.setModelDir(model_dir);
  for (const auto &pl : json_config_.at("pipelines")) {
    // a) 名称
    std::string pipeline_name = pl.at("name").get<std::string>();
    std::cout << "pipeline: " << pipeline_name << "\n";
    nlohmann::json nodes_cfg = pl.at("nodes");
    addPipeline(pipeline_name, frame_buffer_size, nodes_cfg);
  }
  return 0;
}

int32_t FaceCaptureApp::addPipeline(const std::string &pipeline_name,
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
      getFaceDetectionNode(get_config("face_detection_node", nodes_cfg)));
  face_capture_channel->addNode(
      getPersonDetectionNode(get_config("person_detection_node", nodes_cfg)));
  face_capture_channel->addNode(
      getTrackNode(get_config("track_node", nodes_cfg)));
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
  LOGI("add pipeline %s", pipeline_name.c_str());
  return 0;
}

int32_t FaceCaptureApp::getResult(const std::string &pipeline_name,
                                  Packet &result) {
  std::shared_ptr<FaceCaptureResult> face_capture_result =
      std::make_shared<FaceCaptureResult>();
  PtrFrameInfo frame_info =
      pipeline_channels_[pipeline_name]->getProcessedFrame(0);

  auto image =
      frame_info->node_data_["image"].Get<std::shared_ptr<BaseImage>>();
  if (image == nullptr) {
    std::cout << "image is nullptr" << std::endl;
    return -1;
  }
  face_capture_result->image = image;
  face_capture_result->frame_id = frame_info->frame_id_;
  face_capture_result->frame_width = image->getWidth();
  face_capture_result->frame_height = image->getHeight();
  face_capture_result->face_boxes =
      frame_info->node_data_["face_meta"]
          .Get<std::vector<ObjectBoxLandmarkInfo>>();
  face_capture_result->person_boxes =
      frame_info->node_data_["person_meta"].Get<std::vector<ObjectBoxInfo>>();
  face_capture_result->track_results =
      frame_info->node_data_["track_results"].Get<std::vector<TrackerInfo>>();
  pipeline_channels_[pipeline_name]->addFreeFrame(std::move(frame_info));
  result = Packet::Make(face_capture_result);
  return 0;
}

int32_t FaceCaptureApp::release() {
  for (auto &channel : pipeline_channels_) {
    channel.second->stop();
  }
  pipeline_channels_.clear();
  return 0;
}

std::shared_ptr<PipelineNode> FaceCaptureApp::getVideoNode(
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
      std::make_shared<PipelineNode>(Packet::Make(video_decoder));
  video_node->setName("video_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<VideoDecoder> video_decoder =
        packet.Get<std::shared_ptr<VideoDecoder>>();
    std::shared_ptr<BaseImage> image = nullptr;
    int ret = video_decoder->read(image);
    if (ret != 0) {
      std::cout << "video_decoder read failed" << std::endl;
      // assert(false);
    }
    frame_info->node_data_["image"] = Packet::Make(image);
    frame_info->frame_id_ = video_decoder->getFrameId();
    return 0;
  };
  video_node->setProcessFunc(lambda_func);
  video_node->setFristNode(true);

  return video_node;
}

std::shared_ptr<PipelineNode> FaceCaptureApp::getFaceDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> face_detection_model = nullptr;
  if (model_map_.count("face_detection")) {
    face_detection_model = model_map_["face_detection"];
  } else {
    face_detection_model = model_factory_.getModel(ModelType::SCRFD_DET_FACE);
    model_map_["face_detection"] = face_detection_model;
  }
  std::shared_ptr<PipelineNode> face_detection_node =
      node_factory_.createModelNode(face_detection_model);
  face_detection_node->setName("face_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<BaseModel> face_detection_model =
        packet.Get<std::shared_ptr<BaseModel>>();
    auto image =
        frame_info->node_data_["image"].Get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }
    std::shared_ptr<ModelOutputInfo> out_data = nullptr;
    int32_t ret = face_detection_model->inference(image, out_data);
    if (ret != 0) {
      std::cout << "face_detection_model process failed" << std::endl;
      assert(false);
    }
    std::shared_ptr<ModelBoxLandmarkInfo> facemeta =
        std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(out_data);
    frame_info->node_data_["face_meta"] = Packet::Make(facemeta->box_landmarks);
    return 0;
  };
  face_detection_node->setProcessFunc(lambda_func);

  // if (node_config.contains("config_thresh")) {
  //   double thresh = node_config.at("config_thresh");
  //   face_detection_model->setModelThreshold(thresh);
  // }

  return face_detection_node;
}

std::shared_ptr<PipelineNode> FaceCaptureApp::getPersonDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> person_detection_model = nullptr;
  if (model_map_.count("person_detection")) {
    person_detection_model = model_map_["person_detection"];
  } else {
    person_detection_model =
        model_factory_.getModel(ModelType::MBV2_DET_PERSON);
    model_map_["person_detection"] = person_detection_model;
  }
  std::shared_ptr<PipelineNode> person_detection_node =
      node_factory_.createModelNode(person_detection_model);
  person_detection_node->setName("person_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<BaseModel> person_detection_model =
        packet.Get<std::shared_ptr<BaseModel>>();
    auto image =
        frame_info->node_data_["image"].Get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }
    std::shared_ptr<ModelOutputInfo> out_data = nullptr;
    int32_t ret = person_detection_model->inference(image, out_data);
    if (ret != 0) {
      std::cout << "person_detection_model process failed" << std::endl;
      assert(false);
    }
    std::shared_ptr<ModelBoxInfo> person_meta =
        std::dynamic_pointer_cast<ModelBoxInfo>(out_data);
    frame_info->node_data_["person_meta"] = Packet::Make(person_meta->bboxes);
    return 0;
  };
  person_detection_node->setProcessFunc(lambda_func);

  // if (node_config.contains("config_thresh")) {
  //   double thresh = node_config.at("config_thresh");
  //   person_detection_model->setModelThreshold(thresh);
  // }

  return person_detection_node;
}

std::shared_ptr<PipelineNode> FaceCaptureApp::getTrackNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  std::shared_ptr<PipelineNode> track_node =
      std::make_shared<PipelineNode>(Packet::Make(tracker));
  track_node->setName("track_node");
  std::map<TDLObjectType, TDLObjectType> object_pair_config;
  object_pair_config[OBJECT_TYPE_FACE] = OBJECT_TYPE_PERSON;
  tracker->setPairConfig(object_pair_config);

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    auto image =
        frame_info->node_data_["image"].Get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }
    std::shared_ptr<Tracker> tracker = packet.Get<std::shared_ptr<Tracker>>();
    tracker->setImgSize(image->getWidth(), image->getHeight());
    std::vector<ObjectBoxLandmarkInfo> face_infos =
        frame_info->node_data_["face_meta"]
            .Get<std::vector<ObjectBoxLandmarkInfo>>();
    std::vector<ObjectBoxInfo> person_infos =
        frame_info->node_data_["person_meta"].Get<std::vector<ObjectBoxInfo>>();

    std::vector<ObjectBoxInfo> bbox_infos;
    for (auto &face_info : face_infos) {
      ObjectBoxInfo box_info(face_info.class_id, face_info.score, face_info.x1,
                             face_info.y1, face_info.x2, face_info.y2);
      box_info.object_type = face_info.object_type;
      bbox_infos.push_back(box_info);
    }
    for (auto &person_info : person_infos) {
      bbox_infos.push_back(person_info);
    }
    std::vector<TrackerInfo> track_results;
    tracker->track(bbox_infos, frame_info->frame_id_, track_results);
    frame_info->node_data_["track_results"] = Packet::Make(track_results);
    return 0;
  };
  track_node->setProcessFunc(lambda_func);

  return track_node;
}
