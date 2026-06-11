#include "vehicle_adas_app.hpp"
#include <inttypes.h>
#include <cstdio>
#include <json.hpp>
#include "app/app_data_types.hpp"
#include "components/tracker/tracker_types.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
T getNodeData(const std::string &node_name, PtrFrameInfo &frame_info) {
  if (frame_info->node_data_.find(node_name) == frame_info->node_data_.end()) {
    printf("node %s not found\n", node_name.c_str());
    assert(false);
  }
  return frame_info->node_data_[node_name].get<T>();
}

VehicleAdasApp::VehicleAdasApp(const std::string &task_name,
                               const std::string &json_config,
                               bool skip_input_alloc)
    : AppTask(task_name, json_config, skip_input_alloc) {}

int32_t VehicleAdasApp::init() {
  std::string model_dir = json_config_.at("model_dir").get<std::string>();
  int32_t frame_buffer_size =
      json_config_.at("frame_buffer_size").get<int32_t>();
  if (json_config_.contains("model_config")) {
    TDLModelFactory::getInstance().loadModelConfig(
        json_config_.at("model_config"));
  } else {
    TDLModelFactory::getInstance().loadModelConfig();
  }
  TDLModelFactory::getInstance().setModelDir(model_dir);
  for (const auto &pl : json_config_.at("pipelines")) {
    std::string pipeline_name = pl.at("name").get<std::string>();
    std::cout << "pipeline: " << pipeline_name << "\n";
    nlohmann::json nodes_cfg = pl.at("nodes");
    addPipeline(pipeline_name, frame_buffer_size, nodes_cfg);
  }
  return 0;
}

int32_t VehicleAdasApp::addPipeline(const std::string &pipeline_name,
                                    int32_t frame_buffer_size,
                                    const nlohmann::json &nodes_cfg) {
  std::shared_ptr<PipelineChannel> adas_channel =
      std::make_shared<PipelineChannel>(pipeline_name, frame_buffer_size);
  auto get_config = [](const std::string &key, const nlohmann::json &node_cfg) {
    if (node_cfg.contains(key)) {
      return node_cfg.at(key);
    }
    return nlohmann::json();
  };

#ifdef VIDEO_ENABLE
  if (nodes_cfg.contains("video_node")) {
    adas_channel->addNode(getVideoNode(get_config("video_node", nodes_cfg)));
    adas_channel->setExternalFrame(false);
  }
#endif

  adas_channel->addNode(
      getObjectDetectionNode(get_config("object_detection_node", nodes_cfg)));
  adas_channel->addNode(getTrackNode(get_config("track_node", nodes_cfg)));

  if (nodes_cfg.contains("lane_detection_node")) {
    adas_channel->addNode(
        getLaneDetectionNode(get_config("lane_detection_node", nodes_cfg)));
  }

  adas_channel->addNode(getAdasNode(get_config("adas_node", nodes_cfg)));

  adas_channel->start();
  pipeline_channels_[pipeline_name] = adas_channel;

  auto lambda_clear_func = [](PtrFrameInfo &frame_info) {
    frame_info->frame_id_ = 0;
    auto iter = frame_info->node_data_.begin();
    while (iter != frame_info->node_data_.end()) {
      if (iter->first != "image") {
        iter = frame_info->node_data_.erase(iter);
      } else {
        iter++;
      }
    }
  };
  adas_channel->setClearFrameFunc(lambda_clear_func);
  LOGI("add pipeline %s\n", pipeline_name.c_str());
  return 0;
}

int32_t VehicleAdasApp::getResult(const std::string &pipeline_name,
                                  Packet &result) {
  std::shared_ptr<VehicleAdasResult> adas_result =
      std::make_shared<VehicleAdasResult>();
  PtrFrameInfo frame_info =
      pipeline_channels_[pipeline_name]->getProcessedFrame(0);
  if (frame_info == nullptr) {
    LOGW("frame_info is nullptr for pipeline %s\n", pipeline_name.c_str());
    result = Packet::make(adas_result);
    return 0;
  }

  if (frame_info->node_data_.find("image") == frame_info->node_data_.end()) {
    LOGE("image node not found in frame_info for pipeline %s\n",
         pipeline_name.c_str());
    result = Packet::make(adas_result);
    return -1;
  }
  auto image =
      frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();

  adas_result->image = image;
  adas_result->frame_id = frame_info->frame_id_;
  adas_result->frame_width = frame_info->frame_width;
  adas_result->frame_height = frame_info->frame_height;

  if (image == nullptr) {
    result = Packet::make(adas_result);
    return -1;
  }

  if (frame_info->node_data_.find("adas_results") !=
      frame_info->node_data_.end()) {
    adas_result->objects = getNodeData<std::vector<VehicleAdasObjectResult>>(
        "adas_results", frame_info);
  }

  if (frame_info->node_data_.find("lane_state") !=
      frame_info->node_data_.end()) {
    adas_result->lane_state =
        getNodeData<VehicleAdasLaneState>("lane_state", frame_info);
  }

  if (frame_info->node_data_.find("lane_meta") !=
      frame_info->node_data_.end()) {
    auto lane_meta = frame_info->node_data_["lane_meta"]
                         .get<std::shared_ptr<ModelBoxLandmarkInfo>>();
    if (lane_meta) {
      for (auto &box : lane_meta->box_landmarks) {
        VehicleAdasLaneLine line;
        line.x1 = box.landmarks_x[0];
        line.y1 = box.landmarks_y[0];
        line.x2 = box.landmarks_x[1];
        line.y2 = box.landmarks_y[1];
        adas_result->lane_lines.push_back(line);
      }
    }
  }

  adas_result->track_results =
      getNodeData<std::vector<TrackerInfo>>("track_results", frame_info);

  if (getChannelNodeName(pipeline_name, 0) == "video_node") {
    pipeline_channels_[pipeline_name]->addFreeFrame(std::move(frame_info));
  }
  result = Packet::make(adas_result);
  return 0;
}

int32_t VehicleAdasApp::release() {
  for (auto &channel : pipeline_channels_) {
    channel.second->stop();
  }
  pipeline_channels_.clear();
  return 0;
}

#ifdef VIDEO_ENABLE
std::shared_ptr<PipelineNode> VehicleAdasApp::getVideoNode(
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

  std::map<std::string, int> video_decoder_config = {
      {"is_loop", static_cast<int>(node_config.at("is_loop"))}};
  int32_t ret = video_decoder->init(video_path, video_decoder_config);
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
    frame_info->node_data_["image"] = Packet::make(image);

    if (ret != 0) {
      std::cout << "video_decoder read failed" << std::endl;
      return -1;
    }
    frame_info->frame_id_ = video_decoder->getFrameId();
    if (frame_info->frame_id_ == 0) {
      frame_info->frame_width = image->getWidth();
      frame_info->frame_height = image->getHeight();
    }
    return 0;
  };
  video_node->setProcessFunc(lambda_func);
  video_node->setFristNode(true);

  return video_node;
}
#endif

std::shared_ptr<PipelineNode> VehicleAdasApp::getObjectDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> object_detection_model = nullptr;
  if (model_map_.count("object_detection")) {
    object_detection_model = model_map_["object_detection"];
  } else {
    object_detection_model = TDLModelFactory::getInstance().getModel(
        ModelType::YOLOV8N_DET_PERSON_VEHICLE);
    model_map_["object_detection"] = object_detection_model;
  }
  std::shared_ptr<PipelineNode> object_detection_node =
      node_factory_.createModelNode(object_detection_model);
  object_detection_node->setName("object_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    if (frame_info->node_data_.find("image") == frame_info->node_data_.end()) {
      return -1;
    }
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

    std::vector<ObjectBoxInfo> filtered_bboxes;
    for (const auto &bbox : object_meta->bboxes) {
      float width = bbox.x2 - bbox.x1;
      float height = bbox.y2 - bbox.y1;
      if (height > 0 && (width / height) < 3.0f) {
        filtered_bboxes.push_back(bbox);
      }
    }

    frame_info->node_data_["object_meta"] = Packet::make(filtered_bboxes);

    LOGI("frame id:%" PRIu64 ", detect object size: %zu\n",
         frame_info->frame_id_, filtered_bboxes.size());

    return 0;
  };
  object_detection_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    object_detection_model->setModelThreshold(thresh);
  }

  return object_detection_node;
}

std::shared_ptr<PipelineNode> VehicleAdasApp::getTrackNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  std::shared_ptr<PipelineNode> track_node =
      std::make_shared<PipelineNode>(Packet::make(tracker));
  track_node->setName("track_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    if (frame_info->node_data_.find("image") == frame_info->node_data_.end() ||
        frame_info->node_data_.find("object_meta") ==
            frame_info->node_data_.end()) {
      return -1;
    }
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

    std::vector<TrackerInfo> track_results;
    tracker->track(object_infos, frame_info->frame_id_, track_results);
    frame_info->node_data_["track_results"] = Packet::make(track_results);

    LOGI("frame id:%" PRIu64 ", track size: %zu\n", frame_info->frame_id_,
         track_results.size());

    return 0;
  };
  track_node->setProcessFunc(lambda_func);

  return track_node;
}

std::shared_ptr<PipelineNode> VehicleAdasApp::getLaneDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> lane_detection_model = nullptr;
  if (model_map_.count("lane_detection")) {
    lane_detection_model = model_map_["lane_detection"];
  } else {
    lane_detection_model =
        TDLModelFactory::getInstance().getModel(ModelType::LSTR_DET_LANE);
    model_map_["lane_detection"] = lane_detection_model;
  }
  std::shared_ptr<PipelineNode> lane_detection_node =
      node_factory_.createModelNode(lane_detection_model);
  lane_detection_node->setName("lane_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    if (frame_info->node_data_.find("image") == frame_info->node_data_.end()) {
      return -1;
    }
    std::shared_ptr<BaseModel> lane_detection_model =
        packet.get<std::shared_ptr<BaseModel>>();
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }

    std::shared_ptr<ModelOutputInfo> out_data = nullptr;
    int32_t ret = lane_detection_model->inference(image, out_data);
    if (ret != 0) {
      std::cout << "lane_detection_model process failed" << std::endl;
      assert(false);
    }

    std::shared_ptr<ModelBoxLandmarkInfo> lane_meta =
        std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(out_data);

    frame_info->node_data_["lane_meta"] = Packet::make(lane_meta);

    LOGI("frame id:%" PRIu64 ", detect lane size: %zu\n", frame_info->frame_id_,
         lane_meta->box_landmarks.size());

    return 0;
  };
  lane_detection_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    lane_detection_model->setModelThreshold(thresh);
  }

  return lane_detection_node;
}

std::shared_ptr<PipelineNode> VehicleAdasApp::getAdasNode(
    const nlohmann::json &node_config) {
  int det_type =
      node_config.contains("det_type") ? node_config["det_type"].get<int>() : 0;

  std::shared_ptr<VehicleAdas> vehicle_adas = std::make_shared<VehicleAdas>();
  vehicle_adas->init(20, det_type);
  vehicle_adas->updateConfig(node_config);

  std::shared_ptr<PipelineNode> adas_node =
      std::make_shared<PipelineNode>(Packet::make(vehicle_adas));
  adas_node->setName("adas_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    if (frame_info->node_data_.find("image") == frame_info->node_data_.end() ||
        frame_info->node_data_.find("object_meta") ==
            frame_info->node_data_.end() ||
        frame_info->node_data_.find("track_results") ==
            frame_info->node_data_.end()) {
      return -1;
    }

    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    std::shared_ptr<VehicleAdas> vehicle_adas =
        packet.get<std::shared_ptr<VehicleAdas>>();

    std::vector<ObjectBoxInfo> object_infos =
        frame_info->node_data_["object_meta"].get<std::vector<ObjectBoxInfo>>();

    std::vector<TrackerInfo> track_results =
        frame_info->node_data_["track_results"].get<std::vector<TrackerInfo>>();

    std::shared_ptr<ModelBoxLandmarkInfo> lane_meta = nullptr;
    if (frame_info->node_data_.find("lane_meta") !=
        frame_info->node_data_.end()) {
      lane_meta = frame_info->node_data_["lane_meta"]
                      .get<std::shared_ptr<ModelBoxLandmarkInfo>>();
    }

    uint32_t frame_width = frame_info->frame_width;
    uint32_t frame_height = frame_info->frame_height;
    if (image != nullptr) {
      frame_width = image->getWidth();
      frame_height = image->getHeight();
    }

    vehicle_adas->run(frame_info->frame_id_, frame_width, frame_height,
                      object_infos, track_results, lane_meta);

    std::vector<VehicleAdasObjectResult> adas_results;
    vehicle_adas->getResults(adas_results);

    frame_info->node_data_["adas_results"] = Packet::make(adas_results);
    frame_info->node_data_["lane_state"] =
        Packet::make(vehicle_adas->getLaneState());

    LOGI("frame id:%" PRIu64 ", adas results: %zu, lane_state: %d\n",
         frame_info->frame_id_, adas_results.size(),
         vehicle_adas->getLaneState().lane_state);

    return 0;
  };
  adas_node->setProcessFunc(lambda_func);

  return adas_node;
}