#include "consumer_counting_app.hpp"
#include <json.hpp>
#include "app/app_data_types.hpp"
#include "components/snapshot/object_quality.hpp"
#include "components/snapshot/object_snapshot.hpp"
#include "components/tracker/tracker_types.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "consumer_counting.hpp"
#include "utils/tdl_log.hpp"

template <typename T>
T getNodeData(const std::string &node_name, PtrFrameInfo &frame_info) {
  if (frame_info->node_data_.find(node_name) == frame_info->node_data_.end()) {
    printf("node %s not found\n", node_name.c_str());
    assert(false);
  }
  return frame_info->node_data_[node_name].get<T>();
}

ConsumerCountingAPP::ConsumerCountingAPP(const std::string &task_name,
                                         const std::string &json_config)
    : AppTask(task_name, json_config) {}

int32_t ConsumerCountingAPP::init() {
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
    // a) 名称
    std::string pipeline_name = pl.at("name").get<std::string>();
    std::cout << "pipeline: " << pipeline_name << "\n";
    nlohmann::json nodes_cfg = pl.at("nodes");
    addPipeline(pipeline_name, frame_buffer_size, nodes_cfg);
  }
  return 0;
}

int32_t ConsumerCountingAPP::setLine(const std::string &pipeline_name,
                                     const std::string &node_name, int x1,
                                     int y1, int x2, int y2, int mode) {
  std::shared_ptr<PipelineNode> node =
      pipeline_channels_[pipeline_name]->getNode(node_name);

  Packet *packet = node->getWorker();

  std::shared_ptr<ConsumerCounting> consumer_counting =
      packet->get<std::shared_ptr<ConsumerCounting>>();

  consumer_counting->set_counting_line(x1, y1, x2, y2, mode);
  return 0;
}

int32_t ConsumerCountingAPP::addPipeline(const std::string &pipeline_name,
                                         int32_t frame_buffer_size,
                                         const nlohmann::json &nodes_cfg) {
  std::shared_ptr<PipelineChannel> consumer_counting_channel =
      std::make_shared<PipelineChannel>(pipeline_name, frame_buffer_size);
  auto get_config = [](const std::string &key, const nlohmann::json &node_cfg) {
    if (node_cfg.contains(key)) {
      return node_cfg.at(key);
    }
    return nlohmann::json();
  };

#ifdef VIDEO_ENABLE
  if (nodes_cfg.contains("video_node")) {
    consumer_counting_channel->addNode(
        getVideoNode(get_config("video_node", nodes_cfg)));
    consumer_counting_channel->setExternalFrame(false);
  }
#endif

  consumer_counting_channel->addNode(
      getObjectDetectionNode(get_config("object_detection_node", nodes_cfg)));
  consumer_counting_channel->addNode(
      getTrackNode(get_config("track_node", nodes_cfg)));

  if (nodes_cfg.contains("consumer_counting_node")) {
    consumer_counting_channel->addNode(
        ConsumerCountingNode(get_config("consumer_counting_node", nodes_cfg)));
  } else if (nodes_cfg.contains("cross_detection_node")) {
    consumer_counting_channel->addNode(
        CrossDetectionNode(get_config("cross_detection_node", nodes_cfg)));
  }

  consumer_counting_channel->start();
  pipeline_channels_[pipeline_name] = consumer_counting_channel;

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
  consumer_counting_channel->setClearFrameFunc(lambda_clear_func);
  LOGI("add pipeline %s", pipeline_name.c_str());
  return 0;
}

int32_t ConsumerCountingAPP::getResult(const std::string &pipeline_name,
                                       Packet &result) {
  std::shared_ptr<ConsumerCountingResult> consumer_counting_result =
      std::make_shared<ConsumerCountingResult>();
  PtrFrameInfo frame_info =
      pipeline_channels_[pipeline_name]->getProcessedFrame(0);

  auto image =
      frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
  if (image == nullptr) {
    std::cout << "image is nullptr" << std::endl;
    return -1;
  }
  consumer_counting_result->image = image;
  consumer_counting_result->frame_id = frame_info->frame_id_;
  consumer_counting_result->frame_width = image->getWidth();
  consumer_counting_result->frame_height = image->getHeight();
  consumer_counting_result->object_boxes =
      getNodeData<std::vector<ObjectBoxInfo>>("object_meta", frame_info);
  consumer_counting_result->track_results =
      getNodeData<std::vector<TrackerInfo>>("track_results", frame_info);
  consumer_counting_result->counting_line =
      getNodeData<std::vector<int>>("counting_line", frame_info);

  if (frame_info->node_data_.find("counting_result") !=
      frame_info->node_data_.end()) {
    std::vector<uint32_t> counting_result =
        getNodeData<std::vector<uint32_t>>("counting_result", frame_info);
    consumer_counting_result->enter_num = counting_result[0];
    consumer_counting_result->miss_num = counting_result[1];
  }

  if (frame_info->node_data_.find("cross_id") != frame_info->node_data_.end()) {
    consumer_counting_result->cross_id =
        getNodeData<std::vector<uint64_t>>("cross_id", frame_info);
  }

  if (getChannelNodeName(pipeline_name, 0) == "video_node") {
    pipeline_channels_[pipeline_name]->addFreeFrame(std::move(frame_info));
  }
  result = Packet::make(consumer_counting_result);
  return 0;
}

int32_t ConsumerCountingAPP::release() {
  for (auto &channel : pipeline_channels_) {
    channel.second->stop();
  }
  pipeline_channels_.clear();
  return 0;
}

#ifdef VIDEO_ENABLE
std::shared_ptr<PipelineNode> ConsumerCountingAPP::getVideoNode(
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
#endif

std::shared_ptr<PipelineNode> ConsumerCountingAPP::getObjectDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> object_detection_model = nullptr;
  if (model_map_.count(node_config["model"])) {
    object_detection_model = model_map_[node_config["model"]];
  } else {
    if (node_config.at("model") == "head_person_detection") {
      object_detection_model = TDLModelFactory::getInstance().getModel(
          ModelType::YOLOV8N_DET_HEAD_PERSON);
    } else if (node_config.at("model") == "person_vehicle_detection") {
      object_detection_model = TDLModelFactory::getInstance().getModel(
          ModelType::YOLOV8N_DET_PERSON_VEHICLE);
    } else {
      LOGE("Unsupported model type: %s\n",
           node_config.at("model").get<std::string>().c_str());
      return NULL;
    }
    model_map_[node_config.at("model")] = object_detection_model;
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
    return 0;
  };
  object_detection_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    object_detection_model->setModelThreshold(thresh);
  }

  return object_detection_node;
}

std::shared_ptr<PipelineNode> ConsumerCountingAPP::getTrackNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  std::shared_ptr<PipelineNode> track_node =
      std::make_shared<PipelineNode>(Packet::make(tracker));
  track_node->setName("track_node");
  std::map<TDLObjectType, TDLObjectType> object_pair_config;
  object_pair_config[OBJECT_TYPE_HEAD] = OBJECT_TYPE_PERSON;
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

    const std::vector<ObjectBoxInfo> &object_infos =
        frame_info->node_data_["object_meta"].get<std::vector<ObjectBoxInfo>>();

    std::vector<ObjectBoxInfo> bbox_infos;
    for (auto &obj_info : object_infos) {
      bbox_infos.push_back(obj_info);
    }
    std::vector<TrackerInfo> track_results;
    tracker->track(bbox_infos, frame_info->frame_id_, track_results);
    frame_info->node_data_["track_results"] = Packet::make(track_results);
    return 0;
  };
  track_node->setProcessFunc(lambda_func);

  return track_node;
}

std::shared_ptr<PipelineNode> ConsumerCountingAPP::ConsumerCountingNode(
    const nlohmann::json &node_config) {
  int mode = node_config["mode"];
  int x1 = node_config["x1"];
  int y1 = node_config["y1"];
  int x2 = node_config["x2"];
  int y2 = node_config["y2"];

  std::vector<int> counting_line = {x1, y1, x2, y2};

  std::shared_ptr<ConsumerCounting> consumer_counting =
      std::make_shared<ConsumerCounting>(x1, y1, x2, y2, mode);

  std::shared_ptr<PipelineNode> consumer_counting_node =
      std::make_shared<PipelineNode>(Packet::make(consumer_counting));
  consumer_counting_node->setName("consumer_counting_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    const std::vector<TrackerInfo> &track_results =
        frame_info->node_data_["track_results"].get<std::vector<TrackerInfo>>();

    std::shared_ptr<ConsumerCounting> consumer_counting =
        packet.get<std::shared_ptr<ConsumerCounting>>();

    std::vector<int> counting_line;
    consumer_counting->get_counting_line(counting_line);
    frame_info->node_data_["counting_line"] = Packet::make(counting_line);

    consumer_counting->update_consumer_counting_state(track_results);

    std::vector<uint32_t> counting_result;

    counting_result.push_back(consumer_counting->get_enter_num());
    counting_result.push_back(consumer_counting->get_miss_num());

    frame_info->node_data_["counting_result"] = Packet::make(counting_result);

    return 0;
  };
  consumer_counting_node->setProcessFunc(lambda_func);
  return consumer_counting_node;
}

std::shared_ptr<PipelineNode> ConsumerCountingAPP::CrossDetectionNode(
    const nlohmann::json &node_config) {
  int mode = node_config["mode"];
  int x1 = node_config["x1"];
  int y1 = node_config["y1"];
  int x2 = node_config["x2"];
  int y2 = node_config["y2"];

  std::shared_ptr<ConsumerCounting> cross_detection =
      std::make_shared<ConsumerCounting>(x1, y1, x2, y2, mode);

  std::shared_ptr<PipelineNode> cross_detection_node =
      std::make_shared<PipelineNode>(Packet::make(cross_detection));
  cross_detection_node->setName("cross_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    const std::vector<TrackerInfo> &track_results =
        frame_info->node_data_["track_results"].get<std::vector<TrackerInfo>>();

    std::shared_ptr<ConsumerCounting> cross_detection =
        packet.get<std::shared_ptr<ConsumerCounting>>();
    std::vector<int> counting_line;
    cross_detection->get_counting_line(counting_line);
    frame_info->node_data_["counting_line"] = Packet::make(counting_line);

    std::vector<uint64_t> cross_id;
    cross_detection->update_cross_detection_state(track_results, cross_id);

    frame_info->node_data_["cross_id"] = Packet::make(cross_id);

    return 0;
  };
  cross_detection_node->setProcessFunc(lambda_func);
  return cross_detection_node;
}
