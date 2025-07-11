#include "fall_detection_app.hpp"
#include <json.hpp>
#include "app/app_data_types.hpp"
#include "components/snapshot/object_quality.hpp"
#include "components/snapshot/object_snapshot.hpp"
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

FallDetectionApp::FallDetectionApp(const std::string &task_name,
                                   const std::string &json_config)
    : AppTask(task_name, json_config) {}

int32_t FallDetectionApp::init() {
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

int32_t FallDetectionApp::set_fps(float fps) {
  FPS = fps;
  return 0;
}

int32_t FallDetectionApp::addPipeline(const std::string &pipeline_name,
                                      int32_t frame_buffer_size,
                                      const nlohmann::json &nodes_cfg) {
  std::shared_ptr<PipelineChannel> fall_detection_channel =
      std::make_shared<PipelineChannel>(pipeline_name, frame_buffer_size);
  auto get_config = [](const std::string &key, const nlohmann::json &node_cfg) {
    if (node_cfg.contains(key)) {
      return node_cfg.at(key);
    }
    return nlohmann::json();
  };

  if (nodes_cfg.contains("video_node")) {
    fall_detection_channel->addNode(
        getVideoNode(get_config("video_node", nodes_cfg)));
    fall_detection_channel->setExternalFrame(false);
  }
  fall_detection_channel->addNode(getKeypointDetectionNode(
      get_config("keypoint_detection_node", nodes_cfg)));
  fall_detection_channel->addNode(
      getTrackNode(get_config("track_node", nodes_cfg)));
  fall_detection_channel->start();
  pipeline_channels_[pipeline_name] = fall_detection_channel;

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
  fall_detection_channel->setClearFrameFunc(lambda_clear_func);
  LOGI("add pipeline %s", pipeline_name.c_str());
  return 0;
}

int32_t FallDetectionApp::release() {
  for (auto &channel : pipeline_channels_) {
    channel.second->stop();
  }
  pipeline_channels_.clear();
  return 0;
}

std::shared_ptr<PipelineNode> FallDetectionApp::getVideoNode(
    const nlohmann::json &node_config) {
  std::string video_type = node_config.at("video_type");
  std::string video_path = node_config.at("video_path");
  float fps = (float)node_config.at("fps");
  set_fps(fps);
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

std::shared_ptr<PipelineNode> FallDetectionApp::getKeypointDetectionNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<BaseModel> keypoint_detection_model = nullptr;
  if (model_map_.count("keypoint_detection")) {
    keypoint_detection_model = model_map_["keypoint_detection"];
  } else {
    keypoint_detection_model = TDLModelFactory::getInstance().getModel(
        ModelType::KEYPOINT_YOLOV8POSE_PERSON17);
    model_map_["keypoint_detection"] = keypoint_detection_model;
  }
  std::shared_ptr<PipelineNode> keypoint_detection_node =
      node_factory_.createModelNode(keypoint_detection_model);
  keypoint_detection_node->setName("keypoint_detection_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    std::shared_ptr<BaseModel> keypoint_detection_model =
        packet.get<std::shared_ptr<BaseModel>>();
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }
    std::shared_ptr<ModelOutputInfo> out_data = nullptr;
    int32_t ret = keypoint_detection_model->inference(image, out_data);
    if (ret != 0) {
      std::cout << "keypoint_detection_model process failed" << std::endl;
      assert(false);
    }
    std::shared_ptr<ModelBoxLandmarkInfo> keypointmeta =
        std::dynamic_pointer_cast<ModelBoxLandmarkInfo>(out_data);
    frame_info->node_data_["person_boxes_keypoints_meta"] =
        Packet::make(keypointmeta->box_landmarks);
    return 0;
  };
  keypoint_detection_node->setProcessFunc(lambda_func);

  if (node_config.contains("config_thresh")) {
    double thresh = node_config.at("config_thresh");
    keypoint_detection_model->setModelThreshold(thresh);
  }

  return keypoint_detection_node;
}

std::shared_ptr<PipelineNode> FallDetectionApp::getTrackNode(
    const nlohmann::json &node_config) {
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  std::shared_ptr<PipelineNode> track_node =
      std::make_shared<PipelineNode>(Packet::make(tracker));
  track_node->setName("track_node");

  auto lambda_func = [](PtrFrameInfo &frame_info, Packet &packet) -> int32_t {
    auto image =
        frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
    if (image == nullptr) {
      std::cout << "image is nullptr" << std::endl;
      return -1;
    }
    std::shared_ptr<Tracker> tracker = packet.get<std::shared_ptr<Tracker>>();
    tracker->setImgSize(image->getWidth(), image->getHeight());
    const std::vector<ObjectBoxLandmarkInfo> &person_infos =
        frame_info->node_data_["person_boxes_keypoints_meta"]
            .get<std::vector<ObjectBoxLandmarkInfo>>();

    std::vector<ObjectBoxInfo> bbox_infos;
    for (auto &person_info : person_infos) {
      ObjectBoxInfo box_info(person_info.class_id, person_info.score,
                             person_info.x1, person_info.y1, person_info.x2,
                             person_info.y2);
      box_info.object_type = TDLObjectType::OBJECT_TYPE_PERSON;
      bbox_infos.push_back(box_info);
    }

    std::vector<TrackerInfo> track_results;
    tracker->track(bbox_infos, frame_info->frame_id_, track_results);
    frame_info->node_data_["track_results"] = Packet::make(track_results);
    return 0;
  };
  track_node->setProcessFunc(lambda_func);

  return track_node;
}

int32_t FallDetectionApp::detect(
    std::vector<ObjectBoxLandmarkInfo> &person_infos,
    std::vector<TrackerInfo> &track_results,
    std::map<uint64_t, int> &det_results) {
  det_results.clear();
  std::map<uint64_t, int> track_index;
  std::vector<int> new_index;

  for (size_t i = 0; i < track_results.size(); i++) {
    TrackerInfo &t = track_results[i];

    if (t.obj_idx_ != -1) {
      uint64_t track_id = t.track_id_;

      if (t.status_ == TrackStatus::NEW) {
        new_index.push_back(i);
      } else {
        track_index[track_id] = t.obj_idx_;
      }
    }
  }

  for (auto it = muti_person.begin(); it != muti_person.end();) {
    if (track_index.count(it->uid) == 0) {
      it->unmatched_times += 1;

      if (it->unmatched_times == it->MAX_UNMATCHED_TIME) {
        it = muti_person.erase(it);
      } else {
        it->update_queue(it->valid_list, 0);
        it++;
      }

    } else {
      det_results[it->uid] =
          it->detect(person_infos[track_index[it->uid]], FPS);
      it->unmatched_times = 0;
      it++;
    }
  }

  for (uint32_t i = 0; i < new_index.size(); i++) {
    FallDet person(track_results[new_index[i]].track_id_);

    det_results[person.uid] =
        person.detect(person_infos[track_results[new_index[i]].obj_idx_], FPS);

    muti_person.push_back(person);
  }

  return 0;
}

int32_t FallDetectionApp::getResult(const std::string &pipeline_name,
                                    Packet &result) {
  std::shared_ptr<FallDetectionResult> fall_detection_result =
      std::make_shared<FallDetectionResult>();
  PtrFrameInfo frame_info =
      pipeline_channels_[pipeline_name]->getProcessedFrame(0);

  auto image =
      frame_info->node_data_["image"].get<std::shared_ptr<BaseImage>>();
  if (image == nullptr) {
    std::cout << "image is nullptr" << std::endl;
    return -1;
  }
  fall_detection_result->image = image;
  fall_detection_result->frame_id = frame_info->frame_id_;
  fall_detection_result->frame_width = image->getWidth();
  fall_detection_result->frame_height = image->getHeight();
  fall_detection_result->person_boxes_keypoints =
      getNodeData<std::vector<ObjectBoxLandmarkInfo>>(
          "person_boxes_keypoints_meta", frame_info);
  fall_detection_result->track_results =
      getNodeData<std::vector<TrackerInfo>>("track_results", frame_info);

  detect(fall_detection_result->person_boxes_keypoints,
         fall_detection_result->track_results,
         fall_detection_result->det_results);

  if (getChannelNodeName(pipeline_name, 0) == "video_node") {
    pipeline_channels_[pipeline_name]->addFreeFrame(std::move(frame_info));
  }
  result = Packet::make(fall_detection_result);
  return 0;
}
