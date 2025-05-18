#ifndef FALL_DETECTION_APP_HPP
#define FALL_DETECTION_APP_HPP

#include "app/app_task.hpp"
#include "components/tracker/tracker_types.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "fall_detection.hpp"
#include "nn/tdl_model_factory.hpp"

class FallDetectionApp : public AppTask {
 public:
  FallDetectionApp(const std::string &task_name,
                   const std::string &json_config);
  ~FallDetectionApp() {}

  int32_t addPipeline(const std::string &pipeline_name,
                      int32_t frame_buffer_size,
                      const nlohmann::json &nodes_cfg);
  int32_t getResult(const std::string &pipeline_name, Packet &result) override;
  int32_t detect(std::vector<ObjectBoxLandmarkInfo> &person_infos,
                 std::vector<TrackerInfo> &track_results,
                 std::map<uint64_t, int> &det_results);
  int32_t set_fps(float fps);
  int32_t init() override;
  int32_t release() override;

 private:
  std::shared_ptr<PipelineNode> getVideoNode(const nlohmann::json &node_config);
  std::shared_ptr<PipelineNode> getKeypointDetectionNode(
      const nlohmann::json &node_config);
  std::shared_ptr<PipelineNode> getTrackNode(const nlohmann::json &node_config);

  std::string model_dir_;

  std::map<std::string, std::shared_ptr<BaseModel>> model_map_;

  NodeFactory node_factory_;
  std::vector<FallDet> muti_person;
  float FPS = 21.0;
};

#endif
