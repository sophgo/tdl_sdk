#ifndef HUMAN_POSE_APP_HPP
#define HUMAN_POSE_APP_HPP

#include "app/app_task.hpp"
#include "components/tracker/tracker_types.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "human_pose_smooth.hpp"
#include "nn/tdl_model_factory.hpp"

class HumanPoseSmoothApp : public AppTask {
 public:
  HumanPoseSmoothApp(const std::string &task_name,
                     const std::string &json_config);
  ~HumanPoseSmoothApp() {}

  int32_t addPipeline(const std::string &pipeline_name,
                      int32_t frame_buffer_size,
                      const nlohmann::json &nodes_cfg);
  int32_t getResult(const std::string &pipeline_name, Packet &result) override;
  int32_t smooth(std::vector<ObjectBoxLandmarkInfo> &person_infos,
                 std::vector<TrackerInfo> &track_results);

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
  std::vector<HumanKeypoints> muti_keypoints;
  SmoothAlgParam smooth_param_;
  bool enable_smooth_ = true;
};

#endif
