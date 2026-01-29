#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "components/media_analysis/media_analysis_task.hpp"
#include "network/api_poster/api_client.hpp"

class FaceMatchingTask : public MediaAnalysisTask {
 public:
  FaceMatchingTask(const std::string& data_path);
  virtual ~FaceMatchingTask() = default;

  std::string get_event_type() const override { return "face_matching"; }
  json handle_event(const json& request,
                    const std::string& description) override;

  void add_face_info(int registered_id, int face_track_id);

 private:
  std::vector<std::string> get_face_images(const std::string& name);
  void parse_face_info(const std::string& image_path, nlohmann::json& item);

  std::string data_path_;
  std::map<int, int> face_info_map_;
  std::map<std::string, int> name_to_id_map_;
  std::mutex mutex_;
};
