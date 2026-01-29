#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "media_analysis_task.hpp"

class MediaAnalysisEventManager {
 public:
  static MediaAnalysisEventManager* GetInstance();

  void RegisterTask(std::shared_ptr<MediaAnalysisTask> task);
  std::shared_ptr<MediaAnalysisTask> GetTask(const std::string& event_type);

  // Process a JSON request and return the response
  json HandleEvent(const json& request);

  // Get specific task for background processing (e.g. image analysis loop)
  std::shared_ptr<MediaAnalysisTask> GetImageAnalysisTask();

 private:
  MediaAnalysisEventManager() = default;
  ~MediaAnalysisEventManager() = default;
  MediaAnalysisEventManager(const MediaAnalysisEventManager&) = delete;
  MediaAnalysisEventManager& operator=(const MediaAnalysisEventManager&) =
      delete;

  std::map<std::string, std::shared_ptr<MediaAnalysisTask>> tasks_;
  std::shared_ptr<MediaAnalysisTask> image_analysis_task_;
  std::mutex mutex_;
};
