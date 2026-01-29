#include "components/media_analysis/media_analysis_event_manager.hpp"
#include <iostream>

MediaAnalysisEventManager* MediaAnalysisEventManager::GetInstance() {
  static MediaAnalysisEventManager instance;
  return &instance;
}

void MediaAnalysisEventManager::RegisterTask(
    std::shared_ptr<MediaAnalysisTask> task) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (task) {
    tasks_[task->get_event_type()] = task;
    if (task->get_event_type() == "image_analysis") {
      image_analysis_task_ = task;
    }
  }
}

std::shared_ptr<MediaAnalysisTask> MediaAnalysisEventManager::GetTask(
    const std::string& event_type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = tasks_.find(event_type);
  if (it != tasks_.end()) {
    return it->second;
  }
  return nullptr;
}

std::shared_ptr<MediaAnalysisTask>
MediaAnalysisEventManager::GetImageAnalysisTask() {
  std::lock_guard<std::mutex> lock(mutex_);
  return image_analysis_task_;
}

json MediaAnalysisEventManager::HandleEvent(const json& request) {
  std::string event_type;
  if (request.contains("payload") && request["payload"].contains("event")) {
    event_type = request["payload"]["event"];
  } else if (request.contains("task_type")) {
    event_type = request["task_type"];
  }

  std::cout << "Event Manager handling event: " << event_type << std::endl;

  std::string description = "";
  if (request.contains("payload") &&
      request["payload"].contains("description")) {
    description = request["payload"]["description"];
  }

  auto task = GetTask(event_type);
  if (task) {
    return task->handle_event(request, description);
  } else {
    // Ignore registration messages treated as requests
    if (event_type != "web_client") {
      std::cout << "Unknown event type: " << event_type << std::endl;
      json response = request;
      response["error"] = "Unknown event type: " + event_type;
      return response;
    }
  }

  return json();
}
