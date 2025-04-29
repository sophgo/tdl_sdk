#include "app/app_task.hpp"
#include "face_capture/face_capture_app.hpp"

std::shared_ptr<AppTask> AppFactory::createAppTask(
    const std::string &task_name, const std::string &json_config_file) {
  if (task_name == "face_capture") {
    return std::make_shared<FaceCaptureApp>(task_name, json_config_file);
  }
  return nullptr;
}
