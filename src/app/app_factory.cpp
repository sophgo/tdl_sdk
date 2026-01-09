#include "app/app_task.hpp"
#include "consumer_counting/consumer_counting_app.hpp"
#include "face_capture/face_capture_app.hpp"
#include "face_pet_capture/face_pet_capture_app.hpp"
#include "fall_detection/fall_detection_app.hpp"
#include "human_pose_smooth/human_pose_smooth_app.hpp"

std::shared_ptr<AppTask> AppFactory::createAppTask(
    const std::string &task_name, const std::string &json_config_file) {
  if (task_name == "face_capture") {
    return std::make_shared<FaceCaptureApp>(task_name, json_config_file);
  } else if (task_name == "face_pet_capture") {
    return std::make_shared<FacePetCaptureApp>(task_name, json_config_file);
  } else if (task_name == "fall_detection") {
    return std::make_shared<FallDetectionApp>(task_name, json_config_file);
  } else if (task_name == "consumer_counting") {
    return std::make_shared<ConsumerCountingAPP>(task_name, json_config_file);
  } else if (task_name == "cross_detection") {
    return std::make_shared<ConsumerCountingAPP>(task_name, json_config_file);
  } else if (task_name == "human_pose_smooth") {
    return std::make_shared<HumanPoseSmoothApp>(task_name, json_config_file);
  }
  return nullptr;
}
