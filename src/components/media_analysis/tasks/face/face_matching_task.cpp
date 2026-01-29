#include "face_matching_task.hpp"
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <regex>

namespace fs = std::experimental::filesystem;

FaceMatchingTask::FaceMatchingTask(const std::string& data_path)
    : data_path_(data_path) {
  // 初始化时读取registered_info.txt文件
  std::ifstream file(data_path_ + "/registered_info.txt");
  if (file.is_open()) {
    std::string name;
    int registered_id;
    while (file >> name >> registered_id) {
      name_to_id_map_[name] = registered_id;
    }
    file.close();
    LOGI("name_to_id_map_ size: %lu\n", name_to_id_map_.size());
  } else {
    LOGE("无法打开文件: %s\n", (data_path_ + "/registered_info.txt").c_str());
  }

  for (auto& pair : name_to_id_map_) {
    LOGI("name: %s, registered_id: %d\n", pair.first.c_str(), pair.second);
  }
}

void FaceMatchingTask::add_face_info(int registered_id, int face_track_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  face_info_map_[registered_id] = face_track_id;
  LOGI("Updated face info: registered_id=%d, face_track_id=%d, map_size=%lu\n",
       registered_id, face_track_id, face_info_map_.size());
}

std::vector<std::string> FaceMatchingTask::get_face_images(
    const std::string& name) {
  // 通过人名获取注册特征ID
  if (name_to_id_map_.find(name) == name_to_id_map_.end()) {
    LOGI("未找到人名对应的注册ID: %s\n", name.c_str());
    return {};
  }

  int registered_id = name_to_id_map_[name];

  std::vector<std::string> image_paths;

  int face_track_id = -1;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (face_info_map_.find(registered_id) != face_info_map_.end()) {
      face_track_id = face_info_map_[registered_id];
    }
  }

  if (face_track_id != -1) {
    std::string capture_path = data_path_ + "/face";
    if (!fs::exists(capture_path)) {
      return image_paths;
    }

    for (const auto& entry : fs::directory_iterator(capture_path)) {
      if (fs::is_directory(entry.status())) {
        std::string dir_name = entry.path().filename().string();
        if (dir_name.find(std::to_string(face_track_id) + "_") == 0) {
          for (const auto& img_entry : fs::directory_iterator(entry.path())) {
            if (fs::is_regular_file(img_entry.status())) {
              image_paths.push_back(img_entry.path().string());
            }
          }
          break;
        }
      }
    }
  }

  printf("matched face size: %lu\n", image_paths.size());

  return image_paths;
}

json FaceMatchingTask::handle_event(const json& request,
                                    const std::string& description) {
  json response = request;
  response["type"] = "event";
  response["source"] = "c_backend";
  response["destination"] = "python_server";
  response["payload"]["event"] = "face_matching";

  LOGI("task: face_matching, description: %s\n", description.c_str());

  try {
    // 新增逻辑：如果description等于"人名列表"，直接返回所有注册的人名
    if (description == "人名列表") {
      response["payload"]["event"] = "registered_names";
      nlohmann::json message_list = nlohmann::json::array();
      for (const auto& pair : name_to_id_map_) {
        nlohmann::json item;
        message_list.push_back(pair.first);
      }
      response["payload"]["message_list"] = message_list;

      printf("registered_names size: %lu\n", message_list.size());
      return response;
    }

    // 直接使用人名作为参数
    auto matched_images = get_face_images(description);

    LOGI("matched_images size: %lu\n", matched_images.size());

    nlohmann::json message_list = nlohmann::json::array();
    for (const auto& image_path : matched_images) {
      std::string base64_data =
          APIClient::CommonFunctions::loadImageAsBase64(image_path);
      nlohmann::json item;
      item["image"] = !base64_data.empty() ? base64_data : "";
      item["path"] = image_path;
      parse_face_info(image_path, item);
      message_list.push_back(item);
    }

    response["payload"]["message_list"] = message_list;
  } catch (const std::exception& e) {
    response["payload"]["message_list"] = nlohmann::json::array();
  }

  return response;
}

void FaceMatchingTask::parse_face_info(const std::string& image_path,
                                       nlohmann::json& item) {
  try {
    std::string filename = fs::path(image_path).filename().string();
    std::regex pattern(R"((\d{8}_\d{6}).*faceID_(\d+))");
    std::smatch matches;

    if (std::regex_search(filename, matches, pattern) && matches.size() == 3) {
      std::string capture_time = matches[1].str();
      std::string track_id = matches[2].str();

      item["capture_time"] = capture_time;
      item["track_id"] = track_id;

      LOGI("解析成功: capture_time=%s, track_id=%s\n", capture_time.c_str(),
           track_id.c_str());
    } else {
      LOGI("无法解析文件名: %s\n", filename.c_str());
      item["capture_time"] = "";
      item["track_id"] = "";
    }
  } catch (const std::exception& e) {
    LOGI("解析face_info异常: %s\n", e.what());
    item["capture_time"] = "";
    item["track_id"] = "";
  }
}
