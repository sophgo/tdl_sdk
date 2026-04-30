#include "identity_browse_task.hpp"
#include <dirent.h>
#include <sys/stat.h>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include "components/network/api_poster/api_client.hpp"

#include "components/media_analysis/media_analysis_server.hpp"

namespace {
void trim_right(std::string& s) {
  while (!s.empty() &&
         std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.pop_back();
  }
}

std::unordered_map<int, std::string> load_registered_name_map(
    const std::string& registered_info_path) {
  std::unordered_map<int, std::string> registered_name_map;
  std::ifstream ifs(registered_info_path);
  if (!ifs.is_open()) {
    return registered_name_map;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    size_t split_pos = line.find_last_of(" \t");
    if (split_pos == std::string::npos || split_pos + 1 >= line.size()) {
      continue;
    }
    std::string name = line.substr(0, split_pos);
    trim_right(name);
    if (name.empty()) continue;

    try {
      int id = std::stoi(line.substr(split_pos + 1));
      registered_name_map[id] = name;
    } catch (...) {
      continue;
    }
  }

  return registered_name_map;
}
}  // namespace

json IdentityBrowseTask::handle_event(const json& request,
                                      const std::string& description) {
  json response = request;
  response["type"] = "event";
  response["source"] = "c_backend";
  response["destination"] = "web_client";
  response["payload"]["event"] = "browse_identity";

  std::string root_path = MediaAnalysisServer::GetInstance()->get_data_path();
  if (root_path.empty()) {
    response["error"] = "Path is empty";
    return response;
  }

  std::string identity_path = root_path + "/identity";
  std::string registered_info_path = root_path + "/registered_info.txt";
  auto registered_name_map = load_registered_name_map(registered_info_path);
  DIR* dir = opendir(identity_path.c_str());
  if (dir == nullptr) {
    response["error"] = "Cannot open identity directory: " + identity_path;
    return response;
  }

  nlohmann::json message_list = nlohmann::json::array();
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type != DT_DIR || strcmp(entry->d_name, ".") == 0 ||
        strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, "-2") == 0) {
      continue;
    }

    std::string id_dir_path = identity_path + "/" + entry->d_name;
    DIR* id_dir = opendir(id_dir_path.c_str());
    if (id_dir == nullptr) continue;

    struct dirent* img_entry;
    while ((img_entry = readdir(id_dir)) != nullptr) {
      if (img_entry->d_type != DT_REG) continue;
      std::string filename = img_entry->d_name;
      if (filename.size() < 4 ||
          filename.substr(filename.size() - 4) != ".jpg") {
        continue;
      }

      std::string full_path = id_dir_path + "/" + filename;

      // 不再使用 Base64，改为返回代理 URL
      // 这里端口号 hardcode 为 8000，即 MediaAnalysisServer 的默认端口
      std::string proxy_url = "/api/image_proxy?path=" + full_path;

      nlohmann::json item;
      item["image"] = proxy_url;
      item["path"] = full_path;
      int registered_id = -1;
      try {
        registered_id = std::stoi(entry->d_name);
      } catch (...) {
        registered_id = -1;
      }
      item["registered_id"] = registered_id;
      auto it = registered_name_map.find(registered_id);
      if (it != registered_name_map.end()) {
        item["match_id"] = it->second;
      } else {
        item["match_id"] = entry->d_name;
      }
      parse_identity_info(filename, item);
      message_list.push_back(item);
    }
    closedir(id_dir);
  }
  closedir(dir);

  response["payload"]["message_list"] = message_list;
  return response;
}

void IdentityBrowseTask::parse_identity_info(const std::string& filename,
                                             json& item) {
  try {
    // 20260417_145318_frameID_11_registeredID_0_faceID_1_pairID_2_qua_0.502male[1]_glass[0]_age[31]_emotion[Neutral].jpg
    // 20260417_145325_frameID_145_registeredID_0_personID_3_pairID_0_qua_0.460.jpg

    std::regex time_pattern(R"(^(\d{8}_\d{6}))");
    std::regex frame_pattern(R"(_frameID_(\d+))");
    std::regex track_pattern(R"((?:faceID_|personID_)(\d+))");
    std::regex attr_pattern(R"(qua_[\d\.]+(.*)\.jpg$)");

    std::smatch matches;
    if (std::regex_search(filename, matches, time_pattern)) {
      item["capture_time"] = matches[1].str();
    }
    if (std::regex_search(filename, matches, frame_pattern)) {
      item["frame_id"] = matches[1].str();
    }
    if (std::regex_search(filename, matches, track_pattern)) {
      item["track_id"] = matches[1].str();
    }
    if (std::regex_search(filename, matches, attr_pattern)) {
      item["attributes"] = matches[1].str();
    }

    // Determine obj_type
    if (filename.find("faceID_") != std::string::npos) {
      item["obj_type"] = "face";
    } else if (filename.find("personID_") != std::string::npos) {
      item["obj_type"] = "person";
    }

  } catch (const std::exception& e) {
    std::cerr << "Parse error: " << e.what() << std::endl;
  }
}
