#include <dirent.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <cctype>
#include <iomanip>
#include "app/app_data_types.hpp"
#include "app/app_task.hpp"
#include "components/encoder/image_encoder/image_encoder.hpp"
#include "components/media_analysis/media_analysis_event_manager.hpp"
#include "components/media_analysis/media_analysis_server.hpp"
#include "components/snapshot/object_snapshot.hpp"
#include "face_pet_capture/face_pet_capture_app.hpp"
#include "framework/utils/tdl_log.hpp"
#include "matcher/base_matcher.hpp"
#include "opencv2/opencv.hpp"
#include "tasks/face/face_matching_task.hpp"

#define FACE_FEAT_SIZE 256
#define SIMILARITY_THRESHOLD 0.4
// #define SAVE_CAPTURE

// Global variables for online face recognition
std::vector<std::vector<float>> g_registered_faces;
std::vector<int> g_registered_face_ids;

static const char *emotionStr[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                   "Neutral", "Sad",     "Surprise"};

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

bool make_dir(const char *path, mode_t mode = 0755) {
  if (mkdir(path, mode) == 0) {
    return true;
  }
  if (errno == EEXIST) {
    return true;
  }
  return false;
}

static void trim_right(std::string &s) {
  while (!s.empty() &&
         std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.pop_back();
  }
}

static bool load_registered_name_map(
    const std::string &registered_info_path,
    std::unordered_map<int, std::string> &registered_name_map) {
  registered_name_map.clear();
  std::ifstream ifs(registered_info_path);
  if (!ifs.is_open()) {
    return false;
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

  return true;
}

static std::string json_escape(const std::string &input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char c : input) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '\"') {
      out += "\\\"";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

static std::string resolve_registered_label(
    int registered_id,
    const std::unordered_map<int, std::string> &registered_name_map) {
  if (registered_id != -1) {
    auto it = registered_name_map.find(registered_id);
    if (it != registered_name_map.end()) {
      return it->second;
    }
  }
  return std::to_string(registered_id);
}

static void move_unknown_person_images(const char *identity_dir,
                                       int registered_id, uint64_t track_id) {
  if (registered_id == -1) {
    return;
  }

  std::string unknown_dir = std::string(identity_dir) + "/-1";
  DIR *dir = opendir(unknown_dir.c_str());
  if (dir == nullptr) {
    return;
  }

  char dst_dir[512];
  snprintf(dst_dir, sizeof(dst_dir), "%s/%d", identity_dir, registered_id);
  make_dir(dst_dir, 0755);

  std::string track_token = "_personID_" + std::to_string(track_id) + "_";
  std::string old_reg_token = "_registeredID_-1";
  std::string new_reg_token = "_registeredID_" + std::to_string(registered_id);

  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type != DT_REG) continue;

    std::string filename = entry->d_name;
    if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".jpg") {
      continue;
    }
    if (filename.find(track_token) == std::string::npos) {
      continue;
    }

    std::string new_filename = filename;
    size_t reg_pos = new_filename.find(old_reg_token);
    if (reg_pos != std::string::npos) {
      new_filename.replace(reg_pos, old_reg_token.size(), new_reg_token);
    }

    std::string src_path = unknown_dir + "/" + filename;
    std::string dst_path = std::string(dst_dir) + "/" + new_filename;
    if (rename(src_path.c_str(), dst_path.c_str()) != 0) {
      std::cerr << "Failed to move " << src_path << " to " << dst_path << ": "
                << strerror(errno) << std::endl;
    }
  }

  closedir(dir);
}

// Create folder structure based on registered ID
bool create_id_folder(const char *dir_path, int registered_id, char *dst_dir,
                      size_t dst_dir_size) {
  char temp_dir[512];
  snprintf(temp_dir, sizeof(temp_dir), "%s/%d", dir_path, registered_id);
  make_dir(temp_dir, 0755);
  strncpy(dst_dir, temp_dir, dst_dir_size);
  return true;
}

int get_gallery_feature(const char *sz_feat_file,
                        std::vector<float> &g_feature) {
  FILE *fp = fopen(sz_feat_file, "rb");
  if (fp == NULL) {
    printf("read %s failed\n", sz_feat_file);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  int8_t *ptr_feat = (int8_t *)malloc(len);
  fread(ptr_feat, 1, len, fp);
  fclose(fp);
  printf("read %s done,len:%d\n", sz_feat_file, len);
  if (len != FACE_FEAT_SIZE) {
    free(ptr_feat);
    printf("read %s failed,len:%d != %d\n", sz_feat_file, len, FACE_FEAT_SIZE);
    return -1;
  }

  for (size_t i = 0; i < FACE_FEAT_SIZE; i++) {
    g_feature[i] = (float)ptr_feat[i];
  }

  return 0;
}

// Create feature info objects
std::vector<std::shared_ptr<ModelFeatureInfo>> createModelFeatureInfos(
    const std::vector<std::vector<float>> &features) {
  std::vector<std::shared_ptr<ModelFeatureInfo>> feature_infos;

  for (const auto &feature : features) {
    auto feature_info = std::make_shared<ModelFeatureInfo>();
    int feature_dim = feature.size();

    // Allocate memory and convert to UINT8
    feature_info->embedding = new uint8_t[feature_dim * sizeof(float)];
    float *dest = reinterpret_cast<float *>(feature_info->embedding);

    for (int j = 0; j < feature_dim; j++) {
      dest[j] = feature[j];
    }

    feature_info->embedding_num = feature_dim;
    feature_info->embedding_type = TDLDataType::FP32;
    feature_infos.push_back(feature_info);
  }

  return feature_infos;
}

// Register features from gallery directory
void register_feature(const std::string &gallery_dir,
                      std::shared_ptr<BaseMatcher> &feature_matcher,
                      std::vector<int> &folder_ids) {
  std::vector<std::vector<float>> gallery_features_tmp;

  DIR *dir = opendir(gallery_dir.c_str());
  if (dir == nullptr) {
    printf("Failed to open gallery directory: %s\n", gallery_dir.c_str());
    return;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    // Skip non-directories and current/parent directories
    if (entry->d_type != DT_DIR || strcmp(entry->d_name, ".") == 0 ||
        strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    // Try to convert folder name to integer ID
    int folder_id;
    try {
      folder_id = std::stoi(entry->d_name);
    } catch (const std::invalid_argument &e) {
      printf("Skip folder with non-integer name: %s\n", entry->d_name);
      continue;
    } catch (const std::out_of_range &e) {
      printf("Folder name %s is out of integer range\n", entry->d_name);
      continue;
    }

    // Open subdirectory
    std::string subdir_path = gallery_dir + "/" + entry->d_name;
    DIR *subdir = opendir(subdir_path.c_str());
    if (!subdir) {
      printf("Failed to open subdirectory: %s\n", subdir_path.c_str());
      continue;
    }

    // Traverse all files in subdirectory
    struct dirent *file_entry;
    while ((file_entry = readdir(subdir)) != nullptr) {
      // Only process regular files with .bin extension
      if (file_entry->d_type != DT_REG) continue;
      std::string filename = file_entry->d_name;
      if (filename.size() < 4 ||
          filename.substr(filename.size() - 4) != ".bin") {
        continue;
      }

      // Build full bin file path
      std::string bin_file_path = subdir_path + "/" + filename;

      // Read feature
      std::vector<float> g_feature(FACE_FEAT_SIZE);
      if (get_gallery_feature(bin_file_path.c_str(), g_feature) != 0) {
        printf("Skip bin file: %s\n", bin_file_path.c_str());
        continue;
      }

      // Add feature and corresponding folder ID
      gallery_features_tmp.push_back(g_feature);
      folder_ids.push_back(folder_id);
    }
    closedir(subdir);
  }

  closedir(dir);

  if (gallery_features_tmp.size() == 0) {
    printf(
        "Warning: failed to register feature! Local recognition might fail.\n");
  }

  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_features =
      createModelFeatureInfos(gallery_features_tmp);

  feature_matcher->loadGallery(gallery_features);

  printf("register features sucessfully, gallery_features size: %ld\n",
         gallery_features.size());

  return;
}

// Compute cosine similarity between two features
float compute_similarity(const std::vector<float> &feat1,
                         const std::vector<float> &feat2) {
  if (feat1.size() != feat2.size() || feat1.empty()) {
    return -1.0f;
  }

  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;

  for (size_t i = 0; i < feat1.size(); i++) {
    dot_product += feat1[i] * feat2[i];
    norm1 += feat1[i] * feat1[i];
    norm2 += feat2[i] * feat2[i];
  }

  norm1 = sqrt(norm1);
  norm2 = sqrt(norm2);

  if (norm1 > 0 && norm2 > 0) {
    return dot_product / (norm1 * norm2);
  }

  return 0.0f;
}

// Match face feature against registered faces
int match_face_online(const std::vector<float> &query_feature,
                      float &max_similarity) {
  int match_id = -1;

  for (size_t i = 0; i < g_registered_faces.size(); i++) {
    float sim = compute_similarity(query_feature, g_registered_faces[i]);
    if (sim > max_similarity) {
      max_similarity = sim;
      match_id = g_registered_face_ids[i];
    }
  }
  printf("max_similarity: %f, match_id: %d\n", max_similarity, match_id);

  if (max_similarity >= SIMILARITY_THRESHOLD) {
    return match_id;
  }

  return -1;
}

struct FrameTask {
  std::vector<uint8_t> encoded_data;
  uint64_t timestamp;
  int channel_id;
  uint64_t frame_id;
  std::string metadata_json;
};

class AsyncImageSender {
 public:
  AsyncImageSender() : running_(true) {
    worker_thread_ = std::thread(&AsyncImageSender::worker_loop, this);
  }

  ~AsyncImageSender() { stop(); }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    cond_.notify_all();
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

  void push(FrameTask task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() > 10) {
      queue_.pop();
    }
    queue_.push(task);
    cond_.notify_one();
  }

 private:
  void worker_loop() {
    while (true) {
      FrameTask task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || !running_; });

        if (!running_ && queue_.empty()) break;

        if (queue_.empty()) continue;

        task = queue_.front();
        queue_.pop();
      }

      if (!task.encoded_data.empty()) {
        MediaAnalysisServer::GetInstance()->send_image_to_web_client(
            task.encoded_data, task.timestamp, task.channel_id, task.frame_id,
            task.metadata_json);
      }
    }
  }

  std::queue<FrameTask> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::thread worker_thread_;
  bool running_;
};

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <config_file>\n", argv[0]);
    return -1;
  }

  const std::string config_file = argv[1];

  // Read data_path from config file
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open config file: " << config_file << std::endl;
    return -1;
  }
  nlohmann::json j;
  ifs >> j;

  // Check if data_path exists
  if (!j.contains("data_path")) {
    std::cerr << "Config file error: missing 'data_path' field" << std::endl;
    return -1;
  }
  const std::string output_folder_path = j["data_path"];
  make_dir(output_folder_path.c_str(), 0755);

  // Initialize MediaAnalysisServer
  std::cout << "Initializing MediaAnalysisServer..." << std::endl;
  MediaAnalysisServer *server = MediaAnalysisServer::GetInstance();
  server->parse_config(config_file);
  server->init();

  AsyncImageSender sender;

  // Build gallery_dir path
  const std::string gallery_dir = output_folder_path + "/registered_feature";
  make_dir(gallery_dir.c_str(), 0755);

  // Check if gallery_dir exists
  struct stat st;
  bool use_gallery = false;

  DIR *dir = opendir(gallery_dir.c_str());
  if (dir != nullptr) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 &&
          strcmp(entry->d_name, "..") != 0) {
        use_gallery = true;
        break;
      }
    }
    closedir(dir);
  }

  std::shared_ptr<BaseMatcher> feature_matcher;
  std::vector<int> face_feature_ids;

  if (use_gallery) {
    // Use gallery for matching
    std::cout << "Using gallery directory: " << gallery_dir << std::endl;
    feature_matcher = BaseMatcher::getMatcher("bm");
    register_feature(gallery_dir, feature_matcher, face_feature_ids);
  } else {
    // Use online registration (no gallery directory)
    std::cout << "Gallery directory not found, using online face registration"
              << std::endl;
  }

  // Initialize App
  std::shared_ptr<AppTask> app_task =
      AppFactory::createAppTask("face_pet_capture", config_file);

  int ret = app_task->init();
  if (ret != 0) {
    std::cout << "app_task init failed" << std::endl;
    return -1;
  }

  std::vector<std::string> channel_names = app_task->getChannelNames();

  std::shared_ptr<ImageEncoder> encoder =
      (std::dynamic_pointer_cast<FacePetCaptureApp>(app_task))
          ->getImageEncoder(std::string(channel_names[0]));

  // Create output directories
  char identity_dir[512];
  snprintf(identity_dir, sizeof(identity_dir), "%s/identity",
           output_folder_path.c_str());
  make_dir(identity_dir, 0755);

  char image_feature_dir[512];
  snprintf(image_feature_dir, sizeof(image_feature_dir), "%s/image_feature",
           output_folder_path.c_str());
  make_dir(image_feature_dir, 0755);

  char llm_analysis_dir[512];
  snprintf(llm_analysis_dir, sizeof(llm_analysis_dir), "%s/llm_analysis",
           output_folder_path.c_str());
  make_dir(llm_analysis_dir, 0755);
  char latest_analysis_dir[512] = {0};

  char identity_txt_path[512];
  snprintf(identity_txt_path, sizeof(identity_txt_path), "%s/identity_info.txt",
           output_folder_path.c_str());
  std::ofstream identity_ofs(identity_txt_path, std::ios::app);

  char registered_info_path[512];
  snprintf(registered_info_path, sizeof(registered_info_path),
           "%s/registered_info.txt", output_folder_path.c_str());
  std::ofstream registered_info_ofs;
  if (!use_gallery) {
    registered_info_ofs.open(registered_info_path, std::ios::app);
  }
  std::unordered_map<int, std::string> registered_name_map;

  time_t rawtime;
  struct tm *timeinfo;
  char timestamp_str[20];

  std::shared_ptr<ObjectSnapshot> snapshot_comp =
      (std::dynamic_pointer_cast<FacePetCaptureApp>(app_task))
          ->getSnapshot(std::string(channel_names[0]));
  if (snapshot_comp == nullptr) {
    std::cout << "snapshot_comp is null" << std::endl;
    return -1;
  }

  uint64_t counter = 0;
  uint64_t last_counter = 0;
  uint32_t last_time_ms = get_time_in_ms();

  while (true) {
    int processing_channel_num = app_task->getProcessingChannelNum();
    if (processing_channel_num == 0) {
      std::cout << "no processing channel, break" << std::endl;
      break;
    }

    for (const auto &channel_name : channel_names) {
      Packet result;
      int ret = app_task->getResult(channel_name, result);

      counter++;
      int frm_diff = counter - last_counter;
      if (frm_diff >= 30) {
        uint32_t cur_ts_ms = get_time_in_ms();
        float infer_time = (float)(cur_ts_ms - last_time_ms) / frm_diff;
        float fps = (infer_time > 0.0f) ? (1000.0f / infer_time) : 0.0f;
        last_time_ms = cur_ts_ms;
        last_counter = counter;
        printf("frame:%" PRIu64 ", infer time:%.2f ms, fps:%.2f\n", counter,
               infer_time, fps);
      }

      std::shared_ptr<FacePetCaptureResult> cap_result =
          result.get<std::shared_ptr<FacePetCaptureResult>>();
      if (cap_result == nullptr) {
        std::cout << "cap_result is nullptr" << std::endl;
        continue;
      }

      time(&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S", timeinfo);

      const std::map<uint64_t, std::vector<float>> &features =
          cap_result->features;

      // Save snapshots
      for (auto &snapshot : cap_result->face_snapshots) {
        int match_id = -1;
        float max_similarity = 0.0f;

        // Face recognition
        if (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE &&
            cap_result->features.count(snapshot.track_id) &&
            !cap_result->features[snapshot.track_id].empty()) {
          std::vector<float> query_feature =
              cap_result->features.at(snapshot.track_id);

          if (use_gallery) {
            // Use gallery matcher
            std::vector<uint64_t> face_track_ids;
            std::vector<std::vector<float>> query_features_tmp;
            face_track_ids.push_back(snapshot.track_id);
            query_features_tmp.push_back(query_feature);

            std::vector<std::shared_ptr<ModelFeatureInfo>> query_features =
                createModelFeatureInfos(query_features_tmp);
            MatchResult results;
            feature_matcher->queryWithTopK(query_features, 1, results);

            if (!results.indices.empty() && !results.scores.empty() &&
                !results.indices[0].empty() && !results.scores[0].empty()) {
              if (results.scores[0][0] > SIMILARITY_THRESHOLD) {
                match_id = face_feature_ids[results.indices[0][0]];
                max_similarity = results.scores[0][0];
              }
            }
          } else {
            // Use online matching
            match_id = match_face_online(query_feature, max_similarity);

            // If not matched, register as new face
            if (match_id == -1) {
              match_id = g_registered_faces.size();
              g_registered_faces.push_back(query_feature);
              g_registered_face_ids.push_back(match_id);
              printf("New face registered: id=%d, track_id=%lu\n", match_id,
                     snapshot.track_id);

              if (!use_gallery) {
                char dst_dir[512];
                char filename[512];
                create_id_folder(gallery_dir.c_str(), match_id, dst_dir,
                                 sizeof(dst_dir));
                sprintf(filename, "%s/%s_registeredID_%d.bin", dst_dir,
                        timestamp_str, match_id);
                FILE *f = fopen(filename, "wb");
                if (f) {
                  std::vector<int8_t> int8_feat(query_feature.size());
                  for (size_t i = 0; i < query_feature.size(); i++) {
                    int8_feat[i] = (int8_t)query_feature[i];
                  }
                  fwrite(int8_feat.data(), 1, int8_feat.size(), f);
                  fclose(f);
                  printf("Saved online registered face feature to %s\n",
                         filename);
                }

                if (registered_info_ofs.is_open()) {
                  char name_buf[32];
                  snprintf(name_buf, sizeof(name_buf), "人员%d", match_id + 1);
                  registered_info_ofs << name_buf << " " << match_id
                                      << std::endl;
                }
              }
            } else {
              printf("Face matched: id=%d, track_id=%lu, similarity=%.4f\n",
                     match_id, snapshot.track_id, max_similarity);
            }
          }

          // Set face ID for ReID tracking
          if (match_id >= 0) {
            snapshot_comp->setFaceID(snapshot.track_id, snapshot.pair_track_id,
                                     match_id);

            auto task = MediaAnalysisEventManager::GetInstance()->GetTask(
                "face_matching");
            if (task) {
              auto face_task =
                  std::dynamic_pointer_cast<FaceMatchingTask>(task);
              if (face_task) {
                face_task->add_face_info(match_id, snapshot.track_id);
              }
            }
          }
        }

        if (snapshot.object_box_info.object_type == OBJECT_TYPE_PERSON) {
          match_id = snapshot.registered_id;
        }

        // Output identity info to txt
        if (identity_ofs.is_open()) {
          if (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE ||
              snapshot.registered_id != -2) {
            std::string obj_type =
                (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE)
                    ? "face"
                    : "person";
            identity_ofs << snapshot.snapshot_frame_id << "," << obj_type << ","
                         << snapshot.object_box_info.x1 << ","
                         << snapshot.object_box_info.y1 << ","
                         << snapshot.object_box_info.x2 << ","
                         << snapshot.object_box_info.y2 << "," << match_id
                         << std::endl;
          }
        }

        if (snapshot.object_image) {
          char dst_dir[512];
          char filename[512];

          // Create folder based on registered ID (for both face and person)
          create_id_folder(identity_dir, match_id, dst_dir, sizeof(dst_dir));

          // Get attributes if available (for face)
          std::string attr_str = "";
          if (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE) {
            int male = 0;
            int glass = 0;
            int age = 0;
            int emotion = 4;  // Neutral

            if (cap_result->face_attributes.count(snapshot.track_id)) {
              auto &attrs = cap_result->face_attributes[snapshot.track_id];
              male = attrs[OBJECT_ATTRIBUTE_HUMAN_GENDER] > 0.5 ? 1 : 0;
              glass = attrs[OBJECT_ATTRIBUTE_HUMAN_GLASSES] > 0.5 ? 1 : 0;
              age = (int)(attrs[OBJECT_ATTRIBUTE_HUMAN_AGE] * 100);
              emotion = (int)attrs[OBJECT_ATTRIBUTE_HUMAN_EMOTION];
              if (emotion < 0 || emotion > 6) emotion = 4;
            }

            char attr_buf[256];
            snprintf(attr_buf, sizeof(attr_buf),
                     "male[%d]_glass[%d]_age[%d]_emotion[%s]", male, glass, age,
                     emotionStr[emotion]);
            attr_str = std::string(attr_buf);
          }

          std::string obj_type =
              (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE)
                  ? "face"
                  : "person";

          sprintf(filename,
                  "%s/%s_frameID_%" PRIu64
                  "_registeredID_%d"
                  "_%sID_%" PRIu64 "_pairID_%" PRIu64 "_qua_%.3f%s.jpg",
                  dst_dir, timestamp_str, snapshot.snapshot_frame_id, match_id,
                  obj_type.c_str(), snapshot.track_id, snapshot.pair_track_id,
                  snapshot.quality, attr_str.c_str());

          std::vector<uint8_t> snap_buf;
          if (encoder->encodeFrame(snapshot.object_image, snap_buf)) {
#ifdef SAVE_CAPTURE
            std::ofstream ofs(filename, std::ios::binary);
            if (ofs) {
              ofs.write(reinterpret_cast<const char *>(snap_buf.data()),
                        snap_buf.size());
            }
#endif
          }
        }

#ifdef SAVE_CAPTURE
        // Save features
        if (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE &&
            cap_result->features.count(snapshot.track_id) &&
            !cap_result->features[snapshot.track_id].empty()) {
          char dst_dir[512];
          char filename[512];
          create_id_folder(image_feature_dir, match_id, dst_dir,
                           sizeof(dst_dir));

          std::string attr_str = "";
          int male = 0, glass = 0, age = 0, emotion = 4;
          if (cap_result->face_attributes.count(snapshot.track_id)) {
            auto &attrs = cap_result->face_attributes[snapshot.track_id];
            male = attrs[OBJECT_ATTRIBUTE_HUMAN_GENDER] > 0.5 ? 1 : 0;
            glass = attrs[OBJECT_ATTRIBUTE_HUMAN_GLASSES] > 0.5 ? 1 : 0;
            age = (int)(attrs[OBJECT_ATTRIBUTE_HUMAN_AGE] * 100);
            emotion = (int)attrs[OBJECT_ATTRIBUTE_HUMAN_EMOTION];
            if (emotion < 0 || emotion > 6) emotion = 4;
          }
          char attr_buf[256];
          snprintf(attr_buf, sizeof(attr_buf),
                   "_male[%d]_glass[%d]_age[%d]_emotion[%s]", male, glass, age,
                   emotionStr[emotion]);
          attr_str = std::string(attr_buf);

          sprintf(filename,
                  "%s/%s_frameID_%" PRIu64
                  "_registeredID_%d"
                  "_faceID_%" PRIu64 "_pairID_%" PRIu64 "_qua_%.3f%s.bin",
                  dst_dir, timestamp_str, snapshot.snapshot_frame_id, match_id,
                  snapshot.track_id, snapshot.pair_track_id, snapshot.quality,
                  attr_str.c_str());

          FILE *f = fopen(filename, "wb");
          if (f) {
            auto &feature = cap_result->features[snapshot.track_id];
            std::vector<int8_t> int8_feat(feature.size());
            for (size_t i = 0; i < feature.size(); i++) {
              int8_feat[i] = (int8_t)feature[i];
            }
            fwrite(int8_feat.data(), 1, int8_feat.size(), f);
            fclose(f);
            printf("Saved face feature to %s\n", filename);
          }
        }
#endif
      }

#ifdef SAVE_CAPTURE
      // Save full frame for LLM analysis (every 10 frames)
      if (cap_result->frame_id % 75 == 0) {
        char target_dir[512];
        int file_count = 0;

        if (strlen(latest_analysis_dir) > 0) {
          snprintf(target_dir, sizeof(target_dir), "%s/%s", llm_analysis_dir,
                   latest_analysis_dir);
          DIR *latest_subdir = opendir(target_dir);
          if (latest_subdir) {
            struct dirent *subentry;
            while ((subentry = readdir(latest_subdir)) != NULL) {
              if (subentry->d_type == DT_REG) {
                file_count++;
              }
            }
            closedir(latest_subdir);
          }
        }

        if (strlen(latest_analysis_dir) == 0 || file_count >= 10) {
          strncpy(latest_analysis_dir, timestamp_str,
                  sizeof(latest_analysis_dir));
          snprintf(target_dir, sizeof(target_dir), "%s/%s", llm_analysis_dir,
                   latest_analysis_dir);
          make_dir(target_dir, 0755);
        } else {
          snprintf(target_dir, sizeof(target_dir), "%s/%s", llm_analysis_dir,
                   latest_analysis_dir);
        }

        char filename[512];
        snprintf(filename, sizeof(filename), "%s/%08" PRIu64 ".jpg", target_dir,
                 cap_result->frame_id);

        std::vector<uint8_t> full_buf;
        if (encoder->encodeFrame(cap_result->image, full_buf)) {
          std::ofstream ofs(filename, std::ios::binary);
          if (ofs) {
            ofs.write(reinterpret_cast<const char *>(full_buf.data()),
                      full_buf.size());
          }
        }
      }
#endif

      // -----------------------------------------------------------------------
      // Send to MediaAnalysisServer
      // -----------------------------------------------------------------------
      if (cap_result->image) {
        load_registered_name_map(registered_info_path, registered_name_map);

        std::map<int, uint64_t> face_trackId_map;
        std::map<int, uint64_t> person_trackId_map;

        for (size_t i = 0; i < cap_result->track_results.size(); i++) {
          TrackerInfo track_info = cap_result->track_results[i];

          if (track_info.obj_idx_ != -1) {
            if (track_info.box_info_.object_type ==
                TDLObjectType::OBJECT_TYPE_FACE) {
              face_trackId_map[track_info.obj_idx_] = track_info.track_id_;
            } else if (track_info.box_info_.object_type ==
                       TDLObjectType::OBJECT_TYPE_PERSON) {
              person_trackId_map[track_info.obj_idx_ -
                                 cap_result->face_boxes.size()] =
                  track_info.track_id_;
            }
          }
        }

        std::string metadata_json = "{";
        metadata_json +=
            "\"width\":" + std::to_string(cap_result->frame_width) + ",";
        metadata_json +=
            "\"height\":" + std::to_string(cap_result->frame_height) + ",";
        metadata_json +=
            "\"source_width\":" + std::to_string(cap_result->source_width) +
            ",";
        metadata_json +=
            "\"source_height\":" + std::to_string(cap_result->source_height) +
            ",";
        metadata_json += "\"faces\":[";
        for (size_t i = 0; i < cap_result->face_boxes.size(); i++) {
          float x1 = cap_result->face_boxes[i].x1;
          float y1 = cap_result->face_boxes[i].y1;
          float x2 = cap_result->face_boxes[i].x2;
          float y2 = cap_result->face_boxes[i].y2;
          int64_t track_id = -1;
          if (face_trackId_map.find(i) != face_trackId_map.end()) {
            track_id = face_trackId_map[i];
          }
          int registered_id = snapshot_comp->getRegisteredID(track_id);
          std::string registered_label =
              resolve_registered_label(registered_id, registered_name_map);

          metadata_json += "{\"x1\":" + std::to_string(x1) +
                           ",\"y1\":" + std::to_string(y1) +
                           ",\"x2\":" + std::to_string(x2) +
                           ",\"y2\":" + std::to_string(y2) +
                           ",\"track_id\":" + std::to_string(track_id) +
                           ",\"registered_id\":\"" +
                           json_escape(registered_label) + "\"}";
          if (i < cap_result->face_boxes.size() - 1) metadata_json += ",";
        }
        metadata_json += "],";

        metadata_json += "\"persons\":[";
        for (size_t i = 0; i < cap_result->person_boxes.size(); i++) {
          float x1 = cap_result->person_boxes[i].x1;
          float y1 = cap_result->person_boxes[i].y1;
          float x2 = cap_result->person_boxes[i].x2;
          float y2 = cap_result->person_boxes[i].y2;
          int64_t track_id = -1;
          if (person_trackId_map.find(i) != person_trackId_map.end()) {
            track_id = person_trackId_map[i];
          }
          int registered_id = snapshot_comp->getRegisteredID(track_id);
          if (registered_id != -1 && track_id != static_cast<uint64_t>(-1)) {
            move_unknown_person_images(identity_dir, registered_id, track_id);
          }
          std::string registered_label =
              resolve_registered_label(registered_id, registered_name_map);

          metadata_json += "{\"x1\":" + std::to_string(x1) +
                           ",\"y1\":" + std::to_string(y1) +
                           ",\"x2\":" + std::to_string(x2) +
                           ",\"y2\":" + std::to_string(y2) +
                           ",\"track_id\":" + std::to_string(track_id) +
                           ",\"registered_id\":\"" +
                           json_escape(registered_label) + "\"}";
          if (i < cap_result->person_boxes.size() - 1) metadata_json += ",";
        }
        metadata_json += "]";
        metadata_json += "}";

        // std::cout << metadata_json << std::endl;

        // for (int i = 0; i < snapshot.object_box_info.size; i++) {
        //   snapshot.object_box_info[i].registered_id = match_id;
        // }

        std::vector<uint8_t> encoded_data;
        if (encoder->encodeFrame(cap_result->image, encoded_data)) {
          FrameTask task;
          task.encoded_data = std::move(encoded_data);
          task.metadata_json = std::move(metadata_json);

          // Get current timestamp (ms)
          auto now = std::chrono::system_clock::now();
          auto duration = now.time_since_epoch();
          task.timestamp =
              std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                  .count();

          try {
            task.channel_id = std::stoi(channel_name);
          } catch (...) {
            task.channel_id = 0;
          }
          task.frame_id = cap_result->frame_id;

          sender.push(task);
        }
      }

      if (ret != 0) {  // 如果放在前面，无法保存最后强制输出的抓拍图
        std::cout << "get result failed" << std::endl;
        app_task->removeChannel(channel_name);
        continue;
      }
    }
  }

  std::cout << "Stopping..." << std::endl;

  // Close identity log file
  if (identity_ofs.is_open()) {
    identity_ofs.close();
    std::cout << "Identity log file closed." << std::endl;
  }

  if (registered_info_ofs.is_open()) {
    registered_info_ofs.close();
  }

  sender.stop();
  std::cout << "Sender stopped." << std::endl;

  server->stop();
  std::cout << "Server stopped." << std::endl;

  app_task->release();
  std::cout << "App task released." << std::endl;

  return 0;
}
