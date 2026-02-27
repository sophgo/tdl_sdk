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
#include <thread>

#include <iomanip>
#include "app/app_data_types.hpp"
#include "app/app_task.hpp"
#include "components/encoder/image_encoder/image_encoder.hpp"
#include "face_pet_capture/face_pet_capture_app.hpp"
#include "framework/utils/tdl_log.hpp"
#include "matcher/base_matcher.hpp"
#include "opencv2/opencv.hpp"

#define FACE_FEAT_SIZE 256
#define SIMILARITY_THRESHOLD 0.4

// Global variables for online face recognition
std::vector<std::vector<float>> g_registered_faces;
std::vector<int> g_registered_face_ids;

static const char *emotionStr[] = {"Anger",   "Disgust", "Fear",    "Happy",
                                   "Neutral", "Sad",     "Surprise"};

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

bool make_dir(const char *path, mode_t mode = 0755) {
  if (mkdir(path, mode) == 0) {
    return true;
  }
  if (errno == EEXIST) {
    return true;
  }
  return false;
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

  // Build gallery_dir path
  const std::string gallery_dir = output_folder_path + "/registered_feature";

  // Check if gallery_dir exists
  struct stat st;
  bool use_gallery =
      (stat(gallery_dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode));

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

  char identity_txt_path[512];
  snprintf(identity_txt_path, sizeof(identity_txt_path), "%s/identity_info.txt",
           output_folder_path.c_str());
  std::ofstream identity_ofs(identity_txt_path, std::ios::app);

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

  while (true) {
    int processing_channel_num = app_task->getProcessingChannelNum();
    if (processing_channel_num == 0) {
      std::cout << "no processing channel, break" << std::endl;
      break;
    }

    for (const auto &channel_name : channel_names) {
      Packet result;
      int ret = app_task->getResult(channel_name, result);

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
            } else {
              printf("Face matched: id=%d, track_id=%lu, similarity=%.4f\n",
                     match_id, snapshot.track_id, max_similarity);
            }
          }

          // Set face ID for ReID tracking
          if (match_id >= 0) {
            snapshot_comp->setFaceID(snapshot.pair_track_id, match_id);
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
                     "_male[%d]_glass[%d]_age[%d]_emotion[%s]", male, glass,
                     age, emotionStr[emotion]);
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
            std::ofstream ofs(filename, std::ios::binary);
            if (ofs) {
              ofs.write(reinterpret_cast<const char *>(snap_buf.data()),
                        snap_buf.size());
            }
          }
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

  app_task->release();
  std::cout << "App task released." << std::endl;

  return 0;
}
