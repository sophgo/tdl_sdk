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
#include "components/media_analysis/media_analysis_event_manager.hpp"
#include "components/media_analysis/media_analysis_server.hpp"
#include "face_pet_capture/face_pet_capture_app.hpp"
#include "framework/utils/tdl_log.hpp"
#include "matcher/base_matcher.hpp"
#include "opencv2/opencv.hpp"
#include "tasks/face/face_matching_task.hpp"

#define FACE_FEAT_SIZE 256

#define SAVE_CAPTURE_IMAGE

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

bool create_id_folder(const char *dir_path, uint64_t id1, uint64_t id2,
                      char *dst_dir, size_t dst_dir_size) {
  char temp_dir[512];
  char old_dir[512];

  if (id2 <= 0) {
    snprintf(temp_dir, sizeof(temp_dir), "%s/%" PRIu64 "_-1", dir_path, id1);
    make_dir(temp_dir, 0755);
    strncpy(dst_dir, temp_dir, dst_dir_size);
    return true;
  } else {
    snprintf(old_dir, sizeof(old_dir), "%s/%" PRIu64 "_-1", dir_path, id1);
    snprintf(temp_dir, sizeof(temp_dir), "%s/%" PRIu64 "_%" PRIu64, dir_path,
             id1, id2);

    struct stat st;
    if (stat(old_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
      if (rename(old_dir, temp_dir) != 0) {
        fprintf(stderr, "Failed to rename directory from %s to %s: %s\n",
                old_dir, temp_dir, strerror(errno));
        return false;
      }
    } else {
      make_dir(temp_dir, 0755);
    }
    strncpy(dst_dir, temp_dir, dst_dir_size);
    return true;
  }
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

// 创建特征信息对象
std::vector<std::shared_ptr<ModelFeatureInfo>> createModelFeatureInfos(
    const std::vector<std::vector<float>> &features) {
  std::vector<std::shared_ptr<ModelFeatureInfo>> feature_infos;

  for (const auto &feature : features) {
    auto feature_info = std::make_shared<ModelFeatureInfo>();
    int feature_dim = feature.size();
    printf("!!!! feature_dim:%d\n", feature_dim);

    // 分配内存并转换为UINT8
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

// 修复register_feature函数声明和实现
void register_feature(
    const std::string &gallery_dir,
    std::shared_ptr<BaseMatcher> &feature_matcher,  // 修复参数列表
    std::vector<int> &folder_ids) {
  // 修复变量声明 - 移除引用符号并添加分号
  std::vector<std::vector<float>> gallery_features_tmp;

  DIR *dir = opendir(gallery_dir.c_str());
  if (dir == nullptr) {
    printf("Failed to open gallery directory: %s\n", gallery_dir.c_str());
    return;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    // 跳过非目录和当前/上级目录
    if (entry->d_type != DT_DIR || strcmp(entry->d_name, ".") == 0 ||
        strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    // 尝试将文件夹名称转换为整数ID
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

    // 打开子文件夹
    std::string subdir_path = gallery_dir + "/" + entry->d_name;
    DIR *subdir = opendir(subdir_path.c_str());
    if (!subdir) {
      printf("Failed to open subdirectory: %s\n", subdir_path.c_str());
      continue;
    }

    // 遍历子文件夹中的所有文件
    struct dirent *file_entry;
    while ((file_entry = readdir(subdir)) != nullptr) {
      // 仅处理普通文件且后缀为.bin
      if (file_entry->d_type != DT_REG) continue;
      std::string filename = file_entry->d_name;
      if (filename.size() < 4 ||
          filename.substr(filename.size() - 4) != ".bin") {
        continue;
      }

      // 构建完整的bin文件路径
      std::string bin_file_path = subdir_path + "/" + filename;

      // 读取特征
      std::vector<float> g_feature(FACE_FEAT_SIZE);
      if (get_gallery_feature(bin_file_path.c_str(), g_feature) != 0) {
        printf("Skip bin file: %s\n", bin_file_path.c_str());
        continue;
      }

      // 添加特征和对应的文件夹ID
      gallery_features_tmp.push_back(g_feature);
      folder_ids.push_back(folder_id);
    }
    closedir(subdir);  // 关闭子文件夹
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

// -----------------------------------------------------------------------------
// Media Analysis Logic
// -----------------------------------------------------------------------------

struct FrameTask {
  std::vector<uint8_t> encoded_data;
  uint64_t timestamp;
  int channel_id;
  uint64_t frame_id;
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
            task.encoded_data, task.timestamp, task.channel_id, task.frame_id);
      }
    }
  }

  std::queue<FrameTask> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::thread worker_thread_;
  bool running_;
};

std::atomic<bool> g_running(true);

void signal_handler(int signum) { g_running = false; }

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // 修改参数检查，只需要配置文件一个参数
  if (argc != 2) {
    printf("Usage: %s <config_file>\n", argv[0]);
    return -1;
  }

  const std::string config_file = argv[1];

  // 从配置文件读取gallery_dir和data_path
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open config file: " << config_file << std::endl;
    return -1;
  }
  nlohmann::json j;
  ifs >> j;

  // 检查data_path是否存在
  if (!j.contains("data_path")) {
    std::cerr << "Config file error: missing 'data_path' field" << std::endl;
    return -1;
  }
  const std::string output_folder_path = j["data_path"];

  // 构建gallery_dir路径
  const std::string gallery_dir = output_folder_path + "/registered_feature";

  // 检查gallery_dir路径是否存在
  struct stat st;
  if (stat(gallery_dir.c_str(), &st) != 0) {
    std::cerr << "Gallery directory does not exist: " << gallery_dir
              << std::endl;
    return -1;
  }

  signal(SIGINT, signal_handler);

  // Initialize MediaAnalysisServer
  std::cout << "Initializing MediaAnalysisServer..." << std::endl;
  MediaAnalysisServer *server = MediaAnalysisServer::GetInstance();
  server->parse_config(config_file);
  server->init();

  AsyncImageSender sender;

  std::vector<int> face_feature_ids;
  std::shared_ptr<BaseMatcher> feature_matcher = BaseMatcher::getMatcher("bm");
  register_feature(gallery_dir, feature_matcher, face_feature_ids);

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

  // Create output sub-directories
  char face_dir[512], person_dir[512], image_feature_dir[512],
      llm_analysis_dir[512];
  snprintf(face_dir, sizeof(face_dir), "%s/face", output_folder_path.c_str());
  make_dir(face_dir, 0755);
  snprintf(person_dir, sizeof(person_dir), "%s/person",
           output_folder_path.c_str());
  make_dir(person_dir, 0755);
  snprintf(image_feature_dir, sizeof(image_feature_dir), "%s/image_feature",
           output_folder_path.c_str());
  make_dir(image_feature_dir, 0755);
  snprintf(llm_analysis_dir, sizeof(llm_analysis_dir), "%s/llm_analysis",
           output_folder_path.c_str());
  make_dir(llm_analysis_dir, 0755);

  char latest_analysis_dir[20] = {0};

  time_t rawtime;
  struct tm *timeinfo;
  char timestamp_str[20];

  while (g_running) {
    int processing_channel_num = app_task->getProcessingChannelNum();
    if (processing_channel_num == 0) {
      std::cout << "no processing channel, break" << std::endl;
      break;
    }

    for (const auto &channel_name : channel_names) {
      Packet result;
      int ret = app_task->getResult(channel_name, result);

      if (ret != 0) {
        continue;
      }
      std::shared_ptr<FacePetCaptureResult> cap_result =
          result.get<std::shared_ptr<FacePetCaptureResult>>();
      if (cap_result == nullptr) {
        continue;
      }

      time(&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S", timeinfo);

#ifdef SAVE_CAPTURE_IMAGE

      const std::map<uint64_t, std::vector<float>> &features =
          cap_result->features;

      // Save snapshots
      for (auto &snapshot : cap_result->face_snapshots) {
        if (snapshot.object_image) {
          char dst_dir[512];
          char filename[512];

          if (snapshot.object_box_info.object_type == OBJECT_TYPE_PERSON) {
            create_id_folder(person_dir, snapshot.track_id,
                             snapshot.pair_track_id, dst_dir, sizeof(dst_dir));
            sprintf(filename,
                    "%s/%s_frameID_%" PRIu64 "_personID_%" PRIu64
                    "_pairID_%" PRIu64 "_qua_%.3f.jpg",
                    dst_dir, timestamp_str, snapshot.snapshot_frame_id,
                    snapshot.track_id, snapshot.pair_track_id,
                    snapshot.quality);
          } else {
            // Assume FACE
            create_id_folder(face_dir, snapshot.track_id,
                             snapshot.pair_track_id, dst_dir, sizeof(dst_dir));

            // Get attributes if available
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

            sprintf(filename,
                    "%s/%s_frameID_%" PRIu64 "_faceID_%" PRIu64
                    "_pairID_%" PRIu64
                    "_qua_%.3f_male[%d]_glass[%d]_age[%d]_emotion[%s].jpg",
                    dst_dir, timestamp_str, snapshot.snapshot_frame_id,
                    snapshot.track_id, snapshot.pair_track_id, snapshot.quality,
                    male, glass, age, emotionStr[emotion]);
          }

          std::vector<uint8_t> snap_buf;
          if (encoder->encodeFrame(snapshot.object_image, snap_buf)) {
            std::ofstream ofs(filename, std::ios::binary);
            if (ofs) {
              ofs.write(reinterpret_cast<const char *>(snap_buf.data()),
                        snap_buf.size());
            }
          }
        }

        // Save features
        if (snapshot.object_box_info.object_type == OBJECT_TYPE_PERSON &&
            cap_result->features.count(snapshot.track_id) &&
            !cap_result->features[snapshot.track_id].empty()) {
          char dst_dir[512];
          char filename[512];
          create_id_folder(image_feature_dir, snapshot.track_id,
                           snapshot.pair_track_id, dst_dir, sizeof(dst_dir));

          sprintf(filename,
                  "%s/%s_frameID_%" PRIu64 "_personID_%" PRIu64
                  "_pairID_%" PRIu64 "_qua_%.3f.bin",
                  dst_dir, timestamp_str, snapshot.snapshot_frame_id,
                  snapshot.track_id, snapshot.pair_track_id, snapshot.quality);
          printf("!!![2]filename: %s\n", filename);

          FILE *f = fopen(filename, "wb");
          if (f) {
            auto &feature = cap_result->features[snapshot.track_id];
            fwrite(feature.data(), 1, feature.size() * sizeof(float), f);
            fclose(f);
            printf("Saved person feature to %s\n", filename);
          }
        }

        std::vector<uint64_t> face_track_ids;
        std::vector<std::vector<float>> query_features_tmp;
        if (snapshot.object_box_info.object_type == OBJECT_TYPE_FACE &&
            cap_result->features.count(snapshot.track_id) &&
            !cap_result->features[snapshot.track_id].empty()) {
          face_track_ids.push_back(snapshot.track_id);
          query_features_tmp.push_back(cap_result->features[snapshot.track_id]);
        }

        if (query_features_tmp.size() > 0) {
          std::vector<std::shared_ptr<ModelFeatureInfo>> query_features =
              createModelFeatureInfos(query_features_tmp);
          MatchResult results;
          feature_matcher->queryWithTopK(query_features, 1, results);

          for (size_t i = 0; i < results.indices.size(); ++i) {
            std::cout << "  查询特征 " << i << " 的匹配结果:" << std::endl;
            const auto &indices = results.indices[i];
            const auto &scores = results.scores[i];
            for (size_t j = 0; j < indices.size() && j < scores.size(); ++j) {
              std::cout << " 特征库索引: " << std::setw(2) << indices[j]
                        << ", 相似度分数: " << std::fixed
                        << std::setprecision(6) << scores[j] << std::endl;

              // 新增：检查相似度分数是否大于0.45
              if (scores[j] > 0.45) {
                int face_feature_id = indices[j];
                int face_track_id = face_track_ids[i];

                auto task = MediaAnalysisEventManager::GetInstance()->GetTask(
                    "face_matching");
                if (task) {
                  auto face_task =
                      std::dynamic_pointer_cast<FaceMatchingTask>(task);
                  if (face_task) {
                    face_task->add_face_info(face_feature_id, face_track_id);
                  }
                }
              }
              std::cout << std::endl;
            }
          }
        }
      }

      // Save full frame for LLM analysis (every 2 frames)
      if (cap_result->frame_id % 2 == 0) {
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
        std::vector<uint8_t> encoded_data;
        if (encoder->encodeFrame(cap_result->image, encoded_data)) {
          FrameTask task;
          task.encoded_data = std::move(encoded_data);

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
    }
  }

  std::cout << "Stopping..." << std::endl;
  sender.stop();
  server->stop();
  app_task->release();

  return 0;
}
