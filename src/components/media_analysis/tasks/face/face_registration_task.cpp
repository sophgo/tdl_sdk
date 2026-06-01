#include "face_registration_task.hpp"
#include <dirent.h>
#include <sys/stat.h>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "components/media_analysis/media_analysis_server.hpp"
#include "nn/tdl_model_factory.hpp"

namespace fs = std::experimental::filesystem;

namespace {
static const float MIN_FACE_RATIO = 0.05f;
static const float DUPLICATE_THRESHOLD = 0.7f;

void ensureDir(const std::string& dir) {
  if (!fs::exists(dir)) {
    fs::create_directories(dir);
  }
}

std::string base64Decode(const std::string& encoded) {
  static const std::string base64_chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789+/";

  std::string decoded;
  std::vector<int> T(256, -1);
  for (int i = 0; i < 64; i++) T[(unsigned char)base64_chars[i]] = i;

  int val = 0, valb = -8;
  for (unsigned char c : encoded) {
    if (T[c] == -1) break;
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      decoded.push_back(char((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return decoded;
}

float cosineSimilarity(const uint8_t* a, const uint8_t* b, int dim) {
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (int i = 0; i < dim; i++) {
    float va = (float)(int8_t)a[i];
    float vb = (float)(int8_t)b[i];
    dot += va * vb;
    norm_a += va * va;
    norm_b += vb * vb;
  }
  if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

bool loadFeatureBin(const std::string& path, std::vector<uint8_t>& feature,
                    int& dim) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return false;
  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  if (size == 0) return false;
  dim = (int)size;
  feature.resize(size);
  ifs.read(reinterpret_cast<char*>(feature.data()), size);
  return ifs.good();
}
}  // namespace

FaceRegistrationTask::FaceRegistrationTask(const std::string& model_dir,
                                           const std::string& data_path)
    : model_dir_(model_dir), data_path_(data_path) {
  loadRegisteredNames();
}

void FaceRegistrationTask::loadRegisteredNames() {
  std::ifstream file(data_path_ + "/registered_info.txt");
  if (file.is_open()) {
    std::string name;
    int registered_id;
    while (file >> name >> registered_id) {
      name_to_id_map_[name] = registered_id;
    }
    file.close();
  }
}

bool FaceRegistrationTask::initModels() {
  if (models_initialized_) return true;

  TDLModelFactory& factory = TDLModelFactory::getInstance();
  factory.loadModelConfig();
  factory.setModelDir(model_dir_);

  model_fd_ = factory.getModel(ModelType::SCRFD_DET_FACE);
  if (!model_fd_) {
    LOGE("Failed to create face detection model\n");
    return false;
  }

  model_fl_ = factory.getModel(ModelType::KEYPOINT_FACE_V2);
  if (!model_fl_) {
    LOGE("Failed to create face landmark model\n");
    return false;
  }

  model_fe_ = factory.getModel(ModelType::FEATURE_CVIFACE);
  if (!model_fe_) {
    LOGE("Failed to create feature extraction model\n");
    return false;
  }

  models_initialized_ = true;
  return true;
}

FaceRegistrationTask::RegistrationError FaceRegistrationTask::validateFace(
    const std::shared_ptr<BaseImage>& image,
    std::shared_ptr<ModelOutputInfo>& out_fd) {
  RegistrationError err;

  std::vector<std::shared_ptr<BaseImage>> input_images = {image};
  std::vector<std::shared_ptr<ModelOutputInfo>> out_fds;
  int ret = model_fd_->inference(input_images, out_fds);
  if (ret != 0 || out_fds.empty()) {
    err.error_code = "INTERNAL_ERROR";
    err.error_message = "Face detection inference failed";
    return err;
  }

  auto model_box_info = std::dynamic_pointer_cast<ModelBoxInfo>(out_fds[0]);
  if (!model_box_info || model_box_info->bboxes.empty()) {
    err.error_code = "NO_FACE_DETECTED";
    err.error_message = "未检测到人脸，请上传包含清晰正面人脸的图片";
    return err;
  }

  if (model_box_info->bboxes.size() > 1) {
    err.error_code = "MULTIPLE_FACES";
    err.error_message = "检测到多张人脸（" +
                        std::to_string(model_box_info->bboxes.size()) +
                        " 张），请上传只有一个人的照片";
    return err;
  }

  // Quality check: face size relative to image
  const auto& bbox = model_box_info->bboxes[0];
  float face_area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
  float image_area = image->getWidth() * image->getHeight();
  float face_ratio = face_area / image_area;
  if (face_ratio < MIN_FACE_RATIO) {
    err.error_code = "FACE_TOO_SMALL";
    err.error_message =
        "人脸在图片中占比过小，请靠近摄像头或使用更高分辨率的照片";
    return err;
  }

  out_fd = out_fds[0];
  err.error_code = "";
  return err;
}

bool FaceRegistrationTask::checkDuplicate(
    const std::shared_ptr<ModelFeatureInfo>& feature, float threshold) {
  std::string registered_feature_dir = data_path_ + "/registered_feature";
  if (!fs::exists(registered_feature_dir)) return false;

  int query_dim = feature->embedding_num;
  uint8_t* query_embedding = feature->embedding;

  // Check flat bin files first (0.bin, 1.bin, ...)
  DIR* dir = opendir(registered_feature_dir.c_str());
  if (!dir) return false;

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name.size() < 4 || name.substr(name.size() - 4) != ".bin") {
      // Check subdirectories for feature.bin
      if (entry->d_type == DT_DIR && name != "." && name != "..") {
        std::string sub_bin =
            registered_feature_dir + "/" + name + "/feature.bin";
        if (fs::exists(sub_bin)) {
          std::vector<uint8_t> gallery_feat;
          int dim;
          if (loadFeatureBin(sub_bin, gallery_feat, dim) && dim == query_dim) {
            float sim =
                cosineSimilarity(query_embedding, gallery_feat.data(), dim);
            if (sim > threshold) {
              closedir(dir);
              return true;
            }
          }
        }
      }
      continue;
    }

    std::string bin_path = registered_feature_dir + "/" + name;
    std::vector<uint8_t> gallery_feat;
    int dim;
    if (loadFeatureBin(bin_path, gallery_feat, dim) && dim == query_dim) {
      float sim = cosineSimilarity(query_embedding, gallery_feat.data(), dim);
      if (sim > threshold) {
        closedir(dir);
        return true;
      }
    }
  }
  closedir(dir);
  return false;
}

int FaceRegistrationTask::assignRegisteredId() {
  std::lock_guard<std::mutex> lock(mutex_);
  int max_id = -1;
  for (const auto& pair : name_to_id_map_) {
    if (pair.second > max_id) max_id = pair.second;
  }

  // Also scan registered_feature dir for max id
  std::string registered_feature_dir = data_path_ + "/registered_feature";
  if (fs::exists(registered_feature_dir)) {
    DIR* dir = opendir(registered_feature_dir.c_str());
    if (dir) {
      struct dirent* entry;
      while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
          try {
            int id = std::stoi(entry->d_name);
            if (id > max_id) max_id = id;
          } catch (...) {
          }
        }
        // Also check flat bin files like "0.bin"
        if (entry->d_type == DT_REG) {
          std::string name = entry->d_name;
          if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            try {
              int id = std::stoi(name.substr(0, name.size() - 4));
              if (id > max_id) max_id = id;
            } catch (...) {
            }
          }
        }
      }
      closedir(dir);
    }
  }

  return max_id + 1;
}

bool FaceRegistrationTask::saveFeature(
    const std::shared_ptr<ModelFeatureInfo>& feature, int registered_id) {
  std::string feature_dir =
      data_path_ + "/registered_feature/" + std::to_string(registered_id);
  ensureDir(feature_dir);

  std::string feature_path = feature_dir + "/feature.bin";
  std::ofstream ofs(feature_path, std::ios::binary);
  if (!ofs.is_open()) {
    LOGE("Failed to create feature file: %s\n", feature_path.c_str());
    return false;
  }

  switch (feature->embedding_type) {
    case TDLDataType::INT8:
      ofs.write(reinterpret_cast<const char*>(feature->embedding),
                feature->embedding_num * sizeof(int8_t));
      break;
    case TDLDataType::UINT8:
      ofs.write(reinterpret_cast<const char*>(feature->embedding),
                feature->embedding_num * sizeof(uint8_t));
      break;
    case TDLDataType::FP32:
      ofs.write(reinterpret_cast<const char*>(feature->embedding),
                feature->embedding_num * sizeof(float));
      break;
    default:
      LOGE("Unsupported embedding type: %d\n", (int)feature->embedding_type);
      ofs.close();
      return false;
  }

  ofs.close();
  LOGI("Feature saved to: %s\n", feature_path.c_str());
  return true;
}

bool FaceRegistrationTask::appendRegisteredInfo(int registered_id,
                                                const std::string& name) {
  std::string info_path = data_path_ + "/registered_info.txt";
  std::ofstream ofs(info_path, std::ios::app);
  if (!ofs.is_open()) return false;

  ofs << name << " " << registered_id << "\n";
  ofs.close();

  // Also update in-memory map
  {
    std::lock_guard<std::mutex> lock(mutex_);
    name_to_id_map_[name] = registered_id;
  }
  return true;
}

json FaceRegistrationTask::registerFace(const std::string& image_b64,
                                        const std::string& name, bool force) {
  json response;
  response["schema_version"] = "smart_home.ws.v1";

  if (!initModels()) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "服务异常，请稍后重试";
    return response;
  }

  // Step 1: Decode base64 image
  std::string image_data = base64Decode(image_b64);
  if (image_data.empty()) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "图片解码失败，请重新上传";
    return response;
  }

  // Write decoded image to temp file
  std::string tmp_path = data_path_ + "/.tmp_register.jpg";
  std::ofstream tmp_ofs(tmp_path, std::ios::binary);
  if (!tmp_ofs.is_open()) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "服务异常，请稍后重试";
    return response;
  }
  tmp_ofs.write(image_data.c_str(), image_data.size());
  tmp_ofs.close();

  // Read as BaseImage
  std::shared_ptr<BaseImage> image = ImageFactory::readImage(tmp_path);
  fs::remove(tmp_path);

  if (!image) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "图片解码失败，请重新上传";
    return response;
  }

  // Step 2: Face detection & validation
  std::shared_ptr<ModelOutputInfo> out_fd;
  auto err = validateFace(image, out_fd);
  if (!err.error_code.empty()) {
    response["success"] = false;
    response["error_code"] = err.error_code;
    response["error_message"] = err.error_message;
    return response;
  }

  auto model_box_info = std::dynamic_pointer_cast<ModelBoxInfo>(out_fd);
  auto& bbox = model_box_info->bboxes[0];

  if (!force) {
    // Quality check pass (already checked basic size in validateFace)
  }

  // Step 3: Face landmark detection
  // Build ModelBoxLandmarkInfo for landmark inference
  auto box_landmark_info = std::make_shared<ModelBoxLandmarkInfo>();
  box_landmark_info->image_width = image->getWidth();
  box_landmark_info->image_height = image->getHeight();
  ObjectBoxLandmarkInfo obj;
  obj.x1 = bbox.x1;
  obj.y1 = bbox.y1;
  obj.x2 = bbox.x2;
  obj.y2 = bbox.y2;
  box_landmark_info->box_landmarks.push_back(obj);

  std::vector<std::shared_ptr<ModelOutputInfo>> out_fl;
  int ret = model_fl_->inference(image, box_landmark_info, out_fl);
  if (ret != 0 || out_fl.empty()) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "关键点检测失败，请重试或更换图片";
    return response;
  }

  // Step 4: Face alignment using landmarks
  std::shared_ptr<ModelLandmarksInfo> landmarks_meta =
      std::dynamic_pointer_cast<ModelLandmarksInfo>(out_fl[0]);
  if (!landmarks_meta || landmarks_meta->landmarks_x.size() < 5) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "关键点检测结果异常，请重试或更换图片";
    return response;
  }

  float landmarks[10];
  for (int i = 0; i < 5; i++) {
    landmarks[2 * i] = landmarks_meta->landmarks_x[i];
    landmarks[2 * i + 1] = landmarks_meta->landmarks_y[i];
  }

  std::shared_ptr<BaseImage> face_align =
      ImageFactory::alignFace(image, landmarks, nullptr, 5, nullptr);
  if (!face_align) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "人脸对齐失败，请重试或更换图片";
    return response;
  }

  // Step 5: Feature extraction
  std::vector<std::shared_ptr<ModelOutputInfo>> out_fe;
  std::vector<std::shared_ptr<BaseImage>> face_aligns = {face_align};
  ret = model_fe_->inference(face_aligns, out_fe);
  if (ret != 0 || out_fe.empty()) {
    response["success"] = false;
    response["error_code"] = "FEATURE_EXTRACT_FAILED";
    response["error_message"] = "特征提取失败，请重试或更换图片";
    return response;
  }

  std::shared_ptr<ModelFeatureInfo> feature_meta =
      std::dynamic_pointer_cast<ModelFeatureInfo>(out_fe[0]);
  if (!feature_meta || feature_meta->embedding_num == 0) {
    response["success"] = false;
    response["error_code"] = "FEATURE_EXTRACT_FAILED";
    response["error_message"] = "特征提取失败，请重试或更换图片";
    return response;
  }

  // Step 6: Duplicate check
  if (!force && checkDuplicate(feature_meta, DUPLICATE_THRESHOLD)) {
    response["success"] = false;
    response["error_code"] = "ALREADY_EXISTS";
    response["error_message"] = "该人脸已注册，请勿重复注册";
    return response;
  }

  // Step 7: Assign registered_id
  int registered_id = assignRegisteredId();

  // Step 8: Save feature
  if (!saveFeature(feature_meta, registered_id)) {
    response["success"] = false;
    response["error_code"] = "INTERNAL_ERROR";
    response["error_message"] = "特征保存失败，请重试";
    return response;
  }

  // Step 9: Update registered_info.txt
  appendRegisteredInfo(registered_id, name);

  // Step 10: Save registration image
  std::string identity_dir =
      data_path_ + "/identity/" + std::to_string(registered_id);
  ensureDir(identity_dir);
  std::string register_img_path = identity_dir + "/register.jpg";
  ImageFactory::writeImage(register_img_path, image);

  // Build success response
  response["success"] = true;
  response["registered_id"] = registered_id;
  response["name"] = name;
  response["feature_dim"] = feature_meta->embedding_num;

  LOGI("Face registered successfully: registered_id=%d, name=%s\n",
       registered_id, name.c_str());
  return response;
}

json FaceRegistrationTask::handle_event(const json& request,
                                        const std::string& description) {
  // WebSocket-based registration request
  json response = request;
  response["type"] = "response";
  response["source"] = "c_backend";
  response["destination"] = "web_client";

  if (!request.contains("payload") || !request["payload"].contains("params")) {
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "INVALID_REQUEST"},
                                    {"message", "Missing params in request"}};
    return response;
  }

  auto& params = request["payload"]["params"];
  std::string image_b64 = params.value("image", "");
  std::string name = params.value("name", "");
  bool force = params.value("force", false);

  if (image_b64.empty() || name.empty()) {
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {
        {"code", "INVALID_REQUEST"},
        {"message", "Missing image or name parameters"}};
    return response;
  }

  json result = registerFace(image_b64, name, force);
  response["payload"]["result"] =
      result.value("success", false) ? "ok" : "error";
  response["payload"]["data"] = result;

  return response;
}

std::string FaceRegistrationTask::getRegisteredNamesJson() {
  std::lock_guard<std::mutex> lock(mutex_);
  json names_array = json::array();
  for (const auto& pair : name_to_id_map_) {
    names_array.push_back(pair.first);
  }
  return names_array.dump();
}