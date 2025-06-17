#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tdl_model_factory.hpp"
#include "utils/common_utils.hpp"
namespace fs = std::experimental::filesystem;
using ordered_json = nlohmann::ordered_json;

std::string to_upper(const std::string& str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::toupper);
  return result;
}

inline double round2(double val) { return std::round(val * 100.0) / 100.0; }

inline double round1(double val) { return std::round(val * 10.0) / 10.0; }

std::vector<double> round_vector(const std::vector<double>& vec, int n) {
  std::vector<double> result(vec.size());
  double scale = std::pow(10.0, n);
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = std::round(vec[i] * scale) / scale;
  }
  return result;
}

std::string extract_json_prefix(const std::string& model_name) {
  std::regex re("(_\\d+)");
  std::vector<size_t> positions;
  auto it = model_name.cend();
  std::smatch match;
  std::string temp = model_name;

  while (std::regex_search(temp, match, re)) {
    positions.push_back(match.position(0) + (model_name.size() - temp.size()));
    temp = match.suffix();
  }
  if (positions.size() < 2) return model_name;

  size_t cut_pos = positions[positions.size() - 2];
  return model_name.substr(0, cut_pos);
}

std::string extract_model_id(const std::string& model_name) {
  std::regex re("_(\\d)");
  std::smatch match;
  if (std::regex_search(model_name, match, re)) {
    size_t pos = match.position(0);
    return to_upper(model_name.substr(0, pos));
  }
  return to_upper(model_name);
}

std::string common_model_id(std::string model_name) {
  if (model_name.substr(0, 4) == "yolo" &&
      model_name.find("det_coco80") != std::string::npos) {
    size_t pos = model_name.find('_');
    if (model_name.substr(0, 5) == "yolox" ||
        model_name.substr(0, 6) == "yolov7") {
      return to_upper(model_name.substr(0, pos) + "_det_coco80");
      ;
    } else {
      return to_upper(model_name.substr(0, pos - 1) + "_det_coco80");
    }
  }
  return "";
}

void extract_model_info(const std::string& model_dir,
                        std::vector<std::string>& model_names,
                        std::vector<std::string>& model_ids,
                        std::vector<std::string>& model_paths) {
  // 1. 特殊 model_name 到特殊 model_id 的映射（可扩展）
  static const std::unordered_map<std::string, std::string>
      special_model_id_map = {

          {"yolov8n_det_ir_person_384_640_INT8", "YOLOV8N_DET_MONITOR_PERSON"},
          {"yolov8n_det_ir_person_mbv2_384_640_INT8",
           "YOLOV8N_DET_MONITOR_PERSON"},
          {"yolov8n_det_overlook_person_256_448_INT8",
           "YOLOV8N_DET_MONITOR_PERSON"},
          {"yolov8n_det_hand_mv3_384_640_INT8", "YOLOV8N_DET_HAND"},
          {"yolov8n_det_person_vehicle_mv2_035_384_640_INT8",
           "YOLOV8N_DET_PERSON_VEHICLE"},
          {"yolov8n_det_pet_person_035_384_640_INT8", "YOLOV8N_DET_PET_PERSON"},

          {"cls_4_attribute_face_112_112_INT8", "CLS_ATTRIBUTE_FACE"},
          {"cls_sound_nihaoshiyun_126_40_INT8",
           "CLS_SOUND_COMMAND_NIHAOSHIYUN"},
          {"cls_sound_xiaoaixiaoai_126_40_INT8",
           "CLS_SOUND_COMMAND_XIAOAIXIAOAI"},
          {"yolov8n_seg_coco80_640_640_INT8", "YOLOV8_SEG_COCO80"}
          // {"recognition_face_r34_112_112_INT8", "FEATURE_BMFACE_R34"}
      };

  for (const auto& entry : fs::directory_iterator(model_dir)) {
    if (fs::is_regular_file(entry.path())) {
      std::string full_path = entry.path().string();  // 新增
      model_paths.push_back(full_path);               // 存储路径
      std::string filename = entry.path().filename().string();

      size_t last_underscore = filename.rfind('_');
      std::string model_name = filename.substr(0, last_underscore);
      model_names.push_back(model_name);

      // 处理特殊model_id情况（可扩展）
      std::string model_id;
      auto it_id = special_model_id_map.find(model_name);
      std::string com_model_id = common_model_id(model_name);
      if (it_id != special_model_id_map.end()) {
        model_id = it_id->second;
      } else if (com_model_id != "") {
        model_id = com_model_id;
      } else {
        model_id = extract_model_id(model_name);
      }

      model_ids.push_back(model_id);
    }
  }
}

std::pair<ordered_json, std::string> load_json(const std::string& json_path) {
  ordered_json data;
  std::string json_status;

  if (fs::exists(json_path)) {
    json_status = "exist";
    std::ifstream f(json_path);
    if (f) {
      f >> data;
      std::cout << "\n从 " << json_path << " 加载现有的JSON数据\n" << std::endl;
    } else {
      std::cerr << "无法打开 " << json_path << " 文件！" << std::endl;
      // 这里可以选择抛异常或返回空json
      data = ordered_json::object();
    }
  } else {
    json_status = "new";
    std::cout << "JSON文件 " << json_path << " 不存在，将创建一个新文件\n"
              << std::endl;
    data = ordered_json::object();
  }
  return {data, json_status};
}

std::vector<std::shared_ptr<BaseImage>> getInputDatas(std::string& image_path) {
  std::vector<std::shared_ptr<BaseImage>> input_datas;
  std::shared_ptr<BaseImage> frame;

  if (image_path.size() >= 4 &&
      image_path.substr(image_path.size() - 4) == ".txt") {
    std::vector<float> keypoints;
    std::ifstream infile(image_path);
    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      float x, y;
      if (iss >> x >> y) {
        keypoints.push_back(x);
        keypoints.push_back(y);
      }
    }

    if (keypoints.size() != 42) {
      throw std::invalid_argument("txt file err");
    }

    frame = ImageFactory::createImage(42, 1, ImageFormat::GRAY,
                                      TDLDataType::FP32, true);
    float* data_buffer =
        reinterpret_cast<float*>(frame->getVirtualAddress()[0]);

    memcpy(data_buffer, &keypoints[0], 42 * sizeof(float));

  } else if (image_path.size() >= 4 &&
             image_path.substr(image_path.size() - 4) == ".bin") {
    int frame_size = 0;
    FILE* fp = fopen(image_path.c_str(), "rb");
    if (fp) {
      fseek(fp, 0, SEEK_END);
      frame_size = ftell(fp);
      fseek(fp, 0, SEEK_SET);
      frame = ImageFactory::createImage(frame_size, 1, ImageFormat::GRAY,
                                        TDLDataType::UINT8, true);
      uint8_t* data_buffer = frame->getVirtualAddress()[0];
      fread(data_buffer, 1, frame_size, fp);
      fclose(fp);
    }
  } else {
    frame = ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);
  }

  input_datas = {frame};

  return input_datas;
};

// 余弦相似度
float cosine_similarity(const std::vector<float>& v1,
                        const std::vector<float>& v2) {
  float sim = 0, norm1 = 0, norm2 = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    sim += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  if (norm1 == 0 || norm2 == 0) return 0.0f;
  return sim / (sqrt(norm1) * sqrt(norm2));
}

template <typename T>
void embeddingToVec(void* embedding, size_t num,
                    std::vector<float>& feature_vec) {
  T* feature = reinterpret_cast<T*>(embedding);
  for (size_t i = 0; i < num; ++i) {
    feature_vec[i] = (float)feature[i];
  }
  return;
}

// 特征提取函数
std::vector<std::vector<float>> extract_features(
    std::shared_ptr<BaseModel> model,
    const std::vector<std::string>& img_paths) {
  std::vector<std::shared_ptr<BaseImage>> face_aligns;
  for (const auto& path : img_paths) {
    face_aligns.push_back(ImageFactory::readImage(path));
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_fe;
  model->inference(face_aligns, out_fe);  // 一次推理处理整个批次

  std::vector<std::vector<float>> features;
  for (const auto& output : out_fe) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(output);
    std::vector<float> feature_vec(feature_meta->embedding_num);

    switch (feature_meta->embedding_type) {
      case TDLDataType::INT8:
        embeddingToVec<int8_t>(feature_meta->embedding,
                               feature_meta->embedding_num, feature_vec);
        break;
      case TDLDataType::UINT8:
        embeddingToVec<uint8_t>(feature_meta->embedding,
                                feature_meta->embedding_num, feature_vec);
        break;
      case TDLDataType::FP32:
        embeddingToVec<float>(feature_meta->embedding,
                              feature_meta->embedding_num, feature_vec);
        break;
      default:
        assert(false && "Unsupported embedding_type");
    }
    features.push_back(feature_vec);
  }
  return features;
}

// 辅助函数：保存实例分割mask
void save_instance_mask(std::shared_ptr<ModelBoxSegmentationInfo> obj_meta,
                        const std::string& mask_path) {
  if (obj_meta->box_seg.empty()) return;

  int proto_h = obj_meta->mask_height;
  int proto_w = obj_meta->mask_width;

  cv::Mat dst;
  for (uint32_t i = 0; i < obj_meta->box_seg.size(); i++) {
    cv::Mat src(proto_h, proto_w, CV_8UC1, obj_meta->box_seg[i].mask,
                proto_w * sizeof(uint8_t));

    if (i == 0) {
      dst = src.clone();
    } else {
      cv::bitwise_or(dst, src, dst);
    }
  }
  cv::imwrite(mask_path, dst);
}

// 辅助函数：保存语义分割mask
void save_semantic_mask(std::shared_ptr<ModelSegmentationInfo> seg_meta,
                        const std::string& mask_path) {
  uint32_t output_width = seg_meta->output_width;
  uint32_t output_height = seg_meta->output_height;

  // 创建单通道图像，直接使用class_id数据
  cv::Mat mask(output_height, output_width, CV_8UC1, seg_meta->class_id,
               output_width * sizeof(uint8_t));
  cv::imwrite(mask_path, mask);
}

void saveResultsCommon(
    const std::string& model_id, const std::string& model_name,
    const std::string& json_path, const std::string& img_dir,
    const std::string& img_name, const std::string& chip,
    std::function<void(ordered_json&, const std::shared_ptr<ModelOutputInfo>&)>
        fill_func,
    const std::shared_ptr<ModelOutputInfo>& out_data,
    std::function<void(ordered_json&)> fill_global_func = nullptr,
    bool is_array = false) {
  ordered_json data;
  if (fs::exists(json_path)) {
    std::ifstream f(json_path);
    if (f) {
      f >> data;
      std::cout << "从 " << json_path << " 加载现有的JSON数据\n" << std::endl;
    }
  }

  if (data.is_null()) {
    data = ordered_json::object();
  }

  data["model_id"] = model_id;
  data["model_name"] = model_name;
  std::string image_dir_name = fs::path(img_dir).filename().string();
  if (fill_global_func) fill_global_func(data);
  data["image_dir"] = image_dir_name;

  if (!data.contains(chip)) {
    data[chip] = ordered_json::object();
  }

  ordered_json& chip_data = data[chip];

  ordered_json result_json =
      is_array ? ordered_json::array() : ordered_json::object();
  fill_func(result_json, out_data);

  chip_data[img_name] = result_json;

  std::ofstream outfile(json_path);
  outfile << data.dump(4);
  outfile.close();

  std::cout << "更新JSON至 " << json_path << std::endl;
}

void process_face_feature(const std::string& model_id,
                          const std::string& model_name,
                          const std::string& json_path,
                          const std::string& img_dir, const std::string& chip,
                          std::shared_ptr<BaseModel> model,
                          float score_threshold) {
  // 1. 收集所有图片，分组
  std::map<std::string, std::pair<std::string, std::string>> pair_map;
  std::vector<std::string> pair_indices;
  std::regex pat(R"((\d+)-(\d)\.png)");
  for (const auto& entry : fs::directory_iterator(img_dir)) {
    std::string fname = entry.path().filename().string();
    std::smatch m;
    if (std::regex_match(fname, m, pat)) {
      std::string idx = m[1];
      std::string pos = m[2];
      if (pos == "1")
        pair_map[idx].first = fname;
      else if (pos == "2")
        pair_map[idx].second = fname;
    }
  }
  for (const auto& p : pair_map) pair_indices.push_back(p.first);

  // 2. 收集所有有效配对
  std::vector<std::pair<std::string, std::string>> all_pairs;
  for (size_t i = 0; i < pair_indices.size(); ++i) {
    const std::string& idx = pair_indices[i];
    const auto& imgs = pair_map[idx];
    if (!imgs.first.empty() && !imgs.second.empty()) {
      all_pairs.emplace_back(imgs.first, imgs.second);
    }
    size_t next = (i + 1) % pair_indices.size();
    const std::string& next_idx = pair_indices[next];
    const auto& next_imgs = pair_map[next_idx];
    if (!imgs.first.empty() && !next_imgs.second.empty()) {
      all_pairs.emplace_back(imgs.first, next_imgs.second);
    }
  }
  // 3. 全局缓存特征
  std::map<std::string, std::vector<float>> feature_map;
  std::string image_dir_name = fs::path(img_dir).filename().string();

  // 4. 构造JSON（集成特征提取）
  ordered_json root;
  root["model_id"] = model_id;
  root["model_name"] = model_name;
  root["image_dir"] = image_dir_name;
  root["score_threshold"] = round1(score_threshold);
  ordered_json chip_json;

  for (size_t i = 0; i < all_pairs.size(); ++i) {
    const auto& p = all_pairs[i];
    std::string pair_key = "pair" + std::to_string(i + 1);
    std::vector<std::string> paths = {img_dir + "/" + p.first,
                                      img_dir + "/" + p.second};
    std::vector<std::string> names = {p.first, p.second};
    std::cout << "\nProcessing pair " << i + 1 << ": " << p.first << ", "
              << p.second << std::endl;

    // 检查缓存
    bool need_inference = false;
    for (const auto& name : names) {
      if (feature_map.count(name) == 0) {
        need_inference = true;
        break;
      }
    }

    if (need_inference) {
      auto features = extract_features(model, paths);
      for (size_t j = 0; j < names.size(); ++j) {
        feature_map[names[j]] = features[j];
      }
    }

    float sim = cosine_similarity(feature_map[p.first], feature_map[p.second]);
    ordered_json obj;
    obj["images"] = {p.first, p.second};
    obj["similarity"] = round2(sim);
    chip_json[pair_key] = obj;
  }

  root[chip] = chip_json;
  // 保存JSON
  std::ofstream ofs(json_path);
  ofs << std::setw(4) << root << std::endl;
  std::cout << "Face embedding similarity JSON saved: " << json_path
            << std::endl;
}

void processModel(const std::string& model_id, const std::string& model_name,
                  const std::string& json_path, const std::string& img_dir,
                  const std::string& chip, std::shared_ptr<BaseModel> model,
                  float bbox_threshold, float score_threshold,
                  float position_threshold, float model_threshold) {
  if (model_threshold > 0 && model_threshold < 1.0f) {
    printf("setModelThreshold\n");
    getchar();
    model->setModelThreshold(model_threshold);
  }
  ModelType model_type = modelTypeFromString(model_id);

  if (!fs::exists(img_dir)) {
    std::cout << "Image directory does not exist: " << img_dir << std::endl;
    return;
  }

  for (const auto& entry : fs::directory_iterator(img_dir)) {
    std::string image_path = entry.path().string();

    if (model_id.find("SEG") != std::string::npos &&
        image_path.find("_mask_") != std::string::npos) {
      continue;  // gt
    }

    std::vector<std::shared_ptr<BaseImage>> input_datas;
    input_datas = getInputDatas(image_path);
    int img_width = input_datas[0]->getWidth();
    int img_height = input_datas[0]->getHeight();
    std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;

    model->inference(input_datas, out_datas);
    std::string img_name = fs::path(image_path).filename().string();
    std::cout << "Processing image: " << img_name << std::endl;
    ModelOutputType out_type = out_datas[0]->getType();

    // 检测
    if (out_type == ModelOutputType::OBJECT_DETECTION) {
      auto fill_func = [](ordered_json& bboxes_json,
                          const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto obj_meta = std::static_pointer_cast<ModelBoxInfo>(out_data);
        for (const auto& bbox : obj_meta->bboxes) {
          ordered_json bbox_j;

          bbox_j["bbox"] = {round2(bbox.x1), round2(bbox.y1), round2(bbox.x2),
                            round2(bbox.y2)};
          bbox_j["score"] = round2(bbox.score);
          bbox_j["class_id"] = bbox.class_id;

          bboxes_json.push_back(bbox_j);
        }
      };
      auto fill_global = [bbox_threshold, score_threshold](ordered_json& data) {
        data["bbox_threshold"] = round1(bbox_threshold);
        data["score_threshold"] = round1(score_threshold);
      };
      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], fill_global, true);
    }
    // 分类
    else if (out_type == ModelOutputType::CLASSIFICATION) {
      auto fill_func = [](ordered_json& results_json,
                          const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto cls_meta =
            std::static_pointer_cast<ModelClassificationInfo>(out_data);

        ordered_json cls_j;
        cls_j["score"] = round2(cls_meta->topk_scores[0]);
        cls_j["class_id"] = cls_meta->topk_class_ids[0];

        results_json.push_back(cls_j);
      };

      auto fill_global = [score_threshold](ordered_json& data) {
        data["score_threshold"] = round1(score_threshold);
      };

      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], fill_global, true);
    }

    // 属性分类
    else if (out_type == ModelOutputType::CLS_ATTRIBUTE) {
      auto fill_func = [](ordered_json& results_json,
                          const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto face_meta = std::static_pointer_cast<ModelAttributeInfo>(out_data);

        results_json["mask"] =
            round1(face_meta->attributes
                       [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK]);
        results_json["age"] =
            round2(face_meta->attributes
                       [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE]);
        results_json["gender"] =
            round1(face_meta->attributes
                       [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER]);
        results_json["glass"] = round1(
            face_meta->attributes
                [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES]);
      };

      auto fill_global = [score_threshold](ordered_json& data) {
        data["score_threshold"] = round1(score_threshold);
      };

      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], fill_global);
    }

    // 关键点
    else if (out_type == ModelOutputType::OBJECT_LANDMARKS) {
      auto fill_func = [img_width, img_height](
                           ordered_json& img_json,
                           const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto keypoint_meta =
            std::static_pointer_cast<ModelLandmarksInfo>(out_data);
        std::vector<double> norm_x, norm_y;
        double max_x = *std::max_element(keypoint_meta->landmarks_x.begin(),
                                         keypoint_meta->landmarks_x.end());
        double max_y = *std::max_element(keypoint_meta->landmarks_y.begin(),
                                         keypoint_meta->landmarks_y.end());
        bool is_normalized = (max_x <= 1.0 && max_y <= 1.0);

        for (size_t i = 0; i < keypoint_meta->landmarks_x.size(); ++i) {
          if (is_normalized) {
            norm_x.push_back(
                static_cast<double>(keypoint_meta->landmarks_x[i]));
            norm_y.push_back(
                static_cast<double>(keypoint_meta->landmarks_y[i]));
          } else {
            norm_x.push_back(keypoint_meta->landmarks_x[i] / img_width);
            norm_y.push_back(keypoint_meta->landmarks_y[i] / img_height);
          }
        }

        std::vector<double> landmarks_score_d;
        if (!keypoint_meta->attributes.empty()) {
          for (const auto& pair : keypoint_meta->attributes) {
            landmarks_score_d.push_back(pair.second);
          }
        } else {
          for (float score : keypoint_meta->landmarks_score) {
            landmarks_score_d.push_back(static_cast<double>(score));
          }
        }

        img_json["keypoints_x"] = round_vector(norm_x, 4);
        img_json["keypoints_y"] = round_vector(norm_y, 4);

        img_json["keypoints_score"] = round_vector(landmarks_score_d, 2);
      };
      auto fill_global = [score_threshold,
                          position_threshold](ordered_json& data) {
        data["score_threshold"] = round1(score_threshold);
        data["position_threshold"] = round1(position_threshold);
      };
      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], fill_global);
    }

    // 检测+关键点
    else if (out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
      if (model_type == ModelType::KEYPOINT_YOLOV8POSE_PERSON17) {
        auto fill_func = [img_width, img_height](
                             ordered_json& img_json,
                             const std::shared_ptr<ModelOutputInfo>& out_data) {
          auto obj_meta =
              std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
          ObjectBoxLandmarkInfo keypoint_meta = obj_meta->box_landmarks[0];
          std::vector<double> norm_x, norm_y;
          double max_x = *std::max_element(keypoint_meta.landmarks_x.begin(),
                                           keypoint_meta.landmarks_x.end());
          double max_y = *std::max_element(keypoint_meta.landmarks_y.begin(),
                                           keypoint_meta.landmarks_y.end());
          bool is_normalized = (max_x <= 1.0 && max_y <= 1.0);

          for (size_t i = 0; i < keypoint_meta.landmarks_x.size(); ++i) {
            if (is_normalized) {
              norm_x.push_back(
                  static_cast<double>(keypoint_meta.landmarks_x[i]));
              norm_y.push_back(
                  static_cast<double>(keypoint_meta.landmarks_y[i]));
            } else {
              norm_x.push_back(keypoint_meta.landmarks_x[i] / img_width);
              norm_y.push_back(keypoint_meta.landmarks_y[i] / img_height);
            }
          }
          // Convert landmarks_score from std::vector<float> to
          // std::vector<double>
          std::vector<double> landmarks_score_d;
          for (float score : keypoint_meta.landmarks_score) {
            landmarks_score_d.push_back(static_cast<double>(score));
          }

          img_json["keypoints_x"] = round_vector(norm_x, 4);
          img_json["keypoints_y"] = round_vector(norm_y, 4);
          img_json["keypoints_score"] = round_vector(landmarks_score_d, 2);
        };
        auto fill_global = [score_threshold,
                            position_threshold](ordered_json& data) {
          data["score_threshold"] = round1(score_threshold);
          data["position_threshold"] = round1(position_threshold);
        };
        saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                          chip, fill_func, out_datas[0], fill_global);
      } else if (model_type == ModelType::SCRFD_DET_FACE) {
        auto fill_func = [](ordered_json& bboxes_json,
                            const std::shared_ptr<ModelOutputInfo>& out_data) {
          auto obj_meta =
              std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
          for (const auto& box_landmark : obj_meta->box_landmarks) {
            ordered_json bbox_j;
            bbox_j["bbox"] = {round2(box_landmark.x1), round2(box_landmark.y1),
                              round2(box_landmark.x2), round2(box_landmark.y2)};
            bbox_j["score"] = round2(box_landmark.score);
            bboxes_json.push_back(bbox_j);
          }
        };
        auto fill_global = [bbox_threshold,
                            score_threshold](ordered_json& data) {
          data["bbox_threshold"] = round1(bbox_threshold);
          data["score_threshold"] = round1(score_threshold);
        };
        saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                          chip, fill_func, out_datas[0], fill_global, true);
      } else if (model_type == ModelType::LSTR_DET_LANE) {
        auto fill_func = [img_width, img_height](
                             ordered_json& img_json,
                             const std::shared_ptr<ModelOutputInfo>& out_data) {
          auto obj_meta =
              std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
          ObjectBoxLandmarkInfo keypoint_meta = obj_meta->box_landmarks[0];
          std::vector<double> norm_x, norm_y;
          double max_x = *std::max_element(keypoint_meta.landmarks_x.begin(),
                                           keypoint_meta.landmarks_x.end());
          double max_y = *std::max_element(keypoint_meta.landmarks_y.begin(),
                                           keypoint_meta.landmarks_y.end());
          bool is_normalized = (max_x <= 1.0 && max_y <= 1.0);

          for (size_t i = 0; i < keypoint_meta.landmarks_x.size(); ++i) {
            if (is_normalized) {
              norm_x.push_back(
                  static_cast<double>(keypoint_meta.landmarks_x[i]));
              norm_y.push_back(
                  static_cast<double>(keypoint_meta.landmarks_y[i]));
            } else {
              norm_x.push_back(keypoint_meta.landmarks_x[i] / img_width);
              norm_y.push_back(keypoint_meta.landmarks_y[i] / img_height);
            }
          }
          // Convert landmarks_score from std::vector<float> to
          // std::vector<double>
          std::vector<double> landmarks_score_d;
          for (float score : keypoint_meta.landmarks_score) {
            landmarks_score_d.push_back(static_cast<double>(score));
          }

          img_json["keypoints_x"] = round_vector(norm_x, 4);
          img_json["keypoints_y"] = round_vector(norm_y, 4);
          img_json["keypoints_score"] = round_vector(landmarks_score_d, 2);
        };
        auto fill_global = [score_threshold,
                            position_threshold](ordered_json& data) {
          data["score_threshold"] = round1(score_threshold);
          data["position_threshold"] = round1(position_threshold);
        };
        saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                          chip, fill_func, out_datas[0], fill_global);
      }

      else {
        std::cout << "Unsupported model_id: " << static_cast<int>(model_type)
                  << std::endl;
      }
    }
    // 实例分割
    else if (out_type == ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
      auto fill_func = [img_name, img_dir, chip](
                           ordered_json& img_json,
                           const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto obj_meta =
            std::static_pointer_cast<ModelBoxSegmentationInfo>(out_data);

        // 生成mask文件名（将jpg替换为png）
        std::string mask_name = img_name;
        std::string suffix = std::string("_mask_") + chip + std::string(".png");
        mask_name.replace(mask_name.size() - 4, 4, suffix);

        // 保存mask图像到文件
        std::string mask_path = img_dir + "/" + mask_name;
        save_instance_mask(obj_meta, mask_path);

        img_json["mask"] = mask_name;

        ordered_json detection_array = ordered_json::array();
        for (const auto& box_seg : obj_meta->box_seg) {
          ordered_json detection_obj;
          detection_obj["bbox"] = {round2(box_seg.x1), round2(box_seg.y1),
                                   round2(box_seg.x2), round2(box_seg.y2)};
          detection_obj["score"] = round2(box_seg.score);
          detection_obj["class_id"] = box_seg.class_id;
          detection_array.push_back(detection_obj);
        }
        img_json["detection"] = detection_array;
      };

      auto fill_global = [bbox_threshold, score_threshold](ordered_json& data) {
        data["bbox_threshold"] = round1(bbox_threshold);
        data["score_threshold"] = round1(score_threshold);
        data["mask_threshold"] = round1(0.1);
      };

      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], fill_global);
    }

    // 语义分割
    else if (out_type == ModelOutputType::SEGMENTATION) {
      auto fill_func = [img_name, img_dir, chip](
                           ordered_json& img_json,
                           const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto seg_meta =
            std::static_pointer_cast<ModelSegmentationInfo>(out_data);

        // 生成mask文件名（将jpg替换为png）
        std::string mask_name = img_name;
        std::string suffix = std::string("_mask_") + chip + std::string(".png");
        mask_name.replace(mask_name.size() - 4, 4, suffix);

        // 保存mask图像到文件
        std::string mask_path = img_dir + "/" + mask_name;
        save_semantic_mask(seg_meta, mask_path);

        img_json["mask"] = mask_name;
      };

      auto fill_global = [score_threshold](ordered_json& data) {
        data["mask_threshold"] = round1(0.1);
      };

      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], fill_global);
    }

    else if (out_type == ModelOutputType::OCR_INFO) {
      auto fill_func = [](ordered_json& results_json,
                          const std::shared_ptr<ModelOutputInfo>& out_data) {
        auto ocr_meta = std::static_pointer_cast<ModelOcrInfo>(out_data);

        ordered_json ocr_j;
        results_json["characters"] = std::string(ocr_meta->text_info);
      };

      saveResultsCommon(model_id, model_name, json_path, img_dir, img_name,
                        chip, fill_func, out_datas[0], nullptr);
    }

    else {
      std::cout << "Unsupported output type: " << static_cast<int>(out_type)
                << std::endl;
    }

    input_datas[0].reset();
    input_datas.clear();
    out_datas.clear();
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << "Required: model_dir image_dir json_dir chip\n"
              << "Optional:[bbox_threshold] [score_threshold] "
                 "[position_threshold] [model_threshold]"
              << std::endl;
    return -1;
  }
  std::string model_dir = argv[1];
  std::string image_dir = argv[2];
  std::string json_dir = argv[3];
  std::string chip = argv[4];
  float bbox_threshold = 0.5f;
  float score_threshold = 0.1f;
  float position_threshold = 0.1f;
  float model_threshold = -1.0f;

  if (argc >= 6) {
    bbox_threshold = atof(argv[5]);
    score_threshold = atof(argv[6]);
    position_threshold = atof(argv[7]);
    model_threshold = atof(argv[8]);
  }

  std::vector<std::string> model_names, model_ids, model_paths;
  fs::path p(model_dir);
  fs::path parent = p.parent_path();

  extract_model_info(model_dir, model_names, model_ids, model_paths);
  std::cout << "Found " << model_names.size()
            << " models in directory: " << model_dir << std::endl;

  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.setModelDir(parent.string());
  model_factory.loadModelConfig();

  for (size_t i = 0; i < model_ids.size(); ++i) {
    const std::string& model_id_name = model_ids[i];
    std::string json_path = json_dir + "/" + model_names[i] + ".json";
    std::string img_dir = image_dir + "/" + model_names[i];

    std::cout << "Processing model:" << i << " " << model_names[i]
              << ", ID: " << model_ids[i] << ", Image Directory: " << img_dir
              << std::endl;
    std::cout << model_paths[i] << std::endl;

    std::shared_ptr<BaseModel> model =
        model_factory.getModel(model_id_name, model_paths[i]);

    if (!model) {
      printf("--------------------------------------------------");
      printf("Failed to create model %s\n", model_id_name.c_str());
      continue;
    }
    if (model_id_name.find("FEATURE") != std::string::npos) {
      process_face_feature(model_ids[i], model_names[i], json_path, img_dir,
                           chip, model, score_threshold);
      std::cout << "Finished processing face recognition model: "
                << model_id_name << std::endl;
    } else {
      processModel(model_ids[i], model_names[i], json_path, img_dir, chip,
                   model, bbox_threshold, score_threshold, position_threshold,
                   model_threshold);
      std::cout << "Finished processing model " << model_names[i] << std::endl;
    }

    model.reset();
  }
  return 0;
}