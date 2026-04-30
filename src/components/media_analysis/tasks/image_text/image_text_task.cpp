#include "image_text_task.hpp"
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include "network/api_poster/api_client.hpp"
#include "utils/tdl_log.hpp"

// #define LOGI printf

#define CLIP_FEATURE_DIM 512

namespace fs = std::experimental::filesystem;

void ImageTextTask::parse_person_info(const std::string& image_path,
                                      nlohmann::json& item) {
  try {
    std::string filename = fs::path(image_path).filename().string();
    // 修改正则表达式以匹配新的文件名格式 (兼容 personID_ 和 faceID_)
    std::regex pattern(R"((\d{8}_\d{6}).*(?:personID_|faceID_)(\d+))");
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
    LOGI("解析person_info异常: %s\n", e.what());
    item["capture_time"] = "";
    item["track_id"] = "";
  }
}

ImageTextTask::ImageTextTask(const std::string& data_path,
                             const std::string& model_dir,
                             const std::string& txt_dir)
    : data_path_(data_path) {
  matcher_ = BaseMatcher::getMatcher("bm");
  init_text_model(model_dir, txt_dir);
}

void ImageTextTask::init_text_model(const std::string& model_dir,
                                    const std::string& txt_dir) {
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  LOGI("Loading text model FEATURE_MOBILECLIP2_TEXT...\n");
  text_model_ = model_factory.getModel("FEATURE_MOBILECLIP2_TEXT");
  if (text_model_ == nullptr) {
    LOGE("Failed to get model FEATURE_MOBILECLIP2_TEXT\n");
    return;
  }
  LOGI("Text model loaded successfully.\n");

  std::string encoder_file = txt_dir + "/encoder.txt";
  std::string bpe_file = txt_dir + "/vocab.txt";

  LOGI("encoder_file: %s\n", encoder_file.c_str());
  LOGI("bpe_file: %s\n", bpe_file.c_str());

  if (!fs::exists(encoder_file)) {
    LOGE("encoder_file not exists: %s\n", encoder_file.c_str());
    return;
  }
  if (!fs::exists(bpe_file)) {
    LOGE("bpe_file not exists: %s\n", bpe_file.c_str());
    return;
  }

  bpe_ = std::make_shared<BytePairEncoder>(encoder_file, bpe_file);
  LOGI("BytePairEncoder initialized.\n");
}

void ImageTextTask::get_gallery_features(
    std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features,
    std::vector<std::string>& image_paths) {
  std::string feature_path = data_path_ + "/image_feature";

  if (!fs::exists(feature_path)) {
    LOGI("feature_path not exists: %s\n", feature_path.c_str());
    return;
  }

  for (const auto& entry : fs::recursive_directory_iterator(feature_path)) {
    if (fs::is_regular_file(entry.status()) &&
        entry.path().extension() == ".bin") {
      std::ifstream bin_file(entry.path(), std::ios::binary);
      if (!bin_file.is_open()) {
        std::cerr << "无法打开bin文件: " << entry.path() << std::endl;
        continue;
      }

      auto feature_info = std::make_shared<ModelFeatureInfo>();
      feature_info->embedding = new uint8_t[CLIP_FEATURE_DIM * sizeof(float)];
      feature_info->embedding_num = CLIP_FEATURE_DIM;
      feature_info->embedding_type = TDLDataType::FP32;

      bin_file.read(reinterpret_cast<char*>(feature_info->embedding),
                    CLIP_FEATURE_DIM * sizeof(float));

      if (!bin_file) {
        std::cerr << "读取bin文件失败: " << entry.path() << std::endl;
        delete[] feature_info->embedding;
        continue;
      }

      gallery_features.push_back(feature_info);

      std::string image_path = entry.path().string();
      size_t pos = image_path.find("image_feature");
      if (pos != std::string::npos) {
        // 先尝试替换为 identity (sample_app_identity_recognition 结构)
        std::string temp_path = image_path;
        temp_path.replace(pos, 13, "identity");
        pos = temp_path.rfind(".bin");
        if (pos != std::string::npos) {
          temp_path.replace(pos, 4, ".jpg");
        }
        if (fs::exists(temp_path)) {
          image_path = temp_path;
        } else {
          // 退回到 person (sample_app_media_analysis 结构)
          image_path.replace(pos, 13, "person");
          pos = image_path.rfind(".bin");
          if (pos != std::string::npos) {
            image_path.replace(pos, 4, ".jpg");
          }
        }
      }
      image_paths.push_back(image_path);
      bin_file.close();
    }
  }
  std::cout << "成功加载 " << gallery_features.size() << " 个图像特征文件"
            << std::endl;
}

void ImageTextTask::clip_score_post_process(
    std::vector<std::vector<float>>& scores) {
  if (scores.empty()) return;

  const std::size_t rows = scores.size();
  const std::size_t cols = scores[0].size();
  for (auto& row : scores)
    for (auto& v : row) v *= 100.f;

  std::vector<float> colSum(cols, 0.f);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) {
      scores[r][c] = std::exp(scores[r][c]);
      colSum[c] += scores[r][c];
    }
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) scores[r][c] /= (colSum[c] + 1e-8f);
}

json ImageTextTask::handle_event(const json& request,
                                 const std::string& description) {
  json response = request;
  response["type"] = "event";
  response["source"] = "c_backend";
  response["destination"] = "python_server";

  try {
    std::vector<std::vector<int32_t>> tokens;
    bpe_->tokenizerBPE(description, tokens);

    std::vector<std::shared_ptr<BaseImage>> input_texts;
    input_texts.reserve(tokens.size());
    std::shared_ptr<BaseImage> text;
    for (const std::vector<int32_t>& token_vec : tokens) {
      text = ImageFactory::createImage(77, 1, ImageFormat::GRAY,
                                       TDLDataType::INT32, true);
      uint8_t* txt_buffer = text->getVirtualAddress()[0];
      memcpy(txt_buffer, token_vec.data(), 77 * sizeof(int32_t));
      input_texts.push_back(text);
    }

    std::vector<std::shared_ptr<ModelOutputInfo>> text_output_features;
    text_model_->inference(input_texts, text_output_features);

    std::vector<std::shared_ptr<ModelFeatureInfo>> query_features;
    query_features.reserve(text_output_features.size());
    for (const std::shared_ptr<ModelOutputInfo>& out_info :
         text_output_features) {
      std::shared_ptr<ModelFeatureInfo> feature_info =
          std::static_pointer_cast<ModelFeatureInfo>(out_info);
      query_features.push_back(feature_info);
    }

    std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_features;
    std::vector<std::string> image_paths;
    get_gallery_features(gallery_features, image_paths);

    MatchResult results;
    matcher_->loadGallery(gallery_features);
    matcher_->queryWithTopK(query_features, 5, results);
    clip_score_post_process(results.scores);

    std::vector<std::string> matched_images;
    for (size_t i = 0; i < results.indices.size(); ++i) {
      LOGI("  查询特征 %d 的匹配结果:", i);
      const auto& indices = results.indices[i];
      const auto& scores = results.scores[i];
      for (size_t j = 0; j < indices.size() && j < scores.size(); ++j) {
        LOGI(" 特征库图像: %s, 相似度分数: %f\n",
             image_paths[indices[j]].c_str(), scores[j]);
        matched_images.push_back(image_paths[indices[j]]);
      }
      LOGI("\n");
    }

    printf("matched_images size: %d\n", matched_images.size());

    response["payload"]["event"] = "image_and_text_matching";
    nlohmann::json message_list = nlohmann::json::array();
    for (const auto& image_path : matched_images) {
      std::string base64_data =
          APIClient::CommonFunctions::loadImageAsBase64(image_path);
      nlohmann::json item;
      item["image"] = !base64_data.empty() ? base64_data : "";
      item["path"] = image_path;
      parse_person_info(image_path, item);
      message_list.push_back(item);
    }
    response["payload"]["message_list"] = message_list;

  } catch (const std::exception& e) {
    response["payload"]["event"] = "error";
    response["payload"]["message_list"] =
        std::vector<std::string>{"图文匹配失败: " + std::string(e.what())};
  }

  return response;
}
