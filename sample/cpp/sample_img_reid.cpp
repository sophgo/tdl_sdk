#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tdl_model_factory.hpp"

// L2 normalize feature vector before ReID similarity computation.
static void l2_normalize(std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += (double)x * (double)x;
  double n = std::sqrt(s);
  if (n < 1e-12) return;
  float inv = (float)(1.0 / n);
  for (auto& x : v) x *= inv;
}

// Compute cosine similarity between two normalized feature vectors
static float cosine_similarity(const std::vector<float>& v1,
                               const std::vector<float>& v2) {
  if (v1.size() != v2.size() || v1.empty()) {
    return -1.0f;
  }

  float similarity = 0.0f;
  for (size_t i = 0; i < v1.size(); i++) {
    similarity += v1[i] * v2[i];
  }

  return similarity;
}

// Extract ReID feature from an image
static std::vector<float> extract_reid_feature(
    const std::string& image_path, std::shared_ptr<BaseModel> model) {
  std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
  if (!image) {
    printf("Failed to read image: %s\n", image_path.c_str());
    return {};
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_datas;
  std::vector<std::shared_ptr<BaseImage>> input_images = {image};

  int ret = model->inference(input_images, out_datas);
  if (ret != 0) {
    printf("Inference failed for %s, ret = %d\n", image_path.c_str(), ret);
    return {};
  }

  for (size_t i = 0; i < out_datas.size(); i++) {
    if (out_datas[i]->getType() != ModelOutputType::FEATURE_EMBEDDING) continue;

    auto feature_info =
        std::dynamic_pointer_cast<ModelFeatureInfo>(out_datas[i]);
    if (!feature_info) continue;

    void* raw_data = feature_info->embedding;
    int dim = feature_info->embedding_num;

    if (!raw_data || dim <= 0) {
      printf("Invalid feature data for %s\n", image_path.c_str());
      return {};
    }

    std::vector<float> feat(dim, 0.f);

    if (feature_info->embedding_type == TDLDataType::FP32) {
      float* p = reinterpret_cast<float*>(raw_data);
      for (int j = 0; j < dim; j++) feat[j] = p[j];
    } else if (feature_info->embedding_type == TDLDataType::INT8) {
      int8_t* p = reinterpret_cast<int8_t*>(raw_data);
      for (int j = 0; j < dim; j++) feat[j] = (float)p[j];
    } else if (feature_info->embedding_type == TDLDataType::UINT8) {
      uint8_t* p = reinterpret_cast<uint8_t*>(raw_data);
      for (int j = 0; j < dim; j++) feat[j] = (float)p[j];
    } else {
      printf("Unsupported embedding type %d for %s\n",
             (int)feature_info->embedding_type, image_path.c_str());
      return {};
    }

    l2_normalize(feat);
    return feat;
  }

  printf("No feature embedding found for %s\n", image_path.c_str());
  return {};
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <model_dir> <image1_path> <image2_path>\n", argv[0]);
    printf("Example: %s ./models/ ./image1.jpg ./image2.jpg\n", argv[0]);
    printf("ReID model: FEATURE_REID (osnet_cv181x_int8_sym.cvimodel)\n");
    return -1;
  }

  const std::string model_dir = argv[1];
  const std::string image1_path = argv[2];
  const std::string image2_path = argv[3];

  // Initialize model factory and load model
  TDLModelFactory& model_factory = TDLModelFactory::getInstance();
  model_factory.loadModelConfig();
  model_factory.setModelDir(model_dir);

  std::shared_ptr<BaseModel> model =
      model_factory.getModel(ModelType::FEATURE_REID);
  if (!model) {
    printf("Failed to create ReID model\n");
    return -1;
  }

  printf(">>> Extracting feature from image 1: %s\n", image1_path.c_str());
  std::vector<float> feat1 = extract_reid_feature(image1_path, model);
  if (feat1.empty()) {
    printf("Failed to extract feature from image 1\n");
    return -1;
  }

  printf(">>> Extracting feature from image 2: %s\n", image2_path.c_str());
  std::vector<float> feat2 = extract_reid_feature(image2_path, model);
  if (feat2.empty()) {
    printf("Failed to extract feature from image 2\n");
    return -1;
  }

  // Compute similarity
  float similarity = cosine_similarity(feat1, feat2);

  printf(">>> ReID Similarity Result:\n");
  printf("Image 1: %s\n", image1_path.c_str());
  printf("Image 2: %s\n", image2_path.c_str());
  printf("Feature dimension: %zu\n", feat1.size());
  printf("Cosine Similarity: %.4f\n", similarity);

  return 0;
}
