#include "py_matcher.hpp"
#include <string>
#include "py_utils.hpp"

namespace pytdl {

PyMatchResult::PyMatchResult() {}

PyMatchResult::PyMatchResult(const MatchResult& result) : result_(result) {}

std::vector<std::vector<int>> PyMatchResult::getIndices() const {
  return result_.indices;
}

std::vector<std::vector<float>> PyMatchResult::getScores() const {
  return result_.scores;
}

PyMatcher::PyMatcher(std::string matcher_type) {
  matcher_ = BaseMatcher::getMatcher(matcher_type);
  if (matcher_ == nullptr) {
    throw std::runtime_error("Failed to create matcher instance");
  }
}

PyMatcher::~PyMatcher() {
  // 清理存储的特征信息
  clearFeatures(gallery_features_);
  clearFeatures(query_features_);
}

void PyMatcher::clearFeatures(
    std::vector<std::shared_ptr<ModelFeatureInfo>>& features) {
  for (auto& feature : features) {
    if (feature && feature->embedding) {
      delete[] static_cast<uint8_t*>(feature->embedding);
      feature->embedding = nullptr;
    }
  }
  features.clear();
}

int32_t PyMatcher::loadGallery(const py::list& gallery_features) {
  // 清理之前的特征
  clearFeatures(gallery_features_);

  // 将Python列表转换为C++向量
  for (auto feature : gallery_features) {
    py::array feature_array = feature.cast<py::array>();
    auto feature_info = std::make_shared<ModelFeatureInfo>();

    // 支持不同的数据类型
    if (feature_array.dtype().is(py::dtype::of<float>())) {
      feature_info->embedding_type = TDLDataType::FP32;
    } else if (feature_array.dtype().is(py::dtype::of<int8_t>())) {
      feature_info->embedding_type = TDLDataType::INT8;
    } else if (feature_array.dtype().is(py::dtype::of<uint8_t>())) {
      feature_info->embedding_type = TDLDataType::UINT8;
    } else {
      throw std::invalid_argument("特征向量必须是float、int8或uint8类型");
    }

    feature_info->embedding_num = feature_array.size();

    // 为特征分配内存并复制数据
    feature_info->embedding = new uint8_t[feature_array.nbytes()];
    std::memcpy(feature_info->embedding, feature_array.data(),
                feature_array.nbytes());

    gallery_features_.push_back(feature_info);
  }

  return matcher_->loadGallery(gallery_features_);
}

py::list PyMatcher::queryWithTopK(const py::list& query_features,
                                  int32_t topk) {
  // 清理之前的查询特征
  clearFeatures(query_features_);

  // 将Python列表转换为C++向量
  for (auto feature : query_features) {
    py::array feature_array = feature.cast<py::array>();
    auto feature_info = std::make_shared<ModelFeatureInfo>();

    // 支持不同的数据类型
    if (feature_array.dtype().is(py::dtype::of<float>())) {
      feature_info->embedding_type = TDLDataType::FP32;
    } else if (feature_array.dtype().is(py::dtype::of<int8_t>())) {
      feature_info->embedding_type = TDLDataType::INT8;
    } else if (feature_array.dtype().is(py::dtype::of<uint8_t>())) {
      feature_info->embedding_type = TDLDataType::UINT8;
    } else {
      throw std::invalid_argument("特征向量必须是float、int8或uint8类型");
    }

    feature_info->embedding_num = feature_array.size();

    // 为特征分配内存并复制数据
    feature_info->embedding = new uint8_t[feature_array.nbytes()];
    std::memcpy(feature_info->embedding, feature_array.data(),
                feature_array.nbytes());

    query_features_.push_back(feature_info);
  }

  MatchResult results;
  int32_t ret = matcher_->queryWithTopK(query_features_, topk, results);

  if (ret != 0) {
    throw std::runtime_error("Failed to query features, error code: " +
                             std::to_string(ret));
  }

  // 将结果转换为Python列表
  py::list py_indices;
  py::list py_scores;
  for (const auto& indices : results.indices) {
    py::list py_indices_list;
    for (const auto& index : indices) {
      py_indices_list.append(index);
    }
    py_indices.append(py_indices_list);
  }
  for (const auto& scores : results.scores) {
    py::list py_scores_list;
    for (const auto& score : scores) {
      py_scores_list.append(score);
    }
    py_scores.append(py_scores_list);
  }

  return py::make_tuple(py_indices, py_scores);
}

int32_t PyMatcher::updateGallery(const py::list& update_features, int32_t col) {
  // 检查是否只有一个特征
  if (update_features.size() != 1) {
    throw std::invalid_argument(
        "只能更新一个特征，请确保列表中只包含一个特征向量");
  }

  // 获取特征数组
  py::array feature_array = update_features[0].cast<py::array>();

  // 支持不同的数据类型
  TDLDataType data_type;
  if (feature_array.dtype().is(py::dtype::of<float>())) {
    data_type = TDLDataType::FP32;
  } else if (feature_array.dtype().is(py::dtype::of<int8_t>())) {
    data_type = TDLDataType::INT8;
  } else if (feature_array.dtype().is(py::dtype::of<uint8_t>())) {
    data_type = TDLDataType::UINT8;
  } else {
    throw std::invalid_argument("特征向量必须是float、int8或uint8类型");
  }

  // 获取特征数据指针作为void*传递
  void* feature_data = feature_array.mutable_data();

  // 调用原生方法更新特征列
  return matcher_->updateGalleryCol(feature_data, col);
}

int32_t PyMatcher::getGalleryFeatureNum() const {
  return matcher_->getGalleryFeatureNum();
}

int32_t PyMatcher::getQueryFeatureNum() const {
  return matcher_->getQueryFeatureNum();
}

int32_t PyMatcher::getFeatureDim() const { return matcher_->getFeatureDim(); }

PyMatcher createMatcher(std::string matcher_type) {
  return PyMatcher(matcher_type);
}

}  // namespace pytdl