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

PyMatcher::PyMatcher() {
  matcher_ = BaseMatcher::getMatcher();
  if (matcher_ == nullptr) {
    throw std::runtime_error("Failed to create matcher instance");
  }
}

int32_t PyMatcher::loadGallery(const py::list& gallery_features) {
  std::vector<std::shared_ptr<ModelFeatureInfo>> native_features;

  // 将Python列表转换为C++向量
  for (auto feature : gallery_features) {
    py::array feature_array = feature.cast<py::array>();
    if (!feature_array.dtype().is(py::dtype::of<float>())) {
      throw std::invalid_argument("Feature embedding must be float type");
    }

    auto feature_info = std::make_shared<ModelFeatureInfo>();
    feature_info->embedding_num = feature_array.size();
    feature_info->embedding_type = TDLDataType::FP32;

    // 为特征分配内存并复制数据
    feature_info->embedding = new uint8_t[feature_array.nbytes()];
    std::memcpy(feature_info->embedding, feature_array.data(),
                feature_array.nbytes());

    native_features.push_back(feature_info);
  }

  return matcher_->loadGallery(native_features);
}

py::list PyMatcher::queryWithTopK(const py::list& query_features,
                                  int32_t topk) {
  std::vector<std::shared_ptr<ModelFeatureInfo>> native_query_features;

  // 将Python列表转换为C++向量
  for (auto feature : query_features) {
    py::array feature_array = feature.cast<py::array>();
    if (!feature_array.dtype().is(py::dtype::of<float>())) {
      throw std::invalid_argument("Feature embedding must be float type");
    }

    auto feature_info = std::make_shared<ModelFeatureInfo>();
    feature_info->embedding_num = feature_array.size();
    feature_info->embedding_type = TDLDataType::FP32;

    // 为特征分配内存并复制数据
    feature_info->embedding = new uint8_t[feature_array.nbytes()];
    std::memcpy(feature_info->embedding, feature_array.data(),
                feature_array.nbytes());

    native_query_features.push_back(feature_info);
  }

  MatchResult results;
  int32_t ret = matcher_->queryWithTopK(native_query_features, topk, results);

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
  if (!feature_array.dtype().is(py::dtype::of<float>())) {
    throw std::invalid_argument("特征向量必须是float类型");
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

PyMatcher createMatcher() { return PyMatcher(); }

}  // namespace pytdl