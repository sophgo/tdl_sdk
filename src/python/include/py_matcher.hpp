#ifndef PYTHON_MATCHER_HPP_
#define PYTHON_MATCHER_HPP_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "base_matcher.hpp"

namespace py = pybind11;
namespace pytdl {

class PyMatchResult {
 public:
  PyMatchResult();
  PyMatchResult(const MatchResult& result);

  std::vector<std::vector<int>> getIndices() const;
  std::vector<std::vector<float>> getScores() const;

 private:
  MatchResult result_;
};

class PyMatcher {
 public:
  // 构造函数
  PyMatcher(std::string matcher_type);

  // 析构函数
  ~PyMatcher();

  // 加载特征库
  int32_t loadGallery(const py::list& gallery_features);

  // 查询特征
  py::list queryWithTopK(const py::list& query_features, int32_t topk);

  // 更新特征库
  int32_t updateGallery(const py::list& update_features, int32_t col);

  // 获取特征库信息
  int32_t getGalleryFeatureNum() const;
  int32_t getQueryFeatureNum() const;
  int32_t getFeatureDim() const;

 private:
  // 清理特征内存
  void clearFeatures(std::vector<std::shared_ptr<ModelFeatureInfo>>& features);

  std::shared_ptr<BaseMatcher> matcher_;
  std::vector<std::shared_ptr<ModelFeatureInfo>> gallery_features_;
  std::vector<std::shared_ptr<ModelFeatureInfo>> query_features_;
};

// 模块级函数
PyMatcher createMatcher(std::string matcher_type);

}  // namespace pytdl
#endif