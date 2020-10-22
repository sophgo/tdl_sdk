#pragma once
#include "core/object/cvai_object_types.h"
#include "json.hpp"

namespace cviai {
namespace evaluation {
class CocoEval {
 public:
  CocoEval(const char *path_prefix, const char *json_path);
  void getEvalData(const char *path_prefix, const char *json_path);
  uint32_t getTotalImage();
  void getImageIdPair(const int index, std::string *path, int *id);
  void insertObjectData(const int id, const cvai_object_t *obj);
  void resetReadJsonObject();
  void resetWriteJsonObject();
  void saveJsonObject2File(const char *filepath);
  ~CocoEval();

 private:
  const char *m_path_prefix;
  nlohmann::json m_json_read;
  nlohmann::json m_json_write;
};
}  // namespace evaluation
}  // namespace cviai