#include "coco.hpp"

#include <syslog.h>
#include <fstream>
#include <string>

namespace cviai {
namespace evaluation {
CocoEval::CocoEval(const char *path_prefix, const char *json_path) {
  getEvalData(path_prefix, json_path);
}

CocoEval::~CocoEval() {}

void CocoEval::getEvalData(const char *path_prefix, const char *json_path) {
  m_path_prefix = path_prefix;
  m_json_read.clear();
  std::ifstream filestr(json_path);
  filestr >> m_json_read;
  filestr.close();
}

uint32_t CocoEval::getTotalImage() { return m_json_read["images"].size(); }

void CocoEval::getImageIdPair(const int index, std::string *path, int *id) {
  *path = m_path_prefix + std::string("/") + std::string(m_json_read["images"][index]["file_name"]);
  *id = m_json_read["images"][index]["id"];
}

void CocoEval::insertObjectData(const int id, const cvai_object_t *obj) {
  syslog(LOG_INFO, "Image id %d insert object %d\n", id, obj->size);
  for (uint32_t j = 0; j < obj->size; j++) {
    cvai_object_info_t &info = obj->info[j];
    float width = info.bbox.x2 - info.bbox.x1;
    float height = info.bbox.y2 - info.bbox.y1;
    m_json_write.push_back({
        {"image_id", id},
        {"category_id", info.classes},
        {"bbox", {info.bbox.x1, info.bbox.y1, width, height}},
        {"score", info.bbox.score},
    });
  }
}

void CocoEval::resetReadJsonObject() { m_json_read.clear(); }

void CocoEval::resetWriteJsonObject() { m_json_write.clear(); }

void CocoEval::saveJsonObject2File(const char *filepath) {
  std::ofstream result(filepath);
  result << m_json_write;
  result.close();
}
}  // namespace evaluation
}  // namespace cviai