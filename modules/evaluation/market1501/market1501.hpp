#pragma once
#include "core/face/cvai_face_types.h"

#include <string>
#include <vector>

namespace cviai {
namespace evaluation {
typedef struct {
  int cam_id;
  int pid;
  std::string img_path;
  cvai_feature_t feature;
} market_info;

class market1501Eval {
 public:
  market1501Eval(const char *fiilepath);
  int getEvalData(const char *fiilepath);
  uint32_t getImageNum(bool is_query);
  void getPathIdPair(const int index, bool is_query, std::string *path, int *cam_id, int *pid);
  void insertFeature(const int index, bool is_query, const cvai_feature_t *feature);
  void evalCMC();
  void resetData();

 private:
  std::vector<market_info> m_querys;
  std::vector<market_info> m_gallerys;
};
}  // namespace evaluation
}  // namespace cviai