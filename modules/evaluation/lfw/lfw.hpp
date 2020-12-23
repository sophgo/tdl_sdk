#pragma once
#include "core/face/cvai_face_types.h"

#include <string>
#include <vector>

namespace cviai {
namespace evaluation {
typedef struct {
  int label;
  std::string path1;
  std::string path2;
} lfwpair;

class lfwEval {
 public:
  lfwEval();
  int getEvalData(const char *fiilepath, bool label_pos_first);
  uint32_t getTotalImage();
  void getImageLabelPair(const int index, std::string *path1, std::string *path2, int *label);
  void insertFaceData(const int index, const int label, const cvai_face_t *face1,
                      const cvai_face_t *face2);
  void insertLabelScore(const int &index, const int &label, const float &score);
  void resetData();
  void resetEvalData();
  void saveEval2File(const char *filepath);

 private:
  int getMaxFace(const cvai_face_t *face);
  float evalDifference(const cvai_feature_t *features1, const cvai_feature_t *features2);
  void evalAUC(const std::vector<int> &y, std::vector<float> &pred, const uint32_t data_size,
               const char *filepath);
  std::vector<lfwpair> m_data;
  std::vector<int> m_eval_label;
  std::vector<float> m_eval_score;
};
}  // namespace evaluation
}  // namespace cviai