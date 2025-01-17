#ifndef COMMON_OBJ_DEF__HPP_
#define COMMON_OBJ_DEF__HPP_
#include <vector>
#include <string>
#include <map>
typedef struct stObj {
  float x1;
  float y1;
  float x2;
  float y2;
  int label;
  float score;
} ObjectBox;

typedef struct _stObjPts {
  _stObjPts() { score = 0; }
  std::vector<float> x;
  std::vector<float> y;
  float score;
} stObjPts;

typedef struct _stCarplate {
  std::vector<float> scores;
  std::string str_labels;
  int plate_type;
  bool allok;
} stCarplate;

struct anchor_box {
  float x1;
  float y1;
  float x2;
  float y2;
};

struct anchor_cfg {
 public:
  int STRIDE;
  std::vector<int> SCALES;
  int BASE_SIZE;
  std::vector<float> RATIOS;
  int ALLOWED_BORDER;

  anchor_cfg() {
    STRIDE = 0;
    SCALES.clear();
    BASE_SIZE = 0;
    RATIOS.clear();
    ALLOWED_BORDER = 0;
  }
};

void nms_obj(std::vector<ObjectBox> &objs, float nms_thresh,bool ignore_type=false);
void nms_obj_with_conf_gap(std::vector<ObjectBox> &objs, float nms_thresh,float conf_gap);
void nms_obj_with_type(std::vector<ObjectBox> &objs, std::map<int,float> label_nms_thresh);
void nms_contain_obj_with_type(std::vector<ObjectBox> &objs, int obj_type,
                              float nms_thresh,float contain_thresh);
float clip_val(float val, float min_val, float max_val);
void get_max_score_index(float *ptr_probs,int num_cls,float &max_score,int &max_score_ind);

std::vector<std::vector<float>> generate_mmdet_base_anchors(float base_size, float center_offset,
                                                            const std::vector<float> &ratios,
                                                            const std::vector<int> &scales);
std::vector<std::vector<float>> generate_mmdet_grid_anchors(
    int feat_w, int feat_h, int stride, std::vector<std::vector<float>> &base_anchors);

#endif