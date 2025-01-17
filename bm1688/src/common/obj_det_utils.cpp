#include "common/obj_det_utils.hpp"
#include <algorithm>
#include <map>
#include <math.h>
#include <vector>
#include <log/Logger.hpp>

bool cmp(const ObjectBox &a, const ObjectBox &b) {
  if (a.score > b.score)
    return true;
  return false;
}
float clip_val(float val, float min_val, float max_val) {
  if (val < min_val)
    val = min_val;
  if (val > max_val)
    val = max_val;
  return val;
}

ObjectBox cluster_boxes(std::vector<ObjectBox> &c){
  float sumx1=0,sumy1=0,sumx2=0,sumy2=0;
  float sumw = 0;
  for(auto &o:c){
    sumx1 += o.x1*o.score;
    sumy1 += o.y1*o.score;
    sumx2 += o.x2*o.score;
    sumy2 += o.y2*o.score;
    sumw +=o.score;
  }
  ObjectBox res;
  res.x1 = sumx1/sumw;
  res.y1 = sumy1/sumw;
  res.x2 = sumx2/sumw;
  res.y2 = sumy2/sumw;
  res.label = c[0].label;
  res.score = c[0].score;
  return res;
}
void nms_impl_weighted(std::vector<ObjectBox> &objs, float nms_thresh) {
  int num_objs = objs.size();
  std::vector<float> vArea(objs.size());
  for (int i = 0; i < num_objs; ++i) {
    vArea[i] = (objs[i].x2 - objs[i].x1 + 1) * (objs[i].y2 - objs[i].y1 + 1);
  }
  std::vector<std::vector<ObjectBox>> cluster_dets;
  std::vector<int> flags(objs.size(),1);
  for (int i = 0; i < objs.size(); ++i) {
    if(flags[i] == 0)continue;
    std::vector<ObjectBox> c;
    c.push_back(objs[i]);
    for (int j = i + 1; j < objs.size();) {
      if(flags[j] == 0){
        j++;
        continue;
      }
      float xx1 = std::max(objs[i].x1, objs[j].x1);
      float yy1 = std::max(objs[i].y1, objs[j].y1);
      float xx2 = std::min(objs[i].x2, objs[j].x2);
      float yy2 = std::min(objs[i].y2, objs[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w < 0)
        w = 0;
      if (h < 0)
        h = 0;
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= nms_thresh) {
        c.push_back(objs[j]);
        flags[j] = 0;
        // objs.erase(objs.begin() + j);
        // vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
    cluster_dets.push_back(c);
  }
  std::vector<ObjectBox> res;
  for(int i = 0; i < cluster_dets.size();i++){
    ObjectBox r = cluster_boxes(cluster_dets[i]);
    res.push_back(r);
  }
  objs = res;
}

void nms_impl(std::vector<ObjectBox> &objs, float nms_thresh) {
  int num_objs = objs.size();
  std::vector<float> vArea(objs.size());
  for (int i = 0; i < num_objs; ++i) {
    vArea[i] = (objs[i].x2 - objs[i].x1 + 1) * (objs[i].y2 - objs[i].y1 + 1);
  }
  for (int i = 0; i < objs.size(); ++i) {
    for (int j = i + 1; j < objs.size();) {
      float xx1 = std::max(objs[i].x1, objs[j].x1);
      float yy1 = std::max(objs[i].y1, objs[j].y1);
      float xx2 = std::min(objs[i].x2, objs[j].x2);
      float yy2 = std::min(objs[i].y2, objs[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w < 0)
        w = 0;
      if (h < 0)
        h = 0;
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= nms_thresh) {
        objs.erase(objs.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

void nms_contain_impl(std::vector<ObjectBox> &objs, float nms_thresh,float contain_thresh) {
  int num_objs = objs.size();
  std::vector<float> vArea(objs.size());
  for (int i = 0; i < num_objs; ++i) {
    vArea[i] = (objs[i].x2 - objs[i].x1 + 1) * (objs[i].y2 - objs[i].y1 + 1);
  }
  for (int i = 0; i < objs.size(); ++i) {
    for (int j = i + 1; j < objs.size();) {
      float xx1 = std::max(objs[i].x1, objs[j].x1);
      float yy1 = std::max(objs[i].y1, objs[j].y1);
      float xx2 = std::min(objs[i].x2, objs[j].x2);
      float yy2 = std::min(objs[i].y2, objs[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w < 0)
        w = 0;
      if (h < 0)
        h = 0;
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      float contain_ratio = inter/std::min(vArea[i],vArea[j]);
      if (ovr >= nms_thresh && contain_ratio > contain_thresh) {
        objs.erase(objs.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

void nms_obj(std::vector<ObjectBox> &objs, float nms_thresh,bool ignore_type/*=false*/) {
  if(ignore_type){
    nms_impl(objs,nms_thresh);
    return;
  }
  std::map<int, std::vector<ObjectBox>> cls_objs;
  for (auto &obj : objs) {
    cls_objs[obj.label].push_back(obj);
  }
  objs.clear();
  for (auto &kv : cls_objs) {
    std::sort(kv.second.begin(), kv.second.end(), cmp);
    nms_impl(kv.second, nms_thresh);
    objs.insert(objs.end(), kv.second.begin(), kv.second.end());
  }
}

void nms_obj_with_type(std::vector<ObjectBox> &objs, std::map<int,float> label_nms_thresh){
  std::map<int, std::vector<ObjectBox>> cls_objs;
  for (auto &obj : objs) {
    cls_objs[obj.label].push_back(obj);
  }
  objs.clear();
  for (auto &kv : cls_objs) {
    float nms_thresh = label_nms_thresh[kv.first];
    std::sort(kv.second.begin(), kv.second.end(), cmp);
    nms_impl(kv.second, nms_thresh);
    objs.insert(objs.end(), kv.second.begin(), kv.second.end());
  }
}


void nms_obj_with_conf_gap(std::vector<ObjectBox> &objs, float nms_thresh,float conf_gap){
  std::sort(objs.begin(), objs.end(), cmp);
  int num_objs = objs.size();
  std::vector<float> vArea(objs.size());
  for (int i = 0; i < num_objs; ++i) {
    vArea[i] = (objs[i].x2 - objs[i].x1 + 1) * (objs[i].y2 - objs[i].y1 + 1);
  }
  for (int i = 0; i < objs.size(); ++i) {
    for (int j = i + 1; j < objs.size();) {
      float xx1 = std::max(objs[i].x1, objs[j].x1);
      float yy1 = std::max(objs[i].y1, objs[j].y1);
      float xx2 = std::min(objs[i].x2, objs[j].x2);
      float yy2 = std::min(objs[i].y2, objs[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w < 0)
        w = 0;
      if (h < 0)
        h = 0;
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= nms_thresh&&objs[i].score>objs[j].score+conf_gap) {
        objs.erase(objs.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

void nms_contain_obj_with_type(std::vector<ObjectBox> &objs, int obj_type,float nms_thresh,float contain_thresh){
  std::map<int, std::vector<ObjectBox>> cls_objs;
  for (auto &obj : objs) {
    cls_objs[obj.label].push_back(obj);
  }
  objs.clear();
  for (auto &kv : cls_objs) {
    if(obj_type == -1 || obj_type == kv.first){
      std::sort(kv.second.begin(), kv.second.end(), cmp);
      nms_contain_impl(kv.second, nms_thresh,contain_thresh);
    }
    objs.insert(objs.end(), kv.second.begin(), kv.second.end());
  }
}

std::vector<std::vector<float>> generate_mmdet_base_anchors(float base_size, float center_offset,
                                                            const std::vector<float> &ratios,
                                                            const std::vector<int> &scales) {
  std::vector<std::vector<float>> base_anchors;
  float x_center = base_size * center_offset;
  float y_center = base_size * center_offset;

  for (size_t i = 0; i < ratios.size(); i++) {
    float h_ratio = sqrt(ratios[i]);
    float w_ratio = 1 / h_ratio;
    for (size_t j = 0; j < scales.size(); j++) {
      float halfw = base_size * w_ratio * scales[j] / 2;
      float halfh = base_size * h_ratio * scales[j] / 2;
      // x1,y1,x2,y2
      std::vector<float> base_anchor = {x_center - halfw, y_center - halfh, x_center + halfw,
                                        y_center + halfh};
      // LOG(INFO) << "anchor:" << base_anchor[0] << "," << base_anchor[1] << "," << base_anchor[2]
      //           << "," << base_anchor[3];
      base_anchors.emplace_back(base_anchor);
    }
  }
  return base_anchors;
}
// x1,y1,x2,y2
std::vector<std::vector<float>> generate_mmdet_grid_anchors(
    int feat_w, int feat_h, int stride, std::vector<std::vector<float>> &base_anchors) {
  std::vector<std::vector<float>> grid_anchors;
  for (size_t k = 0; k < base_anchors.size(); k++) {
    auto &base_anchor = base_anchors[k];
    for (int ih = 0; ih < feat_h; ih++) {
      int sh = ih * stride;
      for (int iw = 0; iw < feat_w; iw++) {
        int sw = iw * stride;
        std::vector<float> grid_anchor = {base_anchor[0] + sw, base_anchor[1] + sh,
                                          base_anchor[2] + sw, base_anchor[3] + sh};
        // if (grid_anchors.size() < 10)
        // LOG(INFO) << "gridanchor:" << grid_anchor[0] << "," << grid_anchor[1] << ","
        //             << grid_anchor[2] << "," << grid_anchor[3];
        grid_anchors.emplace_back(grid_anchor);
      }
    }
  }
  return grid_anchors;
}