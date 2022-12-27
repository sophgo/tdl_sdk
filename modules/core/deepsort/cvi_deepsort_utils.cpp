#include "cvi_deepsort_utils.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

BBOX bbox_tlwh2tlbr(const BBOX &bbox_tlwh) {
  BBOX bbox_tlbr;
  bbox_tlbr(0) = bbox_tlwh(0);
  bbox_tlbr(1) = bbox_tlwh(1);
  bbox_tlbr(2) = bbox_tlwh(0) + bbox_tlwh(2);
  bbox_tlbr(3) = bbox_tlwh(1) + bbox_tlwh(3);
  return bbox_tlbr;
}

BBOX bbox_tlwh2xyah(const BBOX &bbox_tlwh) {
  BBOX bbox_xyah;
  bbox_xyah(0) = bbox_tlwh(0) + 0.5 * bbox_tlwh(2);
  bbox_xyah(1) = bbox_tlwh(1) + 0.5 * bbox_tlwh(3);
  bbox_xyah(2) = bbox_tlwh(2) / bbox_tlwh(3);
  bbox_xyah(3) = bbox_tlwh(3);
  return bbox_xyah;
}

std::string get_INFO_Vector_Int(const std::vector<int> &idxes, int w) {
  if (idxes.empty()) {
    return std::string("[]");
  }
  std::stringstream ss;
  ss << "[" << std::setw(w) << idxes[0];
  for (size_t i = 1; i < idxes.size(); i++) {
    ss << "," << std::setw(w) << idxes[i];
  }
  ss << "]";
  return std::string(ss.str());
}

std::string get_INFO_Vector_Pair_Int_Int(const std::vector<std::pair<int, int>> &pairs, int w) {
  if (pairs.empty()) {
    return std::string("");
  }
  std::stringstream ss;
  for (size_t i = 0; i < pairs.size(); i++) {
    ss << "<" << std::setw(w) << pairs[i].first << "," << std::setw(w) << pairs[i].second << ">\n";
  }
  return std::string(ss.str());
}

std::string get_INFO_Match_Pair(const std::vector<std::pair<int, int>> &pairs,
                                const std::vector<int> &idxes, int w) {
  if (pairs.empty()) {
    return std::string("");
  }
  std::stringstream ss;
  for (size_t i = 0; i < pairs.size(); i++) {
    ss << "<<" << std::setw(w) << idxes[pairs[i].first] << ">" << std::setw(w) << pairs[i].first
       << "," << std::setw(w) << pairs[i].second << ">\n";
  }
  return std::string(ss.str());
}

stRect tlwh2rect(const BBOX &bbox_tlwh) {
  stRect rct;
  rct.x = bbox_tlwh(0);
  rct.y = bbox_tlwh(1);
  rct.width = bbox_tlwh(2);
  rct.height = bbox_tlwh(3);
  return rct;
}

float get_inter_area(const stRect &box1, const stRect &box2) {
  float xx1 = box1.x > box2.x ? box1.x : box2.x;
  float yy1 = box1.y > box2.y ? box1.y : box2.y;
  float box1_x2 = box1.x + box1.width;
  float box1_y2 = box1.y + box1.height;
  float box2_x2 = box2.x + box2.width;
  float box2_y2 = box2.y + box2.height;

  float xx2 = box1_x2 < box2_x2 ? box1_x2 : box2_x2;
  float yy2 = box1_y2 < box2_y2 ? box1_y2 : box2_y2;
  ;
  float w = xx2 - xx1 + 1;
  float h = yy2 - yy1 + 1;
  if (w <= 0 || h <= 0) return 0;
  float inter = w * h;
  return inter;
}
float cal_iou(const stRect &box1, const stRect &box2) {
  float inter = get_inter_area(box1, box2);
  float area1 = box1.width * box1.height;
  float area2 = box2.width * box2.height;
  float iou = inter / (area1 + area2 - inter);
  return iou;
}

float compute_box_sim(stRect box1, stRect box2) {
  int ctx = box1.width + box2.width;
  int cty = box1.height + box2.height;
  box1.x = ctx - box1.width / 2;
  box1.y = cty - box1.height / 2;

  box2.x = ctx - box2.width / 2;
  box2.y = cty - box2.height / 2;

  float inter_area = get_inter_area(box1, box2);
  float area1 = box1.width * box1.height;
  float area2 = box2.width * box2.height;
  float iou = inter_area / (area1 + area2 - inter_area);
  float aspect_diff =
      fabs(atan(box1.height / float(box1.width)) - atan(box2.height / float(box2.width)));
  float sim = iou - aspect_diff;
  return sim;
}

float cal_iou_bbox(const BBOX &box1, const BBOX &box2) {
  stRect rct1 = tlwh2rect(box1);
  stRect rct2 = tlwh2rect(box2);
  float iou = cal_iou(rct1, rct2);
  return iou;
}
float compute_box_sim_bbox(BBOX box1, BBOX box2) {
  stRect rct1 = tlwh2rect(box1);
  stRect rct2 = tlwh2rect(box2);
  float boxsim = compute_box_sim(rct1, rct2);
  return boxsim;
}

bool is_bbox_crowded(const std::vector<BBOX> &bboxes, int check_idx, float expand_ratio) {
  BBOX selbox = bboxes[check_idx];
  float ctx = selbox(0) + selbox(2) / 2;
  float cty = selbox(1) + selbox(3) / 2;
  selbox(2) *= expand_ratio;
  selbox(3) *= expand_ratio;

  selbox(0) = ctx - selbox(2) / 2;
  selbox(1) = cty - selbox(3) / 2;

  for (size_t i = 0; i < bboxes.size(); i++) {
    if (int(i) == check_idx) continue;
    float iou = cal_iou_bbox(bboxes[i], selbox);
    if (iou > 0) return true;
  }
  return false;
}
