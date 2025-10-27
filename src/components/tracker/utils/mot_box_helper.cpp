#include "utils/mot_box_helper.hpp"
#include "common/object_type_def.hpp"
#include "utils/tdl_log.hpp"
// ctx,cty,aspect,height
DETECTBOX MotBoxHelper::convertToXYAH(const ObjectBoxInfo &box_info) {
  DETECTBOX bbox_xyah;
  bbox_xyah(0) = box_info.x1 + 0.5 * (box_info.x2 - box_info.x1);
  bbox_xyah(1) = box_info.y1 + 0.5 * (box_info.y2 - box_info.y1);
  bbox_xyah(2) = (box_info.x2 - box_info.x1) / (box_info.y2 - box_info.y1);
  bbox_xyah(3) = box_info.y2 - box_info.y1;
  return bbox_xyah;
}

float MotBoxHelper::calculateIOU(const ObjectBoxInfo &box1,
                                 const ObjectBoxInfo &box2) {
  float x1 = std::max(box1.x1, box2.x1);
  float y1 = std::max(box1.y1, box2.y1);
  float x2 = std::min(box1.x2, box2.x2);
  float y2 = std::min(box1.y2, box2.y2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  float union_area = box1_area + box2_area - intersection;

  float iou = intersection / union_area;

  return iou;
}
float MotBoxHelper::calculateIOUOnFirst(const ObjectBoxInfo &box1,
                                        const ObjectBoxInfo &box2) {
  float x1 = std::max(box1.x1, box2.x1);
  float y1 = std::max(box1.y1, box2.y1);
  float x2 = std::min(box1.x2, box2.x2);
  float y2 = std::min(box1.y2, box2.y2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);

  float iou = intersection / box1_area;

  return iou;
}

bool MotBoxHelper::isBboxCrowded(const std::vector<ObjectBoxInfo> &dets,
                                 int check_idx, float expand_ratio) {
  ObjectBoxInfo selbox = dets[check_idx];
  float ctx = selbox.x1 + selbox.x2 / 2;
  float cty = selbox.y1 + selbox.y2 / 2;
  selbox.x2 *= expand_ratio;
  selbox.y2 *= expand_ratio;

  selbox.x1 = ctx - selbox.x2 / 2;
  selbox.y1 = cty - selbox.y2 / 2;

  for (size_t i = 0; i < dets.size(); i++) {
    if (int(i) == check_idx) continue;
    float iou = calculateIOU(dets[i], selbox);
    if (iou > 0) return true;
  }
  return false;
}
float MotBoxHelper::getInterArea(const ObjectBoxInfo &box1,
                                 const ObjectBoxInfo &box2) {
  float xx1 = std::max(box1.x1, box2.x1);
  float yy1 = std::max(box1.y1, box2.y1);

  float xx2 = std::min(box1.x2, box2.x2);
  float yy2 = std::min(box1.y2, box2.y2);

  float w = xx2 - xx1 + 1;
  float h = yy2 - yy1 + 1;
  if (w <= 0 || h <= 0) return 0;
  float inter = w * h;
  return inter;
}
float MotBoxHelper::computeBoxSim(const ObjectBoxInfo &box1,
                                  const ObjectBoxInfo &box2) {
  float inter_area = getInterArea(box1, box2);
  float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  float box1_width = box1.x2 - box1.x1;
  float box1_height = box1.y2 - box1.y1;
  float box2_width = box2.x2 - box2.x1;
  float box2_height = box2.y2 - box2.y1;
  float iou = inter_area / (area1 + area2 - inter_area);
  float aspect_diff =
      fabs(atan(box1_height / box1_width) - atan(box2_height / box2_width));
  float sim = iou - aspect_diff;
  return sim;
}

float MotBoxHelper::calObjectPairScore(ObjectBoxInfo boxa, ObjectBoxInfo boxb,
                                       TDLObjectType typea,
                                       TDLObjectType typeb) {
  if (typea == OBJECT_TYPE_FACE && typeb == OBJECT_TYPE_PERSON) {
    float face_width = boxa.x2 - boxa.x1;
    float face_height = boxa.y2 - boxa.y1;
    float person_width = boxb.x2 - boxb.x1;
    float person_height = boxb.y2 - boxb.y1;

    int face_size = std::max(face_width, face_height);
    float face_ct_y = (boxa.y1 + boxa.y2) / 2;

    if (face_ct_y < boxb.y1) return 0;
    float yoffset = 0.2;
    float ydiff = face_ct_y - boxb.y1 - face_size * (yoffset + 0.5);
    float ydiff_score = 1.0 - ydiff / (boxa.y2 - boxa.y1);

    if (person_height > face_size * 18) return 0;
    int ped_ct_x = (boxb.x1 + boxb.x2) / 2;
    int face_ct_x = (boxa.x1 + boxa.x2) / 2;
    float xdiff = abs(ped_ct_x - face_ct_x);
    float xdiff_score = 1.0 - xdiff / face_size;
    ObjectBoxInfo ped_top_box(
        boxb.class_id, boxb.score, boxb.x1 + person_width * 0.2,
        boxb.y1 + 0.25 * face_size, boxb.x1 + 0.8 * person_width,
        boxa.y1 + 1.25 * face_size);
    float iou_on_first = calculateIOUOnFirst(boxa, ped_top_box);
    if (iou_on_first < 0) {
      LOGI(
          "iou_on_first:%f,boxa[%.1f,%.1f,%.1f,%.1f],ped_top_box[%.1f,%.1f,%."
          "1f,%.1f]",
          iou_on_first, boxa.x1, boxa.y1, boxa.x2, boxa.y2, ped_top_box.x1,
          ped_top_box.y1, ped_top_box.x2, ped_top_box.y2);
      return 0;
    }

    float iou = calculateIOU(boxa, ped_top_box);

    float pair_score = ydiff_score * 0.2 + iou * 0.7 + xdiff_score * 0.1;
    LOGI("pairscore:%f,iou:%f", pair_score, iou);
    return pair_score;
  } else if (typea == OBJECT_TYPE_HEAD && typeb == OBJECT_TYPE_PERSON) {
    float head_width = boxa.x2 - boxa.x1;
    float head_height = boxa.y2 - boxa.y1;
    float person_width = boxb.x2 - boxb.x1;
    // float person_height = boxb.y2 - boxb.y1;

    int head_size = std::max(head_width, head_height);
    float head_ct_x = boxa.x1 + 0.5 * (boxa.x2 - boxa.x1);
    float head_ct_y = boxa.y1 + 0.5 * (boxa.y2 - boxa.y1);
    float ped_center_y = boxb.y1 + 0.5 * (boxb.y2 - boxb.y1);
    if (head_ct_x <= boxb.x1 || head_ct_x >= (boxb.x1 + boxb.x2) ||
        head_ct_y >= ped_center_y)
      return 0;
    float yoffset = 0.2;
    float ydiff = std::abs(boxa.y1 - boxb.y1);
    float ydiff_score = 1.0 - ydiff / (boxa.y2 - boxa.y1);

    if (boxb.y2 - boxb.y1 > head_size * 15) return 0;
    int ped_ct_x = (boxb.x1 + boxb.x2) / 2;
    float xdiff = abs(ped_ct_x - head_ct_x);
    float xdiff_score = 1.0 - xdiff / head_size;
    ObjectBoxInfo ped_top_box(
        boxb.class_id, boxb.score, boxb.x1 + person_width * 0.2,
        boxb.y1 + head_size * yoffset, boxb.x1 + person_width,
        boxb.y1 + head_size * yoffset + head_size * (1 + yoffset));

    if (calculateIOUOnFirst(boxa, ped_top_box) < 0.8) return 0;
    float iou = calculateIOU(boxa, ped_top_box);

    float pair_score = ydiff_score * 0.2 + iou * 0.7 + xdiff_score * 0.1;
    LOGI("pairscore:%f,iou:%f", pair_score, iou);
    return pair_score;
  }

  else {
    return 0;
  }
}