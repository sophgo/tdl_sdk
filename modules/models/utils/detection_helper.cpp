#include "detection_helper.hpp"

#include <algorithm>
#include <cmath>
std::vector<std::vector<float>> DetectionHelper::generateMmdetBaseAnchors(
    float base_size, float center_offset, const std::vector<float> &ratios,
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
      std::vector<float> base_anchor = {x_center - halfw, y_center - halfh,
                                        x_center + halfw, y_center + halfh};

      base_anchors.emplace_back(base_anchor);
    }
  }
  return base_anchors;
}

std::vector<std::vector<float>> DetectionHelper::generateMmdetGridAnchors(
    int feat_w, int feat_h, int stride,
    std::vector<std::vector<float>> &base_anchors) {
  std::vector<std::vector<float>> grid_anchors;
  for (size_t k = 0; k < base_anchors.size(); k++) {
    auto &base_anchor = base_anchors[k];
    for (int ih = 0; ih < feat_h; ih++) {
      int sh = ih * stride;
      for (int iw = 0; iw < feat_w; iw++) {
        int sw = iw * stride;
        std::vector<float> grid_anchor = {
            base_anchor[0] + sw, base_anchor[1] + sh, base_anchor[2] + sw,
            base_anchor[3] + sh};
        // if (grid_anchors.size() < 10)
        //   std::cout << "gridanchor:" << grid_anchor[0] << "," <<
        //   grid_anchor[1] << ","
        //             << grid_anchor[2] << "," << grid_anchor[3] << std::endl;
        grid_anchors.emplace_back(grid_anchor);
      }
    }
  }
  return grid_anchors;
}

void DetectionHelper::nmsFaces(std::vector<cvtdl_face_info_t> &faces,
                               float iou_threshold) {
  std::sort(faces.begin(), faces.end(),
            [](cvtdl_face_info_t &a, cvtdl_face_info_t &b) {
              return a.bbox.score > b.bbox.score;
            });

  int select_idx = 0;
  int num_bbox = faces.size();
  std::vector<int> mask_merged(num_bbox, 0);
  std::vector<int> select_idx_merged(num_bbox, 0);
  std::vector<cvtdl_face_info_t> faces_nms;
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    faces_nms.emplace_back(faces[select_idx]);
    mask_merged[select_idx] = 1;
    select_idx_merged[select_idx] = 1;
    cvtdl_face_info_t select_face = faces[select_idx];
    float area1 =
        static_cast<float>((select_face.bbox.x2 - select_face.bbox.x1 + 1) *
                           (select_face.bbox.y2 - select_face.bbox.y1 + 1));
    float x1 = static_cast<float>(select_face.bbox.x1);
    float y1 = static_cast<float>(select_face.bbox.y1);
    float x2 = static_cast<float>(select_face.bbox.x2);
    float y2 = static_cast<float>(select_face.bbox.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      cvtdl_face_info_t &face_i = faces[i];
      float x = std::max<float>(x1, static_cast<float>(face_i.bbox.x1));
      float y = std::max<float>(y1, static_cast<float>(face_i.bbox.y1));
      float w = std::min<float>(x2, static_cast<float>(face_i.bbox.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(face_i.bbox.y2)) - y + 1;
      if (w <= 0 || h <= 0) {
        continue;
      }

      float area2 = static_cast<float>((face_i.bbox.x2 - face_i.bbox.x1 + 1) *
                                       (face_i.bbox.y2 - face_i.bbox.y1 + 1));
      float area_intersect = w * h;
      if (static_cast<float>(area_intersect) /
              (area1 + area2 - area_intersect) >
          iou_threshold) {
        mask_merged[i] = 1;
        continue;
      }
    }
  }
  for (int i = 0; i < num_bbox; i++) {
    if (select_idx_merged[i] == 0) {
      free(faces[i].pts.x);
      free(faces[i].pts.y);
    }
  }
  faces = faces_nms;
}

void DetectionHelper::nmsObjects(std::vector<cvtdl_bbox_t> &bboxes,
                                 float iou_threshold) {
  std::sort(bboxes.begin(), bboxes.end(),
            [](cvtdl_bbox_t &a, cvtdl_bbox_t &b) { return a.score > b.score; });

  int select_idx = 0;
  int num_bbox = bboxes.size();
  std::vector<int> mask_merged(num_bbox, 0);
  std::vector<int> select_idx_merged(num_bbox, 0);
  std::vector<cvtdl_bbox_t> bboxes_nms;
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms.emplace_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;
    select_idx_merged[select_idx] = 1;
    cvtdl_bbox_t select_bbox = bboxes[select_idx];
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                     (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      cvtdl_bbox_t &bbox_i = bboxes[i];
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) {
        continue;
      }

      float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) *
                                       (bbox_i.y2 - bbox_i.y1 + 1));
      float area_intersect = w * h;
      if (static_cast<float>(area_intersect) /
              (area1 + area2 - area_intersect) >
          iou_threshold) {
        mask_merged[i] = 1;
        continue;
      }
    }
  }
  bboxes = bboxes_nms;
}

void DetectionHelper::nmsObjects(
    std::map<int, std::vector<cvtdl_bbox_t>> &bboxes, float iou_threshold) {
  for (auto &bbox : bboxes) {
    nmsObjects(bbox.second, iou_threshold);
  }
}
