#include "utils/detection_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

std::vector<std::vector<float>> DetectionHelper::generateMmdetBaseAnchors(
    float base_size,
    float center_offset,
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
      std::vector<float> base_anchor = {x_center - halfw, y_center - halfh,
                                        x_center + halfw, y_center + halfh};

      base_anchors.emplace_back(base_anchor);
    }
  }
  return base_anchors;
}

std::vector<std::vector<float>> DetectionHelper::generateMmdetGridAnchors(
    int feat_w,
    int feat_h,
    int stride,
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

// stride->anchor_boxes->[x,y,w,h]
std::vector<std::vector<std::vector<float>>>
DetectionHelper::generateRetinaNetAnchors(
    int min_level,
    int max_level,
    int num_scales,
    const std::vector<std::pair<float, float>> &aspect_ratios,
    float anchor_scale,
    int image_width,
    int image_height) {
  std::vector<std::vector<std::vector<float>>> anchor_bboxes;

  // 遍历各层级（例如 3~7）
  for (int level = min_level; level <= max_level; ++level) {
    int stride = static_cast<int>(std::pow(2, level));
    std::vector<std::vector<float>> boxes_level;

    // 在特征图上按 stride 间隔采样中心点
    for (int y = stride / 2; y < image_height; y += stride) {
      for (int x = stride / 2; x < image_width; x += stride) {
        // 对于每个尺度（octave）和每个宽高比生成一个 AnchorBox
        for (int scale_octave = 0; scale_octave < num_scales; ++scale_octave) {
          float octave_scale = static_cast<float>(scale_octave) / num_scales;
          for (const auto &aspect : aspect_ratios) {
            float base_anchor_size =
                anchor_scale * stride * std::pow(2, octave_scale);
            float box_width = base_anchor_size * aspect.first;
            float box_height = base_anchor_size * aspect.second;
            float anchor_x = x - box_width / 2.0f;
            float anchor_y = y - box_height / 2.0f;
            std::vector<float> anchor_box = {anchor_x, anchor_y, box_width,
                                             box_height};
            boxes_level.emplace_back(anchor_box);
          }
        }
      }
    }

    anchor_bboxes.push_back(boxes_level);
  }
  return anchor_bboxes;
}

void DetectionHelper::nmsObjects(std::vector<ObjectBoxLandmarkInfo> &objects,
                                 float iou_threshold) {
  std::sort(objects.begin(), objects.end(),
            [](ObjectBoxLandmarkInfo &a, ObjectBoxLandmarkInfo &b) {
              return a.score > b.score;
            });

  int select_idx = 0;
  int num_bbox = objects.size();
  std::vector<int> mask_merged(num_bbox, 0);
  std::vector<int> select_idx_merged(num_bbox, 0);
  std::vector<ObjectBoxLandmarkInfo> objects_nms;
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    objects_nms.emplace_back(objects[select_idx]);
    mask_merged[select_idx] = 1;
    select_idx_merged[select_idx] = 1;
    ObjectBoxLandmarkInfo select_object = objects[select_idx];
    float area1 = static_cast<float>((select_object.x2 - select_object.x1 + 1) *
                                     (select_object.y2 - select_object.y1 + 1));
    float x1 = static_cast<float>(select_object.x1);
    float y1 = static_cast<float>(select_object.y1);
    float x2 = static_cast<float>(select_object.x2);
    float y2 = static_cast<float>(select_object.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      ObjectBoxLandmarkInfo &object_i = objects[i];
      float x = std::max<float>(x1, static_cast<float>(object_i.x1));
      float y = std::max<float>(y1, static_cast<float>(object_i.y1));
      float w = std::min<float>(x2, static_cast<float>(object_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(object_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) {
        continue;
      }

      float area2 = static_cast<float>((object_i.x2 - object_i.x1 + 1) *
                                       (object_i.y2 - object_i.y1 + 1));
      float area_intersect = w * h;
      if (static_cast<float>(area_intersect) /
              (area1 + area2 - area_intersect) >
          iou_threshold) {
        mask_merged[i] = 1;
        continue;
      }
    }
  }

  objects = objects_nms;
}

void DetectionHelper::nmsObjects(std::vector<ObjectBoxInfo> &objects,
                                 float iou_threshold) {
  std::sort(
      objects.begin(), objects.end(),
      [](ObjectBoxInfo &a, ObjectBoxInfo &b) { return a.score > b.score; });

  int select_idx = 0;
  int num_bbox = objects.size();
  std::vector<int> mask_merged(num_bbox, 0);
  std::vector<int> select_idx_merged(num_bbox, 0);
  std::vector<ObjectBoxInfo> objects_nms;
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    objects_nms.emplace_back(objects[select_idx]);
    mask_merged[select_idx] = 1;
    select_idx_merged[select_idx] = 1;
    ObjectBoxInfo select_bbox = objects[select_idx];
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                     (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      ObjectBoxInfo &bbox_i = objects[i];
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
  objects = objects_nms;
}

void DetectionHelper::nmsObjects(
    std::map<int, std::vector<ObjectBoxInfo>> &objects, float iou_threshold) {
  for (auto &object : objects) {
    nmsObjects(object.second, iou_threshold);
  }
}
void DetectionHelper::nmsObjects(
    std::vector<ObjectBoxSegmentationInfo> &objects,
    float iou_threshold,
    std::vector<std::pair<int, uint32_t>> &stride_index) {
  std::vector<size_t> indices(objects.size());
  for (size_t i = 0; i < objects.size(); ++i) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&objects](size_t a, size_t b) {
    return objects[a].score > objects[b].score;
  });

  std::vector<ObjectBoxSegmentationInfo> sorted_objects(objects.size());
  std::vector<std::pair<int, uint32_t>> sorted_stride_index(
      stride_index.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    sorted_objects[i] = objects[indices[i]];
    sorted_stride_index[i] = stride_index[indices[i]];
  }

  int select_idx = 0;
  int num_bbox = sorted_objects.size();
  std::vector<int> mask_merged(num_bbox, 0);
  std::vector<int> select_idx_merged(num_bbox, 0);
  std::vector<ObjectBoxSegmentationInfo> objects_nms;
  std::vector<std::pair<int, uint32_t>> stride_index_nms;
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    objects_nms.emplace_back(sorted_objects[select_idx]);
    stride_index_nms.emplace_back(sorted_stride_index[select_idx]);

    mask_merged[select_idx] = 1;
    select_idx_merged[select_idx] = 1;
    ObjectBoxSegmentationInfo select_bbox = sorted_objects[select_idx];
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                     (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      ObjectBoxSegmentationInfo &bbox_i = sorted_objects[i];
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
  objects = objects_nms;
  stride_index = stride_index_nms;
}
void DetectionHelper::nmsObjects(
    std::vector<ObjectBoxLandmarkInfo> &objects,
    float iou_threshold,
    std::vector<std::pair<int, uint32_t>> &stride_index) {
  std::vector<size_t> indices(objects.size());
  for (size_t i = 0; i < objects.size(); ++i) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&objects](size_t a, size_t b) {
    return objects[a].score > objects[b].score;
  });

  std::vector<ObjectBoxLandmarkInfo> sorted_objects(objects.size());
  std::vector<std::pair<int, uint32_t>> sorted_stride_index(
      stride_index.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    sorted_objects[i] = objects[indices[i]];
    sorted_stride_index[i] = stride_index[indices[i]];
  }

  int select_idx = 0;
  int num_bbox = sorted_objects.size();
  std::vector<int> mask_merged(num_bbox, 0);
  std::vector<int> select_idx_merged(num_bbox, 0);
  std::vector<ObjectBoxLandmarkInfo> objects_nms;
  std::vector<std::pair<int, uint32_t>> stride_index_nms;
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    objects_nms.emplace_back(sorted_objects[select_idx]);
    stride_index_nms.emplace_back(sorted_stride_index[select_idx]);

    mask_merged[select_idx] = 1;
    select_idx_merged[select_idx] = 1;
    ObjectBoxLandmarkInfo select_bbox = sorted_objects[select_idx];
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                     (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      ObjectBoxLandmarkInfo &bbox_i = sorted_objects[i];
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
  objects = objects_nms;
  stride_index = stride_index_nms;
}
void DetectionHelper::rescaleBbox(ObjectBoxInfo &bbox,
                                  const std::vector<float> &scale_params) {
  float scale_x = scale_params[0];
  float scale_y = scale_params[1];
  float offset_x = scale_params[2];
  float offset_y = scale_params[3];
  bbox.x1 = bbox.x1 * scale_x + offset_x;
  bbox.y1 = bbox.y1 * scale_y + offset_y;
  bbox.x2 = bbox.x2 * scale_x + offset_x;
  bbox.y2 = bbox.y2 * scale_y + offset_y;
}

void DetectionHelper::rescaleBbox(ObjectBoxSegmentationInfo &bbox,
                                  const std::vector<float> &scale_params) {
  float scale_x = scale_params[0];
  float scale_y = scale_params[1];
  float offset_x = scale_params[2];
  float offset_y = scale_params[3];
  bbox.x1 = bbox.x1 * scale_x + offset_x;
  bbox.y1 = bbox.y1 * scale_y + offset_y;
  bbox.x2 = bbox.x2 * scale_x + offset_x;
  bbox.y2 = bbox.y2 * scale_y + offset_y;
}

void DetectionHelper::rescaleBbox(ObjectBoxLandmarkInfo &bbox,
                                  const std::vector<float> &scale_params) {
  float scale_x = scale_params[0];
  float scale_y = scale_params[1];
  float offset_x = scale_params[2];
  float offset_y = scale_params[3];
  bbox.x1 = bbox.x1 * scale_x + offset_x;
  bbox.y1 = bbox.y1 * scale_y + offset_y;
  bbox.x2 = bbox.x2 * scale_x + offset_x;
  bbox.y2 = bbox.y2 * scale_y + offset_y;

  for (size_t i = 0; i < bbox.landmarks_x.size(); ++i) {
    bbox.landmarks_x[i] = bbox.landmarks_x[i] * scale_x + offset_x;
  }

  for (size_t i = 0; i < bbox.landmarks_y.size(); ++i) {
    bbox.landmarks_y[i] = bbox.landmarks_y[i] * scale_y + offset_y;
  }
}

// void DetectionHelper::convertDetStruct(
//     std::map<int, std::vector<tdl_bbox_t>> &dets, TDLObject *obj,
//     int im_height, int im_width) {
//   int num_obj = 0;
//   for (auto &bbox : dets) {
//     num_obj += bbox.second.size();
//   }
//   memset(obj, 0, sizeof(TDLObject));
//   obj->height = im_height;
//   obj->width = im_width;
//   if (num_obj == 0) {
//     return;
//   }
//   CVI_TDL_MemAllocInit(num_obj, obj);

//   memset(obj->info, 0, sizeof(TDLObjectInfo) * obj->size);

//   int idx = 0;
//   for (auto &bbox : dets) {
//     for (auto &b : bbox.second) {
//       obj->info[idx].bbox.x1 =
//           std::max(0.0f, std::min(b.x1, static_cast<float>(im_width)));
//       obj->info[idx].bbox.y1 =
//           std::max(0.0f, std::min(b.y1, static_cast<float>(im_height)));
//       obj->info[idx].bbox.x2 =
//           std::max(0.0f, std::min(b.x2, static_cast<float>(im_width)));
//       obj->info[idx].bbox.y2 =
//           std::max(0.0f, std::min(b.y2, static_cast<float>(im_height)));
//       obj->info[idx].bbox.score = b.score;
//       obj->info[idx].classes = bbox.first;
//       idx++;
//     }
//   }
// }