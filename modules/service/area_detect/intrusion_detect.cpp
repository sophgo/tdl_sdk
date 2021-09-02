#include "intrusion_detect.hpp"
// #include "cviai_core.h"
#include <cvi_type.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "stdio.h"

#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))

namespace cviai {
namespace service {

ConvexPolygon::~ConvexPolygon() {
  CVI_AI_FreeCpp(&vertices);
  CVI_AI_FreeCpp(&orthogonals);
}

ConvexPolygon::ConvexPolygon(const cvai_pts_t &pts) {
  /* copy vertices */
  this->vertices.size = pts.size;
  this->vertices.x = (float *)malloc(sizeof(float) * pts.size);
  this->vertices.y = (float *)malloc(sizeof(float) * pts.size);
  memcpy(this->vertices.x, pts.x, sizeof(float) * pts.size);
  memcpy(this->vertices.y, pts.y, sizeof(float) * pts.size);

  /* calculate edges */
  float *edge_x = new float[pts.size];
  for (uint32_t i = 0; i < pts.size; i++) {
    uint32_t j = ((i + 1) == pts.size) ? 0 : (i + 1);
    edge_x[i] = pts.x[j] - pts.x[i];
  }
  float *edge_y = new float[pts.size];
  for (uint32_t i = 0; i < pts.size; i++) {
    uint32_t j = ((i + 1) == pts.size) ? 0 : (i + 1);
    edge_y[i] = pts.y[j] - pts.y[i];
  }

  // /* calculate orthogonals */
  this->orthogonals.size = pts.size;
  this->orthogonals.x = (float *)malloc(sizeof(float) * pts.size);
  this->orthogonals.y = (float *)malloc(sizeof(float) * pts.size);
  memcpy(this->orthogonals.x, edge_y, sizeof(float) * pts.size);
  for (uint32_t i = 0; i < pts.size; i++) {
    this->orthogonals.x[i] *= -1.0;
  }
  memcpy(this->orthogonals.y, edge_x, sizeof(float) * pts.size);

  // /* delete useless data */
  delete[] edge_x;
  delete[] edge_y;
}

void ConvexPolygon::show() {
  for (size_t i = 0; i < this->vertices.size; i++) {
    std::cout << "(" << std::setw(4) << this->vertices.x[i] << "," << std::setw(4)
              << this->vertices.y[i] << ")";
    if ((i + 1) != this->vertices.size)
      std::cout << "  &  ";
    else
      std::cout << "\n";
  }
  for (size_t i = 0; i < this->orthogonals.size; i++) {
    std::cout << "(" << std::setw(4) << this->orthogonals.x[i] << "," << std::setw(4)
              << this->orthogonals.y[i] << ")";
    if ((i + 1) != this->orthogonals.size)
      std::cout << "  &  ";
    else
      std::cout << "\n";
  }
}

IntrusionDetect::IntrusionDetect() {
  base.size = 2;
  base.x = (float *)malloc(sizeof(float) * 2);
  base.y = (float *)malloc(sizeof(float) * 2);
  base.x[0] = 1.;
  base.x[1] = 0.;
  base.y[0] = 0.;
  base.y[1] = 1.;
}

IntrusionDetect::~IntrusionDetect() { CVI_AI_FreeCpp(&base); }

void IntrusionDetect::show() {
  std::cout << "Region Num: " << regions.size() << std::endl;
  for (size_t i = 0; i < regions.size(); i++) {
    std::cout << "[" << i << "]\n";
    regions[i]->show();
  }
}

int IntrusionDetect::setRegion(const cvai_pts_t &pts) {
  auto new_region = std::make_shared<ConvexPolygon>(pts);
  regions.push_back(new_region);
  this->show();
  return CVIAI_SUCCESS;
}

bool IntrusionDetect::run(const cvai_bbox_t &bbox) {
  // printf("[RUN] BBox: (%.2f,%.2f,%.2f,%.2f)\n", bbox.x1, bbox.y1, bbox.x2, bbox.y2);
  for (size_t i = 0; i < regions.size(); i++) {
    bool separating = false;
    for (size_t j = 0; j < regions[i]->orthogonals.size; j++) {
      vertex_t o;
      o.x = regions[i]->orthogonals.x[j];
      o.y = regions[i]->orthogonals.y[j];
      if (is_separating_axis(o, regions[i]->vertices, bbox)) {
        separating = true;
        break;
      }
    }
    if (separating) {
      continue;
    }
    for (size_t j = 0; j < base.size; j++) {
      vertex_t o;
      o.x = base.x[j];
      o.y = base.y[j];
      if (is_separating_axis(o, regions[i]->vertices, bbox)) {
        separating = true;
        break;
      }
    }
    if (separating) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

bool IntrusionDetect::is_separating_axis(const vertex_t &axis, const cvai_pts_t &region_pts,
                                         const cvai_bbox_t &bbox) {
  float min_1 = std::numeric_limits<float>::max();
  float max_1 = -std::numeric_limits<float>::max();
  float min_2 = std::numeric_limits<float>::max();
  float max_2 = -std::numeric_limits<float>::max();
  float proj;
  for (uint32_t i = 0; i < region_pts.size; i++) {
    proj = axis.x * region_pts.x[i] + axis.y * region_pts.y[i];
    min_1 = MIN(min_1, proj);
    max_1 = MAX(max_1, proj);
  }
  proj = axis.x * bbox.x1 + axis.y * bbox.y1;
  min_2 = MIN(min_2, proj);
  max_2 = MAX(max_2, proj);
  proj = axis.x * bbox.x2 + axis.y * bbox.y1;
  min_2 = MIN(min_2, proj);
  max_2 = MAX(max_2, proj);
  proj = axis.x * bbox.x2 + axis.y * bbox.y2;
  min_2 = MIN(min_2, proj);
  max_2 = MAX(max_2, proj);
  proj = axis.x * bbox.x1 + axis.y * bbox.y2;
  min_2 = MIN(min_2, proj);
  max_2 = MAX(max_2, proj);

  if ((max_1 >= min_2) && (max_2 >= min_1)) {
    return false;
  } else {
    // printf("[SEP] min_1 = %f | max_1 = %f | min_2 = %f | max_2 = %f\n", min_1, max_1, min_2,
    // max_2);
    return true;
  }
}

}  // namespace service
}  // namespace cviai