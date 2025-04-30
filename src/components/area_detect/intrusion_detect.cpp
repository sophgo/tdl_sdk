#include "area_detect/intrusion_detect.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "assert.h"
#include "stdio.h"

#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))

ConvexPolygon::~ConvexPolygon() {}

bool ConvexPolygon::isConvex(const std::vector<float> &edges_x,
                             const std::vector<float> &edges_y) {
  for (size_t i = 0; i < edges_x.size(); i++) {
    size_t j = (i != edges_x.size() - 1) ? (i + 1) : 0;
    /* 因为顺序是顺时针 */
    if ((edges_x[i] * edges_y[j]) - (edges_x[j] * edges_y[i]) > 0) {
      return false;
    }
  }
  return true;
}

bool ConvexPolygon::setVertices(const PointsInfo &points) {
  /* 计算边 */
  std::vector<float> edge_x(points.x.size());
  for (size_t i = 0; i < points.x.size(); i++) {
    size_t j = ((i + 1) == points.x.size()) ? 0 : (i + 1);
    edge_x[i] = points.x[j] - points.x[i];
  }
  std::vector<float> edge_y(points.y.size());
  for (size_t i = 0; i < points.y.size(); i++) {
    size_t j = ((i + 1) == points.y.size()) ? 0 : (i + 1);
    edge_y[i] = points.y[j] - points.y[i];
  }

  /* 检查凸性条件 */
  if (!isConvex(edge_x, edge_y)) {
    return false;
  }

  /* 复制点 */
  this->points_.x = points.x;
  this->points_.y = points.y;

  /* 计算法线点 */
  this->normal_points_.x = edge_y;
  this->normal_points_.y = edge_x;
  for (size_t i = 0; i < points.x.size(); i++) {
    this->normal_points_.x[i] *= -1.0;
  }

  return true;
}

IntrusionDetect::IntrusionDetect() {
  base_points_.x = {1., 0.};
  base_points_.y = {0., 1.};
}

IntrusionDetect::~IntrusionDetect() {}

int IntrusionDetect::addRegion(const PointsInfo &points,
                               const std::string &region_name) {
  PointsInfo new_pts;
  new_pts.x = points.x;
  new_pts.y.resize(points.y.size());
  /* 将坐标系从图像坐标系转换到欧几里得坐标系 */
  for (size_t i = 0; i < points.y.size(); i++) {
    /* (x,y) 变为 (x,-y) */
    new_pts.y[i] = -1. * points.y[i];
  }
  float area = getSignedGaussArea(new_pts);
  if (area > 0) {
    /* 因为输入点是逆时针顺序，需要翻转 */
    std::reverse(new_pts.x.begin(), new_pts.x.end());
    std::reverse(new_pts.y.begin(), new_pts.y.end());
  } else if (area == 0) {
    std::cout << "Area = 0" << std::endl;
    return 0;
  }

  auto new_region = std::make_shared<ConvexPolygon>();
  if (new_region->setVertices(new_pts)) {
    new_region->region_name_ = region_name;
    regions_.push_back(new_region);
    return 0;
  }

  std::vector<std::vector<int>> convex_idxes;
  if (!partitionIntoConvexPolygons(new_pts, convex_idxes)) {
    std::cout << "partitionIntoConvexPolygons Bug (1)." << std::endl;
    return -1;
  }

  for (size_t i = 0; i < convex_idxes.size(); i++) {
    PointsInfo new_sub_pts;
    new_sub_pts.x.resize(convex_idxes[i].size());
    new_sub_pts.y.resize(convex_idxes[i].size());
    for (size_t j = 0; j < convex_idxes[i].size(); j++) {
      new_sub_pts.x[j] = new_pts.x[convex_idxes[i][j]];
      new_sub_pts.y[j] = new_pts.y[convex_idxes[i][j]];
    }
    auto new_region = std::make_shared<ConvexPolygon>();
    if (!new_region->setVertices(new_sub_pts)) {
      std::cout << "partitionIntoConvexPolygons Bug (2)." << std::endl;
      return -1;
    }
    // 对于分解后的凸多边形，添加序号后缀
    new_region->region_name_ = region_name + "_part" + std::to_string(i + 1);
    regions_.push_back(new_region);
  }

  this->print();
  return 0;
}

void IntrusionDetect::getRegion(std::vector<PointsInfo> &region_info) {
  region_info.clear();
  region_info.resize(regions_.size());

  for (size_t i = 0; i < regions_.size(); i++) {
    PointsInfo &current_region = region_info[i];
    PointsInfo &convex_pts = regions_[i]->points_;

    current_region.x = convex_pts.x;
    current_region.y.resize(convex_pts.y.size());

    // 坐标系转换
    for (size_t j = 0; j < convex_pts.y.size(); j++) {
      current_region.y[j] = -1 * convex_pts.y[j];
    }
  }
}

void IntrusionDetect::clean() { this->regions_.clear(); }

bool IntrusionDetect::isIntrusion(const ObjectBoxInfo &bbox) {
  /* 将坐标系从图像坐标系转换到欧几里得坐标系 */
  ObjectBoxInfo t_bbox = bbox;
  t_bbox.y1 *= -1;
  t_bbox.y2 *= -1;
  for (size_t i = 0; i < regions_.size(); i++) {
    bool separating = false;
    for (size_t j = 0; j < regions_[i]->normal_points_.x.size(); j++) {
      Vertex o;
      o.x = regions_[i]->normal_points_.x[j];
      o.y = regions_[i]->normal_points_.y[j];
      if (isSeparatingAxis(o, regions_[i]->points_, t_bbox)) {
        separating = true;
        break;
      }
    }
    if (separating) {
      continue;
    }
    for (size_t j = 0; j < base_points_.x.size(); j++) {
      Vertex o;
      o.x = base_points_.x[j];
      o.y = base_points_.y[j];
      if (isSeparatingAxis(o, regions_[i]->points_, t_bbox)) {
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

bool IntrusionDetect::isSeparatingAxis(const Vertex &axis,
                                       const PointsInfo &region_pts,
                                       const ObjectBoxInfo &bbox) {
  float min_1 = std::numeric_limits<float>::max();
  float max_1 = -std::numeric_limits<float>::max();
  float min_2 = std::numeric_limits<float>::max();
  float max_2 = -std::numeric_limits<float>::max();
  float proj;
  for (size_t i = 0; i < region_pts.x.size(); i++) {
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
    return true;
  }
}

bool IntrusionDetect::isPointInTriangle(const Vertex &o, const Vertex &v1,
                                        const Vertex &v2, const Vertex &v3) {
  /* o1 = 叉积(v1-o, v2-o)
   * o2 = 叉积(v2-o, v3-o)
   * o3 = 叉积(v3-o, v1-o)
   */
  float o1 = (v1.x - o.x) * (v2.y - o.y) - (v2.x - o.x) * (v1.y - o.y);
  float o2 = (v2.x - o.x) * (v3.y - o.y) - (v3.x - o.x) * (v2.y - o.y);
  float o3 = (v3.x - o.x) * (v1.y - o.y) - (v1.x - o.x) * (v3.y - o.y);
  bool has_pos = (o1 > 0) || (o2 > 0) || (o3 > 0);
  bool has_neg = (o1 < 0) || (o2 < 0) || (o3 < 0);
  return !(has_pos && has_neg);
}

float IntrusionDetect::getSignedGaussArea(const PointsInfo &points) {
  float area = 0;
  for (size_t i = 0; i < points.x.size(); i++) {
    size_t j = (i != points.x.size() - 1) ? (i + 1) : 0;
    area += points.x[i] * points.y[j] - points.x[j] * points.y[i];
  }
  return area / 2;
}

bool IntrusionDetect::triangulateUsingEarClipping(
    const PointsInfo &points, std::vector<std::vector<int>> &triangle_idxes) {
  std::vector<uint32_t> active_idxes;
  for (size_t i = 0; i < points.x.size(); i++) {
    active_idxes.push_back(i);
  }
  for (size_t t = 0; t < points.x.size() - 3; t++) {
    int ear_idx = -1;
    int ear_p, ear_i, ear_q;
    for (size_t i = 0; i < active_idxes.size(); i++) {
      size_t p = (i != 0) ? (i - 1) : (active_idxes.size() - 1);
      size_t q = (i != active_idxes.size() - 1) ? (i + 1) : 0;
      uint32_t idx_p = active_idxes[p];
      uint32_t idx_i = active_idxes[i];
      uint32_t idx_q = active_idxes[q];
      /* 检查凸性: 叉积(v[i]-v[p], v[q]-v[i]) */
      if (0 < (points.x[idx_i] - points.x[idx_p]) *
                      (points.y[idx_q] - points.y[idx_i]) -
                  (points.x[idx_q] - points.x[idx_i]) *
                      (points.y[idx_i] - points.y[idx_p])) {
        continue;
      }
      Vertex v1{.x = points.x[idx_p], .y = points.y[idx_p]};
      Vertex v2{.x = points.x[idx_i], .y = points.y[idx_i]};
      Vertex v3{.x = points.x[idx_q], .y = points.y[idx_q]};
      /* 检查点是否在三角形内 */
      bool is_ear = true;
      for (size_t j = 0; j < active_idxes.size(); j++) {
        if (j == p || j == i || j == q) {
          continue;
        }
        uint32_t idx_j = active_idxes[j];
        Vertex v0{.x = points.x[idx_j], .y = points.y[idx_j]};
        if (isPointInTriangle(v0, v1, v2, v3)) {
          is_ear = false;
          break;
        }
      }
      if (is_ear) {
        ear_idx = (int)i;
        ear_p = (int)idx_p;
        ear_i = (int)idx_i;
        ear_q = (int)idx_q;
        break;
      }
    }
    if (ear_idx == -1) {
      std::cout << "EAR index not found." << std::endl;
      return false;
    }
    triangle_idxes.push_back(std::vector<int>({ear_p, ear_i, ear_q}));
    active_idxes.erase(active_idxes.begin() + ear_idx);
  }

  if (active_idxes.size() != 3) {
    std::cout << "final active index size != 3." << std::endl;
    return false;
  }
  triangle_idxes.push_back(std::vector<int>(
      {(int)active_idxes[0], (int)active_idxes[1], (int)active_idxes[2]}));

  return true;
}

bool IntrusionDetect::partitionIntoConvexPolygons(
    const PointsInfo &points, std::vector<std::vector<int>> &convex_idxes) {
  std::vector<std::vector<int>> triangle_idxes;
  if (!triangulateUsingEarClipping(points, triangle_idxes)) {
    std::cout << "triangulateUsingEarClipping Bug" << std::endl;
    return false;
  }
  bool *valid_triangle = new bool[triangle_idxes.size()];
  std::fill_n(valid_triangle, triangle_idxes.size(), true);
  int a0, a1, a2, b0, b1, b2;
  for (size_t i = 0; i < triangle_idxes.size(); i++) {
    if (!valid_triangle[i]) {
      continue;
    }
    std::cout << std::endl;
    assert(triangle_idxes[i].size() == 3);
    int edge_num = 3;
    a1 = 0;
    while (a1 < edge_num) {
      a0 = (a1 > 0) ? a1 - 1 : edge_num - 1;
      a2 = (a1 + 1) % edge_num;
      bool is_diagonal = false;
      int target_idx = -1;
      for (size_t j = i + 1; j < triangle_idxes.size(); j++) {
        if (!valid_triangle[j]) {
          continue;
        }
        assert(triangle_idxes[j].size() == 3);
        for (b1 = 0; b1 < 3; b1++) {
          b0 = (b1 > 0) ? b1 - 1 : 2;
          b2 = (b1 + 1) % 3;
          if (triangle_idxes[j][b1] == triangle_idxes[i][a2] &&
              triangle_idxes[j][b2] == triangle_idxes[i][a1]) {
            is_diagonal = true;
            target_idx = (int)j;
            break;
          }
        }
        if (is_diagonal) {
          break;
        }
      }
      if (!is_diagonal) {
        a1 += 1;
        continue;
      }
      int a3 = (a2 + 1) % edge_num;
      int b3 = (b2 + 1) % 3;
      assert(b0 == b3);
      Vertex va_01{.x = points.x[triangle_idxes[i][a1]] -
                        points.x[triangle_idxes[i][a0]],
                   .y = points.y[triangle_idxes[i][a1]] -
                        points.y[triangle_idxes[i][a0]]};
      Vertex va_23{.x = points.x[triangle_idxes[i][a3]] -
                        points.x[triangle_idxes[i][a2]],
                   .y = points.y[triangle_idxes[i][a3]] -
                        points.y[triangle_idxes[i][a2]]};
      Vertex vb_01{.x = points.x[triangle_idxes[target_idx][b1]] -
                        points.x[triangle_idxes[target_idx][b0]],
                   .y = points.y[triangle_idxes[target_idx][b1]] -
                        points.y[triangle_idxes[target_idx][b0]]};
      Vertex vb_23{.x = points.x[triangle_idxes[target_idx][b3]] -
                        points.x[triangle_idxes[target_idx][b2]],
                   .y = points.y[triangle_idxes[target_idx][b3]] -
                        points.y[triangle_idxes[target_idx][b2]]};
      if ((va_01.x * vb_23.y - va_01.y * vb_23.x) < 0 &&
          (vb_01.x * va_23.y - vb_01.y * va_23.x) < 0) {
        triangle_idxes[i].insert(triangle_idxes[i].begin() + a1 + 1,
                                 triangle_idxes[target_idx][b3]);
        edge_num += 1;
        valid_triangle[target_idx] = false;
      }
      a1 += 1;
    }
  }

  for (size_t i = 0; i < triangle_idxes.size(); i++) {
    if (!valid_triangle[i]) {
      continue;
    }
    convex_idxes.push_back(triangle_idxes[i]);
  }

  delete[] valid_triangle;
  return true;
}

void IntrusionDetect::print() {
  std::cout << "Region Num: " << regions_.size() << std::endl;
  for (size_t i = 0; i < regions_.size(); i++) {
    std::cout << "[" << i << "]\n";
    regions_[i]->print();
  }
}

void ConvexPolygon::print() {
  std::cout << "区域名称: " << (region_name_.empty() ? "未命名" : region_name_)
            << std::endl;
  for (size_t i = 0; i < this->points_.x.size(); i++) {
    std::cout << "(" << std::setw(4) << this->points_.x[i] << ","
              << std::setw(4) << this->points_.y[i] << ")";
    if ((i + 1) != this->points_.x.size())
      std::cout << "  &  ";
    else
      std::cout << "\n";
  }
  for (size_t i = 0; i < this->normal_points_.x.size(); i++) {
    std::cout << "(" << std::setw(4) << this->normal_points_.x[i] << ","
              << std::setw(4) << this->normal_points_.y[i] << ")";
    if ((i + 1) != this->normal_points_.x.size())
      std::cout << "  &  ";
    else
      std::cout << "\n";
  }
}
