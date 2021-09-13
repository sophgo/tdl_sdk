/** Intrusion Detection base on 2-dimensional convex polygons
 *    collision detection (using the Separating Axis Theorem)
 */
#pragma once
#include <memory>
#include <vector>
#include "core/core/cvai_core_types.h"

namespace cviai {
namespace service {

typedef struct {
  float x;
  float y;
} vertex_t;

class ConvexPolygon {
 public:
  ConvexPolygon() = default;
  ~ConvexPolygon();
  void show();
  bool set_vertices(const cvai_pts_t &pts);

  /* member data */
  cvai_pts_t vertices;
  cvai_pts_t orthogonals;

 private:
  bool is_convex(float *edges_x, float *edges_y, uint32_t size);
};

class IntrusionDetect {
 public:
  IntrusionDetect();
  ~IntrusionDetect();
  int setRegion(const cvai_pts_t &pts);
  void getRegion(cvai_pts_t ***region_info, uint32_t *size);
  bool run(const cvai_bbox_t &bbox);
  void show();

 private:
  bool is_separating_axis(const vertex_t &axis, const cvai_pts_t &region_pts,
                          const cvai_bbox_t &bbox);
  bool is_point_in_triangle(const vertex_t &o, const vertex_t &v1, const vertex_t &v2,
                            const vertex_t &v3);
  float get_SignedGaussArea(const cvai_pts_t &pts);
  bool Triangulate_EC(const cvai_pts_t &pts, std::vector<std::vector<int>> &triangle_idxes);
  bool ConvexPartition_HM(const cvai_pts_t &pts, std::vector<std::vector<int>> &convex_idxes);

  /* member data */
  std::vector<std::shared_ptr<ConvexPolygon>> regions;
  cvai_pts_t base;
};

}  // namespace service
}  // namespace cviai