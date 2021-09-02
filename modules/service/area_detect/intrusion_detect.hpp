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
  ConvexPolygon(const cvai_pts_t &pts);
  ~ConvexPolygon();
  void show();
  cvai_pts_t vertices;
  cvai_pts_t orthogonals;
};

class IntrusionDetect {
 public:
  IntrusionDetect();
  ~IntrusionDetect();
  int setRegion(const cvai_pts_t &pts);
  bool run(const cvai_bbox_t &bbox);
  void show();

 private:
  bool is_separating_axis(const vertex_t &axis, const cvai_pts_t &region_pts,
                          const cvai_bbox_t &bbox);
  std::vector<std::shared_ptr<ConvexPolygon>> regions;
  cvai_pts_t base;
};

}  // namespace service
}  // namespace cviai