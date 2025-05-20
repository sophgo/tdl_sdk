/** Intrusion Detection base_points_ on 2-dimensional convex polygons
 *    collision detection (using the Separating Axis Theorem)
 */
#pragma once
#include <memory>
#include <vector>
#include "common/model_output_types.hpp"

typedef struct {
  float x;
  float y;
} Vertex;

typedef struct {
  std::vector<float> x;  // x坐标数组
  std::vector<float> y;  // y坐标数组
} PointsInfo;

class ConvexPolygon {
 public:
  ConvexPolygon() = default;
  ~ConvexPolygon();
  void print();
  bool setVertices(const PointsInfo &points);

  /* member data */
  PointsInfo points_;
  PointsInfo normal_points_;
  std::string region_name_;

 private:
  bool isConvex(const std::vector<float> &edges_x,
                const std::vector<float> &edges_y);
};

class IntrusionDetect {
 public:
  IntrusionDetect();
  ~IntrusionDetect();
  int addRegion(const PointsInfo &points, const std::string &region_name = "");
  void getRegion(std::vector<PointsInfo> &region_info);
  bool isIntrusion(const ObjectBoxInfo &bbox);
  void clean();
  void print();

 private:
  bool isSeparatingAxis(const Vertex &axis,
                        const PointsInfo &region_pts,
                        const ObjectBoxInfo &bbox);
  bool isPointInTriangle(const Vertex &o,
                         const Vertex &v1,
                         const Vertex &v2,
                         const Vertex &v3);
  float getSignedGaussArea(const PointsInfo &points);
  bool triangulateUsingEarClipping(
      const PointsInfo &points, std::vector<std::vector<int>> &triangle_idxes);
  bool partitionIntoConvexPolygons(const PointsInfo &points,
                                   std::vector<std::vector<int>> &convex_idxes);
  std::vector<std::shared_ptr<ConvexPolygon>> regions_;
  PointsInfo base_points_;
};
