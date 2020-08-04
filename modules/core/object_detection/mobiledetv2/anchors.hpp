#pragma once
#include <cstdint>
#include <initializer_list>
#include <map>
#include <utility>
#include <vector>

namespace cviai {

struct AnchorConfig {
  int stride;
  float octave_scale;
  std::pair<float, float> aspect;
  AnchorConfig(int stride, float octave_scale, std::pair<float, float> aspect);
};

struct AnchorBox {
  float x;
  float y;
  float w;
  float h;
  AnchorBox(float _x, float _y, float _w, float _h) : x(_x), y(_y), w(_w), h(_h) {}
};

class RetinaNetAnchorGenerator {
 public:
  // RetinaNet Anchors class

  RetinaNetAnchorGenerator(int min_level, int max_level, int num_scales,
                           const std::vector<std::pair<float, float>> &aspect_ratios,
                           float anchor_scale, int image_size);

  /**
   * @brief   Create anchor configurations
   * @note
   **/
  void _generate_configs();

  /**
   * @brief   Create anchor boxes
   * @note
   **/
  void _generate_boxes();

  const std::vector<std::vector<AnchorBox>> &get_anchor_boxes() const;

 private:
  int min_level;
  int max_level;
  int num_scales;
  std::vector<std::pair<float, float>> aspect_ratios;
  float anchor_scale;
  int image_size;
  std::vector<std::vector<AnchorBox>> anchor_bboxes;
  std::map<int, std::vector<AnchorConfig>> anchor_configs;
};
}  // namespace cviai