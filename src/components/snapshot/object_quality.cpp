#include "components/snapshot/object_quality.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "framework/utils/pose_helper.hpp"
#include "framework/utils/tdl_log.hpp"
struct FaceQualityConfig {
  // 硬阈值
  float min_area{25 * 25};
  float min_eye_ratio{0.25f};
  float min_mouth_ratio{0.15f};
  // 权重
  float w_pose{0.3f}, w_size{0.5f}, w_motion{0.1f}, w_blur{0.1f};
  // 运动最大贡献
  float max_motion{0.2f};
  // blur 最大扣分
  float max_blur{0.2f};
  // 关键点越界容忍 eps
  float boundary_eps{0.01f};
};

float computePoseScore(const FacePose& pose) {
  // 归一化 yaw/pitch/roll 到 [0,1]
  float yaw = std::min(std::abs(pose.yaw) / 90.f, 1.f);
  float pitch = std::min(std::abs(pose.pitch) / 90.f, 1.f);
  float roll = std::min(std::abs(pose.roll) / 90.f, 1.f);
  // 简单取平均并反转
  return 1.f - (yaw + pitch + roll) / 3.f;
}

float computeSizeScore(float box_w, float box_h, int img_h) {
  // 用脸高 / 图高来打分，范围 [0,1]
  float face_size = std::min(box_w, box_h) - 20;
  if (face_size > 128) {
    face_size = 128;
  }
  float score = face_size / 128;
  return score;
}

float computeLandmarkScore(const std::vector<float>& lx,
                           const std::vector<float>& ly, float x1, float y1,
                           float x2, float y2, const FaceQualityConfig& cfg) {
  // 眼距 & 口距占脸宽的比率
  float box_w = x2 - x1;
  float box_h = y2 - y1;
  float ex = lx[1] - lx[0], ey = ly[1] - ly[0];
  float eye_ratio = std::hypot(ex, ey) / box_w;
  float mx = lx[4] - lx[3], my = ly[4] - ly[3];
  float mouth_ratio = std::hypot(mx, my) / box_w;

  // 边界检测
  float minx = *std::min_element(lx.begin(), lx.end());
  float maxx = *std::max_element(lx.begin(), lx.end());
  float miny = *std::min_element(ly.begin(), ly.end());
  float maxy = *std::max_element(ly.begin(), ly.end());
  float thresh_x = cfg.boundary_eps * box_w;
  float thresh_y = cfg.boundary_eps * box_h;
  if (minx < x1 - thresh_x || maxx > x2 + thresh_x || miny < y1 - thresh_y ||
      maxy > y2 + thresh_y) {
    LOGI(
        "landmark out of boundary, minx: %f, maxx: %f, miny: %f, maxy: %f,x1: "
        "%f,y1: %f,x2: %f,y2: %f,thresh_x: %f,thresh_y: %f",
        minx, maxx, miny, maxy, x1, y1, x2, y2, thresh_x, thresh_y);
    return 0.f;
  }

  // nose should be between eyes and mouth
  if (lx[0] >= lx[2] || lx[1] <= lx[2] || lx[3] >= lx[2] || lx[4] <= lx[2]) {
    LOGI(
        "landmark out of boundary, lx[0]: %f, lx[1]: %f, lx[2]: %f, lx[3]: %f, "
        "lx[4]: %f",
        lx[0], lx[1], lx[2], lx[3], lx[4]);
    return 0.f;
  }

  // 眼距、口距分数线性映射到 [0,1]
  float score_eye =
      std::max(0.f, std::min(1.f, (eye_ratio - cfg.min_eye_ratio) /
                                      (0.5f - cfg.min_eye_ratio)));
  float score_mouth =
      std::max(0.f, std::min(1.f, (mouth_ratio - cfg.min_mouth_ratio) /
                                      (0.5f - cfg.min_mouth_ratio)));
  // TODO(fuquan.ke):add boundary score
  return std::min(score_eye, score_mouth);
}

float computeMotionScore(float vel, const FaceQualityConfig& cfg) {
  // the vel is pixel/frame
  float mv = std::max(0.f, std::min(vel * 0.04f, cfg.max_motion));
  // 运动越小越好，反转为分数
  return 1.f - (mv / cfg.max_motion);
}

float computeBlurScore(float blurness, const FaceQualityConfig& cfg) {
  // 假设 blurness ∈ [0,1]
  return 1.f - std::max(0.f, std::min(blurness, cfg.max_blur)) / cfg.max_blur;
}

float ObjectQualityHelper::getFaceQuality(
    const ObjectBoxInfo& box, const std::vector<float>& landmark_xys,
    int img_width, int img_height,
    const std::map<std::string, float>& other_info) {
  FaceQualityConfig cfg;  // 你可以把它做成成员变量，在外部配置

  // 必要检查：面积
  float area = (box.x2 - box.x1) * (box.y2 - box.y1);
  if (area < cfg.min_area) return 0.f;

  // 拆分 landmarks
  std::vector<float> lx(5), ly(5);
  for (int i = 0; i < 5; i++) {
    lx[i] = landmark_xys[2 * i];
    ly[i] = landmark_xys[2 * i + 1];
  }

  // // 1) 姿态估计
  // FacePose pose;
  // float score_pose = 0;
  // if (PoseHelper::predictFacePose(lx, ly, img_width, img_height, &pose) == 0)
  // {
  //   score_pose = computePoseScore(pose);
  // }

  // 2) 大小分数
  float score_size =
      computeSizeScore(box.x2 - box.x1, box.y2 - box.y1, img_height);

  // 3) 关键点分数
  float score_lmk =
      computeLandmarkScore(lx, ly, box.x1, box.y1, box.x2, box.y2, cfg);

  // 4) 运动分数（如果可用）
  float vel = 0.f;  // 从 tracker 拿到
  if (other_info.find("vel") != other_info.end()) {
    vel = other_info.at("vel");
  }
  float score_mov = computeMotionScore(vel, cfg);

  // 5) 模糊分数（如果可用）
  float blr = 0.f;  // 从脸模糊检测算法拿到
  if (other_info.find("blr") != other_info.end()) {
    blr = other_info.at("blr");
  }
  float score_blur = computeBlurScore(blr, cfg);

  // 6) 加权求和
  float final_score = 0.f;
  if (score_lmk == 0) {
    final_score = 0;
  } else {
    final_score = score_lmk * cfg.w_pose + score_size * cfg.w_size +
                  score_mov * cfg.w_motion + score_blur * cfg.w_blur;
  }

  LOGI(
      "score_size: %f, score_lmk: %f, score_mov: %f, "
      "score_blur: %f, final_score: %f",
      score_size, score_lmk, score_mov, score_blur, final_score);
  // 7) clamp 并返回
  return std::max(0.f, std::min(1.f, final_score));
}

float ObjectQualityHelper::getFaceQuality(
    const ObjectBoxLandmarkInfo& box_landmark, const int img_width,
    const int img_height, const std::map<std::string, float>& other_info) {
  ObjectBoxInfo box_info(box_landmark.class_id, box_landmark.score,
                         box_landmark.x1, box_landmark.y1, box_landmark.x2,
                         box_landmark.y2);
  std::vector<float> landmark_xys;
  if (box_landmark.landmarks_x.size() != 5 ||
      box_landmark.landmarks_y.size() != 5) {
    LOGE(
        "ObjectQualityHelper getFaceQuality landmark_xys size error, "
        "landmarks_x.size(): %d, landmarks_y.size(): %d",
        box_landmark.landmarks_x.size(), box_landmark.landmarks_y.size());
    assert(false);
  }
  for (int i = 0; i < 5; i++) {
    landmark_xys.push_back(box_landmark.landmarks_x[i]);
    landmark_xys.push_back(box_landmark.landmarks_y[i]);
  }
  return getFaceQuality(box_info, landmark_xys, img_width, img_height,
                        other_info);
}
