#include "components/snapshot/object_quality.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "framework/utils/pose_helper.hpp"
#include "framework/utils/tdl_log.hpp"
#include "mot/munkres.hpp"
#include "utils/cost_matrix_helper.hpp"
#include "utils/mot_box_helper.hpp"

#define ABS(x) ((x) >= 0 ? (x) : (-(x)))

struct FaceQualityConfig {
  // 硬阈值
  float min_area{25 * 25};
  float min_eye_ratio{0.25f};
  float min_eye_threshold{0.01};  // 对应1080p, 瞳距20
  float min_mouth_ratio{0.15f};
  // 权重
  float w_lmk{0.3f}, w_pose{0.3f}, w_size{0.2f}, w_motion{0.1f}, w_blur{0.1f};
  // 运动最大贡献
  float max_motion{2.0f};
  // blur 最大扣分
  float max_blur{0.2f};
  // 关键点越界容忍 eps
  float boundary_eps{0.01f};

  float thr_yaw{0.7f};
  float thr_pitch{0.5f};
  float thr_roll{0.65f};
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
                           float x2, float y2, float img_width,
                           const FaceQualityConfig& cfg) {
  // 眼距 & 口距占脸宽的比率
  float box_w = x2 - x1;
  float box_h = y2 - y1;
  float ex = lx[1] - lx[0], ey = ly[1] - ly[0];
  float eye_dis = std::hypot(ex, ey);
  float eye_ratio = eye_dis / box_w;
  float mx = lx[4] - lx[3], my = ly[4] - ly[3];
  float mouth_ratio = std::hypot(mx, my) / box_w;

  if (eye_dis < cfg.min_eye_threshold * img_width) {
    return 0.f;
  }

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

float computeMotionScore(float vel, float reference,
                         const FaceQualityConfig& cfg) {
  // the vel is pixel/frame
  float mv = std::max(0.f, std::min(vel * 10.0f / reference, cfg.max_motion));
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

  // 1) 姿态估计
  FacePose pose;
  float score_pose = 0;
  if (PoseHelper::predictFacePose(lx, ly, &pose) == 0) {
    if (ABS(pose.yaw) > cfg.thr_yaw || ABS(pose.pitch) > cfg.thr_pitch ||
        ABS(pose.roll) > cfg.thr_roll) {
      return 0.0f;
    }
    score_pose =
        1. - (ABS(pose.yaw) + ABS(pose.pitch) + ABS(pose.roll) * 0.5) / 3.;
  }

  // 2) 大小分数
  float score_size =
      computeSizeScore(box.x2 - box.x1, box.y2 - box.y1, img_height);

  // 3) 关键点分数
  float score_lmk = computeLandmarkScore(lx, ly, box.x1, box.y1, box.x2, box.y2,
                                         img_width, cfg);

  // 4) 运动分数（如果可用）
  float vel = 0.f;  // 从 tracker 拿到
  if (other_info.find("vel") != other_info.end()) {
    vel = other_info.at("vel");
  }
  float reference = std::hypot(box.x2 - box.x1, box.y2 - box.y1);
  float score_mov = computeMotionScore(vel, reference, cfg);

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
    final_score = score_lmk * cfg.w_lmk + score_size * cfg.w_size +
                  score_pose * cfg.w_pose + score_mov * cfg.w_motion +
                  score_blur * cfg.w_blur;
  }

  LOGI(
      "score_pose:%f, score_size: %f, score_lmk: %f, score_mov: %f, "
      "score_blur: %f, final_score: %f\n",
      score_pose, score_size, score_lmk, score_mov, score_blur, final_score);
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
        "landmarks_x.size(): %ld, landmarks_y.size(): %ld",
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

void ObjectQualityHelper::getFaceQuality(
    const std::vector<ObjectBoxInfo>& face_bbox,
    const std::vector<ObjectBoxInfo>& head_bbox,
    std::vector<float>& face_quality) {
  face_quality.clear();
  if (face_bbox.size() == 0) {
    return;

  } else if (head_bbox.size() == 0) {
    for (size_t i = 0; i < face_bbox.size(); i++) {
      face_quality.push_back(0.4);
    }

  } else {
    COST_MATRIX cost_matrix(face_bbox.size(), head_bbox.size());

    for (size_t i = 0; i < face_bbox.size(); i++) {
      for (size_t j = 0; j < head_bbox.size(); j++) {
        cost_matrix(i, j) =
            1 - MotBoxHelper::calculateIOU(face_bbox[i], head_bbox[j]);
      }
    }

    Munkres munkres_solver(&cost_matrix);
    if (munkres_solver.solve() == MUNKRES_FAILURE) {
      LOGW("MUNKRES algorithm failed.");
      for (size_t i = 0; i < face_bbox.size(); i++) {
        face_quality.push_back(0.0);
      }
      return;
    }

    for (size_t i = 0; i < face_bbox.size(); i++) {
      int head_idx = munkres_solver.m_match_result[i];
      if (head_idx != -1) {
        float iou_score = 1 - cost_matrix(i, head_idx);

        float head_width = head_bbox[head_idx].x2 - head_bbox[head_idx].x1;
        float head_w_center =
            (head_bbox[head_idx].x1 + head_bbox[head_idx].x2) / 2;
        float face_w_center = (face_bbox[i].x1 + face_bbox[i].x2) / 2;
        float pose_score =
            1 - std::abs(head_w_center - face_w_center) / head_width;

        face_quality.push_back((iou_score + pose_score) * 0.5);

      } else {
        face_quality.push_back(0.4);
      }
    }
  }
}

float ObjectQualityHelper::getFaceQuality(
    const ObjectBoxInfo& box, const std::vector<float>& landmark_xys,
    int img_width, int img_height, bool fl_model,
    const std::map<std::string, float>& other_info) {
  FaceQualityConfig cfg;

  std::vector<float> lx(5), ly(5);
  for (int i = 0; i < 5; i++) {
    lx[i] = landmark_xys[2 * i];
    ly[i] = landmark_xys[2 * i + 1];
  }

  float nose_x = lx[2];
  float left_max = MIN(lx[0], lx[3]);
  float right_max = MAX(lx[1], lx[4]);

  float width = box.x2 - box.x1;
  float height = box.y2 - box.y1;

  float l_ = nose_x - left_max;
  float r_ = right_max - nose_x;

  float eye_diff_x = lx[1] - lx[0];
  float eye_diff_y = ly[1] - ly[0];
  float eye_size = sqrt(eye_diff_x * eye_diff_x + eye_diff_y * eye_diff_y);
  if (eye_size < cfg.min_eye_threshold * img_width) {
    return 0.f;
  }

  float mouth_diff_x = lx[4] - lx[3];
  float mouth_diff_y = ly[4] - ly[3];
  float mouth_size =
      sqrt(mouth_diff_x * mouth_diff_x + mouth_diff_y * mouth_diff_y);

  float vel = 0.f;  // 从 tracker 拿到
  if (other_info.find("vel") != other_info.end()) {
    vel = other_info.at("vel");
  }
  float reference = std::hypot(box.x2 - box.x1, box.y2 - box.y1);
  float score_mov = computeMotionScore(vel, reference, cfg);

  float blurness = 0.f;  // 从脸模糊检测算法拿到
  if (other_info.find("blr") != other_info.end()) {
    blurness = other_info.at("blr");
  }

  FacePose pose;
  float score_pose = 0;
  if (PoseHelper::predictFacePose(lx, ly, &pose) == 0) {
    if (ABS(pose.yaw) > cfg.thr_yaw || ABS(pose.pitch) > cfg.thr_pitch ||
        ABS(pose.roll) > cfg.thr_roll) {
      return 0.0f;
    }
  }

  if (lx[1] > box.x2 || lx[2] > box.x2 || lx[4] > box.x2 || lx[0] < box.x1 ||
      lx[2] < box.x1 || lx[3] < box.x1) {
    return 0.0;
  } else if ((l_ + 0.01 * width) < 0 || (r_ + 0.01 * width) < 0 ||
             (eye_size / width) < 0.25 || (mouth_size / width) < 0.15) {
    return 0.0;
  } else if ((ly[0] < box.y1 || ly[1] < box.y1 || ly[3] > box.y2 ||
              ly[4] > box.y2)) {
    return 0.0;
  } else if (width * height < (25 * 25)) {
    return 0.0;
  } else {
    float face_size = ((box.y2 - box.y1) + (box.x2 - box.x1)) / 2;
    float size_score = 0;
    float pose_score =
        1. - (ABS(pose.yaw) + ABS(pose.pitch) + ABS(pose.roll) * 0.5) / 3.;
    // printf("pose_score_angle: %f, pose->yaw: %f, pose->pitch: %f, pose->roll:
    // %f\n", pose_score, pose->yaw, pose->pitch, pose->roll);

    float area_score;
    float wpose = 0.8;
    float wsize = 0.2;

    float h_ratio = face_size / (float)img_height;

    if (h_ratio < 0.06) {  // 64/1080
      wpose = 0.4;
      area_score = 0;
    } else if (h_ratio < 0.0685)  // 74/1080
    {
      wpose = 0.6;
      // area_score = log(face_size/(float)img_h)/log(4.0);
      area_score = log(h_ratio * 20.0) / log(4.0);
      if (pose_score > 0.8) {
        pose_score = 0.8;
      }
      size_score = 0.75;

    } else {
      area_score = 0.23 + (2.0 - 1.0 / (h_ratio * 4.38 + 0.2)) /
                              5.0;  // 0.23 ~= log(0.0685*20.0)/log(4.0)
      size_score = eye_size / (box.x2 - box.x1);
      size_score += mouth_size / (box.x2 - box.x1);
    }
    if (fl_model && h_ratio > 0.06) {
      wpose = 0.8;
    }

    pose_score =
        pose_score * wpose + wsize * size_score + area_score - blurness * 0.2;

    if (box.x1 < 0.5 * width || img_width - box.x2 < 0.5 * width ||
        box.y1 < 0.5 * height || img_height - box.y2 < 0.5 * height) {
      pose_score -= 0.2;
    }
    return pose_score;
  }
}
