#include "regression_utils.hpp"
#include "core/utils/vpss_helper.h"
namespace cviai {
namespace unitest {

static const float STD_FACE_LANDMARK_X[5] = {38.29459953, 73.53179932, 56.02519989, 41.54930115,
                                             70.72990036};
static const float STD_FACE_LANDMARK_Y[5] = {51.69630051, 51.50139999, 71.73660278, 92.3655014,
                                             92.20410156};

void init_face_meta(cvai_face_t *meta, uint32_t size) {
  memset(meta, 0, sizeof(cvai_face_t));

  meta->size = size;
  meta->height = 112;
  meta->width = 112;
  const int pts_num = 5;
  meta->info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * meta->size);
  for (uint32_t i = 0; i < meta->size; ++i) {
    meta->info[i].bbox.x1 = 0;
    meta->info[i].bbox.x2 = 111;
    meta->info[i].bbox.y1 = 0;
    meta->info[i].bbox.y2 = 111;

    meta->info[i].name[0] = '\0';
    meta->info[i].emotion = EMOTION_UNKNOWN;
    meta->info[i].gender = GENDER_UNKNOWN;
    meta->info[i].race = RACE_UNKNOWN;
    meta->info[i].age = -1;
    meta->info[i].liveness_score = -1;
    meta->info[i].mask_score = -1.0;
    meta->info[i].hardhat_score = -1;
    meta->info[i].face_quality = -1.0;
    meta->info[i].head_pose.yaw = 0;
    meta->info[i].head_pose.pitch = 0;
    meta->info[i].head_pose.roll = 0;
    memset(&meta->info[i].head_pose.facialUnitNormalVector, 0, sizeof(float) * 3);

    memset(&meta->info[i].feature, 0, sizeof(cvai_feature_t));
    if (pts_num > 0) {
      meta->info[i].pts.x = (float *)malloc(sizeof(float) * pts_num);
      meta->info[i].pts.y = (float *)malloc(sizeof(float) * pts_num);
      meta->info[i].pts.size = pts_num;
      for (uint32_t j = 0; j < meta->info[i].pts.size; ++j) {
        meta->info[i].pts.x[j] = STD_FACE_LANDMARK_X[j];
        meta->info[i].pts.y[j] = STD_FACE_LANDMARK_Y[j];
      }
    }
  }
  meta->dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));
  meta->dms->reye_score = 0;
  meta->dms->leye_score = 0;
  meta->dms->yawn_score = 0;
  meta->dms->phone_score = 0;
  meta->dms->smoke_score = 0;
  meta->dms->landmarks_106.size = 106;
  meta->dms->landmarks_5.size = 5;
  meta->dms->landmarks_5.x = (float *)malloc(sizeof(float) * meta->dms->landmarks_5.size);
  meta->dms->landmarks_5.y = (float *)malloc(sizeof(float) * meta->dms->landmarks_5.size);
  for (uint32_t j = 0; j < meta->dms->landmarks_5.size; ++j) {
    meta->dms->landmarks_5.x[0] = 0.0;
    meta->dms->landmarks_5.y[0] = 0.0;
  }
  meta->dms->landmarks_106.x = (float *)malloc(sizeof(float) * meta->dms->landmarks_106.size);
  meta->dms->landmarks_106.y = (float *)malloc(sizeof(float) * meta->dms->landmarks_106.size);
  for (uint32_t j = 0; j < meta->dms->landmarks_106.size; ++j) {
    meta->dms->landmarks_106.x[0] = 0.0;
    meta->dms->landmarks_106.y[0] = 0.0;
  }

  meta->dms->head_pose.yaw = 0;
  meta->dms->head_pose.pitch = 0;
  meta->dms->head_pose.roll = 0;
  meta->dms->dms_od.info = NULL;
  meta->dms->dms_od.size = 0;
}

void init_obj_meta(cvai_object_t *meta, uint32_t size, uint32_t height, uint32_t width,
                   int class_id) {
  memset(meta, 0, sizeof(cvai_object_t));
  meta->size = size;
  meta->height = height;
  meta->width = width;
  meta->info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * meta->size);

  for (uint32_t i = 0; i < meta->size; ++i) {
    meta->info[i].bbox.x1 = 0;
    meta->info[i].bbox.y1 = 0;
    meta->info[i].bbox.x2 = width - 1;
    meta->info[i].bbox.y2 = height - 1;
    meta->info[i].bbox.score = 1.0;
    meta->info[i].classes = class_id;
    meta->info[i].feature.size = 0;
    meta->info[i].feature.ptr = NULL;
    meta->info[i].pedestrian_properity = NULL;
    meta->info[i].vehicle_properity = NULL;
  }
}

void init_vehicle_meta(cvai_object_t *meta) {
  if (meta->info == NULL || meta->height == 0 || meta->width == 0) {
    printf("[WARNING] Please init obj meta first.\n");
    return;
  }
  for (uint32_t i = 0; i < meta->size; ++i) {
    meta->info[i].vehicle_properity = (cvai_vehicle_meta *)malloc(sizeof(cvai_vehicle_meta));
    meta->info[i].vehicle_properity->license_pts.x[0] = 0.0;
    meta->info[i].vehicle_properity->license_pts.x[1] = (float)meta->width - 1.;
    meta->info[i].vehicle_properity->license_pts.x[2] = (float)meta->width - 1.;
    meta->info[i].vehicle_properity->license_pts.x[3] = 0.0;
    meta->info[i].vehicle_properity->license_pts.y[0] = 0.0;
    meta->info[i].vehicle_properity->license_pts.y[1] = 0.0;
    meta->info[i].vehicle_properity->license_pts.y[2] = (float)meta->height - 1.;
    meta->info[i].vehicle_properity->license_pts.y[3] = (float)meta->height - 1.;
  }
}

float iou(cvai_bbox_t &bbox1, cvai_bbox_t &bbox2) {
  float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
  float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
  float inter_x1 = MAX2(bbox1.x1, bbox2.x1);
  float inter_y1 = MAX2(bbox1.y1, bbox2.y1);
  float inter_x2 = MIN2(bbox1.x2, bbox2.x2);
  float inter_y2 = MIN2(bbox1.y2, bbox2.y2);
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  return area_inter / (area1 + area2 - area_inter);
}

}  // namespace unitest
}  // namespace cviai
