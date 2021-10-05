#include "regression_utils.hpp"
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
}

}  // namespace unitest
}  // namespace cviai