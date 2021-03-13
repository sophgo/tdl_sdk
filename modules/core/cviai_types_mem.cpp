#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"

#include <string.h>
// Free

void CVI_AI_FreeCpp(cvai_feature_t *feature) {
  if (feature->ptr != NULL) {
    free(feature->ptr);
    feature->ptr = NULL;
  }
  feature->size = 0;
  feature->type = TYPE_INT8;
}

void CVI_AI_FreeCpp(cvai_pts_t *pts) {
  if (pts->x != NULL) {
    free(pts->x);
    pts->x = NULL;
  }
  if (pts->y != NULL) {
    free(pts->y);
    pts->y = NULL;
  }
  pts->size = 0;
}

void CVI_AI_FreeCpp(cvai_tracker_t *tracker) {
  if (tracker->info != NULL) {
    free(tracker->info);
    tracker->info = NULL;
  }
  tracker->size = 0;
}

void CVI_AI_FreeCpp(cvai_dms_od_t *dms_od) {
  if (dms_od->info != NULL) {
    free(dms_od->info);
    dms_od->info = NULL;
  }
  dms_od->size = 0;
  dms_od->width = 0;
  dms_od->height = 0;
}

void CVI_AI_FreeCpp(cvai_dms_t *dms) {
  CVI_AI_FreeCpp(&dms->landmarks_106);
  CVI_AI_FreeCpp(&dms->landmarks_5);
  CVI_AI_FreeCpp(&dms->dms_od);
}

void CVI_AI_FreeCpp(cvai_face_info_t *face_info) {
  CVI_AI_FreeCpp(&face_info->pts);
  CVI_AI_FreeCpp(&face_info->feature);
}

void CVI_AI_FreeCpp(cvai_face_t *face) {
  if (face->info != NULL) {
    for (uint32_t i = 0; i < face->size; i++) {
      CVI_AI_FreeCpp(&face->info[i]);
    }
    free(face->info);
    face->info = NULL;
  }
  face->size = 0;
  face->width = 0;
  face->height = 0;
}

void CVI_AI_FreeCpp(cvai_object_info_t *obj_info) {
  CVI_AI_FreeCpp(&obj_info->feature);
  if (obj_info->vehicle_properity) {
    free(obj_info->vehicle_properity);
    obj_info->vehicle_properity = NULL;
  }

  if (obj_info->pedestrian_properity) {
    free(obj_info->pedestrian_properity);
    obj_info->pedestrian_properity = NULL;
  }
}

void CVI_AI_FreeCpp(cvai_object_t *obj) {
  if (obj->info != NULL) {
    for (uint32_t i = 0; i < obj->size; i++) {
      CVI_AI_FreeCpp(&obj->info[i]);
    }
    free(obj->info);
    obj->info = NULL;
  }
  obj->size = 0;
  obj->width = 0;
  obj->height = 0;
}

void CVI_AI_FreeFeature(cvai_feature_t *feature) { CVI_AI_FreeCpp(feature); }

void CVI_AI_FreePts(cvai_pts_t *pts) { CVI_AI_FreeCpp(pts); }

void CVI_AI_FreeTracker(cvai_tracker_t *tracker) { CVI_AI_FreeCpp(tracker); }

void CVI_AI_FreeFaceInfo(cvai_face_info_t *face_info) { CVI_AI_FreeCpp(face_info); }

void CVI_AI_FreeFace(cvai_face_t *face) { CVI_AI_FreeCpp(face); }

void CVI_AI_FreeObjectInfo(cvai_object_info_t *obj_info) { CVI_AI_FreeCpp(obj_info); }

void CVI_AI_FreeObject(cvai_object_t *obj) { CVI_AI_FreeCpp(obj); }

void CVI_AI_FreeDMS(cvai_dms_t *dms) { CVI_AI_FreeCpp(dms); }

// Copy

void CVI_AI_CopyInfoCpp(const cvai_face_info_t *info, cvai_face_info_t *infoNew) {
  memcpy(infoNew->name, info->name, sizeof(info->name));
  infoNew->unique_id = info->unique_id;
  infoNew->bbox = info->bbox;
  infoNew->pts.size = info->pts.size;
  if (infoNew->pts.size != 0) {
    uint32_t pts_size = infoNew->pts.size * sizeof(float);
    infoNew->pts.x = (float *)malloc(pts_size);
    infoNew->pts.y = (float *)malloc(pts_size);
    memcpy(infoNew->pts.x, info->pts.x, pts_size);
    memcpy(infoNew->pts.y, info->pts.y, pts_size);
  } else {
    infoNew->pts.x = NULL;
    infoNew->pts.y = NULL;
  }
  infoNew->feature.type = info->feature.type;
  infoNew->feature.size = info->feature.size;
  if (infoNew->feature.size != 0) {
    uint32_t feature_size = infoNew->feature.size * getFeatureTypeSize(infoNew->feature.type);
    infoNew->feature.ptr = (int8_t *)malloc(feature_size);
    memcpy(infoNew->feature.ptr, info->feature.ptr, feature_size);
  } else {
    infoNew->feature.ptr = NULL;
  }
  infoNew->emotion = info->emotion;
  infoNew->gender = info->gender;
  infoNew->race = info->race;
  infoNew->age = info->age;
  infoNew->liveness_score = info->liveness_score;
  infoNew->mask_score = info->mask_score;
  infoNew->face_quality = info->face_quality;
}

void CVI_AI_CopyInfoCpp(const cvai_object_info_t *info, cvai_object_info_t *infoNew) {
  memcpy(infoNew->name, info->name, sizeof(info->name));
  infoNew->unique_id = info->unique_id;
  infoNew->bbox = info->bbox;
  infoNew->feature.type = info->feature.type;
  infoNew->feature.size = info->feature.size;
  if (infoNew->feature.size != 0) {
    uint32_t feature_size = infoNew->feature.size * getFeatureTypeSize(infoNew->feature.type);
    infoNew->feature.ptr = (int8_t *)malloc(feature_size);
    memcpy(infoNew->feature.ptr, info->feature.ptr, feature_size);
  } else {
    infoNew->feature.ptr = NULL;
  }

  if (info->vehicle_properity) {
    infoNew->vehicle_properity = (cvai_vehicle_meta *)malloc(sizeof(cvai_vehicle_meta));
    infoNew->vehicle_properity->license_bbox = info->vehicle_properity->license_bbox;
    memcpy(infoNew->vehicle_properity->license_char, info->vehicle_properity->license_char,
           sizeof(info->vehicle_properity->license_char));
    memcpy(infoNew->vehicle_properity->license_pts.x, info->vehicle_properity->license_pts.x,
           4 * sizeof(float));
  }

  if (info->pedestrian_properity) {
    infoNew->pedestrian_properity = (cvai_pedestrian_meta *)malloc(sizeof(cvai_pedestrian_meta));
    infoNew->pedestrian_properity->fall = info->pedestrian_properity->fall;
    memcpy(infoNew->pedestrian_properity->pose_17.score, info->pedestrian_properity->pose_17.score,
           sizeof(float) * 17);
    memcpy(infoNew->pedestrian_properity->pose_17.x, info->pedestrian_properity->pose_17.x,
           sizeof(float) * 17);
    memcpy(infoNew->pedestrian_properity->pose_17.y, info->pedestrian_properity->pose_17.y,
           sizeof(float) * 17);
  }

  infoNew->classes = info->classes;
}

void CVI_AI_CopyInfoCpp(const cvai_dms_od_info_t *info, cvai_dms_od_info_t *infoNew) {
  memcpy(infoNew->name, info->name, sizeof(info->name));
  infoNew->bbox = info->bbox;
  infoNew->classes = info->classes;
}
void CVI_AI_CopyFaceInfo(const cvai_face_info_t *info, cvai_face_info_t *infoNew) {
  CVI_AI_CopyInfoCpp(info, infoNew);
}

void CVI_AI_CopyObjectInfo(const cvai_object_info_t *info, cvai_object_info_t *infoNew) {
  CVI_AI_CopyInfoCpp(info, infoNew);
}
