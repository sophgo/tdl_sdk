#include "core/cviai_types_free.h"

#include <string.h>

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

void CVI_AI_FreeCpp(cvai_face_info_t *face_info) {
  CVI_AI_FreeCpp(&face_info->face_pts);
  CVI_AI_FreeCpp(&face_info->face_feature);
}

void CVI_AI_FreeCpp(cvai_face_t *face) {
  if (face->info != NULL) {
    CVI_AI_FreeCpp(face->info);
    free(face->info);
    face->info = NULL;
  }
  face->size = 0;
  face->width = 0;
  face->height = 0;
}

void CVI_AI_FreeCpp(cvai_object_info_t *obj_info) {
  memset(obj_info, 0, sizeof(cvai_object_info_t));
}

void CVI_AI_FreeCpp(cvai_object_t *obj) {
  if (obj->info != NULL) {
    free(obj->info);
    obj->info = NULL;
  }
  obj->size = 0;
  obj->width = 0;
  obj->height = 0;
}

void CVI_AI_FreeFeature(cvai_feature_t *feature) { CVI_AI_FreeCpp(feature); }

void CVI_AI_FreePts(cvai_pts_t *pts) { CVI_AI_FreeCpp(pts); }

void CVI_AI_FreeFaceInfo(cvai_face_info_t *face_info) { CVI_AI_FreeCpp(face_info); }

void CVI_AI_FreeFace(cvai_face_t *face) { CVI_AI_FreeCpp(face); }

void CVI_AI_FreeObjectInfo(cvai_object_info_t *obj_info) { CVI_AI_FreeCpp(obj_info); }

void CVI_AI_FreeObject(cvai_object_t *obj) { CVI_AI_FreeCpp(obj); }
