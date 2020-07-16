#include "cviai_types_free.h"

void CVI_AI_FreeCpp(cvi_feature_t *feature) { delete[] feature->ptr; }

void CVI_AI_FreeCpp(cvi_pts_t *pts) {
  delete[] pts->x;
  delete[] pts->y;
}

void CVI_AI_FreeCpp(cvi_face_info_t *face_info) {
  CVI_AI_FreePts(&face_info->face_pts);
  CVI_AI_FreeFeature(&face_info->face_feature);
}

void CVI_AI_FreeCpp(cvi_face_t *face) { CVI_AI_FreeFaceInfo(face->face_info); }

void CVI_AI_FreeCpp(cvi_object_info_t *face_info) {
  // Do nothing
}

void CVI_AI_FreeCpp(cvi_object_t *face) {
  // Do nothing
}

void CVI_AI_FreeFeature(cvi_feature_t *feature) { CVI_AI_FreeCpp(feature); }

void CVI_AI_FreePts(cvi_pts_t *pts) { CVI_AI_FreeCpp(pts); }

void CVI_AI_FreeFaceInfo(cvi_face_info_t *face_info) {
  CVI_AI_FreeCpp(&face_info->face_pts);
  CVI_AI_FreeCpp(&face_info->face_feature);
}

void CVI_AI_FreeFace(cvi_face_t *face) { CVI_AI_FreeCpp(face->face_info); }

void CVI_AI_FreeObjectInfo(cvi_object_info_t *obj_info) { CVI_AI_FreeCpp(obj_info); }

void CVI_AI_FreeObject(cvi_object_t *obj) { CVI_AI_FreeCpp(obj); }
