#ifndef _CVIAI_H_
#define _CVIAI_H_
#include "core/cvi_core_types.h"
#include "face/cvi_face_helper.h"
#include "face/cvi_face_types.h"
#include "object/cvi_object_types.h"

#include "cviai_types_free.h"

// clang-format off
#define CVI_AI_Free(X)                               \
  _Generic((X),                                      \
           cvi_feature_t: CVI_AI_FreeFeature,        \
           cvi_pts_t: CVI_AI_FreePts,                \
           cvi_face_info_t: CVI_AI_FreeFaceInfo,     \
           cvi_face_t: CVI_AI_FreeFace,              \
           cvi_object_info_t: CVI_AI_FreeObjectInfo, \
           cvi_object_t: CVI_AI_FreeObject)(X)
// clang-format on
#endif