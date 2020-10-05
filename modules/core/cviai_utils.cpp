#include "core/cviai_utils.h"

#include "core/cviai_core.h"
#include "core/cviai_types_mem_internal.h"
#include "utils/core_utils.hpp"

#include <string.h>

template <typename FACE>
inline void __attribute__((always_inline)) CVI_AI_InfoCopyToNew(
    const FACE *info, FACE *infoNew,
    typename std::enable_if<std::is_same<FACE, cvai_face_info_t>::value>::type * = 0) {
  CVI_AI_FaceInfoCopyToNew(info, infoNew);
}

template <typename OBJ>
inline void __attribute__((always_inline)) CVI_AI_InfoCopyToNew(
    const OBJ *info, OBJ *infoNew,
    typename std::enable_if<std::is_same<OBJ, cvai_object_info_t>::value>::type * = 0) {
  CVI_AI_ObjInfoCopyToNew(info, infoNew);
}

template <typename T, typename U>
inline int CVI_AI_NMS(const T *input, T *nms, const float threshold, const char method) {
  if (method != 'u' && method != 'm') {
    LOGE("Unsupported NMS method. Only supports u or m");
    return CVI_FAILURE;
  }
  std::vector<U> bboxes;
  std::vector<U> bboxes_nms;
  for (uint32_t i = 0; i < input->size; i++) {
    bboxes.push_back(input->info[i]);
  }
  cviai::NonMaximumSuppression(bboxes, bboxes_nms, threshold, method);
  CVI_AI_Free(nms);
  nms->size = bboxes.size();
  nms->width = input->width;
  nms->height = input->height;
  nms->info = (U *)malloc(nms->size * sizeof(U));
  for (unsigned int i = 0; i < nms->size; i++) {
    CVI_AI_InfoCopyToNew<U>(&bboxes_nms[i], &nms->info[i]);
  }
  return CVI_SUCCESS;
}

int CVI_AI_FaceNMS(const cvai_face_t *face, cvai_face_t *faceNMS, const float threshold,
                   const char method) {
  return CVI_AI_NMS<cvai_face_t, cvai_face_info_t>(face, faceNMS, threshold, method);
}

int CVI_AI_ObjectNMS(const cvai_object_t *obj, cvai_object_t *objNMS, const float threshold,
                     const char method) {
  return CVI_AI_NMS<cvai_object_t, cvai_object_info_t>(obj, objNMS, threshold, method);
}
