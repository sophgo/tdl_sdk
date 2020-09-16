#ifndef _CVIAI_TYPES_MEM_INTERNAL_H_
#define _CVIAI_TYPES_MEM_INTERNAL_H_

inline void CVI_AI_MemAlloc(const uint32_t unit_len, const uint32_t size, const feature_type_e type,
                            cvai_feature_t *feature) {
  if (feature->size != size || feature->type != type) {
    free(feature->ptr);
    feature->ptr = (int8_t *)malloc(unit_len * size);
    feature->size = size;
    feature->type = type;
  }
}

inline void CVI_AI_MemAlloc(const uint32_t size, cvai_pts_t *pts) {
  if (pts->size != size) {
    free(pts->x);
    free(pts->y);
    pts->x = (float *)malloc(sizeof(float) * size);
    pts->y = (float *)malloc(sizeof(float) * size);
    pts->size = size;
  }
}

#endif  // End of _CVIAI_TYPES_MEM_INTERNAL_H_