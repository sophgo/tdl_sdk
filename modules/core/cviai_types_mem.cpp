#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"

#include <cvi_sys.h>
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
  if (face->info) {
    for (uint32_t i = 0; i < face->size; i++) {
      CVI_AI_FreeCpp(&face->info[i]);
    }
    free(face->info);
    face->info = NULL;
  }
  face->size = 0;
  face->width = 0;
  face->height = 0;

  if (face->dms) {
    CVI_AI_FreeCpp(face->dms);
    face->dms = NULL;
  }
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

void CVI_AI_FreeCpp(cvai_image_t *image) {
  if (image->pix[0] != NULL) {
    free(image->pix[0]);
  }
  for (int i = 0; i < 3; i++) {
    image->pix[i] = NULL;
    image->stride[i] = 0;
    image->length[i] = 0;
  }
  image->height = 0;
  image->width = 0;
}

void CVI_AI_FreeCpp(cvai_handpose21_meta_t *handpose) {
  handpose->bbox_x = 0;
  handpose->bbox_y = 0;
  handpose->bbox_w = 0;
  handpose->bbox_h = 0;
  handpose->score = 0;
  handpose->label = -1;
  for (int i = 0; i < 21; i++) {
    handpose->x[i] = 0;
    handpose->xn[i] = 0;
    handpose->y[i] = 0;
    handpose->yn[i] = 0;
  }
  // free(handpose->x);
  // free(handpose->xn);
  // free(handpose->y);
  // free(handpose->yn);
}

void CVI_AI_FreeCpp(cvai_handpose21_meta_ts *handposes) {
  if (handposes->info != NULL) {
    for (uint32_t i = 0; i < handposes->size; i++) {
      CVI_AI_FreeCpp(&handposes->info[i]);
    }
    free(handposes->info);
    handposes->info = NULL;
  }
  handposes->size = 0;
  handposes->width = 0;
  handposes->height = 0;
}

void CVI_AI_FreeFeature(cvai_feature_t *feature) { CVI_AI_FreeCpp(feature); }

void CVI_AI_FreePts(cvai_pts_t *pts) { CVI_AI_FreeCpp(pts); }

void CVI_AI_FreeTracker(cvai_tracker_t *tracker) { CVI_AI_FreeCpp(tracker); }

void CVI_AI_FreeFaceInfo(cvai_face_info_t *face_info) { CVI_AI_FreeCpp(face_info); }

void CVI_AI_FreeFace(cvai_face_t *face) { CVI_AI_FreeCpp(face); }

void CVI_AI_FreeObjectInfo(cvai_object_info_t *obj_info) { CVI_AI_FreeCpp(obj_info); }

void CVI_AI_FreeObject(cvai_object_t *obj) { CVI_AI_FreeCpp(obj); }

void CVI_AI_FreeImage(cvai_image_t *image) { CVI_AI_FreeCpp(image); }

void CVI_AI_FreeDMS(cvai_dms_t *dms) { CVI_AI_FreeCpp(dms); }

void CVI_AI_FreeHandPoses(cvai_handpose21_meta_ts *handposes) { CVI_AI_FreeCpp(handposes); }
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
  infoNew->hardhat_score = info->hardhat_score;
  infoNew->mask_score = info->mask_score;
  infoNew->face_quality = info->face_quality;
  infoNew->head_pose = info->head_pose;
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

void CVI_AI_CopyFaceMeta(const cvai_face_t *src, cvai_face_t *dest) {
  CVI_AI_FreeCpp(dest);
  memset(dest, 0, sizeof(cvai_face_t));
  if (src->size > 0) {
    dest->size = src->size;
    dest->width = src->width;
    dest->height = src->height;
    dest->rescale_type = src->rescale_type;
    if (src->info) {
      dest->info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * src->size);
      memset(dest->info, 0, sizeof(cvai_face_info_t) * src->size);
      for (uint32_t fid = 0; fid < src->size; fid++) {
        CVI_AI_CopyFaceInfo(&src->info[fid], &dest->info[fid]);
      }
    }

    if (src->dms) {
      dest->dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));
      memcpy(dest->dms, src->dms, sizeof(cvai_dms_t));
      cvai_dms_od_info_t *src_dms_od_info = src->dms->dms_od.info;
      if (src_dms_od_info) {
        dest->dms->dms_od.info = (cvai_dms_od_info_t *)malloc(sizeof(cvai_dms_od_info_t));
        memcpy(dest->dms->dms_od.info, src_dms_od_info, sizeof(cvai_dms_od_info_t));
      }
    }
  }
}

void CVI_AI_CopyObjectMeta(const cvai_object_t *src, cvai_object_t *dest) {
  CVI_AI_FreeCpp(dest);
  memset(dest, 0, sizeof(cvai_object_t));
  if (src->size > 0) {
    dest->size = src->size;
    dest->width = src->width;
    dest->height = src->height;
    dest->rescale_type = src->rescale_type;
    if (src->info) {
      dest->info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * src->size);
      memset(dest->info, 0, sizeof(cvai_object_info_t) * src->size);
      for (uint32_t fid = 0; fid < src->size; fid++) {
        CVI_AI_CopyObjectInfo(&src->info[fid], &dest->info[fid]);
      }
    }
  }
}

void CVI_AI_CopyHandPose(const cvai_handpose21_meta_t *src, cvai_handpose21_meta_t *dest) {
  dest->bbox_x = src->bbox_x;
  dest->bbox_y = src->bbox_y;
  dest->bbox_w = src->bbox_w;
  dest->bbox_h = src->bbox_h;
  dest->score = src->score;
  dest->label = src->label;
  memcpy(dest->x, src->x, sizeof(float) * 21);
  memcpy(dest->y, src->y, sizeof(float) * 21);
  memcpy(dest->xn, src->xn, sizeof(float) * 21);
  memcpy(dest->yn, src->yn, sizeof(float) * 21);
}

void CVI_AI_CopyHandPoses(const cvai_handpose21_meta_ts *src, cvai_handpose21_meta_ts *dest) {
  CVI_AI_FreeCpp(dest);
  memset(dest, 0, sizeof(cvai_handpose21_meta_ts));
  dest->width = src->width;
  dest->height = src->height;
  dest->size = src->size;
  if (src->size > 0) {
    dest->info = (cvai_handpose21_meta_t *)malloc(sizeof(cvai_handpose21_meta_t) * src->size);
    memset(dest->info, 0, sizeof(cvai_handpose21_meta_t) * src->size);
    for (uint32_t i = 0; i < src->size; i++) {
      CVI_AI_CopyHandPose(&src->info[i], &dest->info[i]);
    }
  }
}

void CVI_AI_CopyTrackerMeta(const cvai_tracker_t *src, cvai_tracker_t *dst) {
  if (src->size != dst->size) {
    CVI_AI_FreeCpp(dst);
  }
  dst->size = src->size;
  if (dst->size != 0) {
    dst->info = (cvai_tracker_info_t *)malloc(sizeof(cvai_tracker_info_t) * src->size);
    memcpy(dst->info, src->info, sizeof(cvai_tracker_info_t) * src->size);
  } else {
    dst->info = NULL;
  }
}

void CVI_AI_CopyImage(const cvai_image_t *src_image, cvai_image_t *dst_image) {
  if (dst_image->pix[0] != NULL) {
    LOGW("There are already data in destination image. (release them ...)");
    CVI_AI_FreeCpp(dst_image);
  }
  dst_image->pix_format = src_image->pix_format;
  dst_image->height = src_image->height;
  dst_image->width = src_image->width;

  uint32_t image_size = src_image->length[0] + src_image->length[1] + src_image->length[2];
  dst_image->pix[0] = (uint8_t *)malloc(image_size);
  memcpy(dst_image->pix[0], src_image->pix[0], image_size);
  for (int i = 0; i < 3; i++) {
    dst_image->stride[i] = src_image->stride[i];
    dst_image->length[i] = src_image->length[i];
    if (i != 0 && dst_image->length[i] != 0) {
      dst_image->pix[i] = dst_image->pix[i - 1] + dst_image->length[i - 1];
    }
  }
}

void CVI_AI_MapImage(VIDEO_FRAME_INFO_S *frame, bool *p_is_mapped) {
  *p_is_mapped = false;
  CVI_U32 frame_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
    *p_is_mapped = true;
  }
}
void CVI_AI_UnMapImage(VIDEO_FRAME_INFO_S *frame, bool do_unmap) {
  CVI_U32 frame_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame_size);
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
}
CVI_S32 CVI_AI_CopyVpssImage(VIDEO_FRAME_INFO_S *src_frame, cvai_image_t *dst_image) {
  if (src_frame->stVFrame.enPixelFormat != dst_image->pix_format) {
    LOGE("pixel format type not match,src:%d,dst:%d\n", (int)src_frame->stVFrame.enPixelFormat,
         (int)dst_image->pix_format);
    return CVI_FAILURE;
  }
  bool unmap = false;
  CVI_AI_MapImage(src_frame, &unmap);
  CVI_U32 frame_size = src_frame->stVFrame.u32Length[0] + src_frame->stVFrame.u32Length[1] +
                       src_frame->stVFrame.u32Length[2];
  memcpy(dst_image->pix[0], src_frame->stVFrame.pu8VirAddr[0], frame_size);
  CVI_AI_UnMapImage(src_frame, unmap);
  return CVI_SUCCESS;
}