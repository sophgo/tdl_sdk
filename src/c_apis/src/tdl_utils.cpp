#include "tdl_utils.h"
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "encoder/image_encoder.hpp"
#include "tdl_sdk.h"
#include "tdl_type_internal.hpp"
#include "utils/tdl_log.hpp"

int32_t TDL_InitObjectMeta(TDLObject *object_meta, int num_objects,
                           int num_landmark) {
  if (object_meta->info == NULL) {
    object_meta->info =
        (TDLObjectInfo *)malloc(num_objects * sizeof(TDLObjectInfo));
    for (int i = 0; i < num_objects; i++) {
      object_meta->info[i].landmark_properity = NULL;
    }
  }

  for (int i = 0; i < num_objects; i++) {
    if (num_landmark > 0 && object_meta->info[i].landmark_properity == NULL) {
      object_meta->info[i].landmark_properity =
          (TDLLandmarkInfo *)malloc(num_landmark * sizeof(TDLLandmarkInfo));
    }
  }
  object_meta->size = num_objects;
  object_meta->width = 0;
  object_meta->height = 0;
  return 0;
}

int32_t TDL_ReleaseObjectMeta(TDLObject *object_meta) {
  for (size_t i = 0; i < object_meta->size; i++) {
    if (object_meta->info[i].landmark_properity) {
      free(object_meta->info[i].landmark_properity);
      object_meta->info[i].landmark_properity = NULL;
    }
  }

  if (object_meta->info) {
    free(object_meta->info);
    object_meta->info = NULL;
  }

  return 0;
}

int32_t TDL_InitInstanceSegMeta(TDLInstanceSeg *inst_seg_meta, int num_objects,
                                uint32_t mask_size) {
  if (inst_seg_meta->info != NULL) return 0;
  inst_seg_meta->info =
      (TDLInstanceSegInfo *)malloc(num_objects * sizeof(TDLInstanceSegInfo));

  memset(inst_seg_meta->info, 0, num_objects * sizeof(TDLInstanceSegInfo));

  for (int i = 0; i < num_objects; i++) {
    inst_seg_meta->info[i].mask = NULL;
    inst_seg_meta->info[i].mask_point = NULL;
    inst_seg_meta->info[i].mask_point_size = 0;
    inst_seg_meta->info[i].obj_info =
        (TDLObjectInfo *)malloc(sizeof(TDLObjectInfo));
  }

  inst_seg_meta->size = num_objects;
  inst_seg_meta->width = 0;
  inst_seg_meta->height = 0;
  inst_seg_meta->mask_width = 0;
  inst_seg_meta->mask_height = 0;
  return 0;
}

int32_t TDL_ReleaseInstanceSegMeta(TDLInstanceSeg *inst_seg_meta) {
  for (size_t i = 0; i < inst_seg_meta->size; i++) {
    if (inst_seg_meta->info[i].obj_info) {
      free(inst_seg_meta->info[i].obj_info);
      inst_seg_meta->info[i].obj_info = NULL;
    }
    if (inst_seg_meta->info[i].mask) {
      free(inst_seg_meta->info[i].mask);
      inst_seg_meta->info[i].mask = NULL;
    }
    if (inst_seg_meta->info[i].mask_point) {
      free(inst_seg_meta->info[i].mask_point);
      inst_seg_meta->info[i].mask_point = NULL;
    }
  }

  free(inst_seg_meta->info);
  inst_seg_meta->info = NULL;
  return 0;
}

int32_t TDL_InitFaceMeta(TDLFace *face_meta, int num_faces,
                         int num_landmark_per_face) {
  if (face_meta->info != NULL) return 0;
  face_meta->info = (TDLFaceInfo *)malloc(num_faces * sizeof(TDLFaceInfo));
  memset(face_meta->info, 0, num_faces * sizeof(TDLFaceInfo));

  if (num_landmark_per_face > 0) {
    for (int i = 0; i < num_faces; i++) {
      face_meta->info[i].landmarks.x =
          (float *)malloc(num_landmark_per_face * sizeof(float));
      face_meta->info[i].landmarks.y =
          (float *)malloc(num_landmark_per_face * sizeof(float));
    }
  }
  face_meta->size = num_faces;
  return 0;
}

int32_t TDL_ReleaseFaceMeta(TDLFace *face_meta) {
  for (size_t i = 0; i < face_meta->size; i++) {
    free(face_meta->info[i].landmarks.x);
    free(face_meta->info[i].landmarks.y);
    face_meta->info[i].landmarks.x = NULL;
    face_meta->info[i].landmarks.y = NULL;
  }
  free(face_meta->info);
  face_meta->info = NULL;
  return 0;
}

int32_t TDL_InitClassMeta(TDLClass *clas_meta, int num_classes) {
  if (clas_meta->info != NULL) return 0;
  clas_meta->info = (TDLClassInfo *)malloc(num_classes * sizeof(TDLClassInfo));
  memset(clas_meta->info, 0, num_classes * sizeof(TDLClassInfo));
  clas_meta->size = num_classes;
  return 0;
}

int32_t TDL_ReleaseClassMeta(TDLClass *clas_meta) {
  free(clas_meta->info);
  clas_meta->size = 0;
  return 0;
}

int32_t TDL_InitKeypointMeta(TDLKeypoint *keypoint_meta, int num_keypoints) {
  if (keypoint_meta->info) return 0;
  keypoint_meta->info =
      (TDLKeypointInfo *)malloc(num_keypoints * sizeof(TDLKeypointInfo));
  keypoint_meta->size = num_keypoints;
  keypoint_meta->width = 0;
  keypoint_meta->height = 0;
  return 0;
}

int32_t TDL_ReleaseKeypointMeta(TDLKeypoint *keypoint_meta) {
  free(keypoint_meta->info);
  keypoint_meta->info = NULL;
  return 0;
}

int32_t TDL_InitSemanticSegMeta(TDLSegmentation *seg_meta, int output_size) {
  seg_meta->class_id = NULL;
  seg_meta->class_conf = NULL;
  seg_meta->width = 0;
  seg_meta->height = 0;
  seg_meta->output_width = 0;
  seg_meta->output_height = 0;
  return 0;
}

int32_t TDL_ReleaseSemanticSegMeta(TDLSegmentation *seg_meta) {
  if (seg_meta->class_id) {
    free(seg_meta->class_id);
    seg_meta->class_id = NULL;
  }
  if (seg_meta->class_conf) {
    free(seg_meta->class_conf);
    seg_meta->class_conf = NULL;
  }

  return 0;
}

int32_t TDL_InitLaneMeta(TDLLane *lane_meta, int output_size) {
  if (lane_meta->lane) return 0;
  lane_meta->lane = (TDLLanePoint *)malloc(output_size * sizeof(TDLLanePoint));
  memset(lane_meta->lane, 0, output_size * sizeof(TDLLanePoint));

  lane_meta->size = output_size;
  return 0;
}

int32_t TDL_ReleaseLaneMeta(TDLLane *lane_meta) {
  if (lane_meta->lane != NULL) {
    free(lane_meta->lane);
    lane_meta->lane = NULL;
  }
  return 0;
}

int32_t TDL_InitCharacterMeta(TDLOcr *char_meta, int length) {
  if (char_meta->text_info) return 0;
  char_meta->text_info = (char *)malloc(length * sizeof(char));
  memset(char_meta->text_info, 0, length * sizeof(char));
  char_meta->size = length;
  return 0;
};

int32_t TDL_ReleaseCharacterMeta(TDLOcr *char_meta) {
  if (char_meta->text_info != NULL) {
    free(char_meta->text_info);
    char_meta->text_info = NULL;
  }
  return 0;
};

int32_t TDL_InitFeatureMeta(TDLFeature *feature_meta) {
  if (feature_meta->ptr != NULL) return 0;
  feature_meta->ptr = (int8_t *)malloc(sizeof(int8_t));
  return 0;
}

int32_t TDL_ReleaseFeatureMeta(TDLFeature *feature_meta) {
  if (feature_meta->ptr) {
    free(feature_meta->ptr);
    feature_meta->ptr = NULL;
  }
  return 0;
}

int32_t TDL_InitTrackMeta(TDLTracker *track_meta, int num_track) {
  if (track_meta->info) return 0;
  track_meta->info =
      (TDLTrackerInfo *)malloc(num_track * sizeof(TDLTrackerInfo));
  track_meta->out_num = num_track;
  return 0;
}

int32_t TDL_ReleaseTrackMeta(TDLTracker *track_meta) {
  free(track_meta->info);
  track_meta->info = NULL;
  return 0;
}

int32_t TDL_ReleaseCaptureInfo(TDLCaptureInfo *capture_info) {
  TDL_ReleaseFaceMeta(&capture_info->face_meta);
  TDL_ReleaseObjectMeta(&capture_info->person_meta);
  TDL_ReleaseObjectMeta(&capture_info->pet_meta);
  TDL_ReleaseTrackMeta(&capture_info->track_meta);

  for (uint32_t i = 0; i < capture_info->snapshot_size; i++) {
    if (capture_info->snapshot_info[i].object_image) {
      TDL_DestroyImage(capture_info->snapshot_info[i].object_image);
    }
    TDL_ReleaseFeatureMeta(&capture_info->features[i]);
  }
  free(capture_info->features);
  capture_info->features = NULL;

  free(capture_info->snapshot_info);
  capture_info->snapshot_info = NULL;

  capture_info->snapshot_size = 0;
  capture_info->frame_id = 0;
  capture_info->frame_width = 0;
  capture_info->frame_height = 0;
  return 0;
}

int32_t TDL_CaculateSimilarity(const TDLFeature feature1,
                               const TDLFeature feature2, float *similarity) {
  *similarity = 0;
  if (feature1.size != feature2.size) {
    LOGE("feature1.size is not equal to feature2.size");
    return -1;
  }
  float norm1 = 0;
  float norm2 = 0;
  for (size_t i = 0; i < feature1.size; i++) {
    *similarity += feature1.ptr[i] * feature2.ptr[i];
    norm1 += feature1.ptr[i] * feature1.ptr[i];
    norm2 += feature2.ptr[i] * feature2.ptr[i];
  }
  norm1 = sqrt(norm1);
  norm2 = sqrt(norm2);
  *similarity = *similarity / (norm1 * norm2);
  return 0;
}

int32_t TDL_NV21ToGray(TDLImage nv21_image, TDLImage *gray_image) {
  if (nv21_image == NULL) {
    LOGE("nv21_image is NULL");
    return -1;
  }
  TDLImageContext *nv21_image_context = (TDLImageContext *)nv21_image;
  TDLImageContext *gray_image_context = new TDLImageContext();

  gray_image_context->image = ImageFactory::createImage(
      nv21_image_context->image->getWidth(),
      nv21_image_context->image->getHeight(), ImageFormat::GRAY,
      TDLDataType::UINT8, true, InferencePlatform::AUTOMATIC);

  uint8_t *src = nv21_image_context->image->getVirtualAddress()[0];
  uint8_t *dst = gray_image_context->image->getVirtualAddress()[0];
  uint32_t src_stride = nv21_image_context->image->getStrides()[0];
  uint32_t dst_stride = gray_image_context->image->getStrides()[0];
  uint32_t height = nv21_image_context->image->getHeight();
  uint32_t width = nv21_image_context->image->getWidth();
  for (uint32_t i = 0; i < height; i++) {
    memcpy(dst + i * dst_stride, src + i * src_stride, width);
  }
  gray_image_context->image->flushCache();

  *gray_image = gray_image_context;
  return 0;
}

int32_t TDL_BGRPACKEDToGray(TDLImage bgr_packed_image, TDLImage *gray_image) {
  if (bgr_packed_image == NULL) {
    LOGE("bgr_packed_image is NULL");
    return -1;
  }
  TDLImageContext *bgr_packed_image_context =
      (TDLImageContext *)bgr_packed_image;
  TDLImageContext *gray_image_context = new TDLImageContext();

  cv::Mat bgr_mat;
  bool is_rgb;
  ImageFactory::convertToMat(bgr_packed_image_context->image, bgr_mat, is_rgb);
  cv::Mat gray_mat;
  cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
  gray_image_context->image = ImageFactory::createImage(
      gray_mat.cols, gray_mat.rows, ImageFormat::GRAY, TDLDataType::UINT8, true,
      InferencePlatform::AUTOMATIC);
  uint32_t stride = gray_image_context->image->getStrides()[0];
  uint8_t *ptr_dst = gray_image_context->image->getVirtualAddress()[0];
  uint8_t *ptr_src = gray_mat.data;

  for (int r = 0; r < gray_mat.rows; r++) {
    uint8_t *dst = ptr_dst + r * stride;
    memcpy(dst, ptr_src + r * gray_mat.step[0], gray_mat.cols);
  }
  gray_image_context->image->flushCache();

  *gray_image = gray_image_context;
  return 0;
}

int32_t TDL_GetGalleryFeature(const char *gallery_dir,
                              TDLFeatureInfo *feature_info,
                              int32_t feature_size) {
  char *files[100];
  int file_num = 0;
  for (int i = 0; i < 100; i++) {
    files[i] = (char *)malloc(128);
    sprintf(files[i], "%s/%d.bin", gallery_dir, i);
    if (access(files[i], F_OK) == -1) {
      break;
    }

    file_num++;
  }

  feature_info->feature = (TDLFeature *)malloc(file_num * sizeof(TDLFeature));
  feature_info->size = file_num;

  for (int i = 0; i < file_num; i++) {
    memset(&feature_info->feature[i], 0, sizeof(TDLFeature));
    feature_info->feature[i].size = feature_size;
    feature_info->feature[i].type = TDL_TYPE_INT8;
    feature_info->feature[i].ptr = (int8_t *)malloc(feature_size);

    FILE *fp = fopen(files[i], "rb");
    if (fp == NULL) {
      printf("read %s failed\n", files[i]);
      return -1;
    }

    fseek(fp, 0, SEEK_END);
    int len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (len != feature_size) {
      printf("feature size %d is not equal to %d\n", len, feature_size);
      return -1;
    }

    fread(feature_info->feature[i].ptr, 1, len, fp);
    fclose(fp);
    printf("read %s done,len:%d\n", files[i], len);
  }

  for (int j = 0; j < file_num; j++) {
    free(files[j]);
  }

  return 0;
}

int32_t TDL_EncodeFrame(TDLHandle handle, TDLImage image,
                        const char *img_path) {
  TDLContext *context = (TDLContext *)handle;

  if (!context->encoder) {
    printf("ImageEncoder not init!");
    context->encoder = std::make_shared<ImageEncoder>();
  }

  std::vector<uint8_t> encoded_data;

  TDLImageContext *image_context = (TDLImageContext *)image;

  bool ret = context->encoder->encodeFrame(image_context->image, encoded_data);

  if (!ret) {
    std::cerr << "Image encoding failed.\n";
    return -1;
  }

  std::ofstream ofs(std::string(img_path), std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open output file for writing.\n";
    return -1;
  }
  ofs.write(reinterpret_cast<const char *>(encoded_data.data()),
            encoded_data.size());
  ofs.close();
  return 0;
}