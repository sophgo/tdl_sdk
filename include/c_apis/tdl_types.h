#ifndef TDL_TYPES_H
#define TDL_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include "tdl_object_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *TDLHandle;
typedef void *TDLHandleEx;
typedef void *TDLImage;

typedef enum {
  TDL_TYPE_INT8 = 0, /**< Equals to int8_t. */
  TDL_TYPE_UINT8,    /**< Equals to uint8_t. */
  TDL_TYPE_INT16,    /**< Equals to int16_t. */
  TDL_TYPE_UINT16,   /**< Equals to uint16_t. */
  TDL_TYPE_INT32,    /**< Equals to int32_t. */
  TDL_TYPE_UINT32,   /**< Equals to uint32_t. */
  TDL_TYPE_BF16,     /**< Equals to bf16. */
  TDL_TYPE_FP16,     /**< Equals to fp16. */
  TDL_TYPE_FP32,     /**< Equals to fp32. */
  TDL_TYPE_UNKOWN    /**< Equals to unkown. */
} TDLDataTypeE;

typedef enum {
  IMAGE_GRAY = 0,
  IMAGE_RGB_PLANAR,
  IMAGE_RGB_PACKED,
  IMAGE_BGR_PLANAR,
  IMAGE_BGR_PACKED,
  IMAGE_YUV420SP_UV,  // NV12,semi-planar,one Y plane,one interleaved UV
                      // plane,size = width * height * 1.5
  IMAGE_YUV420SP_VU,  // NV21,semi-planar,one Y plane,one interleaved VU
                      // plane,size = width * height * 1.5
  IMAGE_YUV420P_UV,   // I420,planar,one Y plane(w*h),one U
                      // plane(w/2*h/2),one V plane(w/2*h/2),size = width *
                      // height * 1.5
  IMAGE_YUV420P_VU,   // YV12,size = width * height * 1.5
  IMAGE_YUV422P_UV,   // I422_16,size = width * height * 2
  IMAGE_YUV422P_VU,   // YV12_16,size = width * height * 2
  IMAGE_YUV422SP_UV,  // NV16,size = width * height * 2
  IMAGE_YUV422SP_VU,  // NV61,size = width * height * 2

  IMAGE_UNKOWN
} ImageFormatE;

typedef struct {
  float scale_x;
  float scale_y;
  float offset_x;
  float offset_y;
} TDLRescaleConfig;

typedef enum {
  TDL_REJECT = 0,
  TDL_GRABCUT = 1,
  TDL_COLOR = 2,
  TDL_FASTSAM = 3,
} TDLTargetSearchTypeE;
typedef struct {
  uint64_t *mem_addrs;
  uint32_t *mem_sizes;
  uint32_t size;
} TDLModelMemInfo;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
} TDLBox;

typedef struct {
  int8_t *ptr;
  uint32_t size;
  TDLDataTypeE type;
} TDLFeature;

typedef struct {
  uint32_t size;
  TDLFeature *feature;
} TDLFeatureInfo;

typedef struct {
  float *x;
  float *y;
  uint32_t size;
  float score;
} TDLPoints;

typedef struct {
  float x;
  float y;
  float score;
} TDLLandmarkInfo;

typedef struct {
  char name[128];
  TDLBox box;
  bool is_cross;
  bool falling;
  float score;
  int class_id;
  uint64_t track_id;  // track_id为0表示box得分小于跟踪阈值，尚未进行跟踪
  uint32_t landmark_size;
  TDLLandmarkInfo *landmark_properity;
  TDLObjectTypeE obj_type;
} TDLObjectInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  TDLObjectInfo *info;
} TDLObject;

typedef struct {
  char name[128];
  float score;
  uint64_t track_id;
  TDLBox box;
  TDLPoints landmarks;
  TDLFeature feature;

  float gender_score;
  float glass_score;
  float age;
  float liveness_score;
  float hardhat_score;
  float mask_score;
  float emotion_score;
  float recog_score;
  float face_quality;
  float pose_score;
  float blurness;
} TDLFaceInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  TDLFaceInfo *info;
} TDLFace;

typedef struct {
  int32_t class_id;
  float score;
} TDLClassInfo;

typedef struct {
  uint32_t size;

  TDLClassInfo *info;
} TDLClass;

typedef struct {
  float x;
  float y;
  float score;
} TDLKeypointInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  TDLKeypointInfo *info;
} TDLKeypoint;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t output_width;
  uint32_t output_height;
  uint8_t *class_id;
  uint8_t *class_conf;
} TDLSegmentation;

typedef struct {
  uint8_t *mask;
  float *mask_point;
  uint32_t mask_point_size;
  TDLObjectInfo *obj_info;
} TDLInstanceSegInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  uint32_t mask_width;
  uint32_t mask_height;
  TDLInstanceSegInfo *info;
} TDLInstanceSeg;

typedef struct {
  float x[2];
  float y[2];
  float score;
} TDLLanePoint;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  TDLLanePoint *lane;
  int lane_state;
} TDLLane;

typedef struct {
  int w;
  int h;
  float *logits;
} TDLDepthLogits;

typedef struct {
  uint64_t id;
  TDLBox bbox;
} TDLTrackerInfo;
typedef struct {
  uint32_t size;
  int out_num;
  TDLTrackerInfo *info;
} TDLTracker;

typedef struct {
  uint32_t size;
  char *text_info;
} TDLText;

/**
 * @brief VAD 段信息（毫秒）
 * start_ms: 段起始时间（ms）
 * end_ms: 段结束时间（ms），若为 -1 表示仍在进行中
 */
typedef struct {
  int32_t start_ms;
  int32_t end_ms;
} TDLVadSegment;

/**
 * @brief VAD 输出元数据
 * segments: 人声语音段段数组
 * has_speech: 输出是否检测到人声段
 * start_event: 是否存在流式检测语音段开始帧
 * end_event: 是否存在流式检测语音段结束帧
 */
typedef struct {
  uint32_t size;
  TDLVadSegment *segments;
  bool has_speech;
  bool start_event;
  bool end_event;
} TDLVAD;

typedef struct {
  float quality;
  uint64_t snapshot_frame_id;
  uint64_t track_id;
  uint64_t pair_track_id;
  int32_t registered_id;
  TDLObjectTypeE object_type;
  bool male;
  bool glass;
  uint8_t age;
  uint8_t emotion;
  TDLImage object_image;
  TDLBox ori_box;  // 相对于原图的坐标
  uint8_t *encoded_full_image;
  uint32_t full_length;
} TDLSnapshotInfo;

typedef struct {
  uint32_t snapshot_size;
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t source_width;
  uint32_t source_height;
  TDLFace face_meta;
  TDLObject person_meta;
  TDLObject pet_meta;
  TDLTracker track_meta;
  TDLSnapshotInfo *snapshot_info;
  TDLFeature *features;
  TDLImage image;
} TDLCaptureInfo;

typedef struct {
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t enter_num;
  uint32_t miss_num;
  int counting_line[4];
  TDLObject object_meta;
  TDLImage image;
} TDLObjectCountingInfo;

typedef struct {
  float awb[3];  // rgain, ggain, bgain
  float ccm[9];  // rgb[3][3]
  float blc;
} TDLIspMeta;

typedef struct {
  int r;
  int g;
  int b;
} color_rgb;

typedef struct {
  color_rgb color;
  uint32_t size;
} TDLBrush;

typedef struct {
  ImageFormatE dst_image_format;
  TDLDataTypeE dst_pixdata_type;
  int dst_width;
  int dst_height;
  int crop_x;
  int crop_y;
  int crop_width;
  int crop_height;
  float mean[3];
  float scale[3];  // Y=X*scale-mean
  bool keep_aspect_ratio;
} TDLPreprocessParams;
#ifdef __cplusplus
}
#endif
#endif
