# Using Custom AI with AI SDK

To use your custom AI with AI SDK, you need to know how AI SDK fills the result into structures. If you follow the format of the SDK, you'll be able to use your custom AI with the other modules in AI SDK.

## Putting Results into AI Structures

Here we will use ``cvai_face_t`` as an example.

```c
typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvai_face_info_t* info;
} cvai_face_t;
```

This is a structure for face. The length of the ``info`` is stored in ``size``. This is how you malloc a face info array.

```c
cvai_face_t faceMeta;
memset(&faceMeta, 0, sizeof(faceMeta));
faceMeta.size = 10;
faceMeta.info = (cvai_face_info_t*)malloc(faceMeta.size * sizeof(cvai_face_info_t));
```

The ``width`` and the ``height`` in ``cvai_face_t`` is the reference image size for ``cvai_bbox_t`` and ``cvai_pts_t`` in ``cvai_face_info_t``.

```c
typedef struct {
  char name[128];
  uint64_t unique_id;
  cvai_bbox_t bbox;
  cvai_pts_t face_pts;
  cvai_feature_t face_feature;
  cvai_face_emotion_e emotion;
  cvai_face_gender_e gender;
  cvai_face_race_e race;
  float age;
  float liveness_score;
  float mask_score;
  cvai_face_quality_t face_quality;
} cvai_face_info_t;
```

The rescale function in AI SDK will restore the coordinate of the box and the points by calculating the ratio of the ``width``, ``height`` and the new ``width``, ``height`` from ``VIDEO_FRAME_INFO_S``. Make sure to call rescale manually after you fill the data into ``cvai_face_t``.

```c
CVI_S32 CVI_AI_RescaleBBoxCenter(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
CVI_S32 CVI_AI_RescaleBBoxRB(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
```

Now let's take a look at ``cvai_face_info_t``.

```c
typedef struct {
  char name[128];
  uint64_t unique_id;
  cvai_bbox_t bbox;
  cvai_pts_t face_pts;
  cvai_feature_t face_feature;
  cvai_face_emotion_e emotion;
  cvai_face_gender_e gender;
  cvai_face_race_e race;
  float age;
  float liveness_score;
  float mask_score;
  cvai_face_quality_t face_quality;
} cvai_face_info_t;
```

In AI SDK, after face detection is finished, the following data will be available.

```c
  char name[128];
  cvai_bbox_t bbox;
  cvai_pts_t face_pts;
```

The following is the ``cvai_bbox_t`` and ``cvai_pts_t`` structures.

```c
typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
} cvai_bbox_t;

typedef struct {
  float* x;
  float* y;
  uint32_t size;
} cvai_pts_t;
```

This is how you malloc ``x`` and ``y`` for ``cvai_pts_t``.

```c
cvai_pts_t pts;
memset(&pts, 0, sizeof(pts));
pts.size = 5;
pts.x = (float*)malloc(pts.size * sizeof(float));
pts.y = (float*)malloc(pts.size * sizeof(float));
```


After face recognition, the following data will be available.

```c
  cvai_feature_t face_feature;
```

``cvai_feature_t`` is a structure for storing features. The ``size`` represents the number of features, and the ``feature_type_e`` stores the data type that ``ptr`` should cast to when use.

```c
typedef struct {
  int8_t* ptr;
  uint32_t size;
  feature_type_e type;
} cvai_feature_t;
```

This is how you malloc a ``ptr`` for ``cvai_feature_t``.

```c
cvai_feature_t feature;
memset(&feature, 0, sizeof(feature));
feature.size = 256;
feature.type = TYPE_INT8;
feature.ptr = (int8_t*)malloc(feature.size * sizeof(int8_t));
```

This is how you read or write a feature from ``cvai_feature_t``.

```c
if (feature.type == TYPE_INT8) {
  int8_t *ptr = (int8_t*)feature.ptr;
  for (uint32_t i = 0; i < feature.size; i++) {
    ptr[i] = i % 256;
  }
} else {
  // ... do something ...
}
```

The other data will be available if the corresponding API is called.

```c
  cvai_face_emotion_e emotion;  // Available if use face attribute.
  cvai_face_gender_e gender;  // Available if use face attribute.
  cvai_face_race_e race;  // Available if use face attribute.
  float age;  // Available if use face attribute.
  float liveness_score;  // Available if use liveness.
  float mask_score;  // Available if use mask classification.
  cvai_face_quality_t face_quality;  // Available if use face quality.
```

To free the structure, just simply call ``CVI_AI_Free``. It supports the following structures.

```c
#define CVI_AI_Free(X) _Generic((X),                   \
           cvai_feature_t*: CVI_AI_FreeFeature,        \
           cvai_pts_t*: CVI_AI_FreePts,                \
           cvai_tracker_t*: CVI_AI_FreeTracker,        \
           cvai_face_info_t*: CVI_AI_FreeFaceInfo,     \
           cvai_face_t*: CVI_AI_FreeFace,              \
           cvai_object_info_t*: CVI_AI_FreeObjectInfo, \
           cvai_object_t*: CVI_AI_FreeObject)(X)
```
