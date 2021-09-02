# Core

## Multi-threading

We don't recommend to use the same ``VPSS_GRP`` in different threads, so we provide a function to change thread id for any model in one handle.

```c
// Auto assign group id
CVI_S32 CVI_AI_SetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             const uint32_t thread);
// Manually assign group id
CVI_S32 CVI_AI_SetVpssThread2(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                              const uint32_t thread, const VPSS_GRP vpssGroupId);
```

You can get the current thread id for any model.

```c
CVI_S32 CVI_AI_GetVpssThread(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, uint32_t *thread);
```

To get the relationship between thread and the ``VPSS_GRP`` thread uses, you can call ``CVI_AI_GetVpssGrpIds``. The output array ``groups`` gives all the used ``VPSS_GRP`` in order.

```c
  // Get the used group ids by AI SDK.
  uint32_t *groups = NULL;
  uint32_t nums = 0;
  if ((ret = CVI_AI_GetVpssGrpIds(handle, &groups, &nums)) != CVIAI_SUCCESS) {
    printf("Get used group id failed.\n");
    return ret;
  }
  printf("Used group id = %u:\n", nums);
  for (uint32_t i = 0; i < nums; i++) {
    printf("%u ", groups[i]);
  }
  printf("\n");
  free(groups);
```

## Reading Image

There are two ways to read image from file.

1. The ``CVI_AI_ReadImage`` function inside module ``core``
2. IVE library

### ``CVI_AI_ReadImage``

``CVI_AI_ReadImage`` uses ``VB_BLK`` from Middleware SDK. You must release ``VB_BLK`` with ``CVI_VB_ReleaseBlock`` to prevent memory leaks.

```c
  const char *path = "hi.jpg";
  VB_BLK blk;
  VIDEO_FRAME_INFO_S rgb_frame;
  CVI_S32 ret = CVI_AI_ReadImage(path, &blk, &rgb_frame, PIXEL_FORMAT_RGB_888);
  if (ret != CVIAI_SUCCESS) {
    printf("Read image failed with %#x!\n", ret);
    return ret;
  }

  // ...Do something...

  CVI_VB_ReleaseBlock(blk);
```

VI, VPSS, and ``VB_BLK`` shares the same memory pool. It is neccesarily to calculate the required space when initializing Middleware, or you can use the helper function provided by AI SDK.

```c
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  const CVI_S32 image_num = 12;
  // We allocate the pool with size of 24 * (RGB_PACKED_IMAGE in the size of 1920 * 1080)
  // The first four parameters are the input info, and the last four are the output info.
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888,
                         image_num, vpssgrp_width, vpssgrp_height,
                         PIXEL_FORMAT_RGB_888, image_num);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
```

### IVE library

IVE library does not occupied spaces in Middleware's memory pool. IVE uses a different image structure, so it must be converted to ``VIDEO_FRAME_INFO_S`` before use.

```c
  const char *path = "hi.jpg";
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  IVE_IMAGE_S image = CVI_IVE_ReadImage(ive_handle, path, IVE_IMAGE_TYPE_U8C3_PACKAGE);
  if (image.u16Width == 0) {
    printf("Read image failed with %x!\n", ret);
    return ret;
  }
  // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
  VIDEO_FRAME_INFO_S frame;
  ret = CVI_IVE_Image2VideoFrameInfo(&image, &frame, false);
  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }

  // ...Do something...

  // Free image and handles.
  CVI_SYS_FreeI(ive_handle, &image);
  CVI_IVE_DestroyHandle(ive_handle);
```

## Buffer to ``VIDEO_FRAME_INFO_S``

We provide a function ``CVI_AI_Buffer2VBFrame`` in AI SDK to help users to convert pure buffer to ``VIDEO_FRAME_INFO_S``. This works similar to ``CVI_AI_ReadImage``, so ``CVI_VB_ReleaseBlock`` is also required to free the ``VB_BLK``.

```c
CVI_S32 CVI_AI_Buffer2VBFrame(const uint8_t *buffer, uint32_t width, uint32_t height,
                              uint32_t stride, const PIXEL_FORMAT_E inFormat,
                              VB_BLK *blk, VIDEO_FRAME_INFO_S *frame,
                              const PIXEL_FORMAT_E outFormat);
```

## Model related settings

### Set model path

The model path must be set before the corresponding inference function is called.

```c
CVI_S32 CVI_AI_SetModelPath(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                            const char *filepath);
```

### Skip VPSS preprocess

If you already done the "scaling and quantization" step with the Middleware's binding mode, you can skip the VPSS preprocess with the following API for any model.

```c
CVI_S32 CVI_AI_SetSkipVpssPreprocess(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config, bool skip);
```

### Set model threshold

The threshold for any model can be set any time. If you set the threshold after the inference function, AI SDK will use the default threshold saved in the model.

```c
CVI_S32 CVI_AI_SetModelThreshold(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                             float threshold);
```

### Closing models

AI SDK will close the model for you if you destroy the handle, but we still provide the API to close the all the models or indivitually. This is because a user may run out of memory if a user want to switch between features while runtime, providing this API allows users to free spaces without destroying the handle.

```c
CVI_S32 CVI_AI_CloseAllModel(cviai_handle_t handle);

CVI_S32 CVI_AI_CloseModel(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config);
```

### Acquiring VPSS_CHN_ATTR_S from built-in models

The model will be loaded when the function is called. Make sure you complete your settings before calling this funciton. If you want to use VPSS binding mode instead of built-in VPSS instance inside AI SDK, you can get the VPSS_CHN_ATTR_S from the supported built-in models. If the model does not support exporting, ``CVIAI_FAILURE`` will return.

```c
CVI_S32 CVI_AI_GetVpssChnConfig(cviai_handle_t handle, CVI_AI_SUPPORTED_MODEL_E config,
                                const CVI_U32 frameWidth, const CVI_U32 frameHeight,
                                const CVI_U32 idx, cvai_vpssconfig_t *chnConfig);
```

### Inference calls

A model will be loaded when the function is called for the first time. The following functions are the example of the API calls.

```c
// Face recognition
CVI_S32 CVI_AI_FaceAttribute(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                             cvai_face_t *faces);
// Object detection
CVI_S32 CVI_AI_Yolov3(const cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj,
                      cvai_obj_det_type_e det_type);
```

Related sample codes: ``sample_read_fr.c``, ``sample_read_fr2``, ``sample_vi_od.c``

## Model output structures

The output result from a model stores in a apecific structure. Face related data stores in ``cvai_face_t``, and object related data stores in ``cvai_object_t``. THe code snippet shows how to use the structure.

```c
  cvai_face_t face;
  // memset before use.
  memset(&face, 0, sizeof(cvai_face_t));

  CVI_AI_FaceAttribute(handle, frame, &face);

  // Free to avoid memory leaks.
  CVI_AI_Free(&face);
```

``CVI_AI_Free`` is defined as a generic type, it supports the following structures.

```c
#ifdef __cplusplus
#define CVI_AI_Free(X) CVI_AI_FreeCpp(X)
#else
// clang-format off
#define CVI_AI_Free(X) _Generic((X),                   \
           cvai_feature_t*: CVI_AI_FreeFeature,        \
           cvai_pts_t*: CVI_AI_FreePts,                \
           cvai_face_info_t*: CVI_AI_FreeFaceInfo,     \
           cvai_face_t*: CVI_AI_FreeFace,              \
           cvai_object_info_t*: CVI_AI_FreeObjectInfo, \
           cvai_object_t*: CVI_AI_FreeObject)(X)
// clang-format on
#endif
```

These structures are defined in ``include/core/face/cvai_face_types.h``, and `` include/core/object/cvai_object_types.h``. The ``size`` is the length of the variable ``info``. The ``width`` and ``height`` stores in the structure are the reference size used by variable ``info``.

```c
typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvai_face_info_t* info;
} cvai_face_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvai_object_info_t *info;
} cvai_object_t;
```

### Reference size

The structure ``cvai_bbox_t`` stores in ``cvai_face_info_t`` and ``cvai_object_info_t`` are the bounding box of the found face or object. The coordinates store in ``cvai_bbox_t`` correspond to the ``width`` and ``height`` instead of the size of the input frame. Usually the ``width`` and ``height`` will equal to the size of the output model. The structure ``cvai_pts_t`` in ``cvai_face_info_t`` also refers to the ``width`` and ``height``.

```c
typedef struct {
  char name[128];
  cvai_bbox_t bbox;
  cvai_pts_t pts;
  cvai_feature_t feature;
  cvai_face_emotion_e emotion;
  cvai_face_gender_e gender;
  cvai_face_race_e race;
  float age;
  float liveness_score;
  float mask_score;
  cvai_face_quality_t face_quality;
} cvai_face_info_t;

typedef struct {
  char name[128];
  cvai_bbox_t bbox;
  int classes;
} cvai_object_info_t;
```

## Recovering coordinate

To get the coordinate correspond to the frame size, we provide two generic type APIs.

```c
#ifdef __cplusplus
#define CVI_AI_RescaleMetaCenter(videoFrame, X) CVI_AI_RescaleMetaCenterCpp(videoFrame, X)
#define CVI_AI_RescaleMetaRB(videoFrame, X) CVI_AI_RescaleMetaRBCpp(videoFrame, X)
#else
#define CVI_AI_RescaleMetaCenter(videoFrame, X) _Generic((X), \
           cvai_face_t*: CVI_AI_RescaleMetaCenterFace,        \
           cvai_object_t*: CVI_AI_RescaleMetaCenterObj)(videoFrame, X)
#define CVI_AI_RescaleMetaRB(videoFrame, X) _Generic((X),     \
           cvai_face_t*: CVI_AI_RescaleMetaRBFace,            \
           cvai_object_t*: CVI_AI_RescaleMetaRBObj)(videoFrame, X)
#endif
```

If you use the inference calls from AI SDK without VPSS binding mode, you can call ``CVI_AI_RescaleMetaCenter`` to recover the results. However, if you use VPSS binding mode instead of the VPSS inside the inference calls, the function you use depends on the ``stAspectRatio`` settings. If the setting is ``ASPECT_RATIO_AUTO``, you use ``CVI_AI_RescaleMetaCenter``. If the setting is ``ASPECT_RATIO_MANUAL``, you can call ``CVI_AI_RescaleMetaRB`` if you only pad the image
 in the direction of right or bottom.

```c
  cvai_face_t face;
  // memset before use.
  memset(&face, 0, sizeof(cvai_face_t));

  CVI_AI_FaceAttribute(handle, frame, &face);
  CVI_AI_RescaleMetaCenter(frame, &face);

  // Free to avoid memory leaks.
  CVI_AI_Free(&face);
```
