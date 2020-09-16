# FRService

FRService is a module in AI SDK. It provides some handy functions to post-process the results from inference calls. The module also uses a handle.

```c
typedef void *cviai_frservice_handle_t;
```

A ``cviai_handle_t`` is required to create a FRService handle.

```c
  cviai_handle_t handle;
  cviai_frservice_handle_t frs_handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle))!= CVI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }
  if ((ret = CVI_AI_FRService_CreateHandle(&frs_handle, handle)) != CVI_SUCCESS) {
    printf("FRService handle create failed\n");
    return ret;
  }

  // ...Do something...

  // Destroy handle.
  ret |= CVI_AI_FRService_DestroyHandle(frs_handle);
  ret |= CVI_AI_DestroyHandle(handle);
  return ret;
```

Not every function inside FRService requires a handle to work. ``CVI_AI_FRService_DrawRect`` is a function that does not need the FRService handle.

```c
CVI_S32 CVI_AI_FRService_DrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame);
```

## Feature Matching

FRService provides feature matching tool to analyze the result from model that generates feature such as Face Attribute. First, use ``CVI_AI_FRService_RegisterFeatureArray`` to register a feature array for output comparison.

```c
CVI_S32 CVI_AI_FRService_RegisterFeatureArray(cviai_frservice_handle_t handle,
                                              const cvai_frservice_feature_array_t featureArray);
```

Second, use ``CVI_AI_FRService_FaceInfoMatching`` to match to output result with the feature array. The length of the top ``index`` equals to ``k``.

```c
CVI_S32 CVI_AI_FRService_FaceInfoMatching(cviai_frservice_handle_t handle, const cvai_face_t *face,
                                          const uint32_t k, uint32_t **index);
```

The tool also provides raw feature comparison with the registered feature array. Currently only supports ``TYPE_INT8`` and ``TYPE_UINT8`` comparison.

```c
CVI_S32 CVI_AI_FRService_RawMatching(cviai_frservice_handle_t handle, const uint8_t *feature,
                                     const feature_type_e type, const uint32_t k, uint32_t **index);
```

## Digital Zoom

Digital Zoom is an effect tool that zooms in to the largest detected bounding box in a frame. Users can create zoom-in effect easily with this tool.

``face_skip_ratio`` is a threshold that skips the bounding box area smaller than the ``image_size * face_skip_ratio``. ``trans_ratio`` is a value that controls the zoom-in speed. The recommended initial value for these two parameters are ``0.05f`` and ``0.1f``.

```c
CVI_S32 CVI_AI_FRService_DigitalZoom(cviai_frservice_handle_t handle,
                                     const VIDEO_FRAME_INFO_S *inFrame, const cvai_face_t *meta,
                                     const float face_skip_ratio, const float trans_ratio,
                                     VIDEO_FRAME_INFO_S *outFrame);
```

## Draw Rect

``CVI_AI_FRService_DrawRect`` is a function that draws all the bounding boxes and their tag names on the frame.

```c
CVI_S32 CVI_AI_FRService_DrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame);
```
