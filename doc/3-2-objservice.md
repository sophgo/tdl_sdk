# OBJService

OBJService is a module in AI SDK. It provides some handy functions to post-process the results from inference calls. The module also uses a handle.

```c
typedef void *cviai_objservice_handle_t;
```

A ``cviai_handle_t`` is required to create a OBJService handle.

```c
  cviai_handle_t handle;
  cviai_objservice_handle_t objs_handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle))!= CVI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }
  if ((ret = CVI_AI_OBJService_CreateHandle(&objs_handle, handle)) != CVI_SUCCESS) {
    printf("OBJService handle create failed\n");
    return ret;
  }

  // ...Do something...

  // Destroy handle.
  ret |= CVI_AI_OBJService_DestroyHandle(objs_handle);
  ret |= CVI_AI_DestroyHandle(handle);
  return ret;
```

Not every function inside OBJService requires a handle to work. ``CVI_AI_OBJService_DrawRect`` is a function that does not need the OBJService handle.

```c
CVI_S32 CVI_AI_OBJService_DrawRect(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame);
```

## Digital Zoom

Digital Zoom is an effect tool that zooms in to the largest detected bounding box in a frame. Users can create zoom-in effect easily with this tool.

``obj_skip_ratio`` is a threshold that skips the bounding box area smaller than the ``image_size * obj_skip_ratio``. ``trans_ratio`` is a value that controls the zoom-in speed. The recommended initial value for these two parameters are ``0.05f`` and ``0.1f``.

```c
CVI_S32 CVI_AI_OBJService_DigitalZoom(cviai_objservice_handle_t handle,
                                      const VIDEO_FRAME_INFO_S *inFrame, const cvai_object_t *meta,
                                      const float obj_skip_ratio, const float trans_ratio,
                                      VIDEO_FRAME_INFO_S *outFrame);
```

## Draw Rect

``CVI_AI_FRService_DrawRect`` is a function that draws all the bounding boxes and their tag names on the frame.

```c
CVI_S32 CVI_AI_OBJService_DrawRect(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame);
```
