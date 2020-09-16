# Quick Start

## Basic

AI SDK is a C style SDK with prefix ``CVI_AI_``. Let's take a quick example of creating an AI SDK handle. The handle is defined as ``void *`` just like typical C library.

```c
typedef void *cviai_handle_t;
```

Make sure to destroy the handle using ``CVI_AI_DestroyHandle`` to prevent memory leak.

```c
  cviai_handle_t handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle))!= CVI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }

  // ...Do something...

  // Destroy handle.
  ret = CVI_AI_DestroyHandle(handle);
  return ret;
```

Now we know how to create a handle, let's take a look at ``sample_init.c``. When setting a path to handle, it does not actually load the model, it only saves the path to the correspoding network list. You can also get the path you set from the handle. Make sure to free the path given by ``CVI_AI_GetModelPath`` to prevent memory leak.

```c
  const char fake_path[] = "face_quality.cvimodel";
  if ((ret = CVI_AI_SetModelPath(handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, fake_path)) !=
      CVI_SUCCESS) {
    printf("Set model path failed.\n");
    return ret;
  }
  char *path = NULL;
  if ((ret = CVI_AI_GetModelPath(handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, &path)) !=
      CVI_SUCCESS) {
    printf("Get model path failed.\n");
    return ret;
  }
  // Check if the two path are the same.
  if (strcmp(path, fake_path) != 0) {
    ret = CVI_FAILURE;
  }
  // Free the pointer created from CVI_AI_GetModelPath.
  free(path);
```

AI SDK use Vpss hardware to speed up the calculating time on images. Vpss API is a part of Middleware SDK. The SDK doesn't use handle system, but use IDs instead. One AI SDK handle can use multiple Vpss IDs, thus the return value of ``CVI_AI_GetVpssGrpIds`` is an array. Remember to free the array to prevent memory leak.

```c
  // Get the used group ids by AI SDK.
  uint32_t *groups = NULL;
  uint32_t nums = 0;
  if ((ret = CVI_AI_GetVpssGrpIds(handle, &groups, &nums)) != CVI_SUCCESS) {
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

You can also manually assign a group id to AI SDK when creating a handle.

```c
  VPSS_GRP groupId = 2;
  cviai_handle_t handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle, groupId))!= CVI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }
```