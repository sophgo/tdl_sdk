# Quick Start

These are the files you may need to use the AI SDK.

``` shell
# Include header
cviai_experimental.h
cviai.h

# Static library
libcviai_core-static.a
libcviai_evaluation-static.a
libcviai_service-static.a

# Dynamic library
libcviai_core.so
libcviai_evaluation.so
libcviai_service.so
```

Include the headers in your ``*.cpp`` file.

```c
#include "cviai.h"

int main(void) {

  return 0;
}
```

Link the libraries to your binary.

| Function            | Linked libraries                      |
|---------------------|---------------------------------------|
| CVI_AI_*            | libcviai_core.so                      |
| CVI_AI_Service_*    | libcviai_core.so, libcviai_service.so |
| CVI_AI_Eval_*       | libcviai_evaluation.so                |

THe following snippet is a cmake example.

```cmake
project(sample_binary)
set(CVIAI_SDK_ROOT "/path-to-the-directory")
include_directories(${CVIAI_SDK_ROOT}/include)
add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} ${CVIAI_SDK_ROOT}/lib/libcviai_core.so)
```

## Basic

AI SDK is a C style SDK with prefix ``CVI_AI_``. Let's take a quick example of creating an AI SDK handle. The handle is defined as ``void *`` just like typical C library.

```c
typedef void *cviai_handle_t;
```

Make sure to destroy the handle using ``CVI_AI_DestroyHandle`` to prevent memory leak.

```c
  cviai_handle_t handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle))!= CVIAI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }

  // ...Do something...

  // Destroy handle.
  ret = CVI_AI_DestroyHandle(handle);
  return ret;
```

Now we know how to create a handle, let's take a look at ``sample_init.c``. When setting a path to handle, it does not actually load the model, it only saves the path to the correspoding network list. You can also get the path you set from the handle.

```c
  const char fake_path[] = "face_quality.cvimodel";
  if ((ret = CVI_AI_SetModelPath(handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, fake_path)) !=
      CVIAI_SUCCESS) {
    printf("Set model path failed.\n");
    return ret;
  }
  const char *path = CVI_AI_GetModelPath(handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY);

  // Check if the two path are the same.
  if (strcmp(path, fake_path) != 0) {
    ret = CVI_FAILURE;
  }
```

AI SDK use Vpss hardware to speed up the calculating time on images. Vpss API is a part of Middleware SDK. The SDK doesn't use handle system, but use IDs instead. One AI SDK handle can use multiple Vpss IDs, thus the return value of ``CVI_AI_GetVpssGrpIds`` is an array. Remember to free the array to prevent memory leak.

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

You can also manually assign a group id to AI SDK when creating a handle.

```c
  VPSS_GRP groupId = 2;
  cviai_handle_t handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle, groupId))!= CVIAI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }
```
