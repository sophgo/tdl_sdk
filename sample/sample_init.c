#include "cviai.h"

#include <string.h>

int main(void) {
  CVI_S32 ret = CVIAI_SUCCESS;
  cviai_handle_t handle;
  // Create handle
  if ((ret = CVI_AI_CreateHandle(&handle)) != CVIAI_SUCCESS) {
    printf("Handle create failed\n");
    return ret;
  }

  // Get the used group ids by AI SDK.
  VPSS_GRP *groups = NULL;
  uint32_t nums = 0;
  if ((ret = CVI_AI_GetVpssGrpIds(handle, &groups, &nums)) != CVIAI_SUCCESS) {
    printf("Get used group id failed.\n");
    return ret;
  }
  printf("Used group id: %u\n", nums);
  for (uint32_t i = 0; i < nums; i++) {
    printf(" |- [%u] group id used: %u ", i, groups[i]);
  }
  printf("\n");
  free(groups);

  // Destroy handle.
  ret = CVI_AI_DestroyHandle(handle);
  return ret;
}