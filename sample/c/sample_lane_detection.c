#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <model name> <model path> <input image path>\n", argv[0]);
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return -1;
  }
  int ret = 0;

  cvtdl_model_e enOdModelId = TDL_MODEL_LANE_DETECTION_LSTR;
  cvtdl_handle_t tdl_handle = CVI_TDL_CreateHandle(0);

  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, argv[1]);
  if (ret != 0) {
    printf("open lane model failed with %#x!\n", ret);
    goto exit0;
  }

  cvtdl_image_t image = CVI_TDL_ReadImage(argv[2]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  cvtdl_lane_t obj_meta = {0};
  ret = CVI_TDL_LaneDetection(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("CVI_TDL_LaneDetection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
        printf("None to detection\n");
    } else {
       for (int i = 0; i < obj_meta.size; i ++) {
            printf("lane %d\n", i);
            for (int j = 0; j < 2; j ++) {
                printf("%d: %f %f\n", j, obj_meta.lane[i].x[j], obj_meta.lane[i].y[j]);
            }
       }
    }
  }

  CVI_TDL_ReleaseLaneMeta(&obj_meta);
  CVI_TDL_DestroyImage(image);

exit1:
  CVI_TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
