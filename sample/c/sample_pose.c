#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

static int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                       {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
                       {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
                       {1, 3},   {2, 4},   {3, 5},   {4, 6}};

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <model path> <input image path> <output image path>\n", argv[0]);
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    printf("output image path: Path to output image.\n");
    return -1;
  }
  int ret = 0;

  tdl_model_e enOdModelId = TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17;
  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId, argv[1]);
  if (ret != 0) {
    printf("open pose model failed with %#x!\n", ret);
    goto exit0;
  }

  tdl_image_t image = TDL_ReadImage(argv[2]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_object_t obj_meta = {0};
  ret = TDL_Detection(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Pose failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
        printf("None to detection\n");
    } else {
      point_t point[obj_meta.size * 17];
      box_t line[19];
      for (int i = 0; i < obj_meta.size; i++) {
          printf("obj_meta_index : %d, ", i);
          printf("class_id : %d, ", obj_meta.info[i].class_id);
          printf("score : %f, ", obj_meta.info[i].score);
          printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
                                            obj_meta.info[i].box.x2,
                                            obj_meta.info[i].box.y1,
                                            obj_meta.info[i].box.y2);
          for (int j = 0; j < 17; j ++) {
            printf("pose : %d: %f %f %f\n", j, obj_meta.info[i].landmark_properity[j].x,
                obj_meta.info[i].landmark_properity[j].y,
                obj_meta.info[i].landmark_properity[j].score);

            if (obj_meta.info[i].landmark_properity[j].score < 0.5) continue;
            point[i * 17 + j].x = obj_meta.info[i].landmark_properity[j].x;
            point[i * 17 + j].y = obj_meta.info[i].landmark_properity[j].y;
          }

          for (int k = 0; k < 19; k ++) {
            int kps1 = skeleton[k][0];
            int kps2 = skeleton[k][1];

            if (obj_meta.info[i].landmark_properity[kps1].score < 0.5 ||
                obj_meta.info[i].landmark_properity[kps2].score < 0.5) {
              continue;
            }

            line[k].x1 = obj_meta.info[i].landmark_properity[kps1].x;
            line[k].y1 = obj_meta.info[i].landmark_properity[kps1].y;
            line[k].x2 = obj_meta.info[i].landmark_properity[kps2].x;
            line[k].y2 = obj_meta.info[i].landmark_properity[kps2].y;
          }
      }
      TDL_VisualizePoint(point, obj_meta.size * 17, argv[2], argv[3]);
      TDL_VisualizeLine(line, 19, argv[3], argv[3]);
    }
  }

  TDL_ReleaseObjectMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
