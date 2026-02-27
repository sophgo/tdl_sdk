#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include "meta_visualize.h"
#include "pthread_utils.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

static uint32_t g_frame_id = 0;

typedef enum { DETECTION_MODE, TRACKING_MODE } TrackingState;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -v <data_path> -m <sot_model_path> -d <det_model_path> -s "
      "<save_dir>\n",
      prog_name);
  printf(
      "  %s --data_path <path> --sot_model_path <path> --det_model_path "
      "<path> "
      "--save_dir <dir>\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -v, --data_path : The path to the MP4 video file (bm186x platform "
      "only) or the folder path to the image sequence (named as 000001.jpg, "
      "0000002.jpg, ...)\n"
      "  -m, --sot_model_path : sot model path\n"
      "  -d, --det_model_path : detection model path\n"
      "  -s, --save_dir : save dir\n"
      "  -h, --help : print help\n");
}

int main(int argc, char *argv[]) {
  char *data_path = NULL;
  char *sot_model_path = NULL;
  char *det_model_path = NULL;
  char *save_dir = NULL;

  struct option long_options[] = {
      {"data_path", required_argument, 0, 'v'},
      {"sot_model_path", required_argument, 0, 'm'},
      {"det_model_path", required_argument, 0, 'd'},
      {"save_dir", required_argument, 0, 's'},
      {"help", no_argument, 0, 'h'},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "v:m:d:s:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'v':
        data_path = optarg;
        break;
      case 'm':
        sot_model_path = optarg;
        break;
      case 'd':
        det_model_path = optarg;
        break;
      case 's':
        save_dir = optarg;
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      case '?':
        print_usage(argv[0]);
        return -1;
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (!data_path || !sot_model_path || !det_model_path || !save_dir) {
    fprintf(stderr,
            "Error: data_path, sot_model_path, det_model_path, and save_dir "
            "are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("data_path:     %s\n", data_path);
  printf("sot_model_path: %s\n", sot_model_path);
  printf("det_model_path: %s\n", det_model_path);
  printf("save_dir:       %s\n", save_dir);

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  int ret;

  // Open SOT Model
  TDLModel sot_model_id = TDL_MODEL_TRACKING_FEARTRACK;
  ret = TDL_OpenModel(tdl_handle, sot_model_id, sot_model_path, NULL, 0);
  if (ret != 0) {
    printf("open sot model failed with %#x!\n", ret);
    goto exit1;
  }

  // Open DET Model
  TDLModel det_model_id = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  ret = TDL_OpenModel(tdl_handle, det_model_id, det_model_path, NULL, 0);
  if (ret != 0) {
    printf("open det model failed with %#x!\n", ret);
    goto exit1;
  }

  TrackingState state = DETECTION_MODE;
  TDLTracker track_meta = {0};
  int match_counter = 0;
  int img_width = 1920;
  int img_height = 1080;
  // Define Center Box
  box_t center_box;
  int lost_frame_count = 0;

  while (true) {
    TDLImage image = GetVideoFrame(tdl_handle, data_path);
    if (image) {
      g_frame_id++;
      if (g_frame_id % 10 == 0) {
        printf("processing frame %d\n", g_frame_id);
      }

      if (state == DETECTION_MODE) {
        TDLObject obj_meta = {0};
        ret = TDL_Detection(tdl_handle, det_model_id, image, &obj_meta);
        if (ret != 0) {
          printf("TDL_Detection failed with %#x!\n", ret);
          TDL_DestroyImage(image);
          continue;
        }

        if (obj_meta.width > 0 && obj_meta.height > 0) {
          img_width = obj_meta.width;
          img_height = obj_meta.height;
        }
        center_box.x1 = img_width * 0.4;
        center_box.y1 = img_height * 0.38;
        center_box.x2 = img_width * 0.6;
        center_box.y2 = img_height * 0.6;

        // Check if any object center is in center_box
        bool found_match = false;
        for (int i = 0; i < obj_meta.size; i++) {
          float cx = (obj_meta.info[i].box.x1 + obj_meta.info[i].box.x2) / 2.0;
          float cy = (obj_meta.info[i].box.y1 + obj_meta.info[i].box.y2) / 2.0;
          if (cx >= center_box.x1 && cx <= center_box.x2 &&
              cy >= center_box.y1 && cy <= center_box.y2) {
            found_match = true;
            // Don't break, we need to check all for visualization later (or
            // just re-check loop) But for this flag, break is fine if we only
            // care about existence. However, we need to iterate for
            // visualization anyway. Let's just set the flag here and use a
            // helper or re-calc in loop for colors.
            break;
          }
        }

        if (found_match) {
          match_counter++;
          printf("Match found! Counter: %d\n", match_counter);
        } else {
          match_counter = 0;
        }

        if (match_counter >= 3) {
          int values[4];
          values[0] = (int)center_box.x1;
          values[1] = (int)center_box.y1;
          values[2] = (int)center_box.x2;
          values[3] = (int)center_box.y2;

          printf(
              "Match counter reached 3. Initializing tracking with CENTER box: "
              "[%d, %d, %d, %d]\n",
              values[0], values[1], values[2], values[3]);

          ret =
              TDL_SetSingleObjectTracking(tdl_handle, image, &obj_meta, values,
                                          4, g_frame_id, TDL_COLOR, NULL);
          if (ret == 0) {
            printf(
                "TDL_SetSingleObjectTracking successful. Initializing "
                "tracking...\n");
            state = TRACKING_MODE;
            match_counter = 0;
          } else {
            printf("TDL_SetSingleObjectTracking failed with %#x!\n", ret);
            match_counter = 0;
          }
        }

        // Prepare visualization
        int total_boxes = obj_meta.size + 1;
        box_t *boxes = (box_t *)malloc(total_boxes * sizeof(box_t));
        int *colors = (int *)malloc(total_boxes * 3 * sizeof(int));

        // Fill detections
        for (int i = 0; i < obj_meta.size; i++) {
          boxes[i].x1 = obj_meta.info[i].box.x1;
          boxes[i].y1 = obj_meta.info[i].box.y1;
          boxes[i].x2 = obj_meta.info[i].box.x2;
          boxes[i].y2 = obj_meta.info[i].box.y2;

          float cx = (boxes[i].x1 + boxes[i].x2) / 2.0;
          float cy = (boxes[i].y1 + boxes[i].y2) / 2.0;

          if (cx >= center_box.x1 && cx <= center_box.x2 &&
              cy >= center_box.y1 && cy <= center_box.y2) {
            // Yellow
            colors[i * 3 + 0] = 0;    // B
            colors[i * 3 + 1] = 255;  // G
            colors[i * 3 + 2] = 255;  // R
          } else {
            // Green
            colors[i * 3 + 0] = 0;    // B
            colors[i * 3 + 1] = 255;  // G
            colors[i * 3 + 2] = 0;    // R
          }
        }

        // Fill Center Box
        boxes[obj_meta.size] = center_box;
        if (found_match) {
          // Yellow
          colors[obj_meta.size * 3 + 0] = 0;    // B
          colors[obj_meta.size * 3 + 1] = 255;  // G
          colors[obj_meta.size * 3 + 2] = 255;  // R
        } else {
          // Green
          colors[obj_meta.size * 3 + 0] = 0;    // B
          colors[obj_meta.size * 3 + 1] = 255;  // G
          colors[obj_meta.size * 3 + 2] = 0;    // R
        }

        char outpath[128];
        snprintf(outpath, 128, "%s/%07d.jpg", save_dir, g_frame_id);
        VisualizeRectangle(boxes, total_boxes, image, outpath, colors);

        free(boxes);
        free(colors);

        TDL_ReleaseObjectMeta(&obj_meta);

      } else {  // TRACKING_MODE

        ret = TDL_SingleObjectTracking(tdl_handle, image, &track_meta,
                                       g_frame_id);
        if (ret != 0) {
          printf("TDL_SingleObjectTracking failed with %#x!\n", ret);
          state = DETECTION_MODE;
        } else {
          // getchar();
          if (track_meta.info) {
            lost_frame_count = 0;
            // Draw Tracked Box (Red) + Center Box (Red)
            box_t boxes[2];
            int colors[6];

            // Tracked Box (Red)
            boxes[0].x1 = track_meta.info[0].bbox.x1;
            boxes[0].y1 = track_meta.info[0].bbox.y1;
            boxes[0].x2 = track_meta.info[0].bbox.x2;
            boxes[0].y2 = track_meta.info[0].bbox.y2;
            colors[0] = 0;    // B
            colors[1] = 0;    // G
            colors[2] = 255;  // R

            // Center Box (Red)
            boxes[1].x1 = center_box.x1;
            boxes[1].y1 = center_box.y1;
            boxes[1].x2 = center_box.x2;
            boxes[1].y2 = center_box.y2;
            colors[3] = 0;    // B
            colors[4] = 0;    // G
            colors[5] = 255;  // R

            char outpath[128];
            snprintf(outpath, 128, "%s/%07d.jpg", save_dir, g_frame_id);
            VisualizeRectangle(boxes, 2, image, outpath, colors);
            TDL_ReleaseTrackMeta(&track_meta);
          } else {
            lost_frame_count++;
            if (lost_frame_count > 60) {
              printf(
                  "Tracking lost for 2 seconds. Switching to DETECTION "
                  "mode.\n");
              state = DETECTION_MODE;
              lost_frame_count = 0;
            } else {
              printf("Tracking temporarily lost (frame %d). Waiting...\n",
                     lost_frame_count);
            }

            box_t boxes[1];
            int colors[3];
            // Center Box (Red)
            boxes[0].x1 = center_box.x1;
            boxes[0].y1 = center_box.y1;
            boxes[0].x2 = center_box.x2;
            boxes[0].y2 = center_box.y2;
            colors[0] = 0;    // B
            colors[1] = 0;    // G
            colors[2] = 255;  // R

            char outpath[128];
            snprintf(outpath, 128, "%s/%07d.jpg", save_dir, g_frame_id);
            // Pass 0 boxes to just save image
            VisualizeRectangle(boxes, 1, image, outpath, colors);
          }
        }
      }

      TDL_DestroyImage(image);

    } else {
      printf("process done!\n");
      break;
    }
  }

exit1:
  TDL_CloseModel(tdl_handle, sot_model_id);
  TDL_CloseModel(tdl_handle, det_model_id);
  TDL_DestroyHandle(tdl_handle);

  return 0;
}