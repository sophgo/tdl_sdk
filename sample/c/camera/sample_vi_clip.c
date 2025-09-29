#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "sample_utils.h"
#include "tdl_sdk.h"

#define WIDTH 960
#define HEIGHT 540

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "clip_image") != NULL) {
    *model_index = TDL_MODEL_FEATURE_CLIP_IMG;
  } else if (strstr(model_path, "clip_text") != NULL) {
    *model_index = TDL_MODEL_FEATURE_CLIP_TEXT;
  } else if (strstr(model_path, "mobileclip2_B_img") != NULL) {
    *model_index = TDL_MODEL_FEATURE_MOBILECLIP2_IMG;
  } else if (strstr(model_path, "mobileclip2_B_text") != NULL) {
    *model_index = TDL_MODEL_FEATURE_MOBILECLIP2_TEXT;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m <image_model_path> -n <text_model_path> -t <txt_dir> -c "
      "<vi_chn>\n",
      prog_name);
  printf(
      "  %s --image_model_path <path> --text_model_path <path> --txt_dir "
      "<path> --chn <vi_chn>\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --image_model_path   Path to clip image model\n");
  printf("  -n, --text_model_path   Path to clip text model\n");
  printf("  -t, --txt_dir      Path to txt directory\n");
  printf("  -c, --chn          Vi chn\n");
  printf("  -h, --help         Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *image_model_path = NULL;
  char *text_model_path = NULL;
  char *txt_dir = NULL;
  char *vi_chn = 0;
  int chn = 0;

  struct option long_options[] = {
      {"image_model_path", required_argument, 0, 'm'},
      {"text_model_path", required_argument, 0, 'n'},
      {"txt_dir", required_argument, 0, 't'},
      {"pipe", required_argument, 0, 'c'},
      {"help", no_argument, 0, 'h'},
      {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:n:i:t:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'm':
        image_model_path = optarg;
        break;
      case 'n':
        text_model_path = optarg;
        break;
      case 'c':
        vi_chn = optarg;
        break;
      case 't':
        txt_dir = optarg;
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

  if (vi_chn) {
    chn = atoi(vi_chn);
  }

  if (!image_model_path || !text_model_path || !txt_dir) {
    fprintf(stderr,
            "Error: image model path, text model path and txt directory are "
            "required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Image model path:    %s\n", image_model_path);
  printf("  Text model path:    %s\n", text_model_path);
  printf("  Txt directory:   %s\n", txt_dir);

  int ret = 0;
  TDLImage image = NULL;
  TDLFeature obj_meta = {0};
  TDLModel model_id;
  TDLHandle tdl_handle = TDL_CreateHandle(chn);

  if (get_model_info(image_model_path, &model_id) == -1) {
    printf("unsupported model: %s\n", image_model_path);
    return -1;
  }

  ret = InitCamera(tdl_handle, WIDTH, HEIGHT, IMAGE_YUV420SP_UV, 3);
  if (ret != 0) {
    printf("InitCamera %#x!\n", ret);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  ret = TDL_OpenModel(tdl_handle, model_id, image_model_path, NULL);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    DestoryCamera(tdl_handle);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  TDLModel model_id2;
  if (get_model_info(text_model_path, &model_id2) == -1) {
    printf("unsupported model: %s\n", text_model_path);
    DestoryCamera(tdl_handle);
    TDL_CloseModel(tdl_handle, model_id);
    TDL_DestroyHandle(tdl_handle);
    return -1;
  }
  ret = TDL_OpenModel(tdl_handle, model_id2, text_model_path, NULL);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    DestoryCamera(tdl_handle);
    TDL_CloseModel(tdl_handle, model_id);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }
  float *text_feature = NULL;
  int numSentences;
  int embedding_num;
  ret = TDL_ClipText(tdl_handle, model_id2, txt_dir, &text_feature,
                     &numSentences, &embedding_num);
  if (ret != 0) {
    printf("TDL_ClipText failed with %#x!\n", ret);
    DestoryCamera(tdl_handle);
    TDL_CloseModel(tdl_handle, model_id);
    TDL_CloseModel(tdl_handle, model_id2);
    TDL_DestroyHandle(tdl_handle);
    return ret;
  }

  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  printf("按任意键退出...\n");
  while (1) {
    fd_set rfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    int key_pressed = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (key_pressed > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
      break;
    }

    image = GetCameraFrame(tdl_handle, chn);
    if (image == NULL) {
      printf("GetCameraFrame failed\n");
      continue;
    }

    ret = TDL_FeatureExtraction(tdl_handle, model_id, image, &obj_meta);

    if (ret != 0) {
      printf("TDL_FeatureExtraction failed with %#x!\n", ret);
    } else {
      float *image_feature = (float *)(obj_meta.ptr);
      if (image_feature == NULL) {
        printf("image_feature is NULL\n");
      } else {
        int image_rows = 1;
        float *result = NULL;
        ret = TDL_ClipPostprocess(text_feature, numSentences, image_feature,
                                  image_rows, embedding_num, &result);

        if (ret != 0) {
          printf("TDL_ClipPostprocess failed with %#x!\n", ret);
        } else {
          printf("CLIP matching results: ");
          for (int i = 0; i < numSentences; i++) {
            printf("%.4f ", result[i]);
          }
          printf("\n");
        }
        free(result);
      }
      TDL_ReleaseFeatureMeta(&obj_meta);
    }

    TDL_ReleaseFeatureMeta(&obj_meta);
    ReleaseCameraFrame(tdl_handle, chn);
    TDL_DestroyImage(image);
    usleep(40 * 1000);  // Match Vi Frame Rate
  }

  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  TDL_CloseModel(tdl_handle, model_id);
  DestoryCamera(tdl_handle);
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
