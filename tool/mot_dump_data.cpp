#include <inttypes.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "utils/mot_evaluation.hpp"
#include "utils/od.h"

#include <map>
#include <set>
#include <utility>

#include "unistd.h"
extern char *optarg;
extern int optind;
extern int opterr;
extern int optopt;

// #define ENABLE_DEEPSORT_EVALUATION

// #define OUTPUT_MOT_RESULT
#define OUTPUT_MOT_DATA

#define DEFAULT_RESULT_FILE_PATH "cviai_MOT_result.txt"
#define DEFAULT_DUMP_DATA_DIR "cviai_MOT_data"
#define DEFAULT_DUMP_DATA_INFO_NAME "MOT_data_info.txt"

typedef struct {
  TARGET_TYPE_e target_type;
  char *od_m_name;
  char *od_m_path;
  char *fd_m_path;
  char *reid_m_path;
  char *fr_m_path;
  float det_threshold;
  bool enable_DeepSORT;
  char *imagelist_file_path;
  char output_dir[128];
  char output_info_name[128];
  int inference_num;
} ARGS_t;

char *getFileName(char *path) {
  char *retVal = path, *p;
  for (p = path; *p; p++) {
    if (*p == '/' || *p == '\\' || *p == ':') {
      retVal = p + 1;
    }
  }
  return retVal;
}

void usage(char *bin_path) {
  printf(
      "Usage: %s [options]\n"
      "    <target type(=face|person|vehicle|pet)>\n"
      "    <object detection model name>\n"
      "    <object detection model path>\n"
      "    <face detection model path>\n"
      "    <reid model path>\n"
      "    <face recognition model path>\n"
      "    <imagelist path>\n"
      "\n"
      "options:\n"
      "    -t <threshold>     detection threshold (default: 0.5)\n"
      "    -n <number>        inference number (default: -1, inference all)\n"
      "    -d <dir>           dump data directory (default: %s)\n"
      "    -z                 enable DeepSORT (default: disable)\n"
      "    -h                 help\n",
      getFileName(bin_path), DEFAULT_DUMP_DATA_DIR);
}

CVI_S32 parse_args(int argc, char **argv, ARGS_t *args) {
  const char *OPT_STRING = "ht:n:d:z";
  const int ARGS_N = 7;
  /* set default argument value*/
  args->det_threshold = 0.5;
  args->enable_DeepSORT = false;
  args->inference_num = -1;
  sprintf(args->output_dir, "%s", DEFAULT_DUMP_DATA_DIR);
  sprintf(args->output_info_name, "%s", DEFAULT_DUMP_DATA_INFO_NAME);

  char ch;
  while ((ch = getopt(argc, argv, OPT_STRING)) != -1) {
    switch (ch) {
      case 'h': {
        usage(argv[0]);
        return CVI_FAILURE;
      } break;
      case 't': {
        args->det_threshold = atof(optarg);
      } break;
      case 'n': {
        args->inference_num = atoi(optarg);
      } break;
      case 'd': {
        sprintf(args->output_dir, "%s", optarg);
        // args->output_dir = optarg;
      } break;
      case 'z': {
        args->enable_DeepSORT = true;
      } break;
      case '?': {
        printf("error optopt: %c\n", optopt);
        printf("error opterr: %d\n", opterr);
        return CVI_FAILURE;
      }
    }
  }
  if (ARGS_N != (argc - optind)) {
    printf("Args number error (given %d, except %d)\n", (argc - optind), ARGS_N);
    return CVI_FAILURE;
  }
  int i = optind;
  if (CVI_SUCCESS != GET_TARGET_TYPE(&args->target_type, argv[i++])) {
    return CVI_FAILURE;
  }
  args->od_m_name = argv[i++];
  args->od_m_path = argv[i++];
  args->fd_m_path = argv[i++];
  args->reid_m_path = argv[i++];
  args->fr_m_path = argv[i++];
  args->imagelist_file_path = argv[i++];
  return CVI_SUCCESS;
}

void SHOW_ARGS(ARGS_t *args) {
  printf(
      "Target Type: %d\n"
      "Obj Detection Model Name  : %s\n"
      "Obj Detection Model Path  : %s\n"
      "Face Detection Model Path : %s\n"
      "ReID Model Path: %s\n"
      "Face Recognition Model Path : %s\n"
      "Detection Threshold : %f\n"
      "Enable DeepSORT: %s\n"
      "Imagelist File Path : %s\n"
      "Inference Number : %d\n"
      "Output Dir : %s\n"
      "Output Info Name : %s\n",
      args->target_type, args->od_m_name, args->od_m_path, args->fd_m_path, args->reid_m_path,
      args->fr_m_path, args->det_threshold, args->enable_DeepSORT ? "True" : "False",
      args->imagelist_file_path, args->inference_num, args->output_dir, args->output_info_name);
}

int main(int argc, char *argv[]) {
  ARGS_t args;
  if (CVI_SUCCESS != parse_args(argc, argv, &args)) {
    return CVI_FAILURE;
  }
  SHOW_ARGS(&args);

#if 1
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;

  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(args.od_m_name, &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);

  ret |= CVI_AI_OpenModel(ai_handle, od_model_id, args.od_m_path);
  ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, args.fd_m_path);
  ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, args.reid_m_path);
  ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, args.fr_m_path);
  if (ret != CVI_SUCCESS) {
    printf("model open failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, od_model_id, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, false);

  switch (args.target_type) {
    case FACE:
      break;
    case PERSON:
      CVI_AI_SelectDetectClass(ai_handle, od_model_id, 1, CVI_AI_DET_TYPE_PERSON);
      break;
    case VEHICLE:
    case PET:
    default:
      printf("not support target type[%d] now\n", args.target_type);
      return CVI_FAILURE;
  }
  CVI_AI_SetModelThreshold(ai_handle, od_model_id, args.det_threshold);
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, args.det_threshold);

#ifdef OUTPUT_MOT_RESULT
  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle, false);
  cvai_deepsort_config_t ds_conf;
  if (CVI_SUCCESS != GET_PREDEFINED_CONFIG(&ds_conf, args.target_type)) {
    printf("GET_PREDEFINED_CONFIG error!\n");
    return CVI_FAILURE;
  }
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, true);

  FILE *outFile_result;
  outFile_result = fopen(DEFAULT_RESULT_FILE_PATH, "w");
  if (outFile_result == NULL) {
    printf("There is a problem opening the output file: %s.\n", DEFAULT_RESULT_FILE_PATH);
    return CVI_FAILURE;
  }
#endif

  char outFile_data_path[256];
  sprintf(outFile_data_path, "%s/%s", args.output_dir, args.output_info_name);
  FILE *outFile_data;
  outFile_data = fopen(outFile_data_path, "w");
  if (outFile_data == NULL) {
    printf("There is a problem opening the output file: %s.\n", outFile_data_path);
    return CVI_FAILURE;
  }

  char text_buf[256];
  FILE *inFile = fopen(args.imagelist_file_path, "r");
  fscanf(inFile, "%s", text_buf);
  int img_num = atoi(text_buf);
  printf("Images Num: %d\n", img_num);

#ifdef OUTPUT_MOT_RESULT
  fprintf(outFile_result, "%u\n", img_num);
#endif

  fprintf(outFile_data, "%u %d\n", img_num, (int)args.enable_DeepSORT);

  cvai_object_t obj_meta;
  cvai_face_t face_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&face_meta, 0, sizeof(cvai_face_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

#ifdef OUTPUT_MOT_RESULT
  MOT_Evaluation mot_eval_data;
#endif

  for (int counter = 1; counter <= img_num; counter++) {
    if (counter == args.inference_num) {
      break;
    }
    fscanf(inFile, "%s", text_buf);
    printf("[%i] image path = %s\n", counter, text_buf);

    VIDEO_FRAME_INFO_S frame;
    CVI_AI_ReadImage(text_buf, &frame, PIXEL_FORMAT_RGB_888);

    switch (args.target_type) {
      case PERSON: {
        inference(ai_handle, &frame, &obj_meta);
        if (args.enable_DeepSORT) {
          CVI_AI_OSNet(ai_handle, &frame, &obj_meta);
        }
        CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, args.enable_DeepSORT);
      } break;
      case FACE: {
        CVI_AI_RetinaFace(ai_handle, &frame, &face_meta);
        if (args.enable_DeepSORT) {
          CVI_AI_FaceRecognition(ai_handle, &frame, &face_meta);
        }
        for (uint32_t i = 0; i < face_meta.size; i++) {
          printf("face[%u] bbox: x1[%.2f], y1[%.2f], x2[%.2f], y2[%.2f]\n", i,
                 face_meta.info[i].bbox.x1, face_meta.info[i].bbox.y1, face_meta.info[i].bbox.x2,
                 face_meta.info[i].bbox.y2);
        }
#ifdef OUTPUT_MOT_RESULT
#ifdef ENABLE_DEEPSORT_EVALUATION
        CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, enable_DeepSORT);
#else
        CVI_AI_DeepSORT_Face(ai_handle, &face_meta, &tracker_meta, false);
#endif
#endif
      } break;
      default:
        break;
    }

#ifdef OUTPUT_MOT_RESULT
    cvai_tracker_t inact_trackers;
    memset(&inact_trackers, 0, sizeof(cvai_tracker_t));
    CVI_AI_DeepSORT_GetTracker_Inactive(ai_handle, &inact_trackers);
    mot_eval_data.update(tracker_meta, inact_trackers);

    fprintf(outFile_result, "%u\n", tracker_meta.size);
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      cvai_bbox_t *target_bbox =
          (args.target_type == FACE) ? &face_meta.info[i].bbox : &obj_meta.info[i].bbox;
      uint64_t u_id =
          (args.target_type == FACE) ? face_meta.info[i].unique_id : obj_meta.info[i].unique_id;
      fprintf(outFile_result, "%" PRIu64 ",%d,%d,%d,%d,%d,%d,%d,%d,%d\n", u_id,
              (int)target_bbox->x1, (int)target_bbox->y1, (int)target_bbox->x2,
              (int)target_bbox->y2, tracker_meta.info[i].state, (int)tracker_meta.info[i].bbox.x1,
              (int)tracker_meta.info[i].bbox.y1, (int)tracker_meta.info[i].bbox.x2,
              (int)tracker_meta.info[i].bbox.y2);
    }
    fprintf(outFile_result, "%u\n", inact_trackers.size);
    for (uint32_t i = 0; i < inact_trackers.size; i++) {
      fprintf(outFile_result, "%" PRIu64 ",-1,-1,-1,-1,%d,%d,%d,%d,%d\n", inact_trackers.info[i].id,
              inact_trackers.info[i].state, (int)inact_trackers.info[i].bbox.x1,
              (int)inact_trackers.info[i].bbox.y1, (int)inact_trackers.info[i].bbox.x2,
              (int)inact_trackers.info[i].bbox.y2);
    }
    CVI_AI_Free(&inact_trackers);
#endif

    uint32_t bbox_size = (args.target_type == FACE) ? face_meta.size : obj_meta.size;
    fprintf(outFile_data, "%u\n", bbox_size);
    for (uint32_t i = 0; i < bbox_size; i++) {
      cvai_bbox_t *target_bbox =
          (args.target_type == FACE) ? &face_meta.info[i].bbox : &obj_meta.info[i].bbox;
      fprintf(outFile_data, "%f %f %f %f\n", target_bbox->x1, target_bbox->y1, target_bbox->x2,
              target_bbox->y2);
    }
    for (uint32_t i = 0; i < bbox_size; i++) {
      cvai_feature_t *target_feature =
          (args.target_type == FACE) ? &face_meta.info[i].feature : &obj_meta.info[i].feature;
      fprintf(outFile_data, "%u %d %s/feature/%d_%u.bin\n", target_feature->size,
              target_feature->type, args.output_dir, counter, i);
    }

    switch (args.target_type) {
      case FACE:
        CVI_AI_Free(&face_meta);
        break;
      case PERSON:
      case VEHICLE:
      case PET:
        CVI_AI_Free(&obj_meta);
        break;
      default:
        break;
    }
    CVI_AI_Free(&tracker_meta);
    CVI_AI_ReleaseImage(&frame);
  }

#ifdef OUTPUT_MOT_RESULT
  MOT_Performance_t mot_performance;
  mot_eval_data.summary(mot_performance);

  fclose(outFile_result);
#endif

  fclose(outFile_data);

  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
#endif
}
